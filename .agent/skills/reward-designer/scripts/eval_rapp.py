#!/usr/bin/env python3
"""
eval_rapp.py — Evaluate a trained policy under a modified physics parameter.

For each (parameter, value) pair, this script:
  1. Creates the Franka reach environment with default physics
  2. Modifies ONE physics parameter to the test value
  3. Loads the trained policy checkpoint (no training)
  4. Runs inference for N steps
  5. Reports position error and success to a JSON file

Success criterion (matching DrEureka's approach):
  "Does the end-effector stay within X meters of the target on average?"
  This is a task-specific binary check.

Usage:
    /isaac-sim/python.sh eval_rapp.py \
        --checkpoint /tmp/eureka_policy.pt \
        --param-name joint_friction \
        --param-value 0.5 \
        --output /tmp/rapp_result.json
"""

import os
import sys
import json
import math
import argparse

# -- Asset path setup (must happen before any Isaac imports) ---
S3_ROOT_50 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
ISAAC_DIR  = S3_ROOT_50 + "/Isaac"
LAB_DIR    = ISAAC_DIR  + "/IsaacLab"

os.environ["NUCLEUS_ASSET_ROOT_DIR"] = S3_ROOT_50
os.environ["ISAAC_NUCLEUS_DIR"]      = ISAAC_DIR
os.environ["ISAACLAB_NUCLEUS_DIR"]   = LAB_DIR
os.environ["NVIDIA_NUCLEUS_DIR"]     = S3_ROOT_50 + "/NVIDIA"

# -- Launch Isaac Sim headless ----
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

# --- Import Isaac modules ---
import torch
import isaaclab.utils.assets as asset_utils
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

# Force-overwrite asset paths
asset_utils.NUCLEUS_ASSET_ROOT_DIR = S3_ROOT_50
asset_utils.NVIDIA_NUCLEUS_DIR     = S3_ROOT_50 + "/NVIDIA"
asset_utils.ISAAC_NUCLEUS_DIR      = ISAAC_DIR
asset_utils.ISAACLAB_NUCLEUS_DIR   = LAB_DIR
FRANKA_PANDA_CFG.spawn.usd_path = LAB_DIR + "/Robots/FrankaEmika/panda_instanceable.usd"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


# --- Physics parameter modification --------------------------------------------

def apply_parameter(env, param_name, param_value):
    """Modify a single physics parameter on the robot articulation.

    Parameters are modified at the PhysX level via Isaac Lab's articulation API.
    All other parameters remain at their default values.
    """
    robot = env.scene["robot"]
    num_envs = env.num_envs
    num_joints = robot.num_joints
    device = robot.device

    if param_name == "joint_friction":
        values = torch.full((num_envs, num_joints), param_value, device=device)
        robot.write_joint_friction_to_sim(values)

    elif param_name == "joint_armature":
        values = torch.full((num_envs, num_joints), param_value, device=device)
        robot.write_joint_armature_to_sim(values)

    elif param_name == "joint_stiffness_scale":
        default = robot.data.default_joint_stiffness.clone()
        robot.write_joint_stiffness_to_sim(default * param_value)

    elif param_name == "joint_damping_scale":
        default = robot.data.default_joint_damping.clone()
        robot.write_joint_damping_to_sim(default * param_value)

    else:
        raise ValueError(f"Unknown parameter: {param_name}")

    print(f"Applied {param_name} = {param_value}")


# --- Position error computation --------------------------------------------

def compute_position_error(env, robot, ee_body_idx):
    """Compute the distance between end-effector and commanded target.

    Returns mean position error across all environments (meters).
    """
    # End-effector position in world frame
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx, :]  # (num_envs, 3)

    # Convert to environment-local frame (each env has its own origin)
    ee_pos_local = ee_pos_w - env.scene.env_origins  # (num_envs, 3)

    # Commanded target position (in env-local frame)
    command = env.command_manager.get_command("ee_pose")
    target_pos = command[:, :3]  # (num_envs, 3)

    # Euclidean distance
    pos_error = torch.norm(ee_pos_local - target_pos, dim=-1)  # (num_envs,)
    return pos_error.mean().item()


# --- Main evaluation --------------------------------------------------------

def run_rapp_eval(args):
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )

    # --- PPO config (must match what was used in Stage 1 training) ---
    @configclass
    class RAPPEvalRunnerCfg(RslRlOnPolicyRunnerCfg):
        seed: int = 42
        device: str = "cuda:0"
        num_steps_per_env: int = 24
        max_iterations: int = 1
        empirical_normalization: bool = False
        save_interval: int = 9999
        experiment_name: str = "rapp_eval"
        run_name: str = ""
        logger: str = "tensorboard"

        policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[256, 128, 64],
            critic_hidden_dims=[256, 128, 64],
            activation="elu",
        )

        algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=3e-4,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )

    # --- Build environment (same config as eval_headless.py) ---
    @configclass
    class RAPPFrankaReachEnvCfg(ReachEnvCfg):
        def __post_init__(self):
            super().__post_init__()
            self.scene.robot = FRANKA_PANDA_CFG.replace(
                prim_path="/World/envs/env_.*/Robot"
            )
            self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
            self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
            self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]
            self.actions.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                scale=0.5,
                use_default_offset=True,
            )
            self.commands.ee_pose.body_name = "panda_hand"
            self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
            self.scene.num_envs = args.num_envs
            self.scene.env_spacing = 2.0

    env_cfg = RAPPFrankaReachEnvCfg()
    env_cfg.commands.ee_pose.debug_vis = False
    env_cfg.scene.ground.spawn.usd_path = ISAAC_DIR + "/Environments/Grid/default_environment.usd"
    if hasattr(env_cfg.scene, "table"):
        env_cfg.scene.table.spawn.usd_path = (
            ISAAC_DIR + "/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )

    # --- Create environment ---
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # --- Apply physics parameter modification ---
    if args.param_name != "default":
        apply_parameter(env, args.param_name, args.param_value)

    # --- Cache end-effector body index for position error computation ---
    robot = env.scene["robot"]
    ee_body_ids, _ = robot.find_bodies("panda_hand")
    ee_body_idx = ee_body_ids[0]

    # --- Wrap for RSL-RL and build runner ---
    env_wrapped = RslRlVecEnvWrapper(env)

    agent_cfg = RAPPEvalRunnerCfg()

    log_dir = f"/tmp/rapp_eval_{os.getpid()}"
    os.makedirs(log_dir, exist_ok=True)

    runner_dict = {
        "seed": agent_cfg.seed,
        "device": agent_cfg.device,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": 1,
        "empirical_normalization": agent_cfg.empirical_normalization,
        "obs_groups": {},
        "policy": {
            "class_name": agent_cfg.policy.class_name,
            "init_noise_std": agent_cfg.policy.init_noise_std,
            "actor_hidden_dims": agent_cfg.policy.actor_hidden_dims,
            "critic_hidden_dims": agent_cfg.policy.critic_hidden_dims,
            "activation": agent_cfg.policy.activation,
        },
        "algorithm": {
            "class_name": agent_cfg.algorithm.class_name,
            "value_loss_coef": agent_cfg.algorithm.value_loss_coef,
            "use_clipped_value_loss": agent_cfg.algorithm.use_clipped_value_loss,
            "clip_param": agent_cfg.algorithm.clip_param,
            "entropy_coef": agent_cfg.algorithm.entropy_coef,
            "num_learning_epochs": agent_cfg.algorithm.num_learning_epochs,
            "num_mini_batches": agent_cfg.algorithm.num_mini_batches,
            "learning_rate": agent_cfg.algorithm.learning_rate,
            "schedule": agent_cfg.algorithm.schedule,
            "gamma": agent_cfg.algorithm.gamma,
            "lam": agent_cfg.algorithm.lam,
            "desired_kl": agent_cfg.algorithm.desired_kl,
            "max_grad_norm": agent_cfg.algorithm.max_grad_norm,
        },
        "save_interval": agent_cfg.save_interval,
        "experiment_name": agent_cfg.experiment_name,
        "run_name": agent_cfg.run_name,
        "logger": agent_cfg.logger,
    }

    runner = OnPolicyRunner(env_wrapped, runner_dict, log_dir=log_dir, device="cuda:0")

    # --- Load trained policy checkpoint ---
    runner.load(args.checkpoint)
    print(f"Loaded checkpoint from {args.checkpoint}")

    policy = runner.get_inference_policy(device="cuda:0")

    # --- Run inference (no training) ---
    print(f"Running inference: {args.eval_steps} steps, {args.num_envs} envs")

    eval_steps = args.eval_steps
    total_position_error = 0.0
    total_reward = 0.0
    policy_obs = env_wrapped.get_observations()

    with torch.no_grad():
        for step in range(eval_steps):
            actions = policy(policy_obs)
            obs, rewards, dones, truncated, info = env.step(actions)
            policy_obs = env_wrapped.get_observations()

            total_reward += rewards.mean().item()
            total_position_error += compute_position_error(env, robot, ee_body_idx)

    mean_position_error = total_position_error / max(eval_steps, 1)
    mean_reward = total_reward / max(eval_steps, 1)

    # --- Task-specific success check ---
    # "Does the end-effector stay within threshold of the target?"
    success = mean_position_error < args.success_threshold

    # --- Clean up ---
    env.close()
    import shutil
    shutil.rmtree(log_dir, ignore_errors=True)

    # --- Write result ---
    result = {
        "param_name": args.param_name,
        "param_value": args.param_value,
        "mean_position_error": mean_position_error,
        "mean_reward": mean_reward,
        "success": success,
        "success_threshold": args.success_threshold,
        "eval_steps": eval_steps,
        "status": "success",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(result, f, indent=2)

    status_str = "PASS" if success else "FAIL"
    print(f"param={args.param_name} value={args.param_value} "
          f"pos_error={mean_position_error:.4f}m [{status_str}]")
    print(f"Result written to {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAPP single-parameter evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained policy checkpoint from Stage 1")
    parser.add_argument("--param-name", type=str, required=True,
                        help="Physics parameter to modify (or 'default' for baseline)")
    parser.add_argument("--param-value", type=float, default=0.0,
                        help="Value to set the parameter to")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=300,
                        help="Number of inference steps")
    parser.add_argument("--success-threshold", type=float, default=0.10,
                        help="Position error threshold in meters (default: 0.10m = 10cm)")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to write result JSON")
    args = parser.parse_args()

    run_rapp_eval(args)
    simulation_app.close()