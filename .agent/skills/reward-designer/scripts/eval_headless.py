#!/usr/bin/env python3
"""
eval_headless.py — Headless evaluation with RL training (PPO via RSL-RL).
 
This script trains a policy with PPO for N iterations, then evaluates it.
 
Usage:
    docker exec fluxa-isaacsim /isaac-sim/python.sh eval_headless.py \
        --reward-file /path/to/reward_fn.py \
        --num-envs 16 \
        --train-iterations 300 \
        --output /path/to/metrics.json
"""

import os
import sys
import json
import time
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
import omni.usd
import isaaclab.utils.assets as asset_utils
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

# Force-overwrite asset paths
asset_utils.NUCLEUS_ASSET_ROOT_DIR = S3_ROOT_50
asset_utils.NVIDIA_NUCLEUS_DIR     = S3_ROOT_50 + "/NVIDIA"
asset_utils.ISAAC_NUCLEUS_DIR      = ISAAC_DIR
asset_utils.ISAACLAB_NUCLEUS_DIR   = LAB_DIR
FRANKA_PANDA_CFG.spawn.usd_path = LAB_DIR + "/Robots/FrankaEmika/panda_instanceable.usd"

# Optimize CUDA performance
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False

# --- Domain randomization reset functions ----------------------------------------
def randomize_joint_friction_reset(env, env_ids, low: float, high: float, asset_cfg):
    """Sample a fresh joint friction value per env on reset."""
    asset = env.scene[asset_cfg.name]
    num_joints = asset.num_joints
    values = torch.empty(len(env_ids), num_joints, device=asset.device).uniform_(low, high)
    asset.write_joint_friction_coefficient_to_sim(values, env_ids=env_ids)


def randomize_joint_armature_reset(env, env_ids, low: float, high: float, asset_cfg):
    """Sample a fresh joint armature value per env on reset."""
    asset = env.scene[asset_cfg.name]
    num_joints = asset.num_joints
    values = torch.empty(len(env_ids), num_joints, device=asset.device).uniform_(low, high)
    asset.write_joint_armature_to_sim(values, env_ids=env_ids)


def randomize_joint_stiffness_scale_reset(env, env_ids, low: float, high: float, asset_cfg):
    """Scale default joint stiffness by a random factor on reset."""
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_joint_stiffness[env_ids].clone()
    scale = torch.empty(len(env_ids), 1, device=asset.device).uniform_(low, high)
    asset.write_joint_stiffness_to_sim(default * scale, env_ids=env_ids)


def randomize_joint_damping_scale_reset(env, env_ids, low: float, high: float, asset_cfg):
    """Scale default joint damping by a random factor on reset."""
    asset = env.scene[asset_cfg.name]
    default = asset.data.default_joint_damping[env_ids].clone()
    scale = torch.empty(len(env_ids), 1, device=asset.device).uniform_(low, high)
    asset.write_joint_damping_to_sim(default * scale, env_ids=env_ids)


# Map parameter names to their reset functions
DR_FUNC_MAP = {
    "joint_friction": randomize_joint_friction_reset,
    "joint_armature": randomize_joint_armature_reset,
    "joint_stiffness_scale": randomize_joint_stiffness_scale_reset,
    "joint_damping_scale": randomize_joint_damping_scale_reset,
}


def load_dr_config(dr_file_path):
    """Load a DR config .py file and extract parameter ranges."""
    dr_globals = {}
    with open(dr_file_path, 'r') as f:
        exec(f.read(), dr_globals)

    ranges = {}
    for name, val in dr_globals.items():
        if name.endswith('_range') and isinstance(val, list) and len(val) == 2:
            param_name = name[:-len('_range')]
            ranges[param_name] = [float(val[0]), float(val[1])]
    return ranges

# ---------------------------------------------------------------------------


def load_reward_code(reward_file_path):
    """Load and exec the generated reward file, returning reward_dict if it defines one."""
    reward_globals = {}
    with open(reward_file_path, 'r') as f:
        reward_code = f.read()
    
    # Execute the reward code to get function definitions and reward_dict
    exec(reward_code, reward_globals)
    return reward_globals.get("reward_dict", None)


def run_evaluation(args):
    """Run a short rollout and write metrics to a JSON file."""
    
    # RSL-RL imports (must happen after Kit is fully initialized)
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )

    # --- PPO Training Config ---
    @configclass
    class EurekaEvalPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        """Short PPO training config for Eureka reward evaluation.
    
        Not meant for full convergence — just enough iterations to
        differentiate good reward functions from bad ones.
        """
        seed: int = 42
        device: str = "cuda:0"
        num_steps_per_env: int = 24       # steps collected per env per PPO update
        max_iterations: int = 1000         # overridden by --train-iterations CLI arg
        empirical_normalization: bool = False
        save_interval: int = 9999         # effectively never save
        experiment_name: str = "eureka_eval"
        run_name: str = ""
        logger: str = "tensorboard"
    
        policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_hidden_dims=[64, 64],
            critic_hidden_dims=[64, 64],
            activation="elu",
        )
    
        algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.001,
            num_learning_epochs=8,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        )
    
    # --- Load reward function if provided ---
    reward_dict = None
    if args.reward_file and os.path.exists(args.reward_file):
        try:
            reward_dict = load_reward_code(args.reward_file)
            if reward_dict:
                print(f"Loaded reward_dict with {len(reward_dict)} terms from {args.reward_file}")
            else:
                print(f"No reward_dict found in {args.reward_file}, using defaults")
        except Exception as e:
            print(f"Failed to load reward file: {e}")
            # Write error metrics and exit
            write_metrics(args.output, {
                "mean_reward": 0.0,
                "success_rate": 0.0,
                "error": str(e),
                "status": "reward_load_error"
            })
            return

    # --- Build environment config ---
    @configclass
    class EvalFrankaReachEnvCfg(ReachEnvCfg):
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

    env_cfg = EvalFrankaReachEnvCfg()
    env_cfg.commands.ee_pose.debug_vis = False
    env_cfg.scene.ground.spawn.usd_path = ISAAC_DIR + "/Environments/Grid/default_environment.usd"

    if hasattr(env_cfg.scene, "table"):
        env_cfg.scene.table.spawn.usd_path = (
            ISAAC_DIR + "/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )

    # Patch rewards if we loaded a reward_dict
    if reward_dict is not None:
        env_cfg.rewards = reward_dict

    # --- Apply domain randomization if provided ---
    if args.dr_config and os.path.exists(args.dr_config):
        dr_ranges = load_dr_config(args.dr_config)
        print(f"Loaded {len(dr_ranges)} DR ranges from {args.dr_config}")

        for param_name, (low, high) in dr_ranges.items():
            if param_name not in DR_FUNC_MAP:
                print(f"  Warning: No DR function for '{param_name}', skipping")
                continue

            event_name = f"randomize_{param_name}"
            setattr(env_cfg.events, event_name, EventTerm(
                func=DR_FUNC_MAP[param_name],
                mode="reset",
                params={
                    "low": low,
                    "high": high,
                    "asset_cfg": SceneEntityCfg("robot"),
                },
            ))
            print(f"  Added DR event: {param_name} ∈ [{low}, {high}]")

    # --- Create environment ---
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # Wrap for RSL-RL
    env_wrapped = RslRlVecEnvWrapper(env)
    

    # Configure PPO runner
    agent_cfg = EurekaEvalPPORunnerCfg()
    agent_cfg.max_iterations = args.train_iterations
 
    # Create a temp log dir (runner needs one even if we don't save)
    log_dir = f"/tmp/eureka_eval_{os.getpid()}"
    os.makedirs(log_dir, exist_ok=True)
 
    # Convert configclass to dict for OnPolicyRunner
    runner_dict = {
        "seed": agent_cfg.seed,
        "device": agent_cfg.device,
        "num_steps_per_env": agent_cfg.num_steps_per_env,
        "max_iterations": agent_cfg.max_iterations,
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
 
    # --- Train ---
    print(f"Starting PPO training: {args.train_iterations} iterations, "
          f"{args.num_envs} envs, {agent_cfg.num_steps_per_env} steps/env/update")
 
    train_start = time.time()
 
    runner = OnPolicyRunner(env_wrapped, runner_dict, log_dir=log_dir, device="cuda:0")
    runner.learn(num_learning_iterations=args.train_iterations, init_at_random_ep_len=True)

    train_duration = time.time() - train_start
    print(f"Training complete in {train_duration:.1f}s")

    if args.save_policy:
        os.makedirs(os.path.dirname(args.save_policy), exist_ok=True)
        runner.save(args.save_policy)
        print(f"Policy saved to {args.save_policy}")
 
    # --- Clean up ---
    env.close()
 
    # Clean up temp log dir
    import shutil
    shutil.rmtree(log_dir, ignore_errors=True)
 
    # ── Write metrics ────────────────────────────────────────────────────────
    metrics = {
        "policy_checkpoint": args.save_policy,
        "train_iterations": args.train_iterations,
        "train_duration_seconds": train_duration,
        "status": "success",
    }
    write_metrics(args.output, metrics)
    print(f"Metrics written to {args.output}")
 
def write_metrics(output_path, metrics):
    """Write metrics dict to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless Isaac Sim PPO evaluation")
    parser.add_argument("--reward-file", type=str, default=None,
                        help="Path to generated reward_fn.py")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments (default: 16)")
    parser.add_argument("--train-iterations", type=int, default=300,
                        help="PPO training iterations (default: 300)")
    parser.add_argument("--output", type=str, default="/tmp/fluxa_metrics.json",
                        help="Path to write metrics JSON")
    parser.add_argument("--save-policy", type=str, default=None,
                        help="Path to save trained policy checkpoint")
    parser.add_argument("--dr-config", type=str, default=None,
                        help="Path to DR config .py file (applies randomization at reset time)")
    args = parser.parse_args()
 
    run_evaluation(args)
 
    # Exit cleanly
    simulation_app.close()