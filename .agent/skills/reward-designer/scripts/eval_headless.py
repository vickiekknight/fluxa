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
import copy
import statistics
import traceback
from collections import defaultdict
from dataclasses import MISSING

# -- Asset path setup (must happen before any Isaac imports) ---
S3_ROOT_50 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
ISAAC_DIR  = S3_ROOT_50 + "/Isaac"
LAB_DIR    = ISAAC_DIR  + "/IsaacLab"

os.environ["NUCLEUS_ASSET_ROOT_DIR"] = S3_ROOT_50
os.environ["ISAAC_NUCLEUS_DIR"]      = ISAAC_DIR
os.environ["ISAACLAB_NUCLEUS_DIR"]   = LAB_DIR
os.environ["NVIDIA_NUCLEUS_DIR"]     = S3_ROOT_50 + "/NVIDIA"

# End-effector body used for the env config and the ground-truth task metric.
# TODO: should come from workspace-exploration's task_spec.json (ee_body_name),
# which already discovers it -- see the retrofit notes in SKILL.md.
EE_BODY_NAME = "panda_hand"

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
    """Exec the generated reward file and return its module namespace."""
    reward_globals = {}
    with open(reward_file_path, 'r') as f:
        reward_code = f.read()
    exec(reward_code, reward_globals)
    return reward_globals


def resolve_reward_dict(reward_globals, env_cfg, ee_body_name=EE_BODY_NAME):
    """Pull the reward term dict out of a generated reward module.

    reward_template.py places the generated reward_dict INSIDE get_reward_cfg(),
    so it is a function local, not a module global. Reading it off the module
    namespace therefore yields nothing, and the run silently falls back to Isaac
    Lab's default rewards -- which makes every Eureka candidate score the same
    and the whole search degenerate. Prefer the template's documented entry
    point, and fall back to a module-level dict for rewards written that way.
    """
    get_reward_cfg = reward_globals.get("get_reward_cfg")
    if callable(get_reward_cfg):
        return get_reward_cfg(env_cfg, ee_body_name)
    return reward_globals.get("reward_dict")


# --- Ground-truth task metric ----------------------------------------------
# The reach command is expressed in the ROBOT BASE frame, so the error has to be
# computed with combine_frame_transforms -- subtracting env_origins is wrong
# whenever the robot root is offset or rotated within its env. Rather than
# reimplement it, call Isaac Lab's own position_command_error, so the score
# Eureka selects on is the same number Isaac Lab reports.

def make_position_error_fn(env, ee_body_name=EE_BODY_NAME, command_name="ee_pose"):
    """Return a zero-arg closure giving per-env EE position error, in meters."""
    asset_cfg = SceneEntityCfg("robot", body_names=[ee_body_name])
    asset_cfg.resolve(env.scene)

    def position_error():
        return mdp.position_command_error(
            env, command_name=command_name, asset_cfg=asset_cfg
        )

    return position_error


def disable_reward_curriculum(env_cfg):
    """Drop curriculum terms that rewrite reward weights by name.

    Isaac Lab's reach curriculum calls modify_reward_weight on "action_rate" and
    "joint_vel" -- names belonging to the DEFAULT RewardsCfg. Once a generated
    reward dict replaces that config, two things go wrong: a name that no longer
    exists raises at env creation ("Reward term 'joint_vel' not found"), and a
    name that happens to collide gets its weight silently overwritten partway
    through training. Either way the candidate is not evaluated as the LLM wrote
    it, so these terms are dropped whenever a generated reward is installed.
    Setting a term to None is how Isaac Lab's ManagerBase skips it.
    """
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is None:
        return []

    dropped = []
    for term_name, term_cfg in list(vars(curriculum).items()):
        if term_cfg is None:
            continue
        params = getattr(term_cfg, "params", None) or {}
        if "term_name" in params:
            setattr(curriculum, term_name, None)
            dropped.append(term_name)
    return dropped


def needs_body_names(value):
    """Whether a SceneEntityCfg still has body_names unset.

    Tested by type, not identity. RewardsCfg leaves body_names as
    dataclasses.MISSING for the task config to fill in, but MISSING is a plain
    singleton with no __deepcopy__, so copy.deepcopy produces a NEW
    _MISSING_TYPE instance and `is MISSING` silently becomes False. That turned
    every task term into "object of type '_MISSING_TYPE' has no len()" and left
    the ground-truth reward as nothing but the two penalty terms.
    """
    return isinstance(value, type(MISSING))


def make_ground_truth_reward_fn(env, ee_body_name=EE_BODY_NAME):
    """Isaac Lab's hand-written reach reward, evaluated on the live env.

    The paper's headline evaluation metric is the correlation between the
    generated reward and the human-engineered one -- it is how Eureka argues its
    rewards align with (or beat) expert rewards. Reconstructing the baseline from
    Isaac Lab's own RewardsCfg, rather than hardcoding the five terms, means it
    tracks whatever Isaac Lab ships.

    Returns a zero-arg closure giving per-env ground-truth reward, or None if the
    baseline cannot be built (in which case correlation is simply not reported).
    """
    try:
        from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import (
            RewardsCfg,
        )
    except Exception as exc:
        print(f"  (ground-truth reward unavailable: {exc})")
        return None

    terms = []
    for name, term_cfg in vars(RewardsCfg()).items():
        if term_cfg is None or not hasattr(term_cfg, "func"):
            continue

        params, usable = {}, True
        for key, value in (term_cfg.params or {}).items():
            if isinstance(value, SceneEntityCfg):
                entity = copy.deepcopy(value)
                # RewardsCfg leaves body_names as MISSING for the task config to
                # fill in; do the same thing here.
                if needs_body_names(entity.body_names):
                    entity.body_names = [ee_body_name]
                try:
                    entity.resolve(env.scene)
                except Exception as exc:
                    print(f"  (ground-truth term {name!r} skipped: {exc})")
                    usable = False
                    break
                params[key] = entity
            else:
                params[key] = value

        if usable:
            terms.append((name, term_cfg.func, float(term_cfg.weight), params))

    if not terms:
        return None

    print(f"  ground-truth reward: {len(terms)} default terms "
          f"({', '.join(name for name, *_ in terms)})")

    def ground_truth_reward():
        total = None
        for _, func, weight, params in terms:
            value = func(env, **params) * weight
            total = value if total is None else total + value
        # RewardManager scales every term by the control dt, so applying it here
        # too puts the ground-truth signal on the same scale as env.reward_buf.
        # Correlation is scale-invariant, but this keeps the two curves directly
        # comparable when plotted together.
        return total * env.step_dt

    return ground_truth_reward


def reward_correlation(ground_truth, generated):
    """Pearson correlation between the two reward series over training.

    Mirrors the original's np.corrcoef(gt_reward, gpt_reward). Returns None when
    it is undefined -- fewer than two samples, or either series constant (a
    reward that never varies has no correlation to report, and numpy would warn
    and hand back nan).
    """
    if len(ground_truth) < 2 or len(ground_truth) != len(generated):
        return None

    gt = [float(v) for v in ground_truth]
    gen = [float(v) for v in generated]
    if not all(math.isfinite(v) for v in gt + gen):
        return None

    n = len(gt)
    mean_gt = sum(gt) / n
    mean_gen = sum(gen) / n
    covariance = sum((a - mean_gt) * (b - mean_gen) for a, b in zip(gt, gen))
    var_gt = sum((a - mean_gt) ** 2 for a in gt)
    var_gen = sum((b - mean_gen) ** 2 for b in gen)

    # A reward that never varies has no correlation to report.
    if var_gt < 1e-24 or var_gen < 1e-24:
        return None

    value = covariance / math.sqrt(var_gt * var_gen)
    return value if math.isfinite(value) else None


def mean_episode_infos(ep_infos):
    """Average each episode-info key across the collected episodes.

    Mirrors OnPolicyRunner.log's own aggregation, so the numbers recorded here
    are the ones that land in tensorboard.
    """
    if not ep_infos:
        return {}
    means = {}
    for key in ep_infos[0]:
        chunks = []
        for ep_info in ep_infos:
            value = ep_info.get(key)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.tensor([value], dtype=torch.float)
            chunks.append(value.float().flatten())
        if chunks:
            means[key] = float(torch.cat(chunks).mean().item())
    return means


def evaluate_policy(env_wrapped, policy, position_error_fn, eval_steps, success_threshold):
    """Deterministic rollout measuring the ground-truth task metric.

    success_rate is the fraction of (step, env) samples whose end-effector is
    within `success_threshold` of the current target. It is computed from
    simulator state and never from the generated reward, so a candidate cannot
    inflate its own score by reshaping its reward -- this is the analogue of
    Eureka's consecutive_successes, and it is what candidate selection ranks on.
    """
    step_mean_error = []
    step_success = []

    # The whole rollout -- reset, policy, step -- runs inside inference_mode.
    # RSL-RL collects its training rollouts that way, which permanently marks
    # the env's internal buffers as inference tensors; mutating them afterwards
    # outside inference_mode (env.reset() zeroing joint_acc, for one) raises
    # "Inplace update to inference tensor outside InferenceMode". Isaac Lab's
    # own play.py wraps policy and env.step together for the same reason.
    with torch.inference_mode():
        obs, _ = env_wrapped.reset()

        for _ in range(eval_steps):
            actions = policy(obs)
            obs, _, _, _ = env_wrapped.step(actions)

            err = position_error_fn()
            step_mean_error.append(err.mean().item())
            step_success.append((err < success_threshold).float().mean().item())

    n = max(len(step_mean_error), 1)
    return {
        "success_rate": float(sum(step_success) / n),
        "mean_position_error": float(sum(step_mean_error) / n),
        "final_position_error": float(step_mean_error[-1]) if step_mean_error else None,
        "min_position_error": float(min(step_mean_error)) if step_mean_error else None,
        "eval_steps": eval_steps,
        "success_threshold": success_threshold,
    }


def run_evaluation(args):
    """Train a policy on the candidate reward, then score it and write metrics."""
    
    # RSL-RL imports (must happen after Kit is fully initialized)
    from rsl_rl.runners import OnPolicyRunner
    from isaaclab_rl.rsl_rl import (
        RslRlOnPolicyRunnerCfg,
        RslRlPpoActorCriticCfg,
        RslRlPpoAlgorithmCfg,
        RslRlVecEnvWrapper,
    )

    class MetricsRecorder(OnPolicyRunner):
        """OnPolicyRunner that keeps the per-iteration training statistics.

        Eureka's feedback needs each reward term's trajectory across training,
        not one final scalar. The original scraped this out of stdout
        (construct_run_log); overriding log() reads the same numbers straight
        from the locals RSL-RL has already assembled, with no parsing.
        """

        def __init__(self, *args, task_metric_fn=None, gt_reward_fn=None,
                     raw_env=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.task_metric_fn = task_metric_fn
            self.gt_reward_fn = gt_reward_fn
            self.raw_env = raw_env
            self.recorded = {
                "iterations": [],
                "mean_reward": [],
                "mean_episode_length": [],
                "position_error": [],
                # Paired per-iteration samples of the two reward signals, for the
                # correlation the paper reports. Sampled as instantaneous means so
                # both are measured the same way; mean_reward above is an episode
                # return and is not comparable to the ground-truth signal.
                "gt_reward": [],
                "generated_reward": [],
                "reward_components": defaultdict(list),
            }

        def log(self, locs, width=80, pad=35):
            self._record(locs)
            super().log(locs, width, pad)

        def _record(self, locs):
            self.recorded["iterations"].append(int(locs["it"]))

            rewbuffer = locs.get("rewbuffer") or []
            if len(rewbuffer) > 0:
                self.recorded["mean_reward"].append(float(statistics.mean(rewbuffer)))

            lenbuffer = locs.get("lenbuffer") or []
            if len(lenbuffer) > 0:
                self.recorded["mean_episode_length"].append(float(statistics.mean(lenbuffer)))

            # Isaac Lab's RewardManager reports per-term episodic sums as
            # "Episode_Reward/<term_name>"; the term names are the keys of the
            # LLM's own reward_dict, which is exactly what the feedback needs.
            for key, value in mean_episode_infos(locs.get("ep_infos") or []).items():
                if key.startswith("Episode_Reward/"):
                    term = key[len("Episode_Reward/"):]
                    self.recorded["reward_components"][term].append(value)

            # Snapshot of the ground-truth metric at this point in training.
            # Noisy (the policy is still exploring), but it gives the LLM a
            # trajectory; the authoritative number is the post-training eval.
            if self.task_metric_fn is not None:
                try:
                    # Read-only, but kept in inference_mode for consistency with
                    # the rollout the buffers were produced under.
                    with torch.inference_mode():
                        self.recorded["position_error"].append(
                            float(self.task_metric_fn().mean().item())
                        )
                except Exception as exc:
                    print(f"  (task metric snapshot failed: {exc})")

            # Paired snapshot of both reward signals at the same instant.
            if self.gt_reward_fn is not None and self.raw_env is not None:
                try:
                    with torch.inference_mode():
                        gt = float(self.gt_reward_fn().mean().item())
                        generated = float(self.raw_env.reward_buf.mean().item())
                    self.recorded["gt_reward"].append(gt)
                    self.recorded["generated_reward"].append(generated)
                except Exception as exc:
                    print(f"  (ground-truth reward snapshot failed: {exc})")

    # --- PPO Training Config ---
    @configclass
    class EurekaEvalPPORunnerCfg(RslRlOnPolicyRunnerCfg):
        """Short PPO training config for Eureka reward evaluation.
    
        Not meant for full convergence — just enough iterations to
        differentiate good reward functions from bad ones.
        """
        seed: int = 42  # overridden from args below; see env_cfg.seed
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
    # Only exec'd here; the terms are resolved after env_cfg exists, because the
    # template's entry point takes env_cfg as an argument.
    reward_globals = None
    if args.reward_file and os.path.exists(args.reward_file):
        try:
            reward_globals = load_reward_code(args.reward_file)
        except Exception as e:
            print(f"Failed to load reward file: {e}")
            # Write error metrics and exit
            write_metrics(args.output, {
                "mean_reward": 0.0,
                "success_rate": 0.0,
                "reward_components": {},
                "position_error": [],
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "reward_load_error",
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
            self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [EE_BODY_NAME]
            self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [EE_BODY_NAME]
            self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [EE_BODY_NAME]
            self.actions.arm_action = mdp.JointPositionActionCfg(
                asset_name="robot",
                joint_names=["panda_joint.*"],
                scale=0.5,
                use_default_offset=True,
            )
            self.commands.ee_pose.body_name = EE_BODY_NAME
            self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
            self.scene.num_envs = args.num_envs
            self.scene.env_spacing = 2.0

    env_cfg = EvalFrankaReachEnvCfg()

    # Seed the ENVIRONMENT, not just the runner. Isaac Lab applies cfg.seed at
    # env init via configure_seed(), which covers torch, numpy, random and the
    # replicator -- and it happens before the policy is built, so the whole
    # evaluation becomes reproducible. rsl_rl's OnPolicyRunner never reads the
    # `seed` in its config dict (grep it: no occurrences), so the seed we were
    # passing there did nothing and every candidate was being trained
    # unseeded: "Environment seed : None" in the logs.
    env_cfg.seed = args.seed

    env_cfg.commands.ee_pose.debug_vis = False
    env_cfg.scene.ground.spawn.usd_path = ISAAC_DIR + "/Environments/Grid/default_environment.usd"

    if hasattr(env_cfg.scene, "table"):
        env_cfg.scene.table.spawn.usd_path = (
            ISAAC_DIR + "/Props/Mounts/SeattleLabTable/table_instanceable.usd"
        )

    # Patch rewards with the generated terms, now that env_cfg exists.
    if reward_globals is not None:
        try:
            reward_dict = resolve_reward_dict(reward_globals, env_cfg)
        except Exception as e:
            print(f"Failed to resolve reward terms: {e}")
            write_metrics(args.output, {
                "mean_reward": 0.0,
                "success_rate": 0.0,
                "reward_components": {},
                "position_error": [],
                "error": str(e),
                "traceback": traceback.format_exc(),
                "status": "reward_resolve_error",
            })
            return

        if not reward_dict:
            # Fail instead of falling back. A --reward-file was given, so the
            # caller wants THAT reward scored; silently training Isaac Lab's
            # hand-written one instead produces a plausible, good-looking
            # success_rate for a candidate that contributed nothing -- which is
            # exactly how a run once reported success while the generated reward
            # was never applied at all.
            message = (f"{args.reward_file} defines no usable reward terms "
                       f"(no reward_dict, or get_reward_cfg returned empty)")
            print(f"ERROR: {message}")
            write_metrics(args.output, {
                "mean_reward": 0.0,
                "success_rate": 0.0,
                "reward_components": {},
                "position_error": [],
                "error": message,
                "status": "no_reward_terms",
            })
            return

        env_cfg.rewards = reward_dict
        print(f"Loaded {len(reward_dict)} reward terms from {args.reward_file}: "
              f"{sorted(reward_dict)}")
        dropped = disable_reward_curriculum(env_cfg)
        if dropped:
            print(f"  Disabled default reward curriculum terms: {dropped}")

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
    agent_cfg.seed = args.seed
 
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
 
    position_error_fn = make_position_error_fn(env, ee_body_name=EE_BODY_NAME)
    gt_reward_fn = make_ground_truth_reward_fn(env, ee_body_name=EE_BODY_NAME)

    runner = MetricsRecorder(
        env_wrapped, runner_dict, log_dir=log_dir, device="cuda:0",
        task_metric_fn=position_error_fn,
        gt_reward_fn=gt_reward_fn,
        raw_env=env,
    )
    runner.learn(num_learning_iterations=args.train_iterations, init_at_random_ep_len=True)

    train_duration = time.time() - train_start
    print(f"Training complete in {train_duration:.1f}s")

    if args.save_policy:
        os.makedirs(os.path.dirname(args.save_policy), exist_ok=True)
        runner.save(args.save_policy)
        print(f"Policy saved to {args.save_policy}")
 
    # --- Ground-truth evaluation of the trained policy ---
    print(f"Evaluating trained policy: {args.eval_steps} steps, "
          f"success threshold {args.success_threshold}m")
    eval_metrics = evaluate_policy(
        env_wrapped,
        runner.get_inference_policy(device="cuda:0"),
        position_error_fn,
        args.eval_steps,
        args.success_threshold,
    )
    print(f"  success_rate={eval_metrics['success_rate']:.4f}  "
          f"mean_position_error={eval_metrics['mean_position_error']:.4f}m")

    # --- Clean up ---
    env.close()
 
    # Clean up temp log dir
    import shutil
    shutil.rmtree(log_dir, ignore_errors=True)
 
    # ── Write metrics ────────────────────────────────────────────────────────
    recorded = runner.recorded
    metrics = {
        # --- the contract 1_eureka.py consumes ---
        # mean_reward   : selection + logging       (1_eureka.py:289, 401)
        # success_rate  : candidate selection       (1_eureka.py:393)
        # reward_components / position_error
        #               : build_reward_reflection   (1_eureka.py:108, 127)
        "mean_reward": recorded["mean_reward"][-1] if recorded["mean_reward"] else 0.0,
        "success_rate": eval_metrics["success_rate"],
        "reward_components": dict(recorded["reward_components"]),
        "position_error": recorded["position_error"],

        # --- the paper's headline evaluation metric ---
        # How well the generated reward agrees with Isaac Lab's hand-written one.
        # Reported, never fed back: it is an evaluation of the search, not a
        # signal the search is allowed to optimise against.
        "reward_correlation": reward_correlation(
            recorded["gt_reward"], recorded["generated_reward"]),
        "gt_reward_curve": recorded["gt_reward"],
        "generated_reward_curve": recorded["generated_reward"],

        # --- full training curves (W&B / debugging) ---
        "mean_reward_curve": recorded["mean_reward"],
        "mean_episode_length": recorded["mean_episode_length"],
        "iterations": recorded["iterations"],

        # --- post-training evaluation detail ---
        **eval_metrics,

        # --- bookkeeping ---
        "policy_checkpoint": args.save_policy,
        "train_iterations": args.train_iterations,
        "train_duration_seconds": train_duration,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "status": "success",
    }
    write_metrics(args.output, metrics)
    print(f"Metrics written to {args.output}")
 
def sanitize_metrics(metrics):
    """Replace NaN/inf with null and flag the run as failed.

    A generated reward that divides by zero produces NaN rewards. Those must be
    detectable by the host -- both because json.dump would otherwise emit the
    non-standard NaN literal, and because such a candidate has to be discarded
    rather than silently ranked against the others.
    """
    found_nan = False

    def walk(value):
        nonlocal found_nan
        if isinstance(value, dict):
            return {k: walk(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [walk(v) for v in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            found_nan = True
            return None
        return value

    cleaned = walk(metrics)
    if found_nan:
        cleaned["status"] = "nan_reward"
        cleaned["success_rate"] = 0.0
        cleaned["mean_reward"] = 0.0
    return cleaned


def write_metrics(output_path, metrics):
    """Write metrics dict to JSON file, with NaN/inf sanitised out."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(sanitize_metrics(metrics), f, indent=2)
 
 
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
    parser.add_argument("--eval-steps", type=int, default=360,
                        help="Inference steps for the post-training evaluation "
                             "(default: 360 = one 12s reach episode at 30Hz)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed for the environment and policy (default: 42). "
                             "Applied via env_cfg.seed, which Isaac Lab uses to "
                             "seed torch/numpy/random at env init.")
    parser.add_argument("--success-threshold", type=float, default=0.05,
                        help="EE position error counted as a success, in meters "
                             "(default: 0.05). Should come from "
                             "workspace-exploration's success_threshold probe.")
    args = parser.parse_args()

    try:
        run_evaluation(args)
    except Exception:
        # Write the traceback into the metrics file rather than only dying: the
        # host needs it to build execution-error feedback for the LLM, which is
        # how the original Eureka repairs broken candidates.
        tb = traceback.format_exc()
        print(tb)
        write_metrics(args.output, {
            "mean_reward": 0.0,
            "success_rate": 0.0,
            "reward_components": {},
            "position_error": [],
            "traceback": tb,
            "status": "exception",
        })
        # Hard exit. simulation_app.close() reliably hangs once Kit has taken an
        # exception, which made the host wait out its full timeout for a
        # candidate whose verdict was already written. The metrics file is
        # flushed above and there is nothing left worth cleaning up.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    finally:
        # Exit cleanly
        simulation_app.close()