#!/usr/bin/env python3
"""
Run manipulation tasks in Isaac Sim using Isaac Lab framework.

Supports three Fluxa pipeline injection points, all optional:
- --config:        Path to discovered_config.json from workspace-exploration.
                   Overrides Isaac Lab's hardcoded target sampling ranges with
                   the discovered workspace bounds.
- --reward-file:   Path to a Python file containing reward modifications
                   (future: produced by reward-designer Stage 1). The file's
                   code is executed after env_cfg is created and should
                   operate on `env_cfg`.
- --dr-config-file: Path to a Python file containing DR config modifications
                   (future: produced by reward-designer Stage 3). Same
                   contract as --reward-file.

If a path is not given (or missing), that injection is a no-op and Isaac
Lab defaults are used for that component.
"""

import argparse
import asyncio
import json
import os
import sys
import websockets

_SKILLS_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _SKILLS_DIR)
from common.schemas import DiscoveredConfig


RUN_TASK_CODE_TEMPLATE = """
import os

S3_ROOT_50 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
ISAAC_DIR = S3_ROOT_50 + "/Isaac"
LAB_DIR = ISAAC_DIR + "/IsaacLab"

os.environ["NUCLEUS_ASSET_ROOT_DIR"] = S3_ROOT_50
os.environ["ISAAC_NUCLEUS_DIR"] = ISAAC_DIR
os.environ["ISAACLAB_NUCLEUS_DIR"] = LAB_DIR
os.environ["NVIDIA_NUCLEUS_DIR"] = S3_ROOT_50 + "/NVIDIA"

import math
import time
import torch
import omni.usd
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG, UR10_CFG

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg


# 1. Config Definitions
@configclass
class FrankaReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{{ENV_REGEX_NS}}/Robot")
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.commands.ee_pose.body_name = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        self.scene.num_envs = {num_envs}
        self.scene.env_spacing = {env_spacing}

        # === Workspace bounds override (from workspace-exploration) ===
        pos_x_override = {pos_x_override}
        pos_y_override = {pos_y_override}
        pos_z_override = {pos_z_override}
        if pos_x_override is not None:
            self.commands.ee_pose.ranges.pos_x = pos_x_override
            self.commands.ee_pose.ranges.pos_y = pos_y_override
            self.commands.ee_pose.ranges.pos_z = pos_z_override
            print("[reach_task] OVERRIDE: workspace bounds from --config")
            print(f"  x={{pos_x_override}}")
            print(f"  y={{pos_y_override}}")
            print(f"  z={{pos_z_override}}")
        else:
            print("[reach_task] DEFAULT: using Isaac Lab built-in target ranges")
            print(f"  x={{self.commands.ee_pose.ranges.pos_x}}")
            print(f"  y={{self.commands.ee_pose.ranges.pos_y}}")
            print(f"  z={{self.commands.ee_pose.ranges.pos_z}}")


@configclass
class UR10ReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UR10_CFG.replace(prim_path="{{ENV_REGEX_NS}}/Robot")
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["ee_link"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["ee_link"]
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        )
        self.commands.ee_pose.body_name = "ee_link"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        self.scene.num_envs = {num_envs}
        self.scene.env_spacing = {env_spacing}

        # === Workspace bounds override (from workspace-exploration) ===
        pos_x_override = {pos_x_override}
        pos_y_override = {pos_y_override}
        pos_z_override = {pos_z_override}
        if pos_x_override is not None:
            self.commands.ee_pose.ranges.pos_x = pos_x_override
            self.commands.ee_pose.ranges.pos_y = pos_y_override
            self.commands.ee_pose.ranges.pos_z = pos_z_override
            print("[reach_task] OVERRIDE: workspace bounds from --config")
            print(f"  x={{pos_x_override}}")
            print(f"  y={{pos_y_override}}")
            print(f"  z={{pos_z_override}}")
        else:
            print("[reach_task] DEFAULT: using Isaac Lab built-in target ranges")


# 2. Execution Logic
task_name = "{task_name}"
task_configs = {{
    "franka-reach": FrankaReachEnvCfg,
    "franka-reach-play": FrankaReachEnvCfg,
    "ur10-reach": UR10ReachEnvCfg,
    "ur10-reach-play": UR10ReachEnvCfg,
}}

import isaaclab.utils.assets as asset_utils
asset_utils.NUCLEUS_ASSET_ROOT_DIR = S3_ROOT_50
asset_utils.NVIDIA_NUCLEUS_DIR = S3_ROOT_50 + "/NVIDIA"
asset_utils.ISAAC_NUCLEUS_DIR = ISAAC_DIR
asset_utils.ISAACLAB_NUCLEUS_DIR = LAB_DIR

FRANKA_PANDA_CFG.spawn.usd_path = LAB_DIR + "/Robots/FrankaEmika/panda_instanceable.usd"
UR10_CFG.spawn.usd_path = LAB_DIR + "/Robots/UniversalRobots/UR10/ur10_instanceable.usd"

env_cfg = task_configs[task_name]()

# === Joint-limits reset override (from workspace-exploration) ===
safe_config_path = {safe_config_path_override}
if safe_config_path is not None:
    import numpy as _np
    import torch as _torch
    from isaaclab.managers import EventTermCfg as _EventTerm, SceneEntityCfg as _SceneEntityCfg

    _safe = _torch.tensor(_np.load(safe_config_path), dtype=_torch.float32)

    def _reset_joints_from_safe_set(env, env_ids, asset_cfg, safe_configs):
        asset = env.scene[asset_cfg.name]
        safe = safe_configs.to(env.device)
        pick = _torch.randint(0, safe.shape[0], (len(env_ids),), device=env.device)
        q = safe[pick]
        asset.write_joint_state_to_sim(q, _torch.zeros_like(q), env_ids=env_ids)

    env_cfg.events.reset_robot_joints = _EventTerm(
        func=_reset_joints_from_safe_set, mode="reset",
        params={{"asset_cfg": _SceneEntityCfg("robot"), "safe_configs": _safe}},
    )
    print("[reach_task] OVERRIDE: joint reset from safe set,", _safe.shape[0], "configs")
else:
    print("[reach_task] DEFAULT: Isaac Lab built-in joint reset")
    
env_cfg.commands.ee_pose.debug_vis = True
env_cfg.scene.ground.spawn.usd_path = ISAAC_DIR + "/Environments/Grid/default_environment.usd"

if hasattr(env_cfg.scene, "table"):
    env_cfg.scene.table.spawn.usd_path = ISAAC_DIR + "/Props/Mounts/SeattleLabTable/table_instanceable.usd"

# Patches for USD path errors 
for marker_cfg in env_cfg.commands.ee_pose.goal_pose_visualizer_cfg.markers.values():
    if getattr(marker_cfg, "usd_path", None) and marker_cfg.usd_path.startswith("None"):
        marker_cfg.usd_path = marker_cfg.usd_path.replace("None", S3_ROOT_50, 1)

if hasattr(env_cfg.commands.ee_pose, "current_pose_visualizer_cfg"):
    for marker_cfg in env_cfg.commands.ee_pose.current_pose_visualizer_cfg.markers.values():
        if getattr(marker_cfg, "usd_path", None) and marker_cfg.usd_path.startswith("None"):
            marker_cfg.usd_path = marker_cfg.usd_path.replace("None", S3_ROOT_50, 1)

# === Reward modifications (from reward-designer Stage 1) ===
# Injected code, if any, should operate on `env_cfg`
# (e.g., env_cfg.rewards.<term>.weight = ...)
{reward_code}

# === DR config modifications (from reward-designer Stage 3) ===
# Injected code, if any, should operate on `env_cfg`
# (e.g., add EventTerm entries to env_cfg.events)
{dr_config_code}


if task_name in task_configs:
    print("Asset Path: " + FRANKA_PANDA_CFG.spawn.usd_path)
    print("Initializing " + task_name + "...")

    omni.usd.get_context().new_stage()

    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)

    print("✅ Environment Created. Starting Simulation Loop...")

    duration = {duration}
    start_time = time.time()
    obs, info = env.reset()

    try:
        while time.time() - start_time < duration:
            actions = 2.0 * torch.rand(env.num_envs, env.single_action_space.shape[0], device=env.device) - 1.0
            obs, rewards, dones, truncated, info = env.step(actions)
            time.sleep(0.01)
        print("🎯 Duration reached.")
    except Exception as e:
        print(f"❌ Error during loop: {{e}}")
    finally:
        print("📺 Task finished. Leaving stage active for inspection.")
else:
    print(f"❌ Task {task_name} not found.")
"""

def extract_template_overrides(config):
    """Return all template placeholder values as a dict."""
    overrides = {
        # workspace defaults
        "pos_x_override": "None",
        "pos_y_override": "None",
        "pos_z_override": "None",
        
        # joint limits defaults
        "safe_config_path_override": "None",
        
        # success threshold defaults (future)
        # controller gain defaults (future)
    }
    if config is None:
        return overrides

    ws = config.probes.workspace
    if ws is not None:
        overrides["pos_x_override"] = repr(ws.x)
        overrides["pos_y_override"] = repr(ws.y)
        overrides["pos_z_override"] = repr(ws.z)

    # joint limits, success threshold, controller gains: same pattern,
    # added here as each probe is implemented
    jl = config.probes.joint_limits
    if jl is not None:
        overrides["safe_config_path_override"] = repr(jl.safe_config_path)

    return overrides

def load_discovered_config(config_path):
    """
    Load and validate discovered_config.json.

    Returns:
        DiscoveredConfig | None
    """
    if not config_path:
        return None

    if not os.path.exists(config_path):
        print(f"⚠️ WARNING: --config path does not exist: {config_path}")
        return None

    with open(config_path) as f:
        return DiscoveredConfig.model_validate_json(f.read())

# For DrEureka stage of the pipeline
def load_injection_code(file_path, label):
    """Read a Python file for injection into the template.

    Returns a string of Python code. If file_path is missing or empty,
    returns a placeholder comment so the substituted code is a no-op.
    """
    if not file_path:
        return f"# No --{label}-file given — using Isaac Lab defaults."
    if not os.path.exists(file_path):
        print(f"⚠️  WARNING: --{label}-file path does not exist: {file_path}")
        return f"# {label} file missing at {file_path} — using defaults."

    with open(file_path) as f:
        code = f.read()
    print(f"   {label}: {file_path}")
    return f"# === Injected {label} from {file_path} ===\n{code}"


async def send_run_task_command(task_name, num_envs, env_spacing, duration,
                                config_file=None, reward_file=None,
                                dr_config_file=None,
                                host="localhost", port=8765):
    """Send run task command to Isaac Sim with optional injections."""
    uri = f"ws://{host}:{port}"

    # Load all three injections (each is a no-op if its path is None/missing)
    config = load_discovered_config(config_file)
    reward_code = load_injection_code(reward_file, "reward")
    dr_config_code = load_injection_code(dr_config_file, "dr-config")

    overrides = extract_template_overrides(config)
    code = RUN_TASK_CODE_TEMPLATE.format(
        task_name=task_name,
        num_envs=num_envs,
        env_spacing=env_spacing,
        duration=duration,
        reward_code=reward_code,
        dr_config_code=dr_config_code,
        **overrides,
    )

    command = {"type": "execute_python", "code": code}

    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(command))
            print(f"  Sent run task command: {task_name}")
            print(f"   Environments: {num_envs}")
            print(f"   Spacing: {env_spacing}m")
            print(f"   Duration: {duration}s")

            response = await websocket.recv()
            result = json.loads(response)

            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                print(f"Watch progress in WebRTC stream")
                print(f"Task will run for {duration} seconds")
                return True
            else:
                print(f"❌ Error: {result.get('message')}")
                return False

    except ConnectionRefusedError:
        print("❌ ERROR: Cannot connect to Isaac Sim command server")
        print("   Make sure Isaac Sim is running with command server on port 8765")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run manipulation tasks in Isaac Sim")
    parser.add_argument("--task", type=str, required=True,
                        choices=["franka-reach", "franka-reach-play",
                                 "ur10-reach", "ur10-reach-play"],
                        help="Task to run")
    parser.add_argument("--num-envs", type=int, default=16,
                        help="Number of parallel environments (default: 16)")
    parser.add_argument("--env-spacing", type=float, default=2.0,
                        help="Spacing between environments in meters (default: 2.0)")
    parser.add_argument("--duration", type=int, default=60,
                        help="Duration to run task in seconds (default: 60)")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)

    # === Fluxa pipeline injection paths (all optional) ===
    parser.add_argument("--config", type=str, default=None,
                        help="Path to discovered_config.json from workspace-exploration. "
                             "If given, workspace_bounds override Isaac Lab's default "
                             "target sampling ranges.")
    parser.add_argument("--reward-file", type=str, default=None,
                        help="Path to a Python file with reward modifications "
                             "(future: produced by reward-designer Stage 1).")
    parser.add_argument("--dr-config-file", type=str, default=None,
                        help="Path to a Python file with DR config modifications "
                             "(future: produced by reward-designer Stage 3).")

    args = parser.parse_args()

    if args.num_envs < 1:
        print("❌ ERROR: num-envs must be at least 1")
        sys.exit(1)
    if args.env_spacing < 0.5:
        print("⚠️  WARNING: env-spacing very small, environments may overlap")

    print(f"\n{'='*60}")
    print(f"Isaac Lab Manipulation Task")
    print(f"{'='*60}")
    print(f"Task: {args.task}")
    print(f"Environments: {args.num_envs}")
    print(f"Spacing: {args.env_spacing}m")
    print(f"Duration: {args.duration}s")
    if args.config:
        print(f"Workspace config: {args.config}")
    if args.reward_file:
        print(f"Reward file: {args.reward_file}")
    if args.dr_config_file:
        print(f"DR config file: {args.dr_config_file}")
    print(f"{'='*60}\n")

    success = asyncio.run(
        send_run_task_command(
            args.task, args.num_envs, args.env_spacing, args.duration,
            config_file=args.config,
            reward_file=args.reward_file,
            dr_config_file=args.dr_config_file,
            host=args.host, port=args.port,
        )
    )

    if success:
        print(f"\n✅ Task '{args.task}' started successfully!")
        print(f"Will run for {args.duration} seconds")
        print(f"View progress: WebRTC stream")