#!/usr/bin/env python3
"""
Run manipulation tasks in Isaac Sim using Isaac Lab framework
Supports multiple robots: Franka Panda, UR10
"""

import argparse
import asyncio
import websockets
import json
import sys

# Task execution code template
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

# 2. Execution Logic
task_name = "{task_name}"
task_configs = {{"franka-reach": FrankaReachEnvCfg, "franka-reach-play": FrankaReachEnvCfg, "ur10-reach": UR10ReachEnvCfg,
    "ur10-reach-play": UR10ReachEnvCfg,}}

# Sets the global path variables for Isaac Lab
import isaaclab.utils.assets as asset_utils

# FORCE-OVERWRITE all internal variables in the module
asset_utils.NUCLEUS_ASSET_ROOT_DIR = S3_ROOT_50
asset_utils.NVIDIA_NUCLEUS_DIR = S3_ROOT_50 + "/NVIDIA"
asset_utils.ISAAC_NUCLEUS_DIR = ISAAC_DIR
asset_utils.ISAACLAB_NUCLEUS_DIR = LAB_DIR

# Robot Configs
FRANKA_PANDA_CFG.spawn.usd_path = LAB_DIR + "/Robots/FrankaEmika/panda_instanceable.usd"
UR10_CFG.spawn.usd_path = LAB_DIR + "/Robots/UniversalRobots/UR10/ur10_instanceable.usd"

# Set the environment path
env_cfg = task_configs[task_name]()
env_cfg.commands.ee_pose.debug_vis = False
env_cfg.scene.ground.spawn.usd_path = ISAAC_DIR + "/Environments/Grid/default_environment.usd"

if hasattr(env_cfg.scene, "table"):
    env_cfg.scene.table.spawn.usd_path = ISAAC_DIR + "/Props/Mounts/SeattleLabTable/table_instanceable.usd"

if task_name in task_configs:
    print("Asset Path: " + FRANKA_PANDA_CFG.spawn.usd_path)
    print("Initializing " + task_name + "...")
    
    # Force a new stage to clear previous runs
    omni.usd.get_context().new_stage()
    
    from isaaclab.envs import ManagerBasedRLEnv
    env = ManagerBasedRLEnv(cfg=env_cfg)
    
    print("✅ Environment Created. Starting Simulation Loop...")
    
    duration = {duration}
    start_time = time.time()
    obs, info = env.reset()
    
    try:
        while time.time() - start_time < duration:
            # Generate random actions so the arms move jittery (proves it's working)
            actions = 2.0 * torch.rand(env.num_envs, env.single_action_space.shape[0], device=env.device) - 1.0
            obs, rewards, dones, truncated, info = env.step(actions)
            
            # Small sleep to allow the WebRTC stream to process frames
            time.sleep(0.01)
            
        print("🎯 Duration reached.")
    except Exception as e:
        print(f"❌ Error during loop: {{e}}")
    finally:
        # DO NOT call env.close() or simulation_app.close() 
        # to keep the stream alive for inspection
        print("📺 Task finished. Leaving stage active for inspection.")
else:
    print(f"❌ Task {task_name} not found.")
"""

async def send_run_task_command(task_name, num_envs, env_spacing, duration, 
                                 host="localhost", port=8765):
    """Send run task command to Isaac Sim"""
    
    uri = f"ws://{host}:{port}"
    
    # Generate Python code
    code = RUN_TASK_CODE_TEMPLATE.format(
        task_name=task_name,
        num_envs=num_envs,
        env_spacing=env_spacing,
        duration=duration
    )

    # Read generated reward code if provided
    reward_injection = ""
    if reward_file and os.path.exists(reward_file):
        with open(reward_file, 'r') as f:
            reward_injection = f.read()
        print(f"   Reward: {reward_file}")

    dr_injection = ""
    if dr_config_file and os.path.exists(dr_config_file):
        with open(dr_config_file, 'r') as f:
            dr_injection = f.read()
        print(f"   DR Config: {dr_config_file}")

    code = RUN_TASK_CODE_TEMPLATE.format(
        task_name=task_name,
        num_envs=num_envs,
        env_spacing=env_spacing,
        duration=duration,
        reward_code=reward_injection,
        dr_config_code=dr_injection,
    )
    
    command = {
        "type": "execute_python",
        "code": code
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(command))
            print(f"📤 Sent run task command: {task_name}")
            print(f"   Environments: {num_envs}")
            print(f"   Spacing: {env_spacing}m")
            print(f"   Duration: {duration}s")
            
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                print(f"🔍 Watch progress in WebRTC stream")
                print(f"⏱️  Task will run for {duration} seconds")
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
    parser.add_argument("--reward-file", type=str, default=None,
                    help="Path to generated reward_fn.py (on host)")
    parser.add_argument("--dr-config-file", type=str, default=None,
                    help="Path to generated dr_config.py (on host)")
    
    args = parser.parse_args()
    
    # Validate parameters
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
    print(f"{'='*60}\n")
    
    success = asyncio.run(
        send_run_task_command(
            args.task, args.num_envs, args.env_spacing, args.duration,
            args.host, args.port
        )
    )
    
    if success:
        print(f"\n✅ Task '{args.task}' started successfully!")
        print(f"⏱️  Will run for {args.duration} seconds")
        print(f"🌐 View progress: WebRTC stream")