import os, json, time, math, asyncio, websockets, torch

S3_ROOT_50 = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.0"
ISAAC_DIR  = S3_ROOT_50 + "/Isaac"
LAB_DIR    = ISAAC_DIR  + "/IsaacLab"

os.environ["NUCLEUS_ASSET_ROOT_DIR"] = S3_ROOT_50
os.environ["ISAAC_NUCLEUS_DIR"]      = ISAAC_DIR
os.environ["ISAACLAB_NUCLEUS_DIR"]   = LAB_DIR
os.environ["NVIDIA_NUCLEUS_DIR"]     = S3_ROOT_50 + "/NVIDIA"

import omni.usd
import isaaclab.utils.assets as asset_utils
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg
from isaaclab.envs import ManagerBasedRLEnv

asset_utils.NUCLEUS_ASSET_ROOT_DIR = S3_ROOT_50
asset_utils.NVIDIA_NUCLEUS_DIR     = S3_ROOT_50 + "/NVIDIA"
asset_utils.ISAAC_NUCLEUS_DIR      = ISAAC_DIR
asset_utils.ISAACLAB_NUCLEUS_DIR   = LAB_DIR

FRANKA_PANDA_CFG.spawn.usd_path = LAB_DIR + "/Robots/FrankaEmika/panda_instanceable.usd"

# ── Inject generated reward function ──────────────────────────────────────────
FLUXA_REWARD_CODE
# ─────────────────────────────────────────────────────────────────────────────

@configclass
class EvalFrankaReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="/World/envs/env_.*/Robot")
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = ["panda_hand"]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = ["panda_hand"]
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.commands.ee_pose.body_name    = "panda_hand"
        self.commands.ee_pose.ranges.pitch = (math.pi, math.pi)
        self.scene.num_envs   = FLUXA_NUM_ENVS
        self.scene.env_spacing = 2.0
        # Patch rewards if the generated reward_dict exists in scope
        if "reward_dict" in dir():
            self.rewards = reward_dict

omni.usd.get_context().new_stage()
env_cfg = EvalFrankaReachEnvCfg()
env_cfg.commands.ee_pose.debug_vis = False

# Override paths AFTER instantiation
env_cfg.scene.ground.spawn.usd_path = ISAAC_DIR + "/Environments/Grid/default_environment.usd"
if hasattr(env_cfg.scene, "table"):
    env_cfg.scene.table.spawn.usd_path = ISAAC_DIR + "/Props/Mounts/SeattleLabTable/table_instanceable.usd"

env = ManagerBasedRLEnv(cfg=env_cfg)

print("Eval env created. Running rollout...")

duration   = FLUXA_ROLLOUT_DURATION
start_time = time.time()
obs, info  = env.reset()

total_reward  = 0.0
steps         = 0
success_count = 0

try:
    while time.time() - start_time < duration:
        actions = 2.0 * torch.rand(env.num_envs, env.single_action_space.shape[0], device=env.device) - 1.0
        obs, rewards, dones, truncated, info = env.step(actions)
        total_reward  += rewards.mean().item()
        success_count += dones.sum().item()
        steps         += 1
        time.sleep(0.01)
except Exception as e:
    print(f"Rollout error: {{e}}")

mean_reward  = total_reward / max(steps, 1)
success_rate = success_count / max(steps * FLUXA_NUM_ENVS, 1)

print(f"mean_reward={mean_reward:.4f} success_rate={success_rate:.4f}")

# # Send metrics back to host Eureka process
# async def send_metrics():
#     uri = "ws://FLUXA_HOST_IP:FLUXA_RESULTS_PORT"
#     async with websockets.connect(uri) as ws:
#         await ws.send(json.dumps({
#             "mean_reward":  mean_reward,
#             "success_rate": success_rate,
#         }))

# asyncio.run(send_metrics())
# print("📤 Metrics sent back to Eureka.")

# Send metrics back using a plain socket 
import socket
import struct

def send_metrics_sync():
    """Send metrics via raw TCP to avoid asyncio.run() conflicts with Isaac Sim's event loop."""
    payload = json.dumps({
        "mean_reward":  mean_reward,
        "success_rate": success_rate,
    }).encode("utf-8")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    try:
        sock.connect(("FLUXA_HOST_IP", FLUXA_RESULTS_PORT))
        # raw TCP send:
        sock.sendall(payload)
    finally:
        sock.close()

send_metrics_sync()
print("📤 Metrics sent back to Eureka.")

# Soft cleanup AFTER metrics are sent
del env
import gc
gc.collect()
torch.cuda.empty_cache()