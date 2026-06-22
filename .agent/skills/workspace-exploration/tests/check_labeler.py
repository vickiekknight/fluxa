"""Manual ground-truth check for the PhysX self-collision labeler.

Drives a known-free (home) and a known-folded config through the labeler and
prints collide-label + max contact-force magnitude per env. No cuRobo.

Usage:
    /isaac-sim/python.sh tests/check_labeler.py
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg
from isaaclab_assets import FRANKA_PANDA_CFG

from probes.joint_limits_probe import PhysxSelfCollisionLabeler


def _make_franka_cfg():
    cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if cfg.spawn.articulation_props is None:
        cfg.spawn.articulation_props = ArticulationRootPropertiesCfg()
    cfg.spawn.articulation_props.enabled_self_collisions = True
    cfg.spawn.activate_contact_sensors = True
    return cfg


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    robot = _make_franka_cfg()
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=0, track_air_time=False,
    )


def main():
    sim = SimulationContext(SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, 0.0)))
    scene = InteractiveScene(FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    print("joints:", robot.joint_names)

    n = scene.num_envs
    q = robot.data.default_joint_pos.clone()       # (n, 9), env 0 stays home

    # env 1: fold the arm into itself. index 3 = panda_joint4 (elbow),
    # index 5 = panda_joint6 (wrist), index 1 = panda_joint2 (shoulder).
    # All within URDF limits. Tweak if it doesn't collide.
    if n >= 2:
        folded = robot.data.default_joint_pos[1].clone()
        folded[1] = 1.5      # shoulder up, hand toward base
        folded[3] = -3.0     # elbow folded hard
        folded[5] = 3.0      # wrist folded back
        q[1] = folded

    labeler = PhysxSelfCollisionLabeler(sim, scene, robot)
    mask = labeler(q)                                            # (n,) bool
    net = labeler.sensor.data.net_forces_w                       # (n, num_bodies, 3)
    fmag = torch.linalg.norm(net, dim=-1).max(dim=-1).values     # (n,) per-env max

    for i in range(n):
        tag = "home" if i == 0 else ("folded" if i == 1 else f"env{i}")
        print(f"  env {i} ({tag}): collide={bool(mask[i])}  max|F|={fmag[i].item():.4f} N")

    print("\nExpect: env 0 (home) collide=False; env 1 (folded) collide=True.")
    print("If folded reads ~0 N  -> contacts aren't reaching the sensor "
          "(self_collisions / activate_contact_sensors / threshold).")
    print("If home reads collide -> over-reporting "
          "(adjacency filtering lost / gravity on / threshold too low).")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)