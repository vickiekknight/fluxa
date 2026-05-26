"""Probe-only entry point. Hardcoded args for fast iteration.

Usage:
    python scripts/run_probe.py
    python scripts/run_probe.py --num_envs 1000 --n_samples 10000
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

# CLI args
parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--n_samples", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--validate-fk", action="store_true",
                    help="Run FK validation against CuRobo before probing.")
parser.add_argument("--n_validate", type=int, default=50,
                    help="Number of random configs for FK validation.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

# Launch Omniverse BEFORE any isaaclab imports.
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Make sibling packages importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now safe to import sim-touching modules.
import torch
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG

from probes.workspace_probe import workspace_probe
from helpers.io import save_scatter_plot


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    """A scene with N parallel Frankas, no other assets."""
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    # Set up sim and scene.
    sim_cfg = SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)

    scene_cfg = FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    robot = scene["robot"]
    print(f"Spawned scene with {scene.num_envs} envs.")
    print(f"Robot has {robot.num_joints} joints, body names: {robot.body_names}")

    # FK validation (optional; bails before probe runs if it fails)
    if args_cli.validate_fk:
        from tests.test_workspace_integration import run_integration_test
        run_integration_test(
            scene, robot,
            n_configs=args_cli.n_validate,
            seed=args_cli.seed,
        )

    # Run the probe.
    result = workspace_probe(
        scene=scene,
        robot=robot,
        n_samples=args_cli.n_samples,
        seed=args_cli.seed,
        ee_body_name="panda_hand",
    )

    print(f"\n=== Workspace Probe Results ===")
    print(f"N sampled: {result.n_sampled}")
    print(f"Runtime:   {result.runtime_seconds:.2f}s")
    print(f"Bounds (robot-frame, meters):")
    print(f"  x: [{result.bounds['x'][0]:+.3f}, {result.bounds['x'][1]:+.3f}]")
    print(f"  y: [{result.bounds['y'][0]:+.3f}, {result.bounds['y'][1]:+.3f}]")
    print(f"  z: [{result.bounds['z'][0]:+.3f}, {result.bounds['z'][1]:+.3f}]")

    # Save the scatter plot.
    save_scatter_plot(result, "outputs/diagnostics/workspace_scatter.png",
                      title_suffix="Franka, run_probe.py")
    print(f"\nScatter plot saved to outputs/diagnostics/workspace_scatter.png")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    os._exit(0)