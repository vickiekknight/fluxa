"""Probe-only entry point. Hardcoded args for fast iteration.

Usage:
    ./python scripts/run_probe.py
    ./python scripts/run_probe.py --num_envs 1000 --n_samples 10000

    # success-threshold smoke test (flips gravity ON, runs workspace -> success,
    # skips joint-limits since gravity-on is not its validated condition):
    ./python scripts/run_probe.py --success-threshold
    ./python scripts/run_probe.py --success-threshold --n_targets 1000 --st_n_steps 250
"""
import argparse
import os
import sys
import numpy as np

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
parser.add_argument("--validate-collision", action="store_true",
                    help="Run collision validation against CuRobo before probing.")

# --- success-threshold probe ---
parser.add_argument("--success-threshold", action="store_true",
                    help="Run the success-threshold probe. Requires gravity ON, "
                         "so this flips gravity on and skips the joint-limits probe.")
parser.add_argument("--n_targets", type=int, default=500,
                    help="Targets for the success-threshold probe (padded to a "
                         "multiple of num_envs).")
parser.add_argument("--st_n_steps", type=int, default=200,
                    help="Oracle rollout horizon for the success-threshold probe.")
parser.add_argument("--st_statistic", type=str, default="p90",
                    help="Percentile of the error distribution to use as threshold.")
parser.add_argument("--gravity_z", type=float, default=None,
                    help="Override gravity z. Default: -9.81 when --success-threshold "
                         "is set, else 0.0 (the validated condition for the other probes).")

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
from probes.joint_limits_probe import joint_limits_probe
from probes.success_threshold_probe import success_threshold_probe
from helpers.io import save_scatter_plot

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg

def _make_franka_cfg():
    cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if cfg.spawn.articulation_props is None:
        cfg.spawn.articulation_props = ArticulationRootPropertiesCfg()
    cfg.spawn.articulation_props.enabled_self_collisions = True
    cfg.spawn.activate_contact_sensors = True   # required for ContactSensor to report
    return cfg


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    robot = _make_franka_cfg()
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=0,
        track_air_time=False,
    )

def _save_error_hist(errors_m, path, threshold_m, statistic):
    """Quick histogram of EE-position error (mm) with the chosen percentile marked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(errors_m * 1000.0, bins=50, color="#4C72B0", alpha=0.85)
        ax.axvline(threshold_m * 1000.0, color="crimson", ls="--",
                   label=f"{statistic} = {threshold_m * 1000.0:.2f} mm")
        ax.set_xlabel("EE position error (mm)")
        ax.set_ylabel("count")
        ax.set_title("Success-threshold probe: oracle execution error")
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"Error histogram saved to {path}")
    except Exception as e:
        print(f"(skipped histogram: {e})")

def main():
    # Gravity: success-threshold needs it ON; the other probes were validated OFF.
    if args_cli.gravity_z is not None:
        gravity_z = args_cli.gravity_z
    else:
        gravity_z = -9.81 if args_cli.success_threshold else 0.0

    # Set up sim and scene.
    sim_cfg = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, gravity_z))
    sim = SimulationContext(sim_cfg)

    scene_cfg = FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    robot = scene["robot"]
    print(f"Spawned scene with {scene.num_envs} envs.")
    print(f"Robot has {robot.num_joints} joints, body names: {robot.body_names}")

    n_validate = max(args_cli.n_validate, args_cli.num_envs)

    # FK validation (optional; bails before probe runs if it fails)
    if args_cli.validate_fk:
        try:
            from tests.test_workspace_integration import run_integration_test
            run_integration_test(
                scene, robot,
                n_configs=args_cli.n_validate,
                seed=args_cli.seed,
            )
        except AssertionError as e:
            print(f"\n❌ FK Validation Failed! Continuing to next steps...\nError: {e}")

    # Collision validation 
    if args_cli.validate_collision:
        try:
            from tests.test_jointlimits_validation import run_jointlimits_validation
            run_jointlimits_validation(
                sim, scene, robot,
                n=args_cli.n_validate,
                seed=args_cli.seed,
            )
        except AssertionError as e:
            print(f"\n❌ Collision Validation Failed! (Recall threshold missed).")

    # Run the probe.

    # --- Workspace Probe ---
    ws_result = workspace_probe(
        scene=scene,
        robot=robot,
        n_samples=args_cli.n_samples,
        seed=args_cli.seed,
        ee_body_name="panda_hand",
    )

    print(f"\n=== Workspace Probe Results ===")
    print(f"N sampled: {ws_result.n_sampled}")
    print(f"Runtime:   {ws_result.runtime_seconds:.2f}s")
    print(f"Bounds (robot-frame, meters):")
    print(f"  x: [{ws_result.bounds['x'][0]:+.3f}, {ws_result.bounds['x'][1]:+.3f}]")
    print(f"  y: [{ws_result.bounds['y'][0]:+.3f}, {ws_result.bounds['y'][1]:+.3f}]")
    print(f"  z: [{ws_result.bounds['z'][0]:+.3f}, {ws_result.bounds['z'][1]:+.3f}]")

    # Save the scatter plot.
    save_scatter_plot(ws_result, "outputs/diagnostics/workspace_scatter.png",
                      title_suffix="Franka, run_probe.py")
    print(f"\nScatter plot saved to outputs/diagnostics/workspace_scatter.png")

    # --- Joint-limits probe ---
    # validated gravity-OFF; skip when gravity is on -> gravity off isolates
    # the variables joint limits probe is testing. 
    if abs(gravity_z) < 1e-6:
        jl_result = joint_limits_probe(
            sim=sim, scene=scene, robot=robot,
            n_samples=args_cli.n_samples, seed=args_cli.seed,
        )
        print(f"\n=== Joint-Limits Probe Results ===")
        print(f"N sampled:      {jl_result.n_sampled}")
        print(f"N safe:         {jl_result.n_safe}")
        print(f"Collision rate: {jl_result.collision_rate:.1%}")
        print(f"Runtime:        {jl_result.runtime_seconds:.2f}s")
        np.save("outputs/diagnostics/safe_configs.npy", jl_result.safe_configs)
        print("Safe configs saved to outputs/diagnostics/safe_configs.npy")
    else:
        print("\n(joint-limits probe skipped: gravity is on, which is not its "
              "validated condition. Run it in a separate gravity-off invocation.)")

    # --- Success-threshold probe (requires gravity ON) ---
    if args_cli.success_threshold:
        st_result = success_threshold_probe(
            sim=sim, scene=scene, robot=robot,
            workspace_points=ws_result.point_cloud,
            n_targets=args_cli.n_targets,
            seed=args_cli.seed,
            ee_body_name="panda_hand",
            statistic=args_cli.st_statistic,
            n_steps=args_cli.st_n_steps,
        )
        print(f"\n=== Success-Threshold Probe Results ===")
        print(f"Threshold ({st_result.statistic}): "
              f"{st_result.threshold_m * 1000:.2f} mm  ({st_result.threshold_m:.5f} m)")
        print(f"Targets measured: {st_result.n_measured} / {st_result.n_targets}")
        print(f"Convergence rate: {st_result.convergence_rate:.1%} "
              f"(settled within n_steps={st_result.n_steps})")
        print(f"EE frame:         {st_result.ee_frame}")
        print(f"Gravity z:        {st_result.gravity_z}")
        print(f"Position error percentiles (mm):")
        for k, v in st_result.position_error_percentiles_m.items():
            print(f"  {k:>4}: {v * 1000:7.2f}")
        if st_result.position_error_percentiles_m is not None:
            op = st_result.position_error_percentiles_m
            print(f"Orientation error p90: {op['p90']:.2f} deg  (max {op['max']:.2f})")
        print(f"Runtime:          {st_result.runtime_seconds:.2f}s")
 
        np.save("outputs/diagnostics/success_threshold_errors.npy", st_result.errors_m)
        _save_error_hist(st_result.errors_m,
                         "outputs/diagnostics/success_threshold_hist.png",
                         st_result.threshold_m, st_result.statistic)


if __name__ == "__main__":
    # try:
        main()
    # finally:
    #     simulation_app.close()
    # os._exit(0)