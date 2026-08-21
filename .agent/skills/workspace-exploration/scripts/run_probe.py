"""Probe-only entry point. Hardcoded args for fast iteration.

Usage:
    ./python scripts/run_probe.py
    ./python scripts/run_probe.py --num_envs 1000 --n_samples 10000

    # success-threshold smoke test (flips gravity ON, runs workspace -> success,
    # skips joint-limits since gravity-on is not its validated condition):
    ./python scripts/run_probe.py --success-threshold --n_targets 1000 
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
parser.add_argument("--ee_body_name", type=str, default="panda_hand",
                    help="EE body ALL probes measure at.")

# --- success-threshold probe ---
parser.add_argument("--success-threshold", action="store_true",
                    help="Run the success-threshold probe. Requires gravity ON, "
                         "so this flips gravity on and skips the joint-limits probe.")
parser.add_argument("--n_targets", type=int, default=500,
                    help="Targets for the success-threshold probe (padded to a "
                         "multiple of num_envs).")
parser.add_argument("--st_statistic", type=str, default="p90",
                    help="Percentile of the error distribution to use as threshold.")
parser.add_argument("--validate-success-threshold", action="store_true",
                    help="Run CuRobo cross-validation of the success-threshold "
                         "probe's Phase A IK solve before probing. Requires "
                         "--success-threshold.")

# --- controller-gains probe ---
parser.add_argument("--controller-gains", action="store_true",
                    help="Run the controller-gains probe. Requires gravity ON, "
                         "same as --success-threshold. Standalone for now -- "
                         "not yet wired into success_threshold_probe's gains.")

parser.add_argument("--gravity_z", type=float, default=None,
                    help="Override gravity z. Default: -9.81 when --success-threshold "
                         "or --controller-gains is set, else 0.0 (the validated "
                         "condition for the other probes).")

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
from probes.controller_gains_probe import controller_gains_probe
from helpers.io import save_scatter_plot, save_json, load_json

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg

_SKILLS_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _SKILLS_DIR)
from common.schemas import (
    DiscoveredConfig, RobotConfig, ProbeResults,
    SuccessThresholdProbeResult as SuccessThresholdSchema,
)

def _make_franka_cfg():
    cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Isaac Lab's own gains for FRANKA_PANDA_CFG (stiffness=80, damping=4) are
    # tuned for RL torque control, not pose-holding under gravity via IK -- they
    # produce large steady-state gravity droop. Use the same higher gains as
    # FRANKA_PANDA_HIGH_PD_CFG, but keep gravity ON (unlike that preset, which
    # disables it) since this probe measures settled error under gravity.
    cfg.actuators["panda_shoulder"].stiffness = 400.0
    cfg.actuators["panda_shoulder"].damping = 80.0
    cfg.actuators["panda_forearm"].stiffness = 400.0
    cfg.actuators["panda_forearm"].damping = 80.0
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
    """Quick histogram of EE-position error (cm) with the chosen percentile marked."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(errors_m * 100.0, bins=50, color="#4C72B0", alpha=0.85)
        ax.axvline(threshold_m * 100.0, color="crimson", ls="--",
                   label=f"{statistic} = {threshold_m * 100.0:.2f} cm")
        ax.set_xlabel("EE position error (cm)")
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
    # Gravity: success-threshold and controller-gains need it ON; the other
    # probes were validated OFF.
    if args_cli.gravity_z is not None:
        gravity_z = args_cli.gravity_z
    else:
        gravity_z = (-9.81 if (args_cli.success_threshold or args_cli.controller_gains)
                    else 0.0)

    # Set up sim and scene.
    sim_cfg = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, gravity_z))
    sim = SimulationContext(sim_cfg)

    scene_cfg = FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    robot = scene["robot"]
    print(f"Spawned scene with {scene.num_envs} envs.")
    print(f"Robot has {robot.num_joints} joints, body names: {robot.body_names}")

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
        ee_body_name=args_cli.ee_body_name, # "panda_hand"
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

    # --- Controller-gains probe (requires gravity ON) ---
    if args_cli.controller_gains:
        cg_result = controller_gains_probe(
            sim=sim, scene=scene, robot=robot,
            seed=args_cli.seed,
        )
        print(f"\n=== Controller-Gains Probe Results ===")
        print(f"Discovered:       Kp={cg_result.kp:.2f}  Kd={cg_result.kd:.2f}")
        print(f"Robot defaults:   Kp={cg_result.default_kp:.2f}  Kd={cg_result.default_kd:.2f}")
        print(f"Swept range:      Kp={cg_result.kp_range}  Kd={cg_result.kd_range}")
        print(f"Feasible:         {cg_result.n_feasible} / {cg_result.n_candidates}")
        print(f"Settling steps:   {cg_result.settling_steps}")
        print(f"Steady-state err: {cg_result.steady_state_error_rad:.5f} rad")
        print(f"Runtime:          {cg_result.runtime_seconds:.2f}s")
        np.save("outputs/diagnostics/controller_gains_candidates.npy",
               np.stack([cg_result.candidates_kp, cg_result.candidates_kd,
                        cg_result.candidates_feasible.astype(np.float32),
                        cg_result.candidates_steady_state_error_rad], axis=1))
        print("Candidate sweep saved to outputs/diagnostics/controller_gains_candidates.npy")

    # --- Success-threshold probe (requires gravity ON) ---
    if args_cli.success_threshold:
        if args_cli.validate_success_threshold:
            try:
                from tests.test_success_threshold_validation import run_success_threshold_validation
                run_success_threshold_validation(
                    sim, scene, robot, ws_result.point_cloud,
                    n_configs=args_cli.n_validate,
                    seed=args_cli.seed,
                )
            except AssertionError as e:
                print(f"\n❌ Success-Threshold Validation Failed! "
                      f"Continuing to next steps...\nError: {e}")

        st_result = success_threshold_probe(
            sim=sim, scene=scene, robot=robot,
            workspace_points=ws_result.point_cloud,
            n_targets=args_cli.n_targets,
            seed=args_cli.seed,
            ee_body_name=args_cli.ee_body_name,
            statistic=args_cli.st_statistic,
        )
        print(f"\n=== Success-Threshold Probe Results ===")
        print(f"Threshold ({st_result.statistic}): "
              f"{st_result.threshold_m * 100:.2f} cm  ({st_result.threshold_m:.5f} m)")
        print(f"Targets measured: {st_result.n_measured} / {st_result.n_targets}")
        print(f"Convergence rate: {st_result.convergence_rate:.1%}")
        print(f"EE frame:         {st_result.ee_frame}")
        print(f"Gravity z:        {st_result.gravity_z}")
        print(f"Position error percentiles (cm):")
        for k, v in st_result.position_error_percentiles_m.items():
            print(f"  {k:>4}: {v * 100:7.3f}")
        print(f"Runtime:          {st_result.runtime_seconds:.2f}s")
 
        np.save("outputs/diagnostics/success_threshold_errors.npy", st_result.errors_m)
        _save_error_hist(st_result.errors_m,
                         "outputs/diagnostics/success_threshold_hist.png",
                         st_result.threshold_m, st_result.statistic)

        # Merge this probe's section into discovered_config.json. Read-modify-
        # write: run_skill.py's gravity-off pass writes workspace/joint_limits
        # into this same file (success-threshold needs gravity ON, so it can't
        # share that pass -- see setup_scene's docstring) and we must not
        # clobber those sections here.
        existing = load_json("outputs/discovered_config.json")
        discovered = (DiscoveredConfig.model_validate(existing) if existing is not None
                     else DiscoveredConfig(robot=RobotConfig(name="franka"), probes=ProbeResults()))
        discovered.probes.success_threshold = SuccessThresholdSchema(
            ee_frame=st_result.ee_frame,
            threshold_m=st_result.threshold_m,
            statistic=st_result.statistic,
            position_error_percentiles_m=st_result.position_error_percentiles_m,
            n_targets=st_result.n_targets,
            n_measured=st_result.n_measured,
            convergence_rate=st_result.convergence_rate,
            physics_dt=st_result.physics_dt,
            gravity_z=st_result.gravity_z,
            units=st_result.units,
            seed=st_result.seed,
        )
        save_json(discovered.model_dump(), "outputs/discovered_config.json")
        print("success_threshold section written to outputs/discovered_config.json")


if __name__ == "__main__":
    # try:
        main()
    # finally:
    #     simulation_app.close()
    # os._exit(0)