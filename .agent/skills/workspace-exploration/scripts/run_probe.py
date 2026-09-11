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
import traceback
import numpy as np

from isaaclab.app import AppLauncher

_SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _out(rel_path: str) -> str:
    """Resolve an output path against the skill root and ensure its directory
    exists. Everything this script writes goes through here so it works from
    any working directory -- run_skill.py launches it as a subprocess, and
    cwd-relative paths silently wrote to (or failed in) the wrong place."""
    path = os.path.join(_SKILL_ROOT, rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path

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
parser.add_argument("--write-config", action="store_true",
                    help="Record the workspace and joint-limits sections into "
                         "discovered_config.json. Off by default so standalone "
                         "debug runs don't touch the artifact; run_skill.py "
                         "passes it for the gravity-off pass. (The "
                         "success-threshold and controller-gains sections are "
                         "always written by their own flags.)")
parser.add_argument("--task-spec", type=str, default=None,
                    help="Path to outputs/task_spec.json. When given, the "
                         "robot name, EE body, and surface constraint are taken "
                         "from the parsed task instead of this script's "
                         "defaults.")

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
parser.add_argument("--st-ignore-discovered-gains", action="store_true",
                    help="Don't apply the controller_gains section from "
                         "discovered_config.json to the success-threshold "
                         "probe; use the gains the scene was spawned with "
                         "instead. For A/B comparing discovered vs. hardcoded "
                         "gains.")

# --- controller-gains probe ---
parser.add_argument("--controller-gains", action="store_true",
                    help="Run the controller-gains probe. Requires gravity ON, "
                         "same as --success-threshold. Its discovered Kp/Kd are "
                         "written to discovered_config.json and picked up by a "
                         "later --success-threshold run, so run this first.")
parser.add_argument("--cg_pos_tol", type=float, default=2e-2,
                    help="controller-gains: joint-space position-error norm "
                         "(rad) to count as settled. Empirical: 1e-2 yields "
                         "0/1000 feasible on the Franka -- pure PD can't hold "
                         "that tightly against gravity droop at any Kp in range.")
parser.add_argument("--cg_vel_tol", type=float, default=0.2,
                    help="controller-gains: joint-space velocity norm (rad/s) "
                         "to count as settled. Empirical: 5e-2 is below this "
                         "sim's residual jitter floor (3/1000 candidates "
                         "cleared it), so it rejects good gains too.")
parser.add_argument("--cg_n_steps", type=int, default=400,
                    help="controller-gains: physics steps to observe the "
                         "step response over. Needs headroom for the "
                         "heaviest load tested (see controller_gains_probe.py "
                         "docstring) or it silently falls back to a "
                         "not-actually-feasible answer.")
parser.add_argument("--cg_settle_window", type=int, default=150,
                    help="controller-gains: trailing steps that must all be "
                         "within tolerance to count as settled -- must be "
                         "wide enough to overlap real ringing, or it goes "
                         "undetected (see controller_gains_probe.py docstring).")
parser.add_argument("--cg_kd_mult_hi", type=float, default=100.0,
                    help="controller-gains: upper multiplier on the robot's "
                         "default Kd for the sweep range's ceiling.")
parser.add_argument("--controller-gains-load-profile", action="store_true",
                    help="Run controller_gains_probe once per end-effector "
                         "payload in --cg_payloads_kg instead of a single "
                         "unloaded sweep. Requires gravity ON, same as "
                         "--controller-gains.")
parser.add_argument("--cg_payloads_kg", type=float, nargs="+",
                    default=[0.0, 1.5, 3.0],
                    help="controller-gains-load-profile: EE payload masses "
                         "(kg) to sweep gains under. Default spans bare arm "
                         "to Franka's rated 3kg payload.")

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
from probes.controller_gains_probe import controller_gains_probe, controller_gains_load_profile
from helpers.io import save_scatter_plot, save_json, load_json

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg

_SKILLS_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _SKILLS_DIR)
from common.schemas import (
    DiscoveredConfig, RobotConfig, ProbeResults,
    WorkspaceProbeResult as WorkspaceSchema,
    JointLimitsProbeResult as JointLimitsSchema,
    SuccessThresholdProbeResult as SuccessThresholdSchema,
    ControllerGainsProbeResult as ControllerGainsSchema,
)

# Robot-frame z = 0 corresponds to the table surface in v1.
TABLE_HEIGHT = 0.0

# Populated from --task-spec in main() when run_skill.py drives this script.
_task_spec = None
_robot_name = "franka"

def _make_franka_cfg():
    cfg = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Isaac Lab's own gains for FRANKA_PANDA_CFG (stiffness=80, damping=4) are
    # tuned for RL torque control, not pose-holding under gravity via IK -- they
    # produce large steady-state gravity droop. Use the same higher gains as
    # FRANKA_PANDA_HIGH_PD_CFG, but keep gravity ON (unlike that preset, which
    # disables it) since this probe measures settled error under gravity.
    #
    # This is now only the FALLBACK for --success-threshold: when a
    # controller_gains section exists in discovered_config.json, the probe
    # overrides these at runtime with the discovered pair. These 400/80 are
    # hand-tuned for the Franka specifically and mean nothing on another robot,
    # which is exactly why the discovered pair is preferred.
    #
    # Still applied only for --success-threshold: controller_gains_probe
    # searches FROM the robot's own raw ArticulationCfg gains, so it must see
    # the true 80/4 baseline, not this already-elevated value, or its sweep
    # range ends up centered on an already-good answer instead of searching
    # from scratch.
    if args_cli.success_threshold:
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

def _save_gains_sweep_plot(cg_result, path):
    """Kp/Kd sweep: the feasible region, and the error-vs-Kp tradeoff curve."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kp = cg_result.candidates_kp
        kd = cg_result.candidates_kd
        feasible = cg_result.candidates_feasible.astype(bool)
        err = cg_result.candidates_steady_state_error_rad

        # Color scale centered on the decision boundary (near the discovered
        # candidate's own error), not the full range -- a few badly-drooping
        # candidates would otherwise wash out contrast in the region that
        # actually matters (near the feasible/infeasible line).
        err_cap = max(cg_result.steady_state_error_rad * 4, 1e-6)
        err_clipped = np.clip(err, None, err_cap)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax1.scatter(kp[~feasible], kd[~feasible], c=err_clipped[~feasible],
                   cmap="Blues", marker="x", s=25, alpha=0.6,
                   vmin=0, vmax=err_cap, label="infeasible")
        sc = ax1.scatter(kp[feasible], kd[feasible], c=err_clipped[feasible],
                        cmap="Blues", marker="o", s=35, edgecolors="#333333",
                        linewidths=0.5, vmin=0, vmax=err_cap, label="feasible")
        ax1.scatter([cg_result.kp], [cg_result.kd], marker="*", s=300,
                   color="crimson", edgecolors="black", linewidths=0.8,
                   zorder=5, label="discovered")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("Kp"); ax1.set_ylabel("Kd")
        ax1.set_title("Sweep: feasible region")
        ax1.legend(loc="best", fontsize=8)
        fig.colorbar(sc, ax=ax1, label="steady-state error (rad)")

        ax2.scatter(kp[~feasible], err[~feasible], c="#B0B0B0", s=20, alpha=0.5,
                   marker="x", label="infeasible")
        ax2.scatter(kp[feasible], err[feasible], c="#4C72B0", s=25,
                   label="feasible")
        ax2.scatter([cg_result.kp], [cg_result.steady_state_error_rad],
                   marker="*", s=300, color="crimson", edgecolors="black",
                   linewidths=0.8, zorder=5, label="discovered")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("Kp"); ax2.set_ylabel("steady-state error (rad)")
        ax2.set_title("Error vs Kp")
        ax2.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Gains-sweep plot saved to {path}")
    except Exception as e:
        print(f"(skipped gains-sweep plot: {e})")

def _save_gains_cost_plot(cg_result, path):
    """Kp/Kd sweep colored by (steady-state error x settling steps) -- a
    combined accuracy+speed cost, vs. the error-only view in the sweep plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kp = cg_result.candidates_kp
        kd = cg_result.candidates_kd
        feasible = cg_result.candidates_feasible.astype(bool)
        cost = cg_result.candidates_steady_state_error_rad * cg_result.candidates_settling_steps
        discovered_cost = (cg_result.steady_state_error_rad * cg_result.settling_steps)

        cost_cap = max(discovered_cost * 4, 1e-6)
        cost_clipped = np.clip(cost, None, cost_cap)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        ax1.scatter(kp[~feasible], kd[~feasible], c=cost_clipped[~feasible],
                   cmap="Blues", marker="x", s=25, alpha=0.6,
                   vmin=0, vmax=cost_cap, label="infeasible")
        sc = ax1.scatter(kp[feasible], kd[feasible], c=cost_clipped[feasible],
                        cmap="Blues", marker="o", s=35, edgecolors="#333333",
                        linewidths=0.5, vmin=0, vmax=cost_cap, label="feasible")
        ax1.scatter([cg_result.kp], [cg_result.kd], marker="*", s=300,
                   color="crimson", edgecolors="black", linewidths=0.8,
                   zorder=5, label="discovered")
        ax1.set_xscale("log"); ax1.set_yscale("log")
        ax1.set_xlabel("Kp"); ax1.set_ylabel("Kd")
        ax1.set_title("Sweep: error x settling-steps cost")
        ax1.legend(loc="best", fontsize=8)
        fig.colorbar(sc, ax=ax1, label="total cost")

        ax2.scatter(kd[~feasible], cost[~feasible], c="#B0B0B0", s=20, alpha=0.5,
                   marker="x", label="infeasible")
        ax2.scatter(kd[feasible], cost[feasible], c="#4C72B0", s=25,
                   label="feasible")
        ax2.scatter([cg_result.kd], [discovered_cost],
                   marker="*", s=300, color="crimson", edgecolors="black",
                   linewidths=0.8, zorder=5, label="discovered")
        ax2.set_xscale("log"); ax2.set_yscale("log")
        ax2.set_xlabel("Kd"); ax2.set_ylabel("total cost")
        ax2.set_title("Cost vs Kd")
        ax2.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Gains-cost plot saved to {path}")
    except Exception as e:
        print(f"(skipped gains-cost plot: {e})")

def _save_gains_transient_plot(cg_result, pos_tol, vel_tol, path):
    """Full err(t)/vel(t) trajectories for a handful of candidates at roughly
    fixed Kp, spread across Kd -- shows whether Kd actually affects the
    transient (ringing), not just whether it settles by the final window."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        kd = cg_result.representative_kd
        err = cg_result.representative_err_hist_rad     # (n_steps, n_reps)
        vel = cg_result.representative_vel_hist_rad_s
        n_steps = err.shape[0]
        steps = np.arange(n_steps)

        # Sequential: one hue, light (low Kd) -> dark (high Kd) -- Kd is the
        # single continuous variable being compared across these lines.
        order = np.argsort(kd)
        cmap = matplotlib.colormaps["Blues"]
        shades = cmap(np.linspace(0.35, 0.95, len(order)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

        for color, i in zip(shades, order):
            ax1.plot(steps, err[:, i], color=color, lw=1.6,
                    label=f"Kd={kd[i]:.1f}")
            ax2.plot(steps, vel[:, i], color=color, lw=1.6,
                    label=f"Kd={kd[i]:.1f}")

        ax1.axhline(pos_tol, color="crimson", ls="--", lw=1, label="pos_tol")
        ax1.set_xlabel("step"); ax1.set_ylabel("position error (rad)")
        ax1.set_title(f"Transient at Kp≈{cg_result.kp:.0f}: error")
        ax1.legend(loc="best", fontsize=7)

        ax2.axhline(vel_tol, color="crimson", ls="--", lw=1, label="vel_tol")
        ax2.set_xlabel("step"); ax2.set_ylabel("joint velocity (rad/s)")
        ax2.set_title(f"Transient at Kp≈{cg_result.kp:.0f}: velocity")
        ax2.legend(loc="best", fontsize=7)

        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Gains-transient plot saved to {path}")
    except Exception as e:
        print(f"(skipped gains-transient plot: {e})")

def _save_gains_load_profile_plot(profile_result, path):
    """Discovered Kp/Kd, error, and settling time vs. EE payload -- the
    actual profile across loading scenarios."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        payloads = profile_result.payload_masses_kg
        kp = [r.kp for r in profile_result.results]
        kd = [r.kd for r in profile_result.results]
        err = [r.steady_state_error_rad for r in profile_result.results]
        settle = [r.settling_steps for r in profile_result.results]

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))

        axes[0].plot(payloads, kp, "o-", color="#4C72B0")
        axes[0].set_xlabel("EE payload (kg)"); axes[0].set_ylabel("discovered Kp")
        axes[0].set_title("Kp vs load")

        axes[1].plot(payloads, kd, "o-", color="#4C72B0")
        axes[1].set_xlabel("EE payload (kg)"); axes[1].set_ylabel("discovered Kd")
        axes[1].set_title("Kd vs load")

        axes[2].plot(payloads, err, "o-", color="#4C72B0")
        axes[2].set_xlabel("EE payload (kg)"); axes[2].set_ylabel("steady-state error (rad)")
        axes[2].set_title("Error vs load")

        axes[3].plot(payloads, settle, "o-", color="#4C72B0")
        axes[3].set_xlabel("EE payload (kg)"); axes[3].set_ylabel("settling steps")
        axes[3].set_title("Settling time vs load")

        fig.tight_layout()
        fig.savefig(path, dpi=130)
        plt.close(fig)
        print(f"Gains-load-profile plot saved to {path}")
    except Exception as e:
        print(f"(skipped gains-load-profile plot: {e})")

def main():
    # --- Optional parsed task spec (written by run_skill.py) ---
    global _task_spec, _robot_name
    _task_spec = load_json(args_cli.task_spec) if args_cli.task_spec else None
    _robot_name = (_task_spec or {}).get("robot_name", "franka")
    if _task_spec:
        ee = _task_spec.get("ee_body_name")
        if ee:
            args_cli.ee_body_name = ee
        print(f"Using task spec: {_task_spec.get('task_type')} on {_robot_name} "
              f"(ee={args_cli.ee_body_name}, "
              f"constraints={_task_spec.get('constraints')})")

    # Gravity: success-threshold and controller-gains need it ON; the other
    # probes were validated OFF.
    if args_cli.gravity_z is not None:
        gravity_z = args_cli.gravity_z
    else:
        gravity_z = (-9.81 if (args_cli.success_threshold or args_cli.controller_gains
                              or args_cli.controller_gains_load_profile)
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
    save_scatter_plot(ws_result, _out("outputs/diagnostics/workspace_scatter.png"),
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
        np.save(_out("outputs/diagnostics/safe_configs.npy"), jl_result.safe_configs)
        print("Safe configs saved to outputs/diagnostics/safe_configs.npy")
    else:
        jl_result = None
        print("\n(joint-limits probe skipped: gravity is on, which is not its "
              "validated condition. Run it in a separate gravity-off invocation.)")

    # --- Record workspace + joint-limits into discovered_config.json ---
    # Same read-modify-write merge the other two sections use, so this pass
    # can't clobber the gravity-on sections written by a separate invocation.
    if args_cli.write_config:
        # Absolute path: reach_task.py loads this inside the Isaac Sim server
        # process, which may not share this process's working directory.
        safe_path = None
        if jl_result is not None:
            safe_path = os.path.abspath("outputs/diagnostics/safe_configs.npy")

        z_lo, z_hi = ws_result.bounds["z"]
        if _task_spec and _task_spec.get("constraints", {}).get("surface") == "table":
            z_lo = max(z_lo, TABLE_HEIGHT)

        existing = load_json("outputs/discovered_config.json")
        discovered = (DiscoveredConfig.model_validate(existing) if existing is not None
                     else DiscoveredConfig(robot=RobotConfig(name=_robot_name),
                                           probes=ProbeResults()))
        discovered.robot = RobotConfig(name=_robot_name)
        discovered.probes.workspace = WorkspaceSchema(
            x=tuple(ws_result.bounds["x"]),
            y=tuple(ws_result.bounds["y"]),
            z=(z_lo, z_hi),
        )
        if jl_result is not None:
            discovered.probes.joint_limits = JointLimitsSchema(
                n_sampled=jl_result.n_sampled,
                n_safe=jl_result.n_safe,
                collision_rate=jl_result.collision_rate,
                seed=jl_result.seed,
                joint_lower=jl_result.joint_lower.tolist(),
                joint_upper=jl_result.joint_upper.tolist(),
                safe_config_path=safe_path,
            )
        save_json(discovered.model_dump(), "outputs/discovered_config.json")
        print("workspace" + ("/joint_limits" if jl_result is not None else "") +
              " section written to outputs/discovered_config.json")

    # --- Controller-gains probe (requires gravity ON) ---
    if args_cli.controller_gains:
        cg_result = controller_gains_probe(
            sim=sim, scene=scene, robot=robot,
            seed=args_cli.seed,
            pos_tol=args_cli.cg_pos_tol,
            vel_tol=args_cli.cg_vel_tol,
            n_steps=args_cli.cg_n_steps,
            settle_window=args_cli.cg_settle_window,
            kd_range_mult=(0.5, args_cli.cg_kd_mult_hi),
        )
        print(f"\n=== Controller-Gains Probe Results ===")
        print(f"Discovered:       Kp={cg_result.kp:.2f}  Kd={cg_result.kd:.2f}")
        print(f"Robot defaults:   Kp={cg_result.default_kp:.2f}  Kd={cg_result.default_kd:.2f}")
        print(f"Swept range:      Kp={cg_result.kp_range}  Kd={cg_result.kd_range}")
        print(f"Feasible:         {cg_result.n_feasible} / {cg_result.n_candidates}")
        print(f"Settling steps:   {cg_result.settling_steps}")
        print(f"Steady-state err: {cg_result.steady_state_error_rad:.5f} rad")
        print(f"Runtime:          {cg_result.runtime_seconds:.2f}s")
        np.save(_out("outputs/diagnostics/controller_gains_candidates.npy"),
               np.stack([cg_result.candidates_kp, cg_result.candidates_kd,
                        cg_result.candidates_feasible.astype(np.float32),
                        cg_result.candidates_steady_state_error_rad,
                        cg_result.candidates_settling_steps], axis=1))
        print("Candidate sweep saved to outputs/diagnostics/controller_gains_candidates.npy")
        _save_gains_sweep_plot(cg_result, _out("outputs/diagnostics/controller_gains_sweep.png"))
        _save_gains_cost_plot(cg_result, _out("outputs/diagnostics/controller_gains_cost.png"))
        _save_gains_transient_plot(cg_result, args_cli.cg_pos_tol, args_cli.cg_vel_tol,
                                   _out("outputs/diagnostics/controller_gains_transient.png"))

        # Merge into discovered_config.json -- same read-modify-write pattern
        # as success_threshold (this probe also needs gravity ON, so it can't
        # share run_skill.py's gravity-off pass either). Only the single-load
        # sweep writes here; --controller-gains-load-profile is an analysis
        # tool for choosing tuning parameters, not a single answer to persist.
        existing = load_json("outputs/discovered_config.json")
        discovered = (DiscoveredConfig.model_validate(existing) if existing is not None
                     else DiscoveredConfig(robot=RobotConfig(name="franka"), probes=ProbeResults()))
        discovered.probes.controller_gains = ControllerGainsSchema(
            kp=cg_result.kp,
            kd=cg_result.kd,
            default_kp=cg_result.default_kp,
            default_kd=cg_result.default_kd,
            n_candidates=cg_result.n_candidates,
            n_feasible=cg_result.n_feasible,
            steady_state_error_rad=cg_result.steady_state_error_rad,
            settling_steps=cg_result.settling_steps,
            seed=cg_result.seed,
        )
        save_json(discovered.model_dump(), "outputs/discovered_config.json")
        print("controller_gains section written to outputs/discovered_config.json")

    # --- Controller-gains load profile (requires gravity ON) ---
    if args_cli.controller_gains_load_profile:
        profile_result = controller_gains_load_profile(
            sim=sim, scene=scene, robot=robot,
            payload_masses_kg=tuple(args_cli.cg_payloads_kg),
            ee_body_name=args_cli.ee_body_name,
            seed=args_cli.seed,
            pos_tol=args_cli.cg_pos_tol,
            vel_tol=args_cli.cg_vel_tol,
            n_steps=args_cli.cg_n_steps,
            settle_window=args_cli.cg_settle_window,
            kd_range_mult=(0.5, args_cli.cg_kd_mult_hi),
        )
        print(f"\n=== Controller-Gains Load-Profile Results ===")
        for payload_kg, r in zip(profile_result.payload_masses_kg, profile_result.results):
            print(f"  payload={payload_kg:5.2f} kg  Kp={r.kp:8.2f}  Kd={r.kd:7.2f}  "
                 f"feasible={r.n_feasible:4d}/{r.n_candidates}  "
                 f"err={r.steady_state_error_rad:.5f} rad  settle={r.settling_steps} steps")
        _save_gains_load_profile_plot(profile_result,
            _out("outputs/diagnostics/controller_gains_load_profile.png"))

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

        # Use the gains controller_gains_probe discovered, when they're on
        # disk. Falls back to whatever the scene was spawned with (the pair in
        # _make_franka_cfg, which is Franka-specific and hand-tuned) so a
        # standalone run without a prior gains pass still works -- but on any
        # other robot that fallback is meaningless, which is the whole reason
        # to prefer the discovered pair.
        st_kp = st_kd = None
        if not args_cli.st_ignore_discovered_gains:
            _cfg = load_json("outputs/discovered_config.json") or {}
            _cg = (_cfg.get("probes") or {}).get("controller_gains")
            if _cg:
                st_kp, st_kd = _cg.get("kp"), _cg.get("kd")
                print(f"\nUsing discovered controller gains: "
                      f"Kp={st_kp:.2f} Kd={st_kd:.2f}")
            else:
                print("\n(no controller_gains section in discovered_config.json; "
                      "using the gains this scene was spawned with. Run "
                      "--controller-gains first to discover them.)")

        st_result = success_threshold_probe(
            sim=sim, scene=scene, robot=robot,
            workspace_points=ws_result.point_cloud,
            n_targets=args_cli.n_targets,
            seed=args_cli.seed,
            ee_body_name=args_cli.ee_body_name,
            statistic=args_cli.st_statistic,
            arm_stiffness=st_kp,
            arm_damping=st_kd,
        )
        print(f"\n=== Success-Threshold Probe Results ===")
        print(f"Threshold ({st_result.statistic}): "
              f"{st_result.threshold_m * 100:.2f} cm  ({st_result.threshold_m:.5f} m)")
        print(f"Targets measured: {st_result.n_measured} / {st_result.n_targets}")
        print(f"Convergence rate: {st_result.convergence_rate:.1%}")
        print(f"EE frame:         {st_result.ee_frame}")
        print(f"Gravity z:        {st_result.gravity_z}")
        print(f"Arm gains:        Kp={st_result.arm_stiffness:.2f}  Kd={st_result.arm_damping:.2f}")
        print(f"Position error percentiles (cm):")
        for k, v in st_result.position_error_percentiles_m.items():
            print(f"  {k:>4}: {v * 100:7.3f}")
        print(f"Runtime:          {st_result.runtime_seconds:.2f}s")
 
        np.save(_out("outputs/diagnostics/success_threshold_errors.npy"), st_result.errors_m)
        _save_error_hist(st_result.errors_m,
                         _out("outputs/diagnostics/success_threshold_hist.png"),
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
            arm_stiffness=st_result.arm_stiffness,
            arm_damping=st_result.arm_damping,
            units=st_result.units,
            seed=st_result.seed,
        )
        save_json(discovered.model_dump(), "outputs/discovered_config.json")
        print("success_threshold section written to outputs/discovered_config.json")


if __name__ == "__main__":
    # Deliberately NOT calling simulation_app.close(): it hangs in headless
    # mode with a live PhysX scene (that's why it was commented out here
    # originally). os._exit tears the process down without it -- and because
    # it also skips atexit, Kit can't run its shutdown and force the exit code
    # back to 0, which is what made a crashed probe look successful to
    # run_skill.py. Print the traceback ourselves first, since nothing after
    # this point will.
    _rc = 0
    try:
        main()
    except Exception:
        traceback.print_exc()
        _rc = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_rc)