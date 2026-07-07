"""Top-level skill orchestrator: NL description → discovered_config.json.

Usage:
    python scripts/run_skill.py "train the franka to reach random targets on a table"
"""
import argparse
import os
import sys
import numpy as np

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("description", type=str,
                    help="Natural-language task description.")
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--n_samples", type=int, default=2000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--skip-validation", action="store_true",
                    help="Skip FK validation (CuRobo check). Default: validation runs.")
parser.add_argument("--n_validate", type=int, default=50)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass
from isaaclab_assets import FRANKA_PANDA_CFG

from parser.task_parser import parse_task_description
from probes.workspace_probe import workspace_probe
from probes.joint_limits_probe import joint_limits_probe
from helpers.io import save_json, save_scatter_plot

from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim.schemas import ArticulationRootPropertiesCfg

_SKILLS_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.insert(0, _SKILLS_DIR)
from common.schemas import (
    DiscoveredConfig, RobotConfig, ProbeResults,
    WorkspaceProbeResult, JointLimitsProbeResult,
)


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


# Robot-frame z = 0 corresponds to the table surface in v1.
TABLE_HEIGHT = 0.0


def setup_scene(robot_name: str, num_envs: int):
    """Spawn a scene with `num_envs` copies of the requested robot.

    Gravity is disabled: the joint-limits probe labels self-collisions from
    PhysX contact forces, and with a fixed base + no ground + no gravity the
    only contacts are link-link self-collisions (matches run_probe.py, under
    which the ~11% collision rate was validated).
    """
    sim_cfg = SimulationCfg(device="cuda:0", gravity=(0.0, 0.0, 0.0))
    sim = SimulationContext(sim_cfg)

    if robot_name == "franka":
        scene_cfg = FrankaSceneCfg(num_envs=num_envs, env_spacing=2.0)
    else:
        raise NotImplementedError(
            f"v1 supports robot_name='franka' only. Got: {robot_name!r}"
        )

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    return sim, scene, scene["robot"]


def main(user_description: str, num_envs: int, n_samples: int, seed: int):
    # === Stage 1: Parse ===
    task_spec = parse_task_description(user_description)
    save_json(task_spec, "outputs/task_spec.json")
    print(f"Parsed: {task_spec.task_type} on {task_spec.robot_name}")
    if task_spec.constraints:
        print(f"Constraints: {task_spec.constraints}")

    # === Stage 2: Spawn sim ===
    sim, scene, robot = setup_scene(task_spec.robot_name, num_envs)
    print(f"Spawned {scene.num_envs} parallel envs.")

    # Validate FK before any probe runs. Catches silent kinematics regressions
    # (URDF/Isaac Lab version drift) before they poison downstream stages.
    if not args_cli.skip_validation:
        from tests.test_workspace_integration import run_integration_test
        run_integration_test(
            scene, robot,
            n_configs=args_cli.n_validate,
            seed=args_cli.seed,
        )

    # === Stage 3: Run probes ===
    if task_spec.task_type != "reach":
        raise NotImplementedError(
            f"Task type {task_spec.task_type!r} not supported in v1"
        )

    ws_result = workspace_probe(
        scene=scene,
        robot=robot,
        n_samples=n_samples,
        seed=seed,
        ee_body_name=task_spec.ee_body_name,
    )
    jl_result = joint_limits_probe(
        sim=sim, scene=scene, robot=robot,
        n_samples=n_samples, seed=seed,
    )

    # Absolute path: reach_task.py loads this inside the Isaac Sim server
    # process, which may not share this process's working directory.
    os.makedirs("outputs/diagnostics", exist_ok=True)
    safe_path = os.path.abspath("outputs/diagnostics/safe_configs.npy")
    np.save(safe_path, jl_result.safe_configs)
    print(f"Joint-limits: {jl_result.n_safe}/{jl_result.n_sampled} safe "
          f"({jl_result.collision_rate:.1%} collide)")

    # === Stage 4: Apply constraints ===
    if task_spec.constraints.get("surface") == "table":
        z_lo = max(ws_result.bounds["z"][0], TABLE_HEIGHT)
    else:
        z_lo = ws_result.bounds["z"][0]
    z_hi = ws_result.bounds["z"][1]

    # === Stage 5: Write outputs ===
    discovered = DiscoveredConfig(
        robot=RobotConfig(name=task_spec.robot_name),
        probes=ProbeResults(
            workspace=WorkspaceProbeResult(
                x=tuple(ws_result.bounds["x"]),
                y=tuple(ws_result.bounds["y"]),
                z=(z_lo, z_hi),
            ),
            joint_limits=JointLimitsProbeResult(
                n_sampled=jl_result.n_sampled,
                n_safe=jl_result.n_safe,
                collision_rate=jl_result.collision_rate,
                seed=jl_result.seed,
                joint_lower=jl_result.joint_lower.tolist(),
                joint_upper=jl_result.joint_upper.tolist(),
                safe_config_path=safe_path,
            ),
        ),
    )
    save_json(discovered.model_dump(), "outputs/discovered_config.json")

    save_scatter_plot(
        ws_result,
        "outputs/diagnostics/workspace_scatter.png",
        title_suffix=task_spec.robot_name,
    )
    print("\n=== Discovered Config ===")
    print(f"Workspace bounds: {discovered.probes.workspace}")
    print(f"Safe configs:     {jl_result.n_safe} -> {safe_path}")
    print("\nOutputs written:")
    print("  outputs/task_spec.json")
    print("  outputs/discovered_config.json")
    print("  outputs/diagnostics/workspace_scatter.png")


if __name__ == "__main__":
    try:
        main(
            args_cli.description,
            args_cli.num_envs,
            args_cli.n_samples,
            args_cli.seed,
        )
    finally:
        simulation_app.close()
    os._exit(0)