"""Top-level skill orchestrator: NL description → discovered_config.json.

Usage:
    python scripts/run_skill.py "train the franka to reach random targets on a table"
"""
import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("description", type=str,
                    help="Natural-language task description.")
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--n_samples", type=int, default=10000)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_assets import FRANKA_PANDA_CFG  # noqa: E402

from parser.task_parser import parse_task_description  # noqa: E402
from probes.workspace_probe import workspace_probe  # noqa: E402
from utils.io import save_json, save_scatter_plot  # noqa: E402


# Robot-frame z = 0 corresponds to the table surface in v1.
TABLE_HEIGHT = 0.0


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def setup_scene(robot_name: str, num_envs: int):
    """Spawn a scene with `num_envs` copies of the requested robot."""
    sim_cfg = SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)

    if robot_name == "franka":
        scene_cfg = FrankaSceneCfg(num_envs=num_envs, env_spacing=2.0)
    else:
        raise NotImplementedError(
            f"v1 supports robot_name='franka' only. Got: {robot_name!r}"
        )

    scene = InteractiveScene(scene_cfg)
    sim.reset()
    return scene, scene["robot"]


def main(user_description: str, num_envs: int, n_samples: int, seed: int):
    # === Stage 1: Parse ===
    task_spec = parse_task_description(user_description)
    save_json(task_spec, "outputs/task_spec.json")
    print(f"Parsed: {task_spec.task_type} on {task_spec.robot_name}")
    if task_spec.constraints:
        print(f"Constraints: {task_spec.constraints}")

    # === Stage 2: Spawn sim ===
    scene, robot = setup_scene(task_spec.robot_name, num_envs)
    print(f"Spawned {scene.num_envs} parallel envs.")

    # === Stage 3: Run probes ===
    discovered_config = {"robot": {"name": task_spec.robot_name}}

    if task_spec.task_type == "reach":
        result = workspace_probe(
            scene=scene,
            robot=robot,
            n_samples=n_samples,
            seed=seed,
            ee_body_name=task_spec.ee_body_name,
        )
        discovered_config["robot"]["workspace_bounds"] = result.bounds
        save_scatter_plot(
            result,
            "outputs/diagnostics/workspace_scatter.png",
            title_suffix=task_spec.robot_name,
        )
        print(f"Probe ran in {result.runtime_seconds:.2f}s")
    else:
        raise NotImplementedError(
            f"Task type {task_spec.task_type!r} not supported in v1"
        )

    # === Stage 4: Apply constraints ===
    if task_spec.constraints.get("surface") == "table":
        bounds = discovered_config["robot"]["workspace_bounds"]
        original_z_min = bounds["z"][0]
        bounds["z"][0] = max(original_z_min, TABLE_HEIGHT)
        if bounds["z"][0] != original_z_min:
            print(f"Clipped z_min from {original_z_min:.3f} "
                  f"to {bounds['z'][0]:.3f} (table constraint)")

    # === Stage 5: Write outputs ===
    save_json(discovered_config, "outputs/discovered_config.json")
    print("\n=== Discovered Config ===")
    print(f"Workspace bounds: {discovered_config['robot']['workspace_bounds']}")
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