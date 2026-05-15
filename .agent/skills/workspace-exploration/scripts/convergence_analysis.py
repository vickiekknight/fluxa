"""One-off experiment: sweep N to find the right value for workspace_probe.

Run once per robot. Outputs three figures:
  - convergence_curve.png     : bbox volume vs N (the headline metric)
  - convergence_per_axis.png  : x_min, x_max, y_min, y_max, z_min, z_max vs N
  - convergence_extents.png   : bbox width, depth, height vs N

Usage:
    python scripts/convergence_analysis.py
"""
import argparse
import os
import sys

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=1000)
parser.add_argument("--n_seeds", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext, SimulationCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402
from isaaclab_assets import FRANKA_PANDA_CFG  # noqa: E402

from probes.workspace_probe import workspace_probe  # noqa: E402


@configclass
class FrankaSceneCfg(InteractiveSceneCfg):
    robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def bbox_volume(bounds: dict) -> float:
    return ((bounds["x"][1] - bounds["x"][0]) *
            (bounds["y"][1] - bounds["y"][0]) *
            (bounds["z"][1] - bounds["z"][0]))


def plot_volume(N_values, results, out_path: str):
    """Single curve: bbox volume vs N."""
    means = [np.mean([bbox_volume(r) for r in results[N]]) for N in N_values]
    stds = [np.std([bbox_volume(r) for r in results[N]]) for N in N_values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(N_values, means, yerr=stds, marker="o", capsize=5,
                linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xticks(N_values)
    ax.set_xticklabels(N_values)
    ax.set_xlabel("N (number of sampled configurations)")
    ax.set_ylabel("Bounding box volume (m³)")
    ax.set_title("Workspace probe convergence: bbox volume (Franka)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_axis(N_values, results, out_path: str):
    """Six lines on one plot: each face of the bbox vs N."""
    # Compute mean and std for each face across seeds at each N.
    faces = ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    face_keys = [("x", 0), ("x", 1), ("y", 0), ("y", 1), ("z", 0), ("z", 1)]

    means = {f: [] for f in faces}
    stds = {f: [] for f in faces}
    for N in N_values:
        for face, (axis, idx) in zip(faces, face_keys):
            vals = [r[axis][idx] for r in results[N]]
            means[face].append(np.mean(vals))
            stds[face].append(np.std(vals))

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"x_min": "C0", "x_max": "C0",
              "y_min": "C1", "y_max": "C1",
              "z_min": "C2", "z_max": "C2"}
    linestyles = {"x_min": "--", "x_max": "-",
                  "y_min": "--", "y_max": "-",
                  "z_min": "--", "z_max": "-"}

    for face in faces:
        ax.errorbar(N_values, means[face], yerr=stds[face],
                    marker="o", capsize=4, label=face,
                    color=colors[face], linestyle=linestyles[face])

    ax.set_xscale("log")
    ax.set_xticks(N_values)
    ax.set_xticklabels(N_values)
    ax.set_xlabel("N (number of sampled configurations)")
    ax.set_ylabel("Coordinate (m, robot-base frame)")
    ax.set_title("Workspace probe convergence: per-axis bounds (Franka)")
    ax.legend(loc="center right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_extents(N_values, results, out_path: str):
    """Three lines: width (x), depth (y), height (z) of bbox vs N."""
    extents = {"width (x)": [], "depth (y)": [], "height (z)": []}
    extents_std = {"width (x)": [], "depth (y)": [], "height (z)": []}

    for N in N_values:
        widths = [r["x"][1] - r["x"][0] for r in results[N]]
        depths = [r["y"][1] - r["y"][0] for r in results[N]]
        heights = [r["z"][1] - r["z"][0] for r in results[N]]
        extents["width (x)"].append(np.mean(widths))
        extents["depth (y)"].append(np.mean(depths))
        extents["height (z)"].append(np.mean(heights))
        extents_std["width (x)"].append(np.std(widths))
        extents_std["depth (y)"].append(np.std(depths))
        extents_std["height (z)"].append(np.std(heights))

    fig, ax = plt.subplots(figsize=(8, 6))
    for label in extents:
        ax.errorbar(N_values, extents[label], yerr=extents_std[label],
                    marker="o", capsize=4, label=label, linewidth=2)
    ax.set_xscale("log")
    ax.set_xticks(N_values)
    ax.set_xticklabels(N_values)    
    ax.set_xlabel("N (number of sampled configurations)")
    ax.set_ylabel("Bbox extent (m)")
    ax.set_title("Workspace probe convergence: bbox extents (Franka)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    sim_cfg = SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)

    scene_cfg = FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]

    N_values = [1000, 2000, 5000, 10000, 20000]
    print(f"Sweeping N values: {N_values}")
    print(f"Seeds per N: {args_cli.n_seeds}\n")

    # results[N] is a list of bounds dicts, one per seed.
    results = {N: [] for N in N_values}

    for N in N_values:
        for seed in range(args_cli.n_seeds):
            r = workspace_probe(scene, robot, n_samples=N, seed=seed)
            results[N].append(r.bounds)
            vol = bbox_volume(r.bounds)
            print(f"  N={N:>6}, seed={seed}: vol={vol:.4f} m³, "
                  f"x={r.bounds['x']}, y={r.bounds['y']}, z={r.bounds['z']}, "
                  f"runtime={r.runtime_seconds:.2f}s")

    out_dir = "outputs/diagnostics"
    os.makedirs(out_dir, exist_ok=True)

    plot_volume(N_values, results, f"{out_dir}/convergence_curve.png")
    plot_per_axis(N_values, results, f"{out_dir}/convergence_per_axis.png")
    plot_extents(N_values, results, f"{out_dir}/convergence_extents.png")

    print(f"\nFigures saved to {out_dir}/")
    print("  convergence_curve.png     — bbox volume vs N")
    print("  convergence_per_axis.png  — each face of the bbox vs N")
    print("  convergence_extents.png   — bbox width/depth/height vs N")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
    sys.exit(0)