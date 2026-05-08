"""One-off experiment: sweep N to find the right value for workspace_probe.

Plots bbox volume (y) vs N (x), with error bars across seeds.
Run once per robot. Output: outputs/diagnostics/convergence_curve.png.

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


def main():
    sim_cfg = SimulationCfg(device="cuda:0")
    sim = SimulationContext(sim_cfg)

    scene_cfg = FrankaSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    robot = scene["robot"]

    # Choose N values that are multiples of num_envs for clean batching.
    # Small N values are less interesting once num_envs=1000.
    N_values = [1000, 2000, 5000, 10000, 20000]

    print(f"Sweeping N values: {N_values}")
    print(f"Seeds per N: {args_cli.n_seeds}")

    results = {N: [] for N in N_values}
    for N in N_values:
        for seed in range(args_cli.n_seeds):
            r = workspace_probe(scene, robot, n_samples=N, seed=seed)
            vol = bbox_volume(r.bounds)
            results[N].append(vol)
            print(f"  N={N:>6}, seed={seed}: vol={vol:.4f} m³, "
                  f"runtime={r.runtime_seconds:.2f}s")

    means = [np.mean(results[N]) for N in N_values]
    stds = [np.std(results[N]) for N in N_values]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(N_values, means, yerr=stds, marker="o", capsize=5,
                linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_xlabel("N (number of sampled configurations)")
    ax.set_ylabel("Bounding box volume (m³)")
    ax.set_title("Workspace probe convergence (Franka)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = "outputs/diagnostics/convergence_curve.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nConvergence curve saved to {out}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()