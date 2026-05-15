"""Unit tests for workspace_probe pure functions. Runs without Isaac Lab."""
import numpy as np
import torch

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes.workspace_probe import sample_configs, fit_aabb


# Franka URDF joint limits
FRANKA_LIMITS = torch.tensor([
    [-2.8973,  2.8973],
    [-1.7628,  1.7628],
    [-2.8973,  2.8973],
    [-3.0718, -0.0698],
    [-2.8973,  2.8973],
    [-0.0175,  3.7525],
    [-2.8973,  2.8973],
])
FRANKA_LO = FRANKA_LIMITS[:, 0]
FRANKA_HI = FRANKA_LIMITS[:, 1]


def test_samples_within_limits():
    rng = torch.Generator().manual_seed(0)
    cfg = sample_configs(FRANKA_LO, FRANKA_HI, num_envs=4096, generator=rng)
    assert (cfg >= FRANKA_LO).all(), "Sample below lower limit"
    assert (cfg <= FRANKA_HI).all(), "Sample above upper limit"
    assert cfg.shape == (4096, 7)


def test_determinism():
    """Same seed → identical samples."""
    cfg1 = sample_configs(FRANKA_LO, FRANKA_HI, 1024, torch.Generator().manual_seed(42))
    cfg2 = sample_configs(FRANKA_LO, FRANKA_HI, 1024, torch.Generator().manual_seed(42))
    torch.testing.assert_close(cfg1, cfg2)


def test_marginal_uniformity():
    """Each joint's marginal should be ~uniform via KS test."""
    from scipy.stats import kstest
    cfg = sample_configs(FRANKA_LO, FRANKA_HI, 50000, torch.Generator().manual_seed(0))
    normed = (cfg - FRANKA_LO) / (FRANKA_HI - FRANKA_LO)
    for j in range(cfg.shape[1]):
        stat, p = kstest(normed[:, j].numpy(), "uniform")
        assert p > 0.01, f"Joint {j} not uniform (p={p:.4f})"


def test_bbox_contains_all_points():
    pc = np.random.uniform(-1, 1, size=(1000, 3))
    bounds = fit_aabb(pc)
    for axis_idx, axis_name in enumerate(["x", "y", "z"]):
        assert pc[:, axis_idx].min() >= bounds[axis_name][0]
        assert pc[:, axis_idx].max() <= bounds[axis_name][1]


def test_bbox_tight():
    """Bbox bounds equal min/max of point cloud."""
    pc = np.array([[0., 0., 0.], [1., 2., 3.], [-1., -2., -3.]])
    bounds = fit_aabb(pc)
    assert bounds == {"x": [-1.0, 1.0], "y": [-2.0, 2.0], "z": [-3.0, 3.0]}


def test_bbox_single_point():
    pc = np.array([[1., 2., 3.]])
    bounds = fit_aabb(pc)
    assert bounds == {"x": [1.0, 1.0], "y": [2.0, 2.0], "z": [3.0, 3.0]}


if __name__ == "__main__":
    test_samples_within_limits(); print("✓ samples_within_limits")
    test_determinism(); print("✓ determinism")
    test_marginal_uniformity(); print("✓ marginal_uniformity")
    test_bbox_contains_all_points(); print("✓ bbox_contains_all_points")
    test_bbox_tight(); print("✓ bbox_tight")
    test_bbox_single_point(); print("✓ bbox_single_point")
    print("\nAll unit tests passed.")