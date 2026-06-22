"""Unit tests for joint_limits_probe pure logic. Runs without Isaac Lab/cuRobo.

The orchestrator is testable off-sim because the collision labeler is injectable:
a deterministic fake labeler stands in for PhysX.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from probes.joint_limits_probe import sample_configs_sobol, joint_limits_probe

# Franka URDF joint limits (7 arm joints)
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


# ---------- Sobol sampler ----------

def test_sobol_within_limits():
    cfg = sample_configs_sobol(FRANKA_LO, FRANKA_HI, n=4096, seed=0)
    assert (cfg >= FRANKA_LO).all(), "Sample below lower limit"
    assert (cfg <= FRANKA_HI).all(), "Sample above upper limit"
    assert cfg.shape == (4096, 7)


def test_sobol_determinism():
    """Same seed → identical samples (fresh engine each call)."""
    a = sample_configs_sobol(FRANKA_LO, FRANKA_HI, 1024, seed=42)
    b = sample_configs_sobol(FRANKA_LO, FRANKA_HI, 1024, seed=42)
    torch.testing.assert_close(a, b)


def test_sobol_seed_changes_sample():
    a = sample_configs_sobol(FRANKA_LO, FRANKA_HI, 1024, seed=1)
    b = sample_configs_sobol(FRANKA_LO, FRANKA_HI, 1024, seed=2)
    assert not torch.allclose(a, b), "Different seeds gave identical samples"


def test_sobol_coverage():
    """Low-discrepancy: every per-joint bin is populated at moderate N."""
    n, bins = 4096, 16
    cfg = sample_configs_sobol(FRANKA_LO, FRANKA_HI, n, seed=0)
    normed = (cfg - FRANKA_LO) / (FRANKA_HI - FRANKA_LO)     # -> [0, 1)
    for j in range(cfg.shape[1]):
        hist = torch.histc(normed[:, j], bins=bins, min=0.0, max=1.0)
        assert (hist > 0).all(), f"Joint {j} has an empty bin (poor coverage)"


def test_sobol_low_discrepancy_mean():
    """Each coordinate's mean sits very close to 0.5 (tighter than iid uniform)."""
    cfg = sample_configs_sobol(FRANKA_LO, FRANKA_HI, 4096, seed=0)
    normed = (cfg - FRANKA_LO) / (FRANKA_HI - FRANKA_LO)
    means = normed.mean(dim=0)
    assert torch.all((means - 0.5).abs() < 0.02), f"Per-joint means off: {means}"


# ---------- Orchestrator (fake labeler, no sim) ----------

class _FakeData:
    def __init__(self, limits):
        # orchestrator reads robot.data.soft_joint_pos_limits[0] -> (J, 2)
        self.soft_joint_pos_limits = limits.unsqueeze(0)    # (1, J, 2)


class _FakeRobot:
    def __init__(self, limits, device="cpu"):
        self.device = device
        self.data = _FakeData(limits)


class _FakeScene:
    def __init__(self, num_envs):
        self.num_envs = num_envs


def _collide_when_joint3_high(threshold):
    """Fake labeler: a config 'collides' iff arm joint 3 exceeds threshold.
    Deterministic, depends only on the config — stands in for the sim."""
    def labeler(configs):
        return configs[:, 3] > threshold
    return labeler


def test_probe_pads_to_multiple_of_num_envs():
    robot, scene = _FakeRobot(FRANKA_LIMITS), _FakeScene(num_envs=64)
    res = joint_limits_probe(sim=None, scene=scene, robot=robot,
                             n_samples=100, seed=0,            # 100 -> padded to 128
                             labeler=_collide_when_joint3_high(-1.5))
    assert res.n_sampled == 128
    assert res.all_configs.shape == (128, 7)


def test_probe_safe_set_matches_labels():
    robot, scene = _FakeRobot(FRANKA_LIMITS), _FakeScene(num_envs=64)
    thr = -1.5
    res = joint_limits_probe(sim=None, scene=scene, robot=robot,
                             n_samples=512, seed=0,
                             labeler=_collide_when_joint3_high(thr))

    # internal bookkeeping is consistent
    assert res.n_safe + int(res.labels.sum()) == res.n_sampled
    assert res.safe_configs.shape[0] == res.n_safe
    np.testing.assert_allclose(res.collision_rate, res.labels.mean())

    # safe configs are exactly the non-colliding rows, in order
    expected_safe = res.all_configs[res.all_configs[:, 3] <= thr]
    np.testing.assert_array_equal(res.safe_configs, expected_safe)


def test_probe_determinism():
    """Same seed + same labeler → identical labels and safe set."""
    robot, scene = _FakeRobot(FRANKA_LIMITS), _FakeScene(num_envs=64)
    mk = lambda: joint_limits_probe(sim=None, scene=scene, robot=robot,
                                    n_samples=256, seed=7,
                                    labeler=_collide_when_joint3_high(-1.5))
    a, b = mk(), mk()
    np.testing.assert_array_equal(a.labels, b.labels)
    np.testing.assert_array_equal(a.safe_configs, b.safe_configs)


if __name__ == "__main__":
    test_sobol_within_limits(); print("✓ sobol_within_limits")
    test_sobol_determinism(); print("✓ sobol_determinism")
    test_sobol_seed_changes_sample(); print("✓ sobol_seed_changes_sample")
    test_sobol_coverage(); print("✓ sobol_coverage")
    test_sobol_low_discrepancy_mean(); print("✓ sobol_low_discrepancy_mean")
    test_probe_pads_to_multiple_of_num_envs(); print("✓ probe_pads_to_multiple_of_num_envs")
    test_probe_safe_set_matches_labels(); print("✓ probe_safe_set_matches_labels")
    test_probe_determinism(); print("✓ probe_determinism")
    print("\nAll unit tests passed.")