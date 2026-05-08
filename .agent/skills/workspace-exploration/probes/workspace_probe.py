"""Workspace probe: discovers reachable end-effector region via batched FK.

Algorithm:
1. Sample N joint configurations uniformly within URDF joint limits.
2. Write configs to parallel envs and read back EE positions.
3. Convert to robot-base-relative coordinates.
4. Fit axis-aligned bounding box.
"""
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class WorkspaceProbeResult:
    bounds: dict             # {"x": [min, max], "y": [...], "z": [...]}
    point_cloud: np.ndarray  # shape (N, 3), EE positions in robot-base frame
    n_sampled: int
    n_valid: int             # equals n_sampled in v1 (no filtering)
    runtime_seconds: float


def workspace_probe(
    scene,
    robot,
    n_samples: int,
    seed: int = 0,
    ee_body_name: str = "panda_hand",
) -> WorkspaceProbeResult:
    """Probe the reachable workspace of `robot` in `scene` via random-config FK.
    
    Args:
        scene: an Isaac Lab InteractiveScene with `num_envs` parallel envs.
        robot: the Articulation handle inside the scene.
        n_samples: total number of joint configs to sample. Padded up to a
            multiple of `scene.num_envs` if not already.
        seed: RNG seed for reproducibility.
        ee_body_name: name of the end-effector body to read position from.
    
    Returns:
        WorkspaceProbeResult.
    """
    start = time.time()
    device = robot.device
    num_envs = scene.num_envs

    # Round n_samples up to a multiple of num_envs so batching is clean.
    n_batches = (n_samples + num_envs - 1) // num_envs
    n_actual = n_batches * num_envs

    # Joint limits — Isaac Lab exposes per-env soft limits; use env 0's.
    # Shape: (n_joints, 2) with columns [lo, hi].
    joint_limits = robot.data.soft_joint_pos_limits[0]
    lo = joint_limits[:, 0]
    hi = joint_limits[:, 1]
    n_joints = lo.shape[0]

    # Look up EE body index by name.
    ee_body_idx = robot.body_names.index(ee_body_name)

    rng = torch.Generator(device=device).manual_seed(seed)
    all_ee_positions = []

    for _ in range(n_batches):
        # Uniform sample in [lo, hi] for each joint, for each env.
        u = torch.rand((num_envs, n_joints), generator=rng, device=device)
        configs = lo + u * (hi - lo)

        # Write configs to sim (zero velocity).
        zero_vel = torch.zeros_like(configs)
        robot.write_joint_state_to_sim(configs, zero_vel)

        # Propagate kinematic state without stepping physics.
        scene.update(dt=0.0)

        # Read EE world position and convert to robot-base frame.
        ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]   # (num_envs, 3)
        base_pos_w = robot.data.root_pos_w                  # (num_envs, 3)
        ee_pos_rel = ee_pos_w - base_pos_w

        all_ee_positions.append(ee_pos_rel.cpu().numpy())

    point_cloud = np.concatenate(all_ee_positions, axis=0)  # (n_actual, 3)

    bounds = {
        "x": [float(point_cloud[:, 0].min()), float(point_cloud[:, 0].max())],
        "y": [float(point_cloud[:, 1].min()), float(point_cloud[:, 1].max())],
        "z": [float(point_cloud[:, 2].min()), float(point_cloud[:, 2].max())],
    }

    return WorkspaceProbeResult(
        bounds=bounds,
        point_cloud=point_cloud,
        n_sampled=n_actual,
        n_valid=n_actual,
        runtime_seconds=time.time() - start,
    )