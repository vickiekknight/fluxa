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

def sample_configs(lo: torch.Tensor, hi: torch.Tensor, num_envs: int,
                   generator: torch.Generator) -> torch.Tensor:
    """Sample joint configurations uniformly within [lo, hi].
    
    Args:
        lo: shape (n_joints,) lower bounds
        hi: shape (n_joints,) upper bounds
        num_envs: how many configs to sample
        generator: torch.Generator for reproducibility
    
    Returns:
        configs: shape (num_envs, n_joints)
    """
    n_joints = lo.shape[0]
    u = torch.rand((num_envs, n_joints), generator=generator, device=lo.device)
    return lo + u * (hi - lo)

def fit_aabb(point_cloud: np.ndarray) -> dict:
    """Fit an axis-aligned bounding box to a point cloud.
    
    Args:
        point_cloud: shape (N, 3)
    
    Returns:
        dict with keys "x", "y", "z", each a [min, max] list.
    """
    return {
        "x": [float(point_cloud[:, 0].min()), float(point_cloud[:, 0].max())],
        "y": [float(point_cloud[:, 1].min()), float(point_cloud[:, 1].max())],
        "z": [float(point_cloud[:, 2].min()), float(point_cloud[:, 2].max())],
    }


def run_fk_batch(scene, robot, configs: torch.Tensor, ee_body_idx: int) -> torch.Tensor:
    """Write joint configs to sim, propagate kinematics, return EE positions
    in robot-base frame.
    
    Args:
        scene: Isaac Lab InteractiveScene
        robot: Articulation handle
        configs: shape (num_envs, n_joints)
        ee_body_idx: index of the EE body
    
    Returns:
        ee_pos_rel: shape (num_envs, 3), in robot-base frame
    """
    zero_vel = torch.zeros_like(configs)
    robot.write_joint_state_to_sim(configs, zero_vel)
    scene.update(dt=0.0)
    
    ee_pos_w = robot.data.body_pos_w[:, ee_body_idx]
    base_pos_w = robot.data.root_pos_w
    return ee_pos_w - base_pos_w

def workspace_probe(scene, robot, n_samples: int, seed: int = 0,
                    ee_body_name: str = "panda_hand") -> WorkspaceProbeResult:
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
    
    n_batches = (n_samples + num_envs - 1) // num_envs
    n_actual = n_batches * num_envs
    
    joint_limits = robot.data.soft_joint_pos_limits[0]
    lo, hi = joint_limits[:, 0], joint_limits[:, 1]
    
    ee_body_idx = robot.body_names.index(ee_body_name)
    
    rng = torch.Generator(device=device).manual_seed(seed)
    all_ee_positions = []
    
    for batch_idx in range(n_batches):
        configs = sample_configs(lo, hi, num_envs, rng)
        
        ee_pos_rel = run_fk_batch(scene, robot, configs, ee_body_idx)
        
        all_ee_positions.append(ee_pos_rel.cpu().numpy())
    
    point_cloud = np.concatenate(all_ee_positions, axis=0)
    bounds = fit_aabb(point_cloud)
    
    return WorkspaceProbeResult(
        bounds=bounds, point_cloud=point_cloud,
        n_sampled=n_actual, n_valid=n_actual,
        runtime_seconds=time.time() - start,
    )