"""CuRobo cross-validation for success_threshold_probe's Phase A (kinematic
IK solve).

test_workspace_integration.py checks Isaac Lab FK against CuRobo FK on
*random* configs. This checks the same thing on configs that
success_threshold_probe's own DLS solver actually produced -- FKing the
solved joint config through CuRobo (a backend independent of Isaac Lab, which
computed the probe's own `ik_residual`) and comparing against the *intended
target*, not against Isaac Lab's own FK of itself. That's the "solver bug"
half of the "distribution problem vs solver bug" question the probe's
docstring originally posed: if this passes, any large `err` the probe reports
downstream is a Phase B (dynamics/control) issue, not Phase A.

Exposes `run_success_threshold_validation(sim, scene, robot, workspace_points, ...)`
for callers with an already-initialized Isaac Lab scene with a Franka.
"""
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

from probes.success_threshold_probe import solve_kinematic_ik


def run_success_threshold_validation(sim, scene, robot, workspace_points, *,
                                     n_configs: int = 50, seed: int = 0,
                                     ee_body_name: str = "panda_hand",
                                     arm_joint_expr: str = "panda_joint.*",
                                     n_ik_iters: int = 30,
                                     atol: float = 5e-3) -> dict:
    """Cross-check success_threshold_probe's Phase A DLS solve against CuRobo.

    Samples `n_configs` targets from `workspace_points` (same distribution
    success_threshold_probe draws from), runs the probe's own
    `solve_kinematic_ik` to get a joint solution for each, then FKs that
    solution through CuRobo and checks it lands within `atol` of the
    intended target.

    Args:
        sim, scene, robot: Isaac Lab handles, scene must have gravity ON
            (matches success_threshold_probe's validated condition) and
            num_envs >= n_configs.
        workspace_points: (N, 3) reachable-point cloud, e.g. from
            workspace_probe's result -- same distribution the probe samples
            targets from.
        n_configs: number of targets to solve and cross-check.
        seed: RNG seed for target sampling (independent of the probe's own
            seed -- this validates the solver, not a specific probe run).
        atol: absolute tolerance in meters (default 5 mm: ~2mm FK-consistency
            margin, matching test_workspace_integration.py's validated
            tolerance, plus margin for DLS residual convergence).

    Returns:
        dict with max_error_mm, mean_error_mm, n_configs.

    Raises:
        AssertionError if max error exceeds atol.
    """
    print("\n=== Success-Threshold Validation (Isaac Lab DLS solve vs CuRobo FK) ===")

    device = robot.device
    num_envs = scene.num_envs
    assert n_configs <= num_envs, (
        f"Need scene.num_envs ({num_envs}) >= n_configs ({n_configs})."
    )

    # --- 1. CuRobo kinematics (same setup as test_workspace_integration.py) ---
    tensor_args = TensorDeviceType()
    robot_yaml = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_yaml, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    curobo_joint_names = kin_model.joint_names

    missing = [j for j in curobo_joint_names if j not in robot.joint_names]
    assert not missing, (
        f"CuRobo joints not present in Isaac Lab robot: {missing}\n"
        f"  CuRobo:    {curobo_joint_names}\n"
        f"  Isaac Lab: {robot.joint_names}"
    )
    # Map from Isaac Lab's full joint-vector order into CuRobo's own order --
    # do not assume arm_joint_expr's match order coincides with CuRobo's.
    idx_tensor = torch.tensor(
        [robot.joint_names.index(j) for j in curobo_joint_names],
        device=device, dtype=torch.long,
    )

    # --- 2. Same setup success_threshold_probe uses for Phase A ---
    ee_idx = robot.body_names.index(ee_body_name)
    arm_ids, _ = robot.find_joints(arm_joint_expr, preserve_order=True)
    arm_ids_t = torch.as_tensor(arm_ids, device=device, dtype=torch.long)
    ee_jacobi_idx = ee_idx - 1 if robot.is_fixed_base else ee_idx
    q_lim = robot.data.soft_joint_pos_limits[:, arm_ids_t, :]
    default_q = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(default_q)
    dt = sim.get_physics_dt()

    # --- 3. Sample targets from the same point cloud the probe draws from ---
    import numpy as np
    pts = torch.as_tensor(np.asarray(workspace_points), device=device, dtype=torch.float32)
    cpu_gen = torch.Generator().manual_seed(seed)
    sample_idx = torch.randint(0, pts.shape[0], (num_envs,), generator=cpu_gen)
    p_t = pts[sample_idx.to(device)]

    # --- 4. Run the probe's own Phase A solve ---
    q, ik_residual, _diff_ik = solve_kinematic_ik(
        sim, scene, robot, p_t,
        ee_idx=ee_idx, arm_ids_t=arm_ids_t, ee_jacobi_idx=ee_jacobi_idx,
        q_lim=q_lim, default_q=default_q, zero_vel=zero_vel, dt=dt,
        n_ik_iters=n_ik_iters,
    )

    # --- 5. Independent CuRobo FK of the solved configs ---
    arm_configs = q[:n_configs].index_select(1, idx_tensor).to(
        device="cuda:0", dtype=torch.float32).contiguous()
    out = kin_model.get_state(arm_configs)
    ee_curobo = out.ee_position.to(device)                                # (n_configs, 3)

    # --- 6. Compare against the *intended target*, not Isaac Lab's own FK ---
    err = (ee_curobo - p_t[:n_configs]).norm(dim=1)
    max_err_mm = err.max().item() * 1000
    mean_err_mm = err.mean().item() * 1000

    print(f"  N configs:      {n_configs} (seed={seed})")
    print(f"  Isaac Lab ik_residual (own FK) p90: "
          f"{ik_residual[:n_configs].quantile(0.9).item() * 1000:.3f} mm")
    print(f"  CuRobo cross-check max error:  {max_err_mm:.3f} mm")
    print(f"  CuRobo cross-check mean error: {mean_err_mm:.3f} mm")

    if max_err_mm > atol * 1000:
        worst_idx = err.argsort(descending=True)[:5]
        print(f"  ✗ FAILED — exceeds {atol * 1000:.1f} mm tolerance")
        for i in worst_idx.tolist():
            print(f"    cfg {i}: err={err[i].item() * 1000:.2f} mm | "
                  f"target={[f'{v:+.3f}' for v in p_t[i].tolist()]} | "
                  f"curobo={[f'{v:+.3f}' for v in ee_curobo[i].tolist()]}")
        raise AssertionError(
            f"Success-threshold validation failed: max {max_err_mm:.3f} mm > "
            f"{atol * 1000:.1f} mm"
        )

    print(f"  ✓ PASSED — within {atol * 1000:.1f} mm tolerance.\n")
    return {
        "max_error_mm": max_err_mm,
        "mean_error_mm": mean_err_mm,
        "n_configs": n_configs,
    }
