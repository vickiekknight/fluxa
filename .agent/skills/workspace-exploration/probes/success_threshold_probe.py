"""Success-threshold probe: discovers how close the EE must be to a target
to count as success, for this robot under this task's dynamics.

Position-only DLS oracle: sample target positions, drive the arm to each with
Isaac Lab's differential-IK controller under gravity, measure settled EE error,
take a percentile (p90) as the threshold.

Phase A solves each target kinematically (teleporting each iteration, so no
dynamics or gravity are involved) to get a reference joint solution. Phase B
then drives the real, gravity-loaded arm toward that solution via a PD
position target plus an online damped-least-squares correction, at policy
control cadence, and measures where it actually settles.
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SuccessThresholdProbeResult:
    threshold_m: float
    statistic: str
    position_error_percentiles_m: dict
    n_targets: int
    n_measured: int
    convergence_rate: float
    ee_frame: str
    physics_dt: float
    gravity_z: Optional[float]
    # The PD gains this threshold was measured under. Recorded because the
    # threshold is only meaningful alongside them -- the same arm with weaker
    # gains droops under gravity and measures far worse.
    arm_stiffness: float
    arm_damping: float
    units: str
    seed: int
    runtime_seconds: float
    errors_m: np.ndarray
    targets_base: np.ndarray


def _read_gravity_z(sim) -> Optional[float]:
    try:
        return float(sim.cfg.gravity[2])
    except Exception:
        return None


def _percentiles(values: np.ndarray) -> dict:
    ps = [50, 75, 90, 95, 99]
    out = {f"p{p}": float(np.percentile(values, p)) for p in ps}
    out["mean"] = float(values.mean())
    out["max"] = float(values.max())
    return out


def solve_kinematic_ik(sim, scene, robot, p_t, *, ee_idx, arm_ids_t, ee_jacobi_idx,
                       q_lim, default_q, zero_vel, dt, n_ik_iters=30):
    """Pure-kinematic DLS IK solve for target EE positions `p_t` (robot-base
    frame): teleports each iteration so measured == commanded, so no dynamics
    or gravity are involved. Returns (q, residual, diff_ik) -- the solved full
    joint vector, per-env EE position error, and the DifferentialIKController
    left seeded with this target (success_threshold_probe's Phase B reuses it).

    Reused by success_threshold_probe (as Phase A) and by
    tests/test_success_threshold_validation.py (CuRobo FK cross-check).
    """
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms

    num_envs = scene.num_envs
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls",
    )
    diff_ik = DifferentialIKController(ik_cfg, num_envs=num_envs, device=robot.device)

    q = default_q.clone()
    robot.write_joint_state_to_sim(q, zero_vel)
    robot.set_joint_position_target(q)
    scene.write_data_to_sim(); sim.step(render=False); scene.update(dt)

    cur_pos_b, cur_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx])
    diff_ik.reset()
    diff_ik.set_command(p_t, ee_pos=cur_pos_b, ee_quat=cur_quat_b)

    for _ in range(n_ik_iters):                    # ~30 is plenty for DLS
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx])
        J = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids_t]
        q_arm = diff_ik.compute(ee_pos_b, ee_quat_b, J, q[:, arm_ids_t])
        q[:, arm_ids_t] = torch.clamp(q_arm, q_lim[..., 0], q_lim[..., 1])
        robot.write_joint_state_to_sim(q, zero_vel)
        robot.set_joint_position_target(q)
        scene.write_data_to_sim(); sim.step(render=False); scene.update(dt)

    ee_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w,
        robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx])
    residual = torch.linalg.norm(ee_pos_b - p_t, dim=-1)

    return q, residual, diff_ik


def success_threshold_probe(sim, scene, robot, workspace_points, *,
                            n_targets: int, seed: int = 0,
                            ee_body_name: str = "panda_hand",
                            arm_joint_expr: str = "panda_joint.*",
                            statistic: str = "p90",
                            n_ik_iters: int = 30,
                            decimation: int = 2,
                            n_control_steps: int = 300,
                            alpha: float = 0.2,
                            arm_stiffness: float = None,
                            arm_damping: float = None,
                            ee_settle_vel_tol: float = 1e-3) -> SuccessThresholdProbeResult:
    """Measure how close the EE settles to a target, under gravity.

    arm_stiffness/arm_damping: PD gains to apply to the arm joints before
        measuring, normally the pair discovered by controller_gains_probe.
        Applied at runtime via write_joint_stiffness_to_sim rather than baked
        into the ArticulationCfg, so the caller doesn't have to rebuild the
        scene. When None, whatever gains the robot was spawned with are used.

        This matters because the threshold is a property of the controller as
        much as the arm: the same robot with weak gains droops under gravity
        and reports a far worse threshold (that's the bug this probe was
        originally built around). Hardcoding a gain pair here would only ever
        be right for the robot it was tuned on.
    """
    from isaaclab.utils.math import subtract_frame_transforms

    start = time.time()
    device = robot.device
    num_envs = scene.num_envs
    dt = sim.get_physics_dt()

    grav_z = _read_gravity_z(sim)
    if grav_z is not None and abs(grav_z) < 1e-6:
        print("⚠️  success-threshold probe: gravity is ~0. Run with gravity ON.")

    ee_idx = robot.body_names.index(ee_body_name)
    arm_ids, _ = robot.find_joints(arm_joint_expr, preserve_order=True)
    arm_ids_t = torch.as_tensor(arm_ids, device=device, dtype=torch.long)

    if arm_stiffness is not None:
        robot.write_joint_stiffness_to_sim(arm_stiffness, joint_ids=arm_ids)
    if arm_damping is not None:
        robot.write_joint_damping_to_sim(arm_damping, joint_ids=arm_ids)
    applied_kp = float(robot.data.joint_stiffness[0, arm_ids_t].mean())
    applied_kd = float(robot.data.joint_damping[0, arm_ids_t].mean())
    print(f"[success-threshold] arm gains: Kp={applied_kp:.2f} Kd={applied_kd:.2f}"
          f"{' (from controller_gains_probe)' if arm_stiffness is not None else ' (as spawned)'}")
    ee_jacobi_idx = ee_idx - 1 if robot.is_fixed_base else ee_idx
    q_lim = robot.data.soft_joint_pos_limits[:, arm_ids_t, :] 

    default_q = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(default_q)

    pts = torch.as_tensor(np.asarray(workspace_points), device=device, dtype=torch.float32)
    n_batches = (n_targets + num_envs - 1) // num_envs
    n_actual = n_batches * num_envs
    cpu_gen = torch.Generator().manual_seed(seed)
    sample_idx = torch.randint(0, pts.shape[0], (n_actual,), generator=cpu_gen)
    torch.manual_seed(seed)

    err_chunks, tgt_chunks, settled_chunks = [], [], []

    for b in range(n_batches):

        p_t = pts[sample_idx[b * num_envs:(b + 1) * num_envs].to(device)]

        # ---- Phase A: kinematic IK. Teleport each iter so meas == cmd.
        q, _ik_residual, diff_ik = solve_kinematic_ik(
            sim, scene, robot, p_t,
            ee_idx=ee_idx, arm_ids_t=arm_ids_t, ee_jacobi_idx=ee_jacobi_idx,
            q_lim=q_lim, default_q=default_q, zero_vel=zero_vel, dt=dt,
            n_ik_iters=n_ik_iters,
        )

        # ---- Phase B: closed-loop correction under gravity, at policy cadence.
        robot.write_joint_state_to_sim(default_q, zero_vel)
        q_des = q.clone()

        control_dt = decimation * dt
        prev_ee_pos_b = None
        ee_vel = torch.zeros(num_envs, device=device)

        for step in range(n_control_steps):          # = episode_length_s / (dt*decimation)
            robot.set_joint_position_target(q_des)
            for _ in range(decimation):
                scene.write_data_to_sim(); sim.step(render=False); scene.update(dt)

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w,
                robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx])
            q_meas = robot.data.joint_pos[:, arm_ids_t]

            # EE velocity (task-space), not joint velocity: a redundant 7-DOF
            # arm doing a position-only task has null-space motion that never
            # settles in joint space even once the EE itself has stopped.
            if prev_ee_pos_b is not None:
                ee_vel = torch.linalg.norm(ee_pos_b - prev_ee_pos_b, dim=-1) / control_dt
            prev_ee_pos_b = ee_pos_b.clone()

            J = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids_t]
            delta = diff_ik.compute(ee_pos_b, ee_quat_b, J, q_meas) - q_meas
            q_des[:, arm_ids_t] = torch.clamp(
                q_des[:, arm_ids_t] + alpha * delta, q_lim[..., 0], q_lim[..., 1])

        ee_pos_b, _ = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx])

        err = torch.linalg.norm(ee_pos_b - p_t, dim=-1)
        settled = ee_vel < ee_settle_vel_tol

        err_chunks.append(err)
        tgt_chunks.append(p_t.cpu())
        settled_chunks.append(settled.cpu())

    errors = torch.cat(err_chunks).cpu().numpy()
    targets = torch.cat(tgt_chunks).cpu().numpy()
    settled_all = torch.cat(settled_chunks)

    n_measured = int(errors.shape[0])
    convergence_rate = float(settled_all.float().mean().item())

    pct = _percentiles(errors)
    if statistic not in pct:
        raise ValueError(f"statistic must be one of {list(pct)}, got {statistic!r}")
    threshold = pct[statistic]

    return SuccessThresholdProbeResult(
        threshold_m=threshold,
        statistic=statistic,
        position_error_percentiles_m=pct,
        n_targets=n_actual,
        n_measured=n_measured,
        convergence_rate=convergence_rate,
        ee_frame=ee_body_name,
        physics_dt=dt,
        gravity_z=grav_z,
        arm_stiffness=applied_kp,
        arm_damping=applied_kd,
        units="meters",
        seed=seed,
        runtime_seconds=time.time() - start,
        errors_m=errors,
        targets_base=targets,
    )