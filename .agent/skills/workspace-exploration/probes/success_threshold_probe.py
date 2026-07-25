"""Success-threshold probe: discovers how close the EE must be to a target
to count as success, for this particular robot under this task's dynamics.

1. Draw N target positions from the discovered workspace point cloud (positions
   already known reachable, since each came from a real joint config).
2. Reset the arm to its default config, hand the target POSITION to Isaac Lab's
   differential IK (DLS) controller, and roll it out closed-loop under GRAVITY.
   The controller commands joint targets each step; the PD drive tracks them and
   physics settles. This rollout *is* the dynamics -- no separate settle phase.
3. Measure ||EE_settled - target|| at the SAME EE body the reward tracks
   (panda_hand), collect the N errors, and take a percentile (p90 default).

The percentile (not the max) makes the threshold robust to the few hardest
poses while staying tight; p90/p95 is the intended operating point.

Position-only, by design
------------------------
The success criterion this threshold feeds is a POSITION tolerance -- "is the
gripper within X of the target" -- so the probe measures the position-reaching
floor and nothing else. It does NOT constrain gripper orientation.

This matters: the workspace point cloud holds EE positions produced by random
joint configs, so each position is reachable at *some* orientation, not
necessarily gripper-down. Forcing a fixed orientation (`pose` command) asks the
arm to satisfy an orientation many of these targets can't -- the controller then
never settles and the "error" is interrupted flailing, not a control floor.
Position-only (`position_abs`) lets orientation fall out freely, so every
point-cloud target is reachable and the measured error is the real floor.

DLS still needs the *current* EE orientation as a seed at set_command time (it
drives the Jacobian step), so we pass it -- but it is never tracked.

Gravity requirement
-------------------
This probe MUST run with gravity ON. The workspace and joint-limits probes run
gravity-off (clean FK / clean self-collision labelling); this one does not,
because the whole point is the gravitational + PD steady-state error. With
gravity off the achievable error collapses toward the kinematic floor. The probe
warns if it detects gravity is ~0.

Convergence health
------------------
The DLS controller has no success flag. Targets it couldn't settle within the
rollout horizon are counted in ``convergence_rate``. If that rate is low, raise
``n_steps``; a healthy run settles nearly all targets, so the threshold reflects
control floor rather than an under-run horizon.
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class SuccessThresholdProbeResult:
    threshold_m: float                       # the chosen success threshold, meters
    statistic: str                           # which percentile was used, e.g. "p90"
    position_error_percentiles_m: dict       # {"p50","p75","p90","p95","p99","mean","max"}

    n_targets: int                           # targets drawn (padded to a multiple of num_envs)
    n_measured: int                          # targets that produced a valid error
    convergence_rate: float                  # fraction the oracle controller settled within n_steps

    ee_frame: str                            # EE body the error is measured at (must match reward)
    n_steps: int                             # oracle rollout horizon (settling protocol)
    physics_dt: float
    gravity_z: Optional[float]               # recorded for auditability
    units: str
    seed: int
    runtime_seconds: float

    errors_m: np.ndarray                     # (n_measured,) raw errors, for diagnostics/validation
    targets_base: np.ndarray                 # (n_measured, 3) target positions, base frame


def _read_gravity_z(sim) -> Optional[float]:
    """Best-effort read of the sim's gravity (z component); None if not found."""
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


def success_threshold_probe(sim, scene, robot, workspace_points, *,
                            n_targets: int, seed: int = 0,
                            ee_body_name: str = "panda_hand",
                            arm_joint_expr: str = "panda_joint.*",
                            statistic: str = "p90",
                            n_steps: int = 200,
                            settle_tol_m: float = 1e-4) -> SuccessThresholdProbeResult:
    """Discover the success threshold as a percentile of oracle position error.

    Args:
        sim, scene, robot: an Isaac Lab sim/scene/Articulation, GRAVITY ON.
        workspace_points: (M, 3) reachable EE positions in robot-base frame --
            i.e. WorkspaceProbeResult.point_cloud. Targets are drawn from here so
            they are position-reachable by construction.
        n_targets: number of target positions to probe (padded to a multiple of
            scene.num_envs; each batch is rolled out fully in parallel).
        ee_body_name: EE body to measure at. MUST match reach_task.py's reward
            body_names, or the threshold is measured at the wrong frame.
        arm_joint_expr: regex selecting the actuated arm joints (franka default).
        statistic: which percentile of the error distribution becomes the
            threshold. "p90" (default) or "p95" per the probe design.
        n_steps: closed-loop rollout horizon. Should exceed DLS settling time;
            watch convergence_rate and raise this if it's low.
        settle_tol_m: per-step EE displacement below which a target is deemed
            settled (feeds convergence_rate; does not gate the threshold).
    """
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.utils.math import subtract_frame_transforms

    start = time.time()
    device = robot.device
    num_envs = scene.num_envs
    dt = sim.get_physics_dt()

    grav_z = _read_gravity_z(sim)
    if grav_z is not None and abs(grav_z) < 1e-6:
        print("⚠️  success-threshold probe: gravity is ~0. The measured error "
              "will collapse toward the kinematic floor. Run this probe with "
              "gravity ON for a meaningful threshold.")

    # --- indices ------------------------------------------------------------
    ee_idx = robot.body_names.index(ee_body_name)
    arm_ids, _ = robot.find_joints(arm_joint_expr, preserve_order=True)
    arm_ids_t = torch.as_tensor(arm_ids, device=device, dtype=torch.long)
    # Fixed base -> the base link is absent from the Jacobian, so shift by one.
    ee_jacobi_idx = ee_idx - 1 if robot.is_fixed_base else ee_idx
    print(f"[debug] ee_idx={ee_idx}  ee_jacobi_idx={ee_jacobi_idx}  arm_ids={arm_ids}")

    default_q = robot.data.default_joint_pos.clone()     # (num_envs, J)
    zero_vel = torch.zeros_like(default_q)

    # --- DLS controller: POSITION ONLY --------------------------------------
    # position_abs tracks the target position and lets orientation fall out
    # freely. set_command still needs the current EE quaternion as a seed, so we
    # pass it each batch below -- it is never tracked.
    ik_cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls",
    )
    diff_ik = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)

    # --- target sampling (reachable positions from the workspace cloud) ------
    pts = torch.as_tensor(np.asarray(workspace_points), device=device, dtype=torch.float32)
    n_batches = (n_targets + num_envs - 1) // num_envs
    n_actual = n_batches * num_envs
    cpu_gen = torch.Generator().manual_seed(seed)
    sample_idx = torch.randint(0, pts.shape[0], (n_actual,), generator=cpu_gen)

    err_chunks, tgt_chunks, settled_chunks = [], [], []

    for b in range(n_batches):
        p_t = pts[sample_idx[b * num_envs:(b + 1) * num_envs].to(device)]   # (num_envs, 3)

        # reset arm to a consistent start and let it register
        robot.write_joint_state_to_sim(default_q, zero_vel)
        robot.set_joint_position_target(default_q)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)

        # seed orientation for the position command = current EE quat (base frame)
        cur_pos_b, cur_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx],
        )
        diff_ik.reset()
        diff_ik.set_command(p_t, ee_pos=cur_pos_b, ee_quat=cur_quat_b)

        prev_ee_w = robot.data.body_pos_w[:, ee_idx].clone()
        last_step_disp = torch.full((num_envs,), float("inf"), device=device)

        for _ in range(n_steps):
            ee_pos_w = robot.data.body_pos_w[:, ee_idx]
            ee_quat_w = robot.data.body_quat_w[:, ee_idx]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w, ee_quat_w
            )
            jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, arm_ids_t]
            joint_pos = robot.data.joint_pos[:, arm_ids_t]

            joint_pos_des = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            robot.set_joint_position_target(joint_pos_des, joint_ids=arm_ids)
            scene.write_data_to_sim()

            for _ in range(4):
                sim.step(render=False)
                scene.update(dt)

            if b == 0 and _ % 25 == 0:
                ee_now = robot.data.body_pos_w[:, ee_idx] - robot.data.root_pos_w
                med_err = torch.linalg.norm(ee_now - p_t, dim=-1).median()
                jmove = torch.linalg.norm(
                    robot.data.joint_pos[:, arm_ids_t] - default_q[:, arm_ids_t], dim=-1
                ).median()
                des_gap = torch.linalg.norm(
                    joint_pos_des - robot.data.joint_pos[:, arm_ids_t], dim=-1
                ).median()
                print(f"[debug] step {_:4d}  med_err={med_err*1000:7.1f}mm  "
                      f"jmove={jmove:.3f}rad  des_gap={des_gap:.4f}")

            cur_ee_w = robot.data.body_pos_w[:, ee_idx]
            last_step_disp = torch.linalg.norm(cur_ee_w - prev_ee_w, dim=-1)
            prev_ee_w = cur_ee_w.clone()

        # measure in the same base frame the workspace point cloud uses
        ee_rel = robot.data.body_pos_w[:, ee_idx] - robot.data.root_pos_w   # (num_envs, 3)
        err = torch.linalg.norm(ee_rel - p_t, dim=-1)                       # (num_envs,)

        err_chunks.append(err)
        tgt_chunks.append(p_t)
        settled_chunks.append(last_step_disp < settle_tol_m)

    errors = torch.cat(err_chunks).cpu().numpy()
    targets = torch.cat(tgt_chunks).cpu().numpy()
    settled_all = torch.cat(settled_chunks)

    n_measured = int(errors.shape[0])
    convergence_rate = float(settled_all.float().mean().item())
    if convergence_rate < 0.8:
        print(f"⚠️  success-threshold probe: only {convergence_rate:.0%} of targets "
              f"settled within n_steps={n_steps}. Consider raising n_steps.")

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
        n_steps=n_steps,
        physics_dt=dt,
        gravity_z=grav_z,
        units="meters",
        seed=seed,
        runtime_seconds=time.time() - start,
        errors_m=errors,
        targets_base=targets,
    )