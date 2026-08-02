"""Success-threshold probe: discovers how close the EE must be to a target
to count as success, for this robot under this task's dynamics.

Position-only DLS oracle: sample target positions, drive the arm to each with
Isaac Lab's differential-IK controller under gravity, measure settled EE error,
take a percentile (p90) as the threshold.

DEBUG SCAFFOLDING (temporary):
  ISOLATION_TEST=True  -> targets are FK of small perturbations around home,
                          i.e. provably reachable with a known-good solution.
                          Used to tell "distribution problem" from "solver bug".
  DEBUG=True           -> per-step trace on batch 0.
Remove both once the probe is validated.
"""
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

DEBUG = True
ISOLATION_TEST = True          # True: near-home reachable targets (bug-isolation)
ISO_DELTA_RAD = 0.3            # perturbation magnitude around home for the test


@dataclass
class SuccessThresholdProbeResult:
    threshold_m: float
    statistic: str
    position_error_percentiles_m: dict
    n_targets: int
    n_measured: int
    convergence_rate: float
    ee_frame: str
    command_type: str
    n_steps: int
    physics_dt: float
    gravity_z: Optional[float]
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


def success_threshold_probe(sim, scene, robot, workspace_points, *,
                            n_targets: int, seed: int = 0,
                            ee_body_name: str = "panda_hand",
                            arm_joint_expr: str = "panda_joint.*",
                            statistic: str = "p90",
                            n_steps: int = 200,
                            settle_tol_m: float = 1e-4) -> SuccessThresholdProbeResult:
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
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
    ee_jacobi_idx = ee_idx - 1 if robot.is_fixed_base else ee_idx

    default_q = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(default_q)

    if DEBUG:
        print(f"[debug] ee_idx={ee_idx}  ee_jacobi_idx={ee_jacobi_idx}  arm_ids={arm_ids}")
        print(f"[debug] ISOLATION_TEST={ISOLATION_TEST}")

    ik_cfg = DifferentialIKControllerCfg(
        command_type="position", use_relative_mode=False, ik_method="dls",
    )
    diff_ik = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)

    pts = torch.as_tensor(np.asarray(workspace_points), device=device, dtype=torch.float32)
    n_batches = (n_targets + num_envs - 1) // num_envs
    n_actual = n_batches * num_envs
    cpu_gen = torch.Generator().manual_seed(seed)
    sample_idx = torch.randint(0, pts.shape[0], (n_actual,), generator=cpu_gen)
    torch.manual_seed(seed)

    err_chunks, tgt_chunks, settled_chunks = [], [], []

    for b in range(n_batches):
        # q_known is only defined in the isolation branch; None otherwise.
        q_known = None

        if ISOLATION_TEST:
            # Target = FK of a small perturbation around home: provably reachable,
            # known-good joint solution (q_known), inside the DLS basin.
            delta = ISO_DELTA_RAD * (2 * torch.rand((num_envs, len(arm_ids)), device=device) - 1)
            q_known = default_q.clone()
            q_known[:, arm_ids_t] = default_q[:, arm_ids_t] + delta
            robot.write_joint_state_to_sim(q_known, zero_vel)
            robot.set_joint_position_target(q_known)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(dt)
            p_t = (robot.data.body_pos_w[:, ee_idx] - robot.data.root_pos_w).clone()
        else:
            p_t = pts[sample_idx[b * num_envs:(b + 1) * num_envs].to(device)]

        # reset arm to home; rollout always starts here
        robot.write_joint_state_to_sim(default_q, zero_vel)
        robot.set_joint_position_target(default_q)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)

        cur_pos_b, cur_quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            robot.data.body_pos_w[:, ee_idx], robot.data.body_quat_w[:, ee_idx],
        )
        diff_ik.reset()
        diff_ik.set_command(p_t, ee_pos=cur_pos_b, ee_quat=cur_quat_b)

        prev_ee_w = robot.data.body_pos_w[:, ee_idx].clone()
        last_step_disp = torch.full((num_envs,), float("inf"), device=device)

        for step in range(n_steps):
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
            for _sub in range(4):                     # let the PD reach q_des
                sim.step(render=False)
                scene.update(dt)

            if DEBUG and b == 0 and step % 25 == 0:
                ee_now = robot.data.body_pos_w[:, ee_idx] - robot.data.root_pos_w
                med_err = torch.linalg.norm(ee_now - p_t, dim=-1).median()
                cmd_gap = torch.linalg.norm(ee_pos_b - p_t, dim=-1).median()
                des_gap = torch.linalg.norm(
                    joint_pos_des - robot.data.joint_pos[:, arm_ids_t], dim=-1
                ).median()
                msg = (f"[debug] step {step:4d}  med_err={med_err*1000:7.1f}mm  "
                       f"cmd_gap={cmd_gap*1000:7.1f}mm  des_gap={des_gap:.4f}")
                if q_known is not None:
                    sol_gap = torch.linalg.norm(
                        joint_pos_des - q_known[:, arm_ids_t], dim=-1
                    ).median()
                    msg += f"  sol_gap={sol_gap:.4f}rad"
                print(msg)

            cur_ee_w = robot.data.body_pos_w[:, ee_idx]
            last_step_disp = torch.linalg.norm(cur_ee_w - prev_ee_w, dim=-1)
            prev_ee_w = cur_ee_w.clone()

        ee_rel = robot.data.body_pos_w[:, ee_idx] - robot.data.root_pos_w
        err = torch.linalg.norm(ee_rel - p_t, dim=-1)

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
        command_type=ik_cfg.command_type,

    )