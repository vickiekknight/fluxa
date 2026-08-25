"""Controller-gains probe: discovers the (Kp, Kd) PD gains that make the arm
track a commanded joint target smoothly under gravity -- not maximally stiff,
just the least-aggressive gain that settles cleanly without ringing.

Sweeps candidate (Kp, Kd) pairs in parallel, one pair per env (Sobol-sampled),
via Isaac Lab's per-env `write_joint_stiffness_to_sim`/`write_joint_damping_to_sim`.
Every env is commanded the *same* joint-space step target, so the only
difference between envs is their gain -- isolating the controller's dynamics
from anything IK/task-space related (that's success_threshold_probe's job).

Design choices, and why:
  - One (Kp, Kd) pair applied uniformly across all arm joints (matches Isaac
    Lab's own FRANKA_PANDA_HIGH_PD_CFG pattern), not per-joint.
  - Kp and Kd are swept fully independently, not coupled via a fixed damping
    ratio -- a fixed ratio would just be a different hardcoded, Franka-tuned
    assumption. The goal is a probe that works for any robot's ArticulationCfg.
  - The sweep range is a multiplier on whatever gains are already loaded on
    the robot's ArticulationCfg, not hardcoded absolute bounds -- same reason.
"""
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class ControllerGainsProbeResult:
    kp: float
    kd: float
    default_kp: float
    default_kd: float
    n_candidates: int
    n_feasible: int
    steady_state_error_rad: float
    settling_steps: int
    kp_range: tuple
    kd_range: tuple
    seed: int
    runtime_seconds: float
    candidates_kp: np.ndarray
    candidates_kd: np.ndarray
    candidates_feasible: np.ndarray
    candidates_steady_state_error_rad: np.ndarray
    candidates_settling_steps: np.ndarray
    representative_kp: np.ndarray
    representative_kd: np.ndarray
    representative_err_hist_rad: np.ndarray      # (n_steps, n_representative)
    representative_vel_hist_rad_s: np.ndarray    # (n_steps, n_representative)


def controller_gains_probe(sim, scene, robot, *,
                           arm_joint_expr: str = "panda_joint.*",
                           step_delta_rad: float = 0.3,
                           n_steps: int = 400,
                           kp_range_mult: tuple = (0.5, 100.0),
                           kd_range_mult: tuple = (0.5, 100.0),
                           pos_tol: float = 1e-2,
                           vel_tol: float = 5e-2,
                           settle_window: int = 150,
                           n_representative: int = 5,
                           kp_band_frac: float = 0.15,
                           default_kp: float = None,
                           default_kd: float = None,
                           seed: int = 0) -> ControllerGainsProbeResult:
    """Sweep (Kp, Kd) in parallel across envs; pick whichever feasible pair
    has the lowest (steady-state error x settling steps) cost -- accurate and
    fast, not just the smallest Kp that happens to clear tolerance.

    "Feasible" is settling cleanly without a separate overshoot metric: a
    candidate must reach AND STAY within (pos_tol, vel_tol) for the trailing
    `settle_window` steps. A ringing/oscillating response won't clear that
    window as long as it's wide enough to overlap the actual ringing --
    confirmed empirically that this matters: at too narrow a window, ringing
    that died out before the window started went undetected entirely, and Kd
    barely affected which candidates passed. See `settle_window`'s docstring.

    Selecting by cost rather than "smallest feasible Kp" matters because Kd
    doesn't affect steady-state error at all (confirmed empirically: it's a
    near-perfect function of Kp alone), so a Kp-only selection rule leaves Kd
    to whatever a candidate happened to be paired with in the Sobol draw --
    it was found this way to differ from the actual lowest-cost region by
    4-5x on real data.

    Args:
        arm_joint_expr: regex for the arm joints to tune (all get the same
            swept Kp/Kd).
        step_delta_rad: joint-space step size, applied once from a resting
            pose under gravity. Same step for every env, only gains differ.
        n_steps: physics steps to observe the step response over. Needs
            headroom for the heaviest load you'll ever test with the same
            budget: a 3kg EE payload was observed to need meaningfully longer
            than a bare arm to settle, and a too-small n_steps/settle_window
            for that case silently falls back to "least-bad infeasible
            candidate" rather than a real answer (tell-tale sign: the
            returned settling_steps equals n_steps exactly, and/or
            steady_state_error_rad exceeds pos_tol).
        kp_range_mult, kd_range_mult: independent (lo, hi) multipliers
            applied to the robot's own currently-loaded Kp/Kd (from its
            ArticulationCfg) to build each sweep range. Kept separate rather
            than one shared range: the two searches don't need to span the
            same multiplicative distance, and empirically for Franka the
            useful Kd region sits closer to its range ceiling than Kp's does.
        pos_tol: joint-space position-error norm (rad) below which a
            candidate counts as settled.
        vel_tol: joint-space velocity norm (rad/s) below which a candidate
            counts as settled.
        settle_window: trailing steps that must all be within (pos_tol,
            vel_tol) to count as settled. Must be wide enough to overlap
            whatever ringing an underdamped candidate exhibits -- too narrow
            and a candidate can be actively oscillating for most of the run
            and still pass on a clean tail alone (confirmed empirically: with
            window=20, Kd was picking up almost no signal at all, because
            ringing that lasted to ~step 120 had already died out by step 180).
        n_representative: number of candidates near the discovered Kp, spread
            across the sampled Kd range at that Kp, to save full per-step
            trajectories for -- lets you inspect the transient (not just the
            tail the feasibility check looks at) to see whether Kd actually
            affects ringing/overshoot the way 2nd-order intuition predicts.
        kp_band_frac: representative candidates are drawn from envs with Kp
            within +/-kp_band_frac of the discovered Kp.
        default_kp, default_kd: override the "current robot gains" reference
            point the sweep range is built around, instead of reading it live
            from env 0. Required when calling this repeatedly on the same
            scene (e.g. controller_gains_load_profile): after one call, env
            0's live stiffness/damping is whatever random candidate it was
            last assigned, not the robot's true baseline -- reading it again
            would center the next sweep on leftover noise instead of a stable
            reference.
        seed: RNG seed for the Sobol gain sample and the step direction.

    Returns:
        ControllerGainsProbeResult with the discovered (kp, kd) and the full
        per-candidate sweep data for diagnostics.
    """
    start = time.time()
    device = robot.device
    num_envs = scene.num_envs
    dt = sim.get_physics_dt()

    arm_ids, _ = robot.find_joints(arm_joint_expr, preserve_order=True)
    arm_ids_t = torch.as_tensor(arm_ids, device=device, dtype=torch.long)
    n_arm = len(arm_ids)

    if default_kp is None:
        default_kp = float(robot.data.joint_stiffness[0, arm_ids_t].mean())
    if default_kd is None:
        default_kd = float(robot.data.joint_damping[0, arm_ids_t].mean())
    kp_lo, kp_hi = default_kp * kp_range_mult[0], default_kp * kp_range_mult[1]
    kd_lo, kd_hi = default_kd * kd_range_mult[0], default_kd * kd_range_mult[1]

    # --- Sample one (Kp, Kd) candidate per env, log-uniform: the ranges span
    # orders of magnitude, and linear interpolation over that would crowd
    # almost every sample into the high end of the absolute range.
    engine = torch.quasirandom.SobolEngine(dimension=2, scramble=True, seed=seed)
    u = engine.draw(num_envs).to(device=device, dtype=torch.float32)   # (num_envs, 2) in [0,1]
    log_kp_lo, log_kp_hi = torch.log(torch.tensor(kp_lo)), torch.log(torch.tensor(kp_hi))
    log_kd_lo, log_kd_hi = torch.log(torch.tensor(kd_lo)), torch.log(torch.tensor(kd_hi))
    kp_per_env = torch.exp(log_kp_lo + u[:, 0] * (log_kp_hi - log_kp_lo))
    kd_per_env = torch.exp(log_kd_lo + u[:, 1] * (log_kd_hi - log_kd_lo))

    stiffness = kp_per_env.unsqueeze(1).expand(num_envs, n_arm).contiguous()
    damping = kd_per_env.unsqueeze(1).expand(num_envs, n_arm).contiguous()
    robot.write_joint_stiffness_to_sim(stiffness, joint_ids=arm_ids)
    robot.write_joint_damping_to_sim(damping, joint_ids=arm_ids)

    # --- Same joint-space step target for every env; only gains differ ---
    default_q = robot.data.default_joint_pos.clone()
    zero_vel = torch.zeros_like(default_q)
    cpu_gen = torch.Generator().manual_seed(seed)
    step = step_delta_rad * (2 * torch.rand(n_arm, generator=cpu_gen).to(device) - 1)
    q_target = default_q.clone()
    q_target[:, arm_ids_t] = default_q[:, arm_ids_t] + step

    robot.write_joint_state_to_sim(default_q, zero_vel)
    robot.set_joint_position_target(q_target)

    err_hist = torch.empty(n_steps, num_envs, device=device)
    vel_hist = torch.empty(n_steps, num_envs, device=device)

    for t in range(n_steps):
        scene.write_data_to_sim(); sim.step(render=False); scene.update(dt)
        q_meas = robot.data.joint_pos[:, arm_ids_t]
        qdot_meas = robot.data.joint_vel[:, arm_ids_t]
        err_hist[t] = torch.linalg.norm(q_meas - q_target[:, arm_ids_t], dim=-1)
        vel_hist[t] = torch.linalg.norm(qdot_meas, dim=-1)

    # --- Feasible: settled and stable over a trailing window, wide enough to
    # overlap real ringing (confirmed empirically -- see settle_window's
    # docstring), not the whole remaining episode (a single numerical blip
    # right at the very end of a 100+-step tail shouldn't disqualify an
    # otherwise well-settled candidate).
    is_ok = (err_hist < pos_tol) & (vel_hist < vel_tol)                # (n_steps, num_envs)
    window = min(settle_window, n_steps)
    feasible = is_ok[-window:].all(dim=0)                              # (num_envs,)

    # Diagnostic only: how long a *fully* unbroken settle (first crossing to
    # the very end) would have taken, for whichever candidate is chosen below.
    suffix_all_ok = torch.flip(
        torch.cummin(torch.flip(is_ok.int(), dims=[0]), dim=0).values, dims=[0]
    ).bool()
    fully_settled = suffix_all_ok.any(dim=0)
    settled_step = torch.argmax(suffix_all_ok.int(), dim=0)

    steady_state_error = err_hist[-1]                                  # (num_envs,)
    steady_state_vel = vel_hist[-1]                                    # (num_envs,)
    # Never-fully-settled candidates are capped at n_steps (worst case in this
    # run), not left at their misleading raw argmax of an all-False array.
    settling_steps_all = torch.where(fully_settled, settled_step,
                                     torch.full_like(settled_step, n_steps))

    # Cost = settling speed, Kp as tie-break -- NOT error x settling_steps.
    # Feasibility already gates accuracy (a feasible candidate is guaranteed
    # err < pos_tol), and error keeps shrinking without bound as Kp grows (the
    # 1/Kp gravity-droop relationship has no plateau), so multiplying by it
    # would just make "minimize cost" degenerate into "maximize Kp" -- found
    # this empirically: the discovered Kp sat at the range ceiling for every
    # load, independent of load, which is what gave Kd its arbitrary look.
    # Settling_steps doesn't have that runaway relationship with Kp (ringing
    # can make settling slower even at very high Kp with insufficient Kd), so
    # it's the part of "smooth" actually worth optimizing once accuracy is
    # already satisfied. Kp only breaks ties among equally-fast candidates.
    big = float(kp_hi) * 10 + 1
    cost = settling_steps_all.to(torch.float32) * big + kp_per_env

    if feasible.any():
        idx_pool = feasible.nonzero(as_tuple=True)[0]
        best_idx = idx_pool[torch.argmin(cost[idx_pool])]
    else:
        final_pos_ok = int((steady_state_error < pos_tol).sum().item())
        final_vel_ok = int((steady_state_vel < vel_tol).sum().item())
        print(f"⚠️  controller-gains probe: no candidate settled within "
              f"n_steps={n_steps} (trailing {window}-step window). At the "
              f"final step: {final_pos_ok}/{num_envs} met pos_tol={pos_tol}, "
              f"{final_vel_ok}/{num_envs} met vel_tol={vel_tol} -- whichever "
              f"count is low tells you which to loosen (or widen "
              f"kp_range_mult/kd_range_mult / increase n_steps if both are low). "
              f"Returning the lowest-cost candidate anyway, but it is NOT "
              f"validated feasible -- check settling_steps/steady_state_error_rad "
              f"against n_steps/pos_tol before trusting this result.")
        best_idx = torch.argmin(cost)

    # --- Representative candidates: fixed-Kp band around the discovered Kp,
    # spread across the Kd values sampled at that Kp. Saves the FULL err(t)/
    # vel(t) trajectory (not just the tail the feasibility check uses) so you
    # can see whether low Kd actually rings during the transient even when it
    # still clears the trailing-window check by the end.
    ref_kp = float(kp_per_env[best_idx])
    in_band = ((kp_per_env > ref_kp * (1 - kp_band_frac)) &
              (kp_per_env < ref_kp * (1 + kp_band_frac)))
    band_idx = in_band.nonzero(as_tuple=True)[0]
    if band_idx.numel() > 0:
        order = torch.argsort(kd_per_env[band_idx])
        n_reps = min(n_representative, band_idx.numel())
        positions = torch.linspace(0, band_idx.numel() - 1, n_reps).round().long()
        rep_idx = band_idx[order[positions]]
    else:
        rep_idx = best_idx.unsqueeze(0)

    return ControllerGainsProbeResult(
        kp=float(kp_per_env[best_idx]),
        kd=float(kd_per_env[best_idx]),
        default_kp=default_kp,
        default_kd=default_kd,
        n_candidates=num_envs,
        n_feasible=int(feasible.sum().item()),
        steady_state_error_rad=float(steady_state_error[best_idx]),
        settling_steps=int(settled_step[best_idx]) if fully_settled[best_idx] else n_steps,
        kp_range=(kp_lo, kp_hi),
        kd_range=(kd_lo, kd_hi),
        seed=seed,
        runtime_seconds=time.time() - start,
        candidates_kp=kp_per_env.cpu().numpy(),
        candidates_kd=kd_per_env.cpu().numpy(),
        candidates_feasible=feasible.cpu().numpy(),
        candidates_steady_state_error_rad=steady_state_error.cpu().numpy(),
        candidates_settling_steps=settling_steps_all.cpu().numpy(),
        representative_kp=kp_per_env[rep_idx].cpu().numpy(),
        representative_kd=kd_per_env[rep_idx].cpu().numpy(),
        representative_err_hist_rad=err_hist[:, rep_idx].cpu().numpy(),
        representative_vel_hist_rad_s=vel_hist[:, rep_idx].cpu().numpy(),
    )


@dataclass
class ControllerGainsLoadProfileResult:
    payload_masses_kg: list
    ee_body_name: str
    results: list          # one ControllerGainsProbeResult per payload, same order


def controller_gains_load_profile(sim, scene, robot, *,
                                  payload_masses_kg: tuple = (0.0, 1.5, 3.0),
                                  ee_body_name: str = "panda_hand",
                                  **gains_probe_kwargs) -> ControllerGainsLoadProfileResult:
    """Run controller_gains_probe once per end-effector payload mass, so the
    discovered (Kp, Kd) can be compared across loading scenarios.

    Adds each payload directly to the EE link's mass via the raw PhysX tensor
    API (`root_physx_view.set_masses`) -- Isaac Lab's `Articulation` wrapper
    doesn't expose a mass setter at all, only `get_masses()`, so this bypasses
    the wrapper the same way `get_jacobians()` already does elsewhere in this
    probe. Per `set_masses`' own docstring, sparse (single-link) updates
    aren't supported: the full (num_envs, max_links) tensor must be read,
    mutated, and written back each time.

    The *same* seed is used for every payload (passed through in
    `gains_probe_kwargs` or defaulted by controller_gains_probe), so each
    payload's sweep tests the identical (Kp, Kd) candidate grid -- differences
    across the returned results are attributable to the load, not to
    resampling. Original masses are restored before returning.

    Args:
        payload_masses_kg: added end-effector mass per scenario, in kg (added
            on top of whatever the EE link's own mass already is). Defaults
            span 0 (bare arm, matches every prior run this session) up to
            Franka's rated 3kg payload.
        ee_body_name: link the payload is added to.
        **gains_probe_kwargs: forwarded to controller_gains_probe (e.g.
            pos_tol, vel_tol, settle_window, seed, kp_range_mult, ...).

    Returns:
        ControllerGainsLoadProfileResult with one ControllerGainsProbeResult
        per payload, in the same order as payload_masses_kg.
    """
    device = robot.device
    num_envs = scene.num_envs
    ee_idx = robot.body_names.index(ee_body_name)

    # Capture the TRUE baseline gains once, before any sweep mutates env 0's
    # live stiffness/damping -- see default_kp/default_kd's docstring on
    # controller_gains_probe for why re-reading it between calls is wrong.
    arm_joint_expr = gains_probe_kwargs.get("arm_joint_expr", "panda_joint.*")
    arm_ids, _ = robot.find_joints(arm_joint_expr, preserve_order=True)
    arm_ids_t = torch.as_tensor(arm_ids, device=device, dtype=torch.long)
    baseline_kp = float(robot.data.joint_stiffness[0, arm_ids_t].mean())
    baseline_kd = float(robot.data.joint_damping[0, arm_ids_t].mean())

    baseline_masses = robot.root_physx_view.get_masses().clone()   # (num_envs, max_links)
    assert baseline_masses.shape[1] == len(robot.body_names), (
        f"root_physx_view link count ({baseline_masses.shape[1]}) doesn't "
        f"match robot.body_names ({len(robot.body_names)}) -- mass-index "
        f"mapping is unverified for this robot, refusing to guess ee_idx."
    )
    indices = torch.arange(num_envs, device=baseline_masses.device, dtype=torch.int32)

    results = []
    try:
        for payload_kg in payload_masses_kg:
            masses = baseline_masses.clone()
            masses[:, ee_idx] = baseline_masses[:, ee_idx] + payload_kg
            robot.root_physx_view.set_masses(masses, indices)

            print(f"\n[controller-gains load-profile] payload={payload_kg:.2f} kg "
                 f"at {ee_body_name} (mass {float(baseline_masses[0, ee_idx]):.3f} "
                 f"-> {float(masses[0, ee_idx]):.3f} kg)")
            result = controller_gains_probe(sim, scene, robot,
                                            default_kp=baseline_kp, default_kd=baseline_kd,
                                            **gains_probe_kwargs)
            results.append(result)
    finally:
        robot.root_physx_view.set_masses(baseline_masses, indices)

    return ControllerGainsLoadProfileResult(
        payload_masses_kg=list(payload_masses_kg),
        ee_body_name=ee_body_name,
        results=results,
    )
