"""Integration test: validate the PhysX self-collision labeler against CuRobo.

Exposes `run_jointlimits_validation(sim, scene, robot, ...)` for callers that
already have an initialized Isaac Lab scene with a Franka built FOR COLLISION
LABELING: self-collisions enabled, contact reporting on, a ContactSensor named
"contact_forces", gravity off, fixed base (see run_probe.py / run_skill.py).
The scene must already be sim.reset()-ed.

This is the collision analog of test_workspace_integration.run_integration_test.
PhysX checks the real convex collision meshes for every non-adjacent link pair;
CuRobo represents the robot as spheres. By default CuRobo's franka.yml ALSO
skips a large set of non-adjacent pairs via `self_collision_ignore` (tuned for
motion planning), which makes it blind to wrist-folded-into-forearm poses that
uniform Sobol sampling generates -- the source of a large false-negative count.

Here we relax `self_collision_ignore` down to only the joint-connected / rigid
pairs (exactly what PhysX auto-excludes), so both checkers evaluate the same
pair set. Any residual disagreement is then the genuine sphere-vs-mesh boundary
error, not a structural blind spot.

Like the FK test, we vary only the 7 arm joints (CuRobo's franka.yml is 7-DOF)
and leave the fingers at their default position.
"""
import torch

from curobo.util_file import get_robot_configs_path, join_path, load_yaml
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from probes.joint_limits_probe import sample_configs_sobol, PhysxSelfCollisionLabeler


# Relaxed self-collision ignore matrix: keep ONLY joint-connected / rigid pairs
# excluded (these are what PhysX also auto-excludes). Everything else -- notably
# link5/link6 vs hand & fingers, plus arm pairs (0,2),(1,3),(1,4),(2,4),(3,6),
# (4,6),(4,7),(5,7) -- is now CHECKED, matching PhysX's pair set.
_RELAXED_SELF_COLLISION_IGNORE = {
    "panda_link0": ["panda_link1"],
    "panda_link1": ["panda_link2"],
    "panda_link2": ["panda_link3"],
    "panda_link3": ["panda_link4"],
    "panda_link4": ["panda_link5", "panda_link8"],
    "panda_link5": ["panda_link6"],
    "panda_link6": ["panda_link7"],
    "panda_link7": ["panda_hand", "panda_leftfinger", "panda_rightfinger", "attached_object"],
    "panda_hand": ["panda_leftfinger", "panda_rightfinger", "attached_object"],
    "panda_leftfinger": ["panda_rightfinger", "attached_object"],
    "panda_rightfinger": ["attached_object"],
}


def _make_robot_dict():
    """Stock franka.yml as a dict, with self_collision_ignore relaxed to only
    joint-connected / rigid pairs. Installed franka.yml is left untouched."""
    robot_dict = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot_dict["robot_cfg"]["kinematics"]["self_collision_ignore"] = \
        _RELAXED_SELF_COLLISION_IGNORE
    return robot_dict


def _build_robot_world(robot_dict, activation_distance=0.0):
    """RobotWorld from the (already-relaxed) dict at a given self-collision
    activation distance (0.0 == strict contact)."""
    return RobotWorld(
        RobotWorldConfig.load_from_config(
            robot_config=robot_dict,    # dict, not "franka.yml" -> uses our edit
            world_model=None,           # self-collision only
            self_collision_activation_distance=activation_distance,
        )
    )


def _diagnose_misses(arm_configs, lo, hi, p, c, joint_names):
    """Are PhysX's collisions that cuRobo misses (false negatives) clustered
    near joint limits?

    cuRobo does NOT clamp q (verified in v0.7.7 source: forward() runs the FK
    kernel on raw q), so both checkers see identical angles. A limit-clustering
    pattern therefore means one of two things, NOT a clamping bug:
      (a) cuRobo's spheres under-cover the extreme folded poses that live near
          the limits  -> cuRobo is wrong, PhysX is right;
      (b) the PhysX labeler throws artifact contacts at those extreme poses
          -> PhysX is wrong, and the safe set is excluding good configs.
    Only a mesh-mesh oracle (Pinocchio/Coal) can say which. This readout just
    tells us whether the residual is structured (near limits) or diffuse.
    """
    import numpy as np
    q = arm_configs.detach().cpu().numpy()                     # (n, 7), cuRobo joint order
    lo_np, hi_np = lo.detach().cpu().numpy(), hi.detach().cpu().numpy()
    u = (q - lo_np) / (hi_np - lo_np)                          # (n, 7) in [0, 1]
    prox = np.minimum(u, 1.0 - u)                             # 0 == on a limit, 0.5 == mid-range
    near_any = (prox < 0.05).any(axis=1)                      # within 5% of ANY limit

    fn, tp = (p & ~c), (p & c)

    def _row(mask, name):
        if mask.sum() == 0:
            print(f"     {name:16s} n=0"); return
        closest = prox[mask].min(axis=1)                      # closest-limit proximity per config
        print(f"     {name:16s} n={int(mask.sum()):4d} | median closest-limit "
              f"prox={np.median(closest):.3f} | within 5% of a limit: "
              f"{100 * near_any[mask].mean():3.0f}%")

    print("\n  -- joint-limit proximity (0.00 = on a limit, 0.50 = mid-range) --")
    _row(np.ones_like(p), "all configs")
    _row(tp,              "PhysX & cuRobo")   # agreement collisions
    _row(fn,              "PhysX-only (FN)")  # the misses in question

    if fn.sum():
        per_joint = (prox[fn] < 0.05).mean(axis=0) * 100
        print("     FN configs, % with each joint within 5% of a limit:")
        for jn, pj in zip(joint_names, per_joint):
            print(f"        {jn:18s} {pj:4.0f}%")
        idx = list(np.where(fn)[0][:5])
        print(f"     first FN indices {idx} (normalized joint pos; * = within 5% of a limit):")
        for i in idx:
            cells = " ".join(f"{v:4.2f}{'*' if min(v, 1 - v) < 0.05 else ' '}" for v in u[i])
            print(f"        cfg {i:4d}: {cells}")


def _curobo_self_collision_mask(robot_world, arm_configs, threshold=0.0):
    """Bool mask (B,), True == self-colliding.

    Self-collision-ONLY path (world cost is None with world_model=None). cuRobo
    returns penetration as a cost: 0.0 == clear, > 0 == colliding. Sanity-check
    the sign on a known-folded config before trusting the mask.
    """
    q = arm_configs.to(device="cuda:0", dtype=torch.float32).contiguous()   # (B, dof)
    state = robot_world.get_kinematics(q)                                    # CudaRobotModelState
    d_self = robot_world.get_self_collision_distance(
        state.link_spheres_tensor.unsqueeze(1)                              # (B, 1, n_spheres, 4)
    ).squeeze(1)                                                            # (B,)
    if d_self.dim() > 1:                       # defensive: collapse any residual dims
        d_self = d_self.view(d_self.shape[0], -1).max(dim=1)[0]
    return d_self > threshold


def run_jointlimits_validation(sim, scene, robot, n: int = 512, seed: int = 42,
                               min_recall: float = 0.95) -> dict:
    print("\n=== Joint-Limits Validation (PhysX vs CuRobo self-collision) ===")

    # --- 1. CuRobo: RobotWorld with relaxed ignore list (parity with PhysX pairs) ---
    robot_dict = _make_robot_dict()
    robot_world = _build_robot_world(robot_dict, activation_distance=0.0)
    curobo_joint_names = robot_world.kinematics.joint_names

    # --- 2. Map CuRobo's 7 arm joints onto Isaac Lab's joint vector ---
    missing = [j for j in curobo_joint_names if j not in robot.joint_names]
    assert not missing, (
        f"CuRobo joints not present in Isaac Lab robot: {missing}\n"
        f"  CuRobo:    {curobo_joint_names}\n"
        f"  Isaac Lab: {robot.joint_names}"
    )
    isaac_indices = [robot.joint_names.index(j) for j in curobo_joint_names]
    idx_tensor = torch.tensor(isaac_indices, device=robot.device, dtype=torch.long)
    n_arm = len(isaac_indices)

    print(f"  CuRobo DOF: {n_arm} | Isaac Lab joints: {len(robot.joint_names)}")
    print(f"  Arm-joint indices in Isaac Lab: {isaac_indices}")

    # --- 3. Sobol-sample the 7 arm joints within URDF limits ---
    num_envs = scene.num_envs
    assert num_envs >= 1
    n = ((n + num_envs - 1) // num_envs) * num_envs        # pad to full PhysX batches

    if isaac_indices == list(range(n_arm)):
        arm_limits = robot.data.soft_joint_pos_limits[0, :n_arm]
    else:
        arm_limits = torch.index_select(
            robot.data.soft_joint_pos_limits[0], 0, idx_tensor
        )
    lo, hi = arm_limits[:, 0].contiguous(), arm_limits[:, 1].contiguous()
    arm_configs = sample_configs_sobol(lo, hi, n, seed).to(device="cuda:0", dtype=torch.float32)

    # --- 4. CuRobo reference labels (single batched call) ---
    c = _curobo_self_collision_mask(robot_world, arm_configs).cpu().numpy()

    # --- 5. PhysX labels: full 9-joint vector per chunk, fingers at default ---
    labeler = PhysxSelfCollisionLabeler(sim, scene, robot)
    physx_chunks = []
    for i in range(0, n, num_envs):
        arm_chunk = arm_configs[i:i + num_envs]             # (num_envs, 7)
        full_q = robot.data.default_joint_pos.clone()       # (num_envs, 9)
        full_q.index_copy_(1, idx_tensor, arm_chunk)        # vary only the arm
        physx_chunks.append(labeler(full_q))
    p = torch.cat(physx_chunks).cpu().numpy()

    # --- 6. Confusion matrix ---
    tp = int((p & c).sum())      # both collide
    fn = int((p & ~c).sum())     # PhysX collide, CuRobo clear  <- should drop sharply now
    fp = int((~p & c).sum())     # PhysX clear, CuRobo collide  <- sphere/buffer inflation
    tn = int((~p & ~c).sum())

    print(f"  N configs:                     {n} (seed={seed})")
    print(f"  PhysX collide & CuRobo collide : {tp}")
    print(f"  PhysX collide, CuRobo clear    : {fn}   (expect ~0 with relaxed ignores)")
    print(f"  PhysX clear,  CuRobo collide   : {fp}   (sphere/buffer inflation)")
    print(f"  both clear                     : {tn}")
    print(f"  agreement:                     {(tp + tn) / n:.3f}")

    # --- 7b. Diagnostic: is the residual structured (near joint limits)? ---
    _diagnose_misses(arm_configs, lo, hi, p, c, curobo_joint_names)

    # --- 7. Assert: CuRobo's collision set ~contains PhysX's (superset property) ---
    n_physx_collide = tp + fn
    if n_physx_collide == 0:
        print("  \u26a0 PhysX found no self-collisions in this sample. Increase n, or "
              "verify the labeler fires on a known-colliding config.")
        recall = float("nan")
    else:
        recall = tp / n_physx_collide
        print(f"  CuRobo recall of PhysX collisions: {recall:.3f}")
        if recall < min_recall:
            miss_idx = [i for i in range(n) if p[i] and not c[i]][:5]
            print(f"  \u2717 FAILED \u2014 recall {recall:.3f} < {min_recall}. "
                  f"First missed config indices: {miss_idx}")
            raise AssertionError(
                f"Self-collision validation failed: CuRobo recall {recall:.3f} "
                f"< {min_recall}. Likely the sign convention in "
                f"_curobo_self_collision_mask is flipped, the joint mapping is "
                f"wrong, or the PhysX labeler is over-reporting."
            )
        print(f"  \u2713 PASSED \u2014 CuRobo recall \u2265 {min_recall}.\n")

    return {
        "n": n, "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "agreement": (tp + tn) / n,
        "curobo_recall_of_physx": recall,
        "physx_collision_rate": float(p.mean()),
        "curobo_collision_rate": float(c.mean()),
    }