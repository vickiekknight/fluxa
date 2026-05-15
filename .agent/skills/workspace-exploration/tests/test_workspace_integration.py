"""Integration test: validate Isaac Lab FK (used by workspace_probe) against
CuRobo as an independent reference, on the same joint configurations.

Run from inside your Isaac Lab test harness that provides `scene` and `robot`.
"""
import torch

# CuRobo
from curobo.kinematics import Kinematics, KinematicsCfg
from curobo.types import JointState


# =====================================================================
# 1. CuRobo kinematics setup (reference FK)
# =====================================================================
config = KinematicsCfg.from_robot_yaml_file("franka.yml")
curobo_kin = Kinematics(config)

print(f"CuRobo DOF:         {curobo_kin.get_dof()}")
print(f"CuRobo joint names: {curobo_kin.joint_names}")
print(f"CuRobo tool frames: {curobo_kin.tool_frames}")

# =====================================================================
# 2. Isaac Lab side — `scene` and `robot` come from the test harness
#    (the same setup workspace_probe runs inside).
# =====================================================================
print(f"Isaac Lab joints:   {robot.joint_names}")
print(f"Isaac Lab bodies:   {robot.body_names}")
ee_idx = robot.body_names.index("panda_hand")

# CRITICAL: joint order must match between CuRobo and Isaac Lab, otherwise
# the same column of `test_configs` means different things to each side
# and you'll see huge "errors" that are really just permutation bugs.
assert list(curobo_kin.joint_names) == list(robot.joint_names), (
    f"Joint order mismatch!\n"
    f"  CuRobo:    {curobo_kin.joint_names}\n"
    f"  Isaac Lab: {robot.joint_names}"
)

# =====================================================================
# 3. Test configurations — pick a mix: home, zeros, near-limits, random
# =====================================================================
test_configs = torch.tensor([
    [ 0.0, -0.785,  0.0, -2.356,  0.0, 1.571,  0.785],   # home
    [ 0.0,  0.0,    0.0,  0.0,    0.0, 0.0,    0.0],     # all zeros
    [ 0.5,  0.3,   -0.2, -1.5,    0.4, 1.2,    0.0],     # arbitrary 1
    [-0.5,  0.5,    0.3, -2.0,   -0.4, 1.0,   -0.5],     # arbitrary 2
    [ 1.5, -0.5,    1.0, -2.5,    0.8, 2.0,    1.0],     # arbitrary 3
    # ...add ~50 total for a meaningful sample; one trick is to seed RNG
    # and generate random configs within limits here
], device="cuda:0", dtype=torch.float32)
N = test_configs.shape[0]

# =====================================================================
# 4. CuRobo reference FK
# =====================================================================
state = curobo_kin.compute_kinematics(
    JointState.from_position(test_configs, joint_names=curobo_kin.joint_names)
)
ee_curobo = state.tool_poses.get_link_pose("panda_hand").position  # (N, 3)

# =====================================================================
# 5. Probe-path FK via Isaac Lab — exactly what workspace_probe does
# =====================================================================
assert N <= scene.num_envs, (
    f"This test needs scene.num_envs ({scene.num_envs}) >= N ({N})."
)

isaac_configs = test_configs.to(robot.device)

# Pad to num_envs so the write covers all envs; we'll only read the first N.
padded_q = torch.zeros((scene.num_envs, isaac_configs.shape[1]), device=robot.device)
padded_q[:N] = isaac_configs
zero_vel = torch.zeros_like(padded_q)

robot.write_joint_state_to_sim(padded_q, zero_vel)
scene.update(dt=0.0)

ee_isaac_all = robot.data.body_pos_w[:, ee_idx] - robot.data.root_pos_w  # (num_envs, 3)
ee_isaac = ee_isaac_all[:N]                                              # (N, 3)

# =====================================================================
# 6. Compare
# =====================================================================
err = (ee_isaac - ee_curobo.to(robot.device)).norm(dim=1)  # (N,)

print(f"\nFK comparison over {N} configs:")
print(f"  max error : {err.max().item() * 1000:.3f} mm")
print(f"  mean error: {err.mean().item() * 1000:.3f} mm")
for i, e in enumerate(err.tolist()):
    print(f"    cfg {i}: {e * 1000:.2f} mm")

# 2 mm tolerance is generous and catches frame/link/permutation bugs without
# false-flagging numerical noise. If you see ~constant error across all configs
# (e.g., every error ≈ 0.10 m), it's almost certainly a link mismatch — check
# that CuRobo's tool frame and Isaac Lab's body lookup are the same physical link.
torch.testing.assert_close(
    ee_isaac, ee_curobo.to(robot.device), atol=2e-3, rtol=0,
)
print("\n✓ Isaac Lab FK matches CuRobo within tolerance — probe FK validated.")