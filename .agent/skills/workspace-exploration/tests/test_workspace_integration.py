"""Integration test: validate Isaac Lab FK against CuRobo on the same configs.

Exposes `run_integration_test(scene, robot, ...)` for callers that already
have an initialized Isaac Lab scene with a Franka.
"""
import torch

from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.types.base import TensorDeviceType
from curobo.types.robot import RobotConfig
from curobo.util_file import get_robot_path, join_path, load_yaml

def run_integration_test(scene, robot, n_configs: int = 50, seed: int = 0,
                         atol: float = 2e-3) -> dict:
    """Compare Isaac Lab FK against CuRobo on `n_configs` random configs.

    Samples uniform joint configs within URDF limits (same distribution as
    `workspace_probe`), runs FK through both Isaac Lab and CuRobo, and
    asserts agreement within `atol` meters.

    Args:
        scene: Isaac Lab InteractiveScene. Must have num_envs >= n_configs.
        robot: Articulation handle from scene["robot"].
        n_configs: number of random joint configs to test.
        seed: RNG seed for reproducibility.
        atol: absolute tolerance in meters (default 2 mm).

    Returns:
        dict with max_error_mm, mean_error_mm, n_configs.
    
    Raises:
        AssertionError if max error exceeds atol or joint order mismatches.
    """
    print("\n=== FK Integration Test (Isaac Lab vs CuRobo) ===")

    # --- 1. CuRobo kinematics ---
    tensor_args = TensorDeviceType()
    robot_yaml = load_yaml(join_path(get_robot_path(), "franka.yml"))["robot_cfg"]
    robot_cfg = RobotConfig.from_dict(robot_yaml, tensor_args)
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    curobo_joint_names = kin_model.joint_names
    
    # --- 2. Map CuRobo's joints onto Isaac Lab's joint vector ---
    # CuRobo's franka.yml uses 7 arm DOF; Isaac Lab's Franka has 9 joints
    # (7 arm + 2 finger). We test the 7 arm joints and leave fingers at default.
    missing = [j for j in curobo_joint_names if j not in robot.joint_names]
    assert not missing, (
        f"CuRobo joints not present in Isaac Lab robot: {missing}\n"
        f"  CuRobo:    {curobo_joint_names}\n"
        f"  Isaac Lab: {robot.joint_names}"
    )
    isaac_indices = [robot.joint_names.index(j) for j in curobo_joint_names]
    idx_tensor = torch.tensor(isaac_indices, device=robot.device, dtype=torch.long)
    
    print(f"  CuRobo DOF: {len(curobo_joint_names)} | "
          f"Isaac Lab joints: {len(robot.joint_names)}")
    print(f"  Arm-joint indices in Isaac Lab: {isaac_indices}")

    # ee_link comes from the cuRobo config. For the comparison below to be
    # frame-consistent it must (a) name a body present in Isaac Lab, and
    # (b) be expressed in a base frame matching Isaac Lab's robot root.
    # If ee_link is not an Isaac body (e.g. panda_grasptarget), either point it
    # at panda_hand in franka.yml, or pull that link explicitly from the cuRobo
    # state via out.link_pose["panda_hand"].position (link must be tracked).
    ee_link = robot_yaml["kinematics"]["ee_link"]
    assert ee_link in robot.body_names, (
        f"CuRobo ee_link '{ee_link}' is not an Isaac Lab body name.\n"
        f"  Isaac Lab bodies: {robot.body_names}"
    )
    ee_idx = robot.body_names.index(ee_link)
    assert n_configs <= scene.num_envs, (
        f"Need scene.num_envs ({scene.num_envs}) >= n_configs ({n_configs})."
    )


    # --- 3. Sample random configs within URDF joint limits ---
    # Match workspace_probe's sampling distribution. We sample only the 7
    # arm joints (matching CuRobo's DOF) and leave fingers at their default
    # position when writing back to Isaac Lab.
    rng = torch.Generator(device=robot.device).manual_seed(seed)

    # Avoid Python-list fancy indexing on Isaac Lab's GPU tensors — it
    # deadlocks against PhysX's CUDA pipeline. Use slicing if indices are
    # contiguous, otherwise index_select with a CUDA tensor.
    if isaac_indices == list(range(len(isaac_indices))):
        arm_limits = robot.data.soft_joint_pos_limits[0, :len(isaac_indices)]
    else:
        arm_limits = torch.index_select(
            robot.data.soft_joint_pos_limits[0], 0, idx_tensor
        )

    lo, hi = arm_limits[:, 0], arm_limits[:, 1]                          # (7,), (7,)
    u = torch.rand((n_configs, len(isaac_indices)),
                   generator=rng, device=robot.device)
    arm_configs = lo + u * (hi - lo)                                     # (N, 7)

    # --- 4. CuRobo reference FK on the 7-DOF arm configs ---
    q = arm_configs.to(device="cuda:0", dtype=torch.float32).contiguous()  # (N, 7)
    out = kin_model.get_state(q)                                           # CudaRobotModelState
    ee_curobo = out.ee_position                                           # (N, 3)

    # --- 5. Isaac Lab FK: build full 9-joint vector, vary only the arm 7 ---
    # Fingers stay at default; they don't affect panda_hand pose.
    # index_copy_ avoids the Python-list fancy-indexing GPU deadlock.
    full_q = robot.data.default_joint_pos.clone()                        # (num_envs, 9)
    full_q[:n_configs].index_copy_(1, idx_tensor, arm_configs)
    zero_vel = torch.zeros_like(full_q)

    robot.write_joint_state_to_sim(full_q, zero_vel)
    scene.update(dt=0.0)

    ee_isaac = (
        robot.data.body_pos_w[:, ee_idx] - robot.data.root_pos_w
    )[:n_configs]                                                        # (N, 3)

    # --- 6. Compare ---
    err = (ee_isaac - ee_curobo.to(robot.device)).norm(dim=1)
    max_err_mm = err.max().item() * 1000
    mean_err_mm = err.mean().item() * 1000

    print(f"  EE link:    {ee_link}")
    print(f"  N configs:  {n_configs} (seed={seed})")
    print(f"  Max error:  {max_err_mm:.3f} mm")
    print(f"  Mean error: {mean_err_mm:.3f} mm")

    if err.max().item() > atol:
        worst_idx = err.argsort(descending=True)[:5]
        print(f"  ✗ FAILED — exceeds {atol * 1000:.1f} mm tolerance")
        print(f"  Worst 5 configs:")
        for i in worst_idx.tolist():
            print(f"    cfg {i}: err={err[i].item() * 1000:.2f} mm | "
                  f"isaac={[f'{v:+.3f}' for v in ee_isaac[i].tolist()]} | "
                  f"curobo={[f'{v:+.3f}' for v in ee_curobo[i].tolist()]}")
        raise AssertionError(
            f"FK validation failed: max {max_err_mm:.3f} mm > {atol * 1000:.1f} mm"
        )

    print(f"  ✓ PASSED — within {atol * 1000:.1f} mm tolerance.\n")
    return {
        "max_error_mm": max_err_mm,
        "mean_error_mm": mean_err_mm,
        "n_configs": n_configs,
    }