import torch
import torch.nn as nn
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

# --- LLM GENERATED CODE START ---
# Populated by 1_eureka.py with the model's reward functions AND its
# module-level `reward_dict`. There is a single injection point on purpose: the
# previous template split the response into "functions" and "the dict" by
# scanning for a line starting with `reward_dict`, which broke as soon as the
# model wrapped its terms in a factory function or named that factory
# `reward_dict` itself. inject_code now validates the whole block with ast
# instead of slicing it by text.

def position_tracking_coarse(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    temperature: float,
) -> torch.Tensor:
    """Coarse exponential reward for end-effector position tracking across the workspace."""
    pos_error = mdp.position_command_error(env, command_name=command_name, asset_cfg=asset_cfg)
    return torch.exp(-pos_error / temperature)


def position_tracking_fine(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    temperature: float,
) -> torch.Tensor:
    """Fine-scale exponential reward providing strong gradients near the 5cm target."""
    pos_error = mdp.position_command_error(env, command_name=command_name, asset_cfg=asset_cfg)
    return torch.exp(-pos_error / temperature)


def orientation_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    temperature: float,
) -> torch.Tensor:
    """Exponential reward for aligning the end-effector orientation with the command."""
    ori_error = mdp.orientation_command_error(env, command_name=command_name, asset_cfg=asset_cfg)
    return torch.exp(-ori_error / temperature)


def pose_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    pos_threshold: float,
    ori_threshold: float,
) -> torch.Tensor:
    """Bonus when the end-effector stays within the target position and orientation tolerances."""
    pos_error = mdp.position_command_error(env, command_name=command_name, asset_cfg=asset_cfg)
    ori_error = mdp.orientation_command_error(env, command_name=command_name, asset_cfg=asset_cfg)
    success = (pos_error < pos_threshold) & (ori_error < ori_threshold)
    return success.to(dtype=torch.float32)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """L2 penalty on action changes between steps to ensure smooth motions and avoid jerks."""
    return mdp.action_rate_l2(env)


def joint_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """L2 penalty on joint velocities to prevent erratic movements."""
    return mdp.joint_vel_l2(env, asset_cfg=asset_cfg)


reward_dict = {
    "position_tracking_coarse": RewTerm(
        func=position_tracking_coarse,
        weight=1.5,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "temperature": 0.2,
        },
    ),
    "position_tracking_fine": RewTerm(
        func=position_tracking_fine,
        weight=2.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "temperature": 0.05,
        },
    ),
    "orientation_tracking": RewTerm(
        func=orientation_tracking,
        weight=1.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "temperature": 0.25,
        },
    ),
    "pose_success": RewTerm(
        func=pose_success,
        weight=3.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=["panda_hand"]),
            "pos_threshold": 0.05,
            "ori_threshold": 0.2,
        },
    ),
    "action_rate": RewTerm(
        func=action_rate_penalty,
        weight=-0.01,
        params={},
    ),
    "joint_velocity": RewTerm(
        func=joint_velocity_penalty,
        weight=-0.0005,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
        },
    ),
}


# --- LLM GENERATED CODE END ---


def get_reward_cfg(env_cfg, robot_ee_body_name: str):
    """Return the generated reward terms.

    Kept as the entry point because manipulation-tasks and eval_headless.py both
    call it; the dictionary itself is defined at module level above.
    """
    return reward_dict


def compute_success(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Standardized success check, independent of the generated reward.

    Finds the end-effector by name rather than assuming index 0.
    """
    target_pos = env.command_manager.get_command("ee_pose")[:, :3]

    robot = env.scene["robot"]
    ee_name = env.cfg.robot_ee_body_name
    ee_index, _ = robot.find_bodies(ee_name)

    ee_pos = robot.data.body_pos_w[:, ee_index[0]]

    distance = torch.norm(target_pos - ee_pos, dim=-1)
    # TODO: threshold should come from workspace-exploration's
    # success_threshold probe rather than being hardcoded here.
    return (distance < 0.05).float()
