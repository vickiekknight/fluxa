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

# INSERT_REWARD_CODE_HERE

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
