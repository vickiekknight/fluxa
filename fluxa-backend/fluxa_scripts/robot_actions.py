# fluxa_scripts/robot_actions.py
import torch
import numpy as np
import json
import isaacsim

GRIPPER_CLOSED = torch.tensor([0.00, 0.00], dtype=torch.float32)
GRIPPER_OPEN = torch.tensor([0.04, 0.04], dtype=torch.float32)

def move_arm(my_franka, joint_positions_7_dof, gripper_positions=None):
    from isaacsim.core.utils.types import ArticulationAction
    
    """Move the arm in the existing simulation."""
    if not torch.is_tensor(joint_positions_7_dof):
        joint_positions_7_dof = torch.tensor(joint_positions_7_dof, dtype=torch.float32)

    if gripper_positions is None:
        gripper_positions = GRIPPER_OPEN
    elif not torch.is_tensor(gripper_positions):
        gripper_positions = torch.tensor(gripper_positions, dtype=torch.float32)

    full_joint_positions = torch.cat((joint_positions_7_dof, gripper_positions))
    action = ArticulationAction(joint_positions=full_joint_positions)
    my_franka.apply_action(action)

def reset_arm_to_home(my_franka):
    """Reset arm to home pose."""
    home_pose_full_np = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.3, 0.8, 0.04, 0.04])
    home_pose_tensor = torch.tensor(home_pose_full_np, dtype=torch.float32)
    my_franka.set_joint_positions(home_pose_tensor)

def run_pickup_trajectory(my_world, my_franka, left_sensor, right_sensor, save_path):
    """Run the cloth pick-up trajectory and collect tactile data."""
    tactile_trace = []

    # Joint poses
    home_pose = torch.tensor([0.0, -0.7, 0.0, -2.0, 0.0, 1.3, 0.8], dtype=torch.float32)
    approach_pose = torch.tensor([0.2,  0.3, 0.0,  -0.8, 0.0, 3.0, 1.77], dtype=torch.float32)
    lift_pose = torch.tensor([0.2,  0.2, 0.0,  -0.6, 0.0, 3.0, 1.77], dtype=torch.float32)

    phase = "approach"
    max_steps = 600
    i = 0
    reset_needed = False
    trajectory_complete = False

    # Reset arm to home
    reset_arm_to_home(my_franka)

    while i < max_steps and not trajectory_complete:
        my_world.step(render=True)
        i += 1

        # Collect contact sensor data
        left_contacts = left_sensor.get_current_frame()
        right_contacts = right_sensor.get_current_frame()

        if left_contacts or right_contacts:
            frame_data = {
                "frame_index": i,
                "phase": phase,
                "left_finger_contacts": left_contacts,
                "right_finger_contacts": right_contacts
            }
            tactile_trace.append(frame_data)

        # Trajectory phases
        if phase == "approach" and i == 50:
            move_arm(my_franka, approach_pose, GRIPPER_OPEN)
            phase = "close_gripper"
            print("Approaching cloth...")

        elif phase == "close_gripper" and i == 150:
            move_arm(my_franka, approach_pose, GRIPPER_CLOSED)
            phase = "grip_wait"
            print("Gripper closed on cloth.")

        elif phase == "grip_wait" and i >= 260:
            phase = "lift"
            print("Waiting before lift...")

        elif phase == "lift" and i == 270:
            move_arm(my_franka, lift_pose, GRIPPER_CLOSED)
            phase = "release"
            print("Lifting cloth...")

        elif phase == "release" and i == 450:
            move_arm(my_franka, lift_pose, GRIPPER_OPEN)
            phase = "home"
            print("Releasing cloth...")

        elif phase == "home" and i >= 510:
            move_arm(my_franka, home_pose, GRIPPER_OPEN)
            # Save tactile data
            with open(save_path, "w") as f:
                json.dump(tactile_trace, f, indent=2)
            print(f"✅ Tactile data saved to {save_path} ({len(tactile_trace)} frames)")
            trajectory_complete = True

    return tactile_trace
