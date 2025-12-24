# cloth_franka_tactile.py

from isaacsim.simulation_app import SimulationApp

# Start Isaac Sim with streaming enabled
CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "hide_ui": False,  # Show the GUI
    "renderer": "RaytracedLighting",
    "display_options": 3094,  
}

simulation_app = SimulationApp(launch_config=CONFIG)

import omni.ui as ui
import numpy as np
import torch
import json

from isaacsim.core.utils.extensions import enable_extension

from isaacsim.core.api import World
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.materials.particle_material import ParticleMaterial
from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
from omni.physx.scripts import deformableUtils, physicsUtils
from omni.physx.scripts.physicsUtils import *
from omni.physx import get_physx_simulation_interface
from isaacsim.sensors.physics import ContactSensor
from pxr import Usd, Gf, UsdGeom, PhysxSchema, PhysicsSchemaTools, UsdPhysics

#--- Adjust viewport to match window size ---------------------#

# List of panels in Isaac Sim to close
panels_to_close = ["Stage","Render Settings",
                   "Property","Content","Console"]

# Close them if they exist
for panel_name in panels_to_close:
    win = ui.Workspace.get_window(panel_name)
    if win:
        win.visible = False

# Maximize the viewport
viewport = ui.Workspace.get_window("Viewport")
if viewport:
    dock_id = viewport.dock_id
    width = ui.Workspace.get_main_window_width()
    height = ui.Workspace.get_main_window_height()
    ui.Workspace.set_dock_id_width(dock_id, width)
    ui.Workspace.set_dock_id_height(dock_id, height)
#--------------------------------------------------------------#

# Enable WebRTC Livestream extension
simulation_app.set_setting("/app/window/drawMouse", True)
enable_extension("omni.kit.livestream.webrtc")
simulation_app.update()

# Create world and robot
my_world = World(stage_units_in_meters=1.0, backend="torch", device="cuda", physics_dt=1.0 / 120.0)
# my_world.get_physics_context().set_gravity(0)
my_world.get_physics_context()
stage = simulation_app.context.get_stage()
my_world.scene.add_default_ground_plane()

# Add Franka robot
my_franka = my_world.scene.add(Franka(prim_path="/World/Franka", name="my_franka"))

# Add a Cube for the cloth to lay on
cube_size = 0.3
cube_height = 1.0
my_world.scene.add(
    FixedCuboid(
        prim_path="/World/Xform/Cube",
        name="cloth_table",
        position=np.array([0.9, 0.0, cube_height / 2.0]),
        scale=np.array([cube_size, cube_size, cube_height]),
        color=np.array([1.0, 1.0, 1.0]),
    )
)

# ------------------ Create particle cloth ------------------ #

# 1. Create the cloth mesh geometry
cloth_path = "/World/cloth"
plane_mesh = UsdGeom.Mesh.Define(stage, cloth_path)

# Create a triangle mesh (5x5 grid, 1.0 scale)
tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(
    dimx=10, dimy=10, scale=0.5
)

plane_mesh.GetPointsAttr().Set(tri_points)
plane_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
plane_mesh.GetFaceVertexCountsAttr().Set([3] * (len(tri_indices) // 3))

# Set position and orientation
init_loc = Gf.Vec3f(0.9, 0.0, 1.5)
physicsUtils.setup_transform_as_scale_orient_translate(plane_mesh)
physicsUtils.set_or_add_translate_op(plane_mesh, init_loc)

# 2. Create particle material (defines cloth physics properties)
particle_material = ParticleMaterial(
    prim_path="/World/particleMaterial",
    drag=0.1,
    lift=0.3,
    friction=2.5,
)

# 3. Create particle system (required for cloth simulation)
radius = 0.5 * (0.6 / 5.0)
restOffset = radius
contactOffset = restOffset * 1.5

particle_system = SingleParticleSystem(
    prim_path="/World/particleSystem",
    simulation_owner=my_world.get_physics_context().prim_path,
    rest_offset=restOffset,
    contact_offset=contactOffset,
    solid_rest_offset=restOffset,
    fluid_rest_offset=restOffset,
    particle_contact_offset=contactOffset,
)

# 4. Create the cloth prim (combines mesh, particle system, and material)
cloth = SingleClothPrim(
    name="my_cloth",
    prim_path=cloth_path,
    particle_system=particle_system,
    particle_material=particle_material,
)

# 5. Add to scene
my_world.scene.add(cloth)

# -------------------------------------------------------- #

my_world.reset()
print("✅ Scene created and streaming!")

# --- Add Contact Sensors to Gripper Fingers --- #
print("\n🔍 Adding contact sensors to gripper fingers...")

# Contact sensors for left and right fingers
left_finger_sensor = my_world.scene.add(
    ContactSensor(
        prim_path="/World/Franka/panda_leftfinger/contact_sensor",
        name="left_finger_contact_sensor",
        min_threshold=0,
        max_threshold=10000000,
        radius=0.05,  # Adjust based on finger size
        translation=np.array([0.0, 0.0, 0.0])  # Relative to finger
    )
)

right_finger_sensor = my_world.scene.add(
    ContactSensor(
        prim_path="/World/Franka/panda_rightfinger/contact_sensor",
        name="right_finger_contact_sensor",
        min_threshold=0,
        max_threshold=10000000,
        radius=0.05,
        translation=np.array([0.0, 0.0, 0.0])
    )
)

# Enable raw contact data
left_finger_sensor.add_raw_contact_data_to_frame()
right_finger_sensor.add_raw_contact_data_to_frame()

print("✅ Contact sensors added to both gripper fingers")
print(f"   Left finger sensor: /World/Franka/panda_leftfinger/contact_sensor")
print(f"   Right finger sensor: /World/Franka/panda_rightfinger/contact_sensor")

# --- Tactile data storage --- #
tactile_trace = []

# A helper function for arm motion
def move_arm(joint_positions_7_dof, gripper_positions=None):
    if not torch.is_tensor(joint_positions_7_dof):
        joint_positions_7_dof = torch.tensor(joint_positions_7_dof, dtype=torch.float32)

    if gripper_positions is None:
        gripper_positions = torch.tensor([0.04, 0.04], dtype=torch.float32)
    elif not torch.is_tensor(gripper_positions):
        gripper_positions = torch.tensor(gripper_positions, dtype=torch.float32)
    
    full_joint_positions = torch.cat((joint_positions_7_dof, gripper_positions))
    my_franka.apply_action(ArticulationAction(joint_positions=full_joint_positions))

# Joint poses
home_pose_np = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.3, 0.8])
approach_pose_np = np.array([0.2,  0.3, 0.0,  -0.8, 0.0, 3.0, 1.77])
lift_pose_np = np.array([0.2,  0.2, 0.0,  -0.6, 0.0, 3.0, 1.77])

home_pose = torch.tensor(home_pose_np, dtype=torch.float32)
approach_pose = torch.tensor(approach_pose_np, dtype=torch.float32)
lift_pose = torch.tensor(lift_pose_np, dtype=torch.float32)

GRIPPER_CLOSED = torch.tensor([0.00, 0.00], dtype=torch.float32)
GRIPPER_OPEN = torch.tensor([0.04, 0.04], dtype=torch.float32)

def reset_arm_to_home():
    home_pose_full_np = np.array([0.0, -0.7, 0.0, -2.0, 0.0, 1.3, 0.8, 0.04, 0.04])
    home_pose_tensor = torch.tensor(home_pose_full_np, dtype=torch.float32)
    my_franka.set_joint_positions(home_pose_tensor)

# Initialization
reset_arm_to_home()

print("⏳ Waiting for cloth to settle...")
settle_steps = 200
for _ in range(settle_steps):
    my_world.step(render=True)

print("✅ Cloth settled. Starting main loop...")

i = 0
reset_needed = False
phase = "approach"

# Main simulation loop
max_steps = 600
trajectory_complete = False

while i < max_steps and not trajectory_complete:
    my_world.step(render=True)
    
    # The contact callback is automatically called during step()
    # Contact data is automatically collected in tactile_trace via the callback
    
    if my_world.is_stopped() and not reset_needed:
        reset_needed = True
        
    if my_world.is_playing():
        if reset_needed:
            my_world.reset()
            reset_arm_to_home() 
            i = 0
            phase = "home"
            reset_needed = False

        i += 1

        # Collect contact sensor data
        left_contacts = left_finger_sensor.get_current_frame()
        right_contacts = right_finger_sensor.get_current_frame()
        
        # Store tactile data if there are contacts
        if left_contacts or right_contacts:
            frame_data = {
                "frame_index": i,
                "phase": phase,
                "left_finger_contacts": left_contacts,
                "right_finger_contacts": right_contacts
            }
            tactile_trace.append(frame_data)

        # Go down to cloth level
        if phase == "approach" and i == 50:
            move_arm(approach_pose, gripper_positions=GRIPPER_OPEN)
            phase = "close_gripper"
            print("Approaching cloth...")
        
        # Close gripper immediately
        elif phase == "close_gripper" and i == 150:
            move_arm(approach_pose, gripper_positions=GRIPPER_CLOSED) 
            phase = "grip_wait"
            print("Gripper closed on cloth.")
            
        # Wait a moment to ensure the gripper has closed
        elif phase == "grip_wait" and i >= 260:
            phase = "lift"
            print("Waiting before lift...")

        # Lift cloth
        elif phase == "lift" and i == 270:
            move_arm(lift_pose, gripper_positions=GRIPPER_CLOSED)
            phase = "release"
            print("Lifting cloth...")

        # Release cloth
        elif phase == "release" and i == 450:
            move_arm(lift_pose, gripper_positions=GRIPPER_OPEN) 
            phase = "home"
            print("Releasing cloth...")
            
        elif phase == "home" and i >= 510:
            move_arm(home_pose, gripper_positions=GRIPPER_OPEN)
            
            # Save tactile data at end of trajectory
            with open("/isaac-sim/fluxa_scripts/tactile_data/tactile_data.json", "w") as f:
                json.dump(tactile_trace, f, indent=2)
            print(f"✅ Tactile data saved to /tactile_data/tactile_data.json ({len(tactile_trace)} frames with contacts)")
            
            # Mark trajectory as complete to exit the loop
            trajectory_complete = True
            print("Trajectory complete! Exiting...")

# Cleanup when done
print("Simulation finished.")
print(f"Total frames with contact data: {len(tactile_trace)}")