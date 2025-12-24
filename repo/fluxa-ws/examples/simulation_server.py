# fluxa_scripts/simulation_server.py
from isaacsim.simulation_app import SimulationApp

# --- CONFIG ---
CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "hide_ui": False,
    "renderer": "RaytracedLighting",
}

# Placeholders for objects
simulation_app = None
my_world = None
my_franka = None
left_sensor = None
right_sensor = None

def init_simulation():
    """Initialize the simulation and return all objects."""
    global simulation_app, my_world, my_franka, left_sensor, right_sensor

    simulation_app = SimulationApp(launch_config=CONFIG)

    from isaacsim.core.api import World
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.sensors.physics import ContactSensor
    from isaacsim.core.api.objects import FixedCuboid
    from isaacsim.core.prims import SingleClothPrim, SingleParticleSystem
    from isaacsim.core.api.materials.particle_material import ParticleMaterial
    from omni.physx.scripts import deformableUtils, physicsUtils
    from pxr import UsdGeom, Gf
    import numpy as np
    import omni.ui as ui
    from isaacsim.core.utils.extensions import enable_extension

    # Close unnecessary panels
    for panel_name in ["Stage","Render Settings","Property","Content","Console"]:
        win = ui.Workspace.get_window(panel_name)
        if win:
            win.visible = False
    viewport = ui.Workspace.get_window("Viewport")
    if viewport:
        dock_id = viewport.dock_id
        ui.Workspace.set_dock_id_width(dock_id, ui.Workspace.get_main_window_width())
        ui.Workspace.set_dock_id_height(dock_id, ui.Workspace.get_main_window_height())

    # Enable livestream
    enable_extension("omni.kit.livestream.webrtc")
    simulation_app.update()

    # --- CREATE WORLD ---
    my_world = World(stage_units_in_meters=1.0, backend="torch", device="cuda", physics_dt=1.0/120.0)
    stage = simulation_app.context.get_stage()
    my_world.scene.add_default_ground_plane()

    # --- ADD ROBOT ---
    my_franka = my_world.scene.add(Franka(prim_path="/World/Franka", name="my_franka"))

    # --- ADD CLOTH TABLE ---
    cube_size, cube_height = 0.3, 1.0
    my_world.scene.add(FixedCuboid(
        prim_path="/World/Xform/Cube",
        name="cloth_table",
        position=np.array([0.9, 0.0, cube_height / 2.0]),
        scale=np.array([cube_size, cube_size, cube_height]),
        color=np.array([1.0, 1.0, 1.0]),
    ))

    # --- CREATE PARTICLE CLOTH ---
    cloth_path = "/World/cloth"
    plane_mesh = UsdGeom.Mesh.Define(stage, cloth_path)
    tri_points, tri_indices = deformableUtils.create_triangle_mesh_square(dimx=10, dimy=10, scale=0.5)
    plane_mesh.GetPointsAttr().Set(tri_points)
    plane_mesh.GetFaceVertexIndicesAttr().Set(tri_indices)
    plane_mesh.GetFaceVertexCountsAttr().Set([3]*(len(tri_indices)//3))
    physicsUtils.setup_transform_as_scale_orient_translate(plane_mesh)
    physicsUtils.set_or_add_translate_op(plane_mesh, Gf.Vec3f(0.9, 0.0, 1.5))

    particle_material = ParticleMaterial(
        prim_path="/World/particleMaterial",
        drag=0.1,
        lift=0.3,
        friction=2.5
    )

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
        particle_contact_offset=contactOffset
    )

    cloth = SingleClothPrim(
        name="my_cloth",
        prim_path=cloth_path,
        particle_system=particle_system,
        particle_material=particle_material
    )
    my_world.scene.add(cloth)
    my_world.reset()

    # --- CONTACT SENSORS ---
    left_sensor = my_world.scene.add(ContactSensor(
        prim_path="/World/Franka/panda_leftfinger/contact_sensor",
        name="left_finger_contact_sensor",
        min_threshold=0,
        max_threshold=10000000,
        radius=0.05
    ))
    right_sensor = my_world.scene.add(ContactSensor(
        prim_path="/World/Franka/panda_rightfinger/contact_sensor",
        name="right_finger_contact_sensor",
        min_threshold=0,
        max_threshold=10000000,
        radius=0.05
    ))
    left_sensor.add_raw_contact_data_to_frame()
    right_sensor.add_raw_contact_data_to_frame()

    print("✅ Simulation initialized and livestream ready!")
    return simulation_app, my_world, my_franka, left_sensor, right_sensor

def run_simulation_loop():
    """Call this in a separate script to run the simulation loop."""
    global my_world, simulation_app
    try:
        while True:
            my_world.step(render=True)
    except KeyboardInterrupt:
        simulation_app.close()
