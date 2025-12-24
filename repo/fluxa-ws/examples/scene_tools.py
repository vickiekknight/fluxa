# fluxa_scripts/scene_tools.py

import numpy as np

def add_franka_robot(world, position=[0.0, 0.0, 0.0], name="my_franka"):
    """
    Add a Franka robot to the scene.
    
    Args:
        world: Isaac Sim World instance
        position: [x, y, z] position in meters
        name: Unique name for the robot
    
    Returns:
        Robot object
    """
    from isaacsim.robot.manipulators.examples.franka import Franka

    robot = world.scene.add(
        Franka(
            prim_path=f"/World/{name}",
            name=name,
            position=np.array(position, dtype=np.float32)
        )
    )
    print(f"✅ Added Franka robot '{name}' at position {position}")
    return {"success": True, "name": name, "position": position}

def add_cube(world, position=[0.5, 0.0, 0.5], size=0.1, color=[1.0, 0.0, 0.0], 
             dynamic=True, name="cube"):
    """
    Add a cube to the scene.
    
    Args:
        world: Isaac Sim World instance
        position: [x, y, z] position in meters
        size: Cube size in meters
        color: [r, g, b] color (0-1 range)
        dynamic: If True, cube can move; if False, it's fixed
        name: Unique name for the cube
    
    Returns:
        Cube object info
    """
    from isaacsim.core.api.objects import FixedCuboid, DynamicCuboid

    cube_class = DynamicCuboid if dynamic else FixedCuboid
    
    cube = world.scene.add(
        cube_class(
            prim_path=f"/World/{name}",
            name=name,
            position=np.array(position, dtype=np.float32),
            scale=np.array([size, size, size], dtype=np.float32),
            color=np.array(color, dtype=np.float32)
        )
    )
    
    cube_type = "dynamic" if dynamic else "fixed"
    print(f"✅ Added {cube_type} cube '{name}' at position {position}")
    return {
        "success": True, 
        "name": name, 
        "position": position, 
        "size": size, 
        "dynamic": dynamic
    }

def add_table(world, position=[0.5, 0.0, 0.0], width=0.6, depth=0.6, 
              height=0.4, name="table"):
    """
    Add a table (fixed cuboid) to the scene.
    
    Args:
        world: Isaac Sim World instance
        position: [x, y, z] position for table center
        width: Table width (x-axis)
        depth: Table depth (y-axis)
        height: Table height (z-axis)
        name: Unique name for the table
    
    Returns:
        Table object info
    """
    from isaacsim.core.api.objects import FixedCuboid

    # Position table so its top surface is at the specified height
    table_position = np.array([position[0], position[1], height / 2.0])
    
    table = world.scene.add(
        FixedCuboid(
            prim_path=f"/World/{name}",
            name=name,
            position=table_position,
            scale=np.array([width, depth, height], dtype=np.float32),
            color=np.array([0.6, 0.4, 0.2], dtype=np.float32)  # Brown
        )
    )
    
    print(f"✅ Added table '{name}' at position {position}")
    return {
        "success": True,
        "name": name,
        "position": position.tolist(),
        "dimensions": {"width": width, "depth": depth, "height": height}
    }

def clear_scene(world):
    """
    Clear all objects from the scene (except ground plane).
    
    Args:
        world: Isaac Sim World instance
    
    Returns:
        Status message
    """
    world.scene.clear()
    print("✅ Scene cleared")
    return {"success": True, "message": "Scene cleared"}

def reset_simulation(world):
    """
    Reset the simulation to initial state.
    
    Args:
        world: Isaac Sim World instance
    
    Returns:
        Status message
    """
    world.reset()
    print("✅ Simulation reset")
    return {"success": True, "message": "Simulation reset"}