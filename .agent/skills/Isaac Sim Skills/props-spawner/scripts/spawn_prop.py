#!/usr/bin/env python3
"""
Spawn props (cubes, spheres, cylinders, cones) in Isaac Sim
"""

import argparse
import asyncio
import websockets
import json
import sys
import random

# Prop spawning code template
SPAWN_PROP_CODE_TEMPLATE = """
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf, Sdf

prop_type = "{prop_type}"
position = {position}
size = {size}
height = {height}
color = {color}
mass = {mass}
prim_path = "{prim_path}"
physics_enabled = {physics_enabled}

stage = omni.usd.get_context().get_stage()

# Check if path already exists, if so, generate unique name
if prim_path == "auto":
    base_path = f"/World/{{prop_type.capitalize()}}"
    counter = 1
    prim_path = base_path
    while stage.GetPrimAtPath(prim_path):
        prim_path = f"{{base_path}}_{{counter}}"
        counter += 1
elif stage.GetPrimAtPath(prim_path):
    print(f"⚠️  Prim already exists at {{prim_path}}, generating unique name")
    base_path = prim_path
    counter = 1
    while stage.GetPrimAtPath(prim_path):
        prim_path = f"{{base_path}}_{{counter}}"
        counter += 1

# Create the geometry
if prop_type == "cube":
    geom = UsdGeom.Cube.Define(stage, prim_path)
    geom.CreateSizeAttr().Set(size)
elif prop_type == "sphere":
    geom = UsdGeom.Sphere.Define(stage, prim_path)
    geom.CreateRadiusAttr().Set(size)
elif prop_type == "cylinder":
    geom = UsdGeom.Cylinder.Define(stage, prim_path)
    geom.CreateRadiusAttr().Set(size)
    geom.CreateHeightAttr().Set(height)
elif prop_type == "cone":
    geom = UsdGeom.Cone.Define(stage, prim_path)
    geom.CreateRadiusAttr().Set(size)
    geom.CreateHeightAttr().Set(height)
else:
    print(f"❌ Unknown prop type: {{prop_type}}")
    geom = None

if geom:
    # Set color
    if color:
        geom.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    
    # Set position
    prim = geom.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    # Add physics if requested
    if physics_enabled:
        # Add collision
        UsdPhysics.CollisionAPI.Apply(prim)
        
        # Add rigid body (makes it dynamic)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        
        # Set mass
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(mass)
    
    print(f"✅ SUCCESS: Created {{prop_type}} at {{prim_path}}")
    print(f"   Position: {{position}}")
    print(f"   Size: {{size}}")
    if prop_type in ["cylinder", "cone"]:
        print(f"   Height: {{height}}")
    print(f"   Color: {{color if color else 'random'}}")
    print(f"   Physics: {{physics_enabled}}")
    print(f"   Mass: {{mass}} kg")
"""

async def send_spawn_prop_command(prop_type, position, size, height, color, mass, prim_path, 
                                   physics_enabled, host="localhost", port=8765):
    """Send spawn prop command to Isaac Sim"""
    
    uri = f"ws://{host}:{port}"
    
    # Generate Python code
    code = SPAWN_PROP_CODE_TEMPLATE.format(
        prop_type=prop_type,
        position=position,
        size=size,
        height=height,
        color=color if color else None,
        mass=mass,
        prim_path=prim_path if prim_path else "auto",
        physics_enabled=physics_enabled
    )
    
    command = {
        "type": "execute_python",
        "code": code
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(command))
            print(f"📤 Sent spawn {prop_type} command")
            
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                print(f"🔍 Check WebRTC stream to see the {prop_type}")
                return True
            else:
                print(f"❌ Error: {result.get('message')}")
                return False
                
    except ConnectionRefusedError:
        print("❌ ERROR: Cannot connect to Isaac Sim command server")
        print("   Make sure Isaac Sim is running with command server on port 8765")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spawn props in Isaac Sim")
    parser.add_argument("--type", type=str, required=True,
                       choices=["cube", "sphere", "cylinder", "cone"],
                       help="Prop type")
    parser.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 1.0],
                       metavar=("X", "Y", "Z"), help="Position in meters")
    parser.add_argument("--size", type=float, default=0.5,
                       help="Size/radius in meters (default: 0.5)")
    parser.add_argument("--height", type=float, default=1.0,
                       help="Height for cylinder/cone in meters (default: 1.0)")
    parser.add_argument("--color", type=float, nargs=3, default=None,
                       metavar=("R", "G", "B"),
                       help="RGB color 0-1 range (default: random)")
    parser.add_argument("--mass", type=float, default=1.0,
                       help="Mass in kg (default: 1.0)")
    parser.add_argument("--path", type=str, default=None,
                       help="Custom USD prim path (default: auto-generated)")
    parser.add_argument("--physics", dest='physics', action='store_true',
                       help="Enable physics (default)")
    parser.add_argument("--no-physics", dest='physics', action='store_false',
                       help="Disable physics")
    parser.set_defaults(physics=True)
    
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    
    args = parser.parse_args()
    
    # Generate random color if not provided
    color = args.color
    if color is None:
        color = [random.random(), random.random(), random.random()]
        print(f"Generated random color: RGB({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})")
    
    success = asyncio.run(
        send_spawn_prop_command(
            args.type, args.pos, args.size, args.height, color, args.mass,
            args.path, args.physics, args.host, args.port
        )
    )
    
    sys.exit(0 if success else 1)



