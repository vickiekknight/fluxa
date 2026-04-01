#!/usr/bin/env python3
"""
Create physics scene in Isaac Sim
"""

import argparse
import asyncio
import websockets
import json
import sys

# Physics scene creation code template
PHYSICS_SCENE_CODE_TEMPLATE = """
import omni.usd
from pxr import UsdGeom, UsdPhysics, Gf

floor = {floor}
gravity = {gravity}

stage = omni.usd.get_context().get_stage()

# Create /World if needed
world_path = "/World"
if not stage.GetPrimAtPath(world_path):
    UsdGeom.Xform.Define(stage, world_path)

# Create physics scene
physics_scene_path = "/World/PhysicsScene"
if not stage.GetPrimAtPath(physics_scene_path):
    physics_scene = UsdPhysics.Scene.Define(stage, physics_scene_path)
    physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
    physics_scene.CreateGravityMagnitudeAttr().Set(abs(gravity[2]))
    print(f"Created physics scene at {{physics_scene_path}}")

# Create ground plane
if floor:
    ground_path = "/World/GroundPlane"
    if not stage.GetPrimAtPath(ground_path):
        ground_xform = UsdGeom.Xform.Define(stage, ground_path)
        plane_path = f"{{ground_path}}/CollisionPlane"
        plane_geom = UsdGeom.Mesh.Define(stage, plane_path)
        
        plane_size = 100.0
        points = [
            Gf.Vec3f(-plane_size, -plane_size, 0),
            Gf.Vec3f(plane_size, -plane_size, 0),
            Gf.Vec3f(plane_size, plane_size, 0),
            Gf.Vec3f(-plane_size, plane_size, 0)
        ]
        plane_geom.CreatePointsAttr().Set(points)
        plane_geom.CreateFaceVertexCountsAttr().Set([4])
        plane_geom.CreateFaceVertexIndicesAttr().Set([0, 1, 2, 3])
        
        UsdPhysics.CollisionAPI.Apply(plane_geom.GetPrim())
        print(f"Created ground plane at {{ground_path}}")

print("✅ SUCCESS: Physics scene created")
"""

async def send_create_physics_scene_command(floor, gravity, host="localhost", port=8765):
    """Send create physics scene command to Isaac Sim"""
    
    uri = f"ws://{host}:{port}"
    
    # Generate Python code
    code = PHYSICS_SCENE_CODE_TEMPLATE.format(
        floor=floor,
        gravity=gravity
    )
    
    command = {
        "type": "execute_python",
        "code": code
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(command))
            print(f"📤 Sent create physics scene command")
            
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                return True
            else:
                print(f"❌ Error: {result.get('message')}")
                return False
                
    except ConnectionRefusedError:
        print("❌ ERROR: Cannot connect to Isaac Sim command server")
        return False
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create physics scene in Isaac Sim")
    parser.add_argument("--floor", dest='floor', action='store_true', help="Create ground plane")
    parser.add_argument("--no-floor", dest='floor', action='store_false', help="Don't create ground plane")
    parser.set_defaults(floor=True)
    parser.add_argument("--gravity", type=float, nargs=3, default=[0.0, 0.0, -9.81],
                       metavar=("X", "Y", "Z"))
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    
    args = parser.parse_args()
    
    success = asyncio.run(
        send_create_physics_scene_command(args.floor, args.gravity, args.host, args.port)
    )
    
    sys.exit(0 if success else 1)