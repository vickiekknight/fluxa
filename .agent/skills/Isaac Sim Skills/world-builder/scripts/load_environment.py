#!/usr/bin/env python3
"""
Load pre-built environments into Isaac Sim
"""

import argparse
import asyncio
import websockets
import json
import sys

# Environment loading code template
LOAD_ENV_CODE_TEMPLATE = """
from isaacsim.storage.native import get_assets_root_path
from isaacsim.core.utils.stage import add_reference_to_stage
import omni.usd
from pxr import UsdGeom, UsdLux, Gf
import omni.kit.viewport.utility as viewport_utils

environment_name = "{environment_name}"
position = {position}
prim_path = "{prim_path}"

# Environment configurations
environment_configs = {{
    "simple_room": {{"path": "/Isaac/Environments/Simple_Room/simple_room.usd", "camera_pos": [6.0, 6.0, 4.0], "camera_target": [0.0, 0.0, 0.5]}},
    "room": {{"path": "/Isaac/Environments/Simple_Room/simple_room.usd", "camera_pos": [6.0, 6.0, 4.0], "camera_target": [0.0, 0.0, 0.5]}},
    "simple_grid": {{"path": "/Isaac/Environments/Grid/default_environment.usd", "camera_pos": [8.0, 8.0, 5.0], "camera_target": [0.0, 0.0, 0.0]}},
    "grid": {{"path": "/Isaac/Environments/Grid/default_environment.usd", "camera_pos": [8.0, 8.0, 5.0], "camera_target": [0.0, 0.0, 0.0]}},
    "warehouse": {{"path": "/Isaac/Environments/Simple_Warehouse/warehouse.usd", "camera_pos": [20.0, 20.0, 12.0], "camera_target": [0.0, 0.0, 2.0]}},
    "simple_warehouse": {{"path": "/Isaac/Environments/Simple_Warehouse/warehouse.usd", "camera_pos": [20.0, 20.0, 12.0], "camera_target": [0.0, 0.0, 2.0]}},
    "warehouse_forklifts": {{"path": "/Isaac/Environments/Simple_Warehouse/warehouse_with_forklifts.usd", "camera_pos": [25.0, 25.0, 15.0], "camera_target": [0.0, 0.0, 2.0]}},
    "warehouse_shelves": {{"path": "/Isaac/Environments/Simple_Warehouse/warehouse_multiple_shelves.usd", "camera_pos": [30.0, 30.0, 18.0], "camera_target": [0.0, 0.0, 2.0]}},
    "full_warehouse": {{"path": "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd", "camera_pos": [40.0, 40.0, 25.0], "camera_target": [0.0, 0.0, 3.0]}},
    "hospital": {{"path": "/Isaac/Environments/Hospital/hospital.usd", "camera_pos": [30.0, 30.0, 18.0], "camera_target": [0.0, 0.0, 2.0]}},
    "office": {{"path": "/Isaac/Environments/Office/office.usd", "camera_pos": [20.0, 20.0, 12.0], "camera_target": [0.0, 0.0, 2.0]}},
    "jetracer": {{"path": "/Isaac/Environments/Jetracer/jetracer_track_solid.usd", "camera_pos": [15.0, 15.0, 10.0], "camera_target": [0.0, 0.0, 0.0]}},
    "digital_twin": {{"path": "/Isaac/Environments/Small_Warehouse/small_warehouse_digital_twin.usd", "camera_pos": [20.0, 20.0, 12.0], "camera_target": [0.0, 0.0, 2.0]}},
}}

env_key = environment_name.lower().replace(" ", "_").replace("-", "_")

if env_key not in environment_configs:
    print(f"❌ Unknown environment: {{environment_name}}")
    print(f"Available: {{sorted(set(k for k in environment_configs.keys() if '_' not in k or k.count('_') <= 1))}}")
else:
    env_config = environment_configs[env_key]
    assets_root_path = get_assets_root_path()
    asset_path = assets_root_path + env_config["path"]
    
    # Load environment
    add_reference_to_stage(asset_path, prim_path)
    
    # Apply position offset
    if position != [0, 0, 0]:
        stage = omni.usd.get_context().get_stage()
        prim = stage.GetPrimAtPath(prim_path)
        if prim:
            xformable = UsdGeom.Xformable(prim)
            xformable.AddTranslateOp().Set(Gf.Vec3d(*position))
    
    # Add default lighting if needed
    stage = omni.usd.get_context().get_stage()
    has_dome_light = any(p.GetTypeName() == "DomeLight" for p in stage.Traverse())
    
    if not has_dome_light:
        dome_light = UsdLux.DomeLight.Define(stage, "/World/DefaultDomeLight")
        dome_light.GetIntensityAttr().Set(1000.0)
        print("Added dome light")
    
    # Set camera
    camera_pos = env_config.get("camera_pos", [10.0, 10.0, 6.0])
    camera_target = env_config.get("camera_target", [0.0, 0.0, 0.0])
    camera_pos = [camera_pos[i] + position[i] for i in range(3)]
    camera_target = [camera_target[i] + position[i] for i in range(3)]
    
    viewport_api = viewport_utils.get_active_viewport()
    if viewport_api:
        camera_path = "/World/Camera"
        camera_prim = stage.GetPrimAtPath(camera_path)
        if not camera_prim or not camera_prim.IsValid():
            camera = UsdGeom.Camera.Define(stage, camera_path)
        
        # Set camera transform
        cam_pos = Gf.Vec3d(*camera_pos)
        cam_target = Gf.Vec3d(*camera_target)
        forward = (cam_target - cam_pos).GetNormalized()
        world_up = Gf.Vec3d(0, 0, 1)
        right = Gf.Cross(forward, world_up).GetNormalized()
        up = Gf.Cross(right, forward).GetNormalized()
        
        matrix = Gf.Matrix4d()
        matrix.SetIdentity()
        matrix.SetRow(0, Gf.Vec4d(right[0], right[1], right[2], 0))
        matrix.SetRow(1, Gf.Vec4d(up[0], up[1], up[2], 0))
        matrix.SetRow(2, Gf.Vec4d(-forward[0], -forward[1], -forward[2], 0))
        matrix.SetRow(3, Gf.Vec4d(cam_pos[0], cam_pos[1], cam_pos[2], 1))
        
        camera_prim = stage.GetPrimAtPath(camera_path)
        xformable = UsdGeom.Xformable(camera_prim)
        xformable.ClearXformOpOrder()
        xformable.AddTransformOp().Set(matrix)
        
        viewport_api.set_active_camera(camera_path)
    
    print(f"✅ SUCCESS: Loaded environment '{{environment_name}}' at {{prim_path}}")
"""

async def send_load_environment_command(environment_name, position, prim_path, host="localhost", port=8765):
    """Send load environment command to Isaac Sim"""
    
    uri = f"ws://{host}:{port}"
    
    # Generate Python code
    code = LOAD_ENV_CODE_TEMPLATE.format(
        environment_name=environment_name,
        position=position,
        prim_path=prim_path
    )
    
    command = {
        "type": "execute_python",
        "code": code
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(command))
            print(f"📤 Sent load environment command: {environment_name}")
            
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                print(f"🔍 Check WebRTC stream to see the environment")
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
    parser = argparse.ArgumentParser(description="Load environment in Isaac Sim")
    parser.add_argument("--environment", type=str, required=True,
                       help="Environment name (warehouse, hospital, office, room, grid, etc.)")
    parser.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                       metavar=("X", "Y", "Z"))
    parser.add_argument("--path", type=str, default="/World/Environment",
                       help="Custom prim path")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    
    args = parser.parse_args()
    
    success = asyncio.run(
        send_load_environment_command(args.environment, args.pos, args.path, args.host, args.port)
    )
    
    sys.exit(0 if success else 1)