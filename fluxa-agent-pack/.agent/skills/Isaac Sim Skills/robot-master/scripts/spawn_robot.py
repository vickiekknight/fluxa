#!/usr/bin/env python3
"""
Spawn robots by sending Python code to running Isaac Sim instance
"""

import argparse
import asyncio
import websockets
import json
import sys

# Robot spawning code template
SPAWN_CODE_TEMPLATE = """
import omni.usd
from pxr import UsdGeom, Gf
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path

ROBOT_CONFIGS = {{
    "franka": {{"asset": "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd", "default_path": "/World/Franka"}},
    "g1": {{"asset": "/Isaac/Robots/Unitree/G1/g1.usd", "default_path": "/World/G1"}},
    "go1": {{"asset": "/Isaac/Robots/Unitree/Go1/go1.usd", "default_path": "/World/Go1"}},
    "go2": {{"asset": "/Isaac/Robots/Unitree/Go2/go2.usd", "default_path": "/World/Go2"}},
    "jetbot": {{"asset": "/Isaac/Robots/NVIDIA/Jetbot/jetbot.usd", "default_path": "/World/Jetbot"}},
    "carter": {{"asset": "/Isaac/Robots/NVIDIA/Carter/carter_v1.usd", "default_path": "/World/Carter"}},
    "h1": {{"asset": "/Isaac/Robots/Unitree/H1/h1.usd", "default_path": "/World/H1"}}
}}

robot_type = "{robot_type}"
position = {position}
prim_path = {prim_path}

stage = omni.usd.get_context().get_stage()
assets_root = get_assets_root_path()

config = ROBOT_CONFIGS.get(robot_type.lower())
if not config:
    print(f"❌ Unknown robot type: {{robot_type}}")
else:
    asset_path = assets_root + config["asset"]
    target_path = prim_path or config["default_path"]
    
    # Check if already exists
    if stage.GetPrimAtPath(target_path):
        print(f"⚠️  Prim already exists at {{target_path}}")
    else:
        # Add to stage
        add_reference_to_stage(asset_path, target_path)
        
        # Set position
        prim = stage.GetPrimAtPath(target_path)
        if prim:
            xformable = UsdGeom.Xformable(prim)
            translate_op = xformable.AddTranslateOp()
            translate_op.Set(Gf.Vec3d(*position))
        
        print(f"✅ SUCCESS: {{robot_type}} spawned at {{target_path}}")
"""

async def send_spawn_command(robot_type, position, prim_path=None, host="localhost", port=8765):
    """Send spawn command to Isaac Sim"""
    
    uri = f"ws://{host}:{port}"
    
    # Generate Python code to execute
    code = SPAWN_CODE_TEMPLATE.format(
        robot_type=robot_type,
        position=position,
        prim_path=f'"{prim_path}"' if prim_path else 'None'
    )
    
    command = {
        "type": "execute_python",
        "code": code
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send command
            await websocket.send(json.dumps(command))
            print(f"📤 Sent spawn command: {robot_type} at {position}")
            
            # Wait for response
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                print(f"🔍 Check WebRTC stream to see the robot")
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
    parser = argparse.ArgumentParser(description="Spawn robots in running Isaac Sim")
    parser.add_argument("--type", type=str, default="franka", 
                       choices=["franka", "g1", "go1", "go2", "jetbot", "carter", "h1"])
    parser.add_argument("--pos", type=float, nargs=3, default=[0.0, 0.0, 0.0], 
                       metavar=("X", "Y", "Z"))
    parser.add_argument("--path", type=str, default=None,
                       help="Custom USD prim path")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    
    args = parser.parse_args()
    
    success = asyncio.run(
        send_spawn_command(args.type, args.pos, args.path, args.host, args.port)
    )
    
    sys.exit(0 if success else 1)