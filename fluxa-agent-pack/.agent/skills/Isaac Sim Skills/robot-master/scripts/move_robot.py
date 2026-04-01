#!/usr/bin/env python3
"""
Move robot by sending Python code to running Isaac Sim instance
"""

import argparse
import asyncio
import websockets
import json
import sys

# Robot moving code template
MOVE_CODE_TEMPLATE = """
import omni.usd
from pxr import UsdGeom, Gf

ROBOT_CONFIGS = {{
    "franka": "/World/Franka",
    "g1": "/World/G1",
    "go1": "/World/Go1",
    "go2": "/World/Go2",
    "jetbot": "/World/Jetbot",
    "carter": "/World/Carter",
    "h1": "/World/H1"
}}

robot_type = "{robot_type}"
position = {position}
custom_path = {custom_path}

stage = omni.usd.get_context().get_stage()
# Find target path
target_path = custom_path or ROBOT_CONFIGS.get(robot_type.lower())

if not target_path:
    print(f"❌ Unknown robot type or path: {{robot_type}}")
else:
    prim = stage.GetPrimAtPath(target_path)
    if not prim:
        print(f"⚠️  Prim not found at {{target_path}}")
    else:
        # Move it
        xformable = UsdGeom.Xformable(prim)
        
        # Try to find existing translate op first (so we don't accumulate them)
        found_xform = False
        for op in xformable.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                op.Set(Gf.Vec3d(*position))
                found_xform = True
                break
        
        if not found_xform:
            xformable.AddTranslateOp().Set(Gf.Vec3d(*position))
            
        print(f"✅ SUCCESS: Moved {{target_path}} to {{position}}")
"""

async def send_move_command(robot_type, position, custom_path=None, host="localhost", port=8765):
    """Send move command to Isaac Sim"""
    
    uri = f"ws://{host}:{port}"
    
    # Generate Python code to execute
    code = MOVE_CODE_TEMPLATE.format(
        robot_type=robot_type,
        position=position,
        custom_path=f'"{custom_path}"' if custom_path else 'None'
    )
    
    command = {
        "type": "execute_python",
        "code": code
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            # Send command
            await websocket.send(json.dumps(command))
            print(f"📤 Sent move command: {robot_type} to {position}")
            
            # Wait for response
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                print(f"🔍 Check WebRTC stream for updated position")
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
    parser = argparse.ArgumentParser(description="Move robots in running Isaac Sim")
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
        send_move_command(args.type, args.pos, args.path, args.host, args.port)
    )
    
    sys.exit(0 if success else 1)
