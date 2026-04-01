#!/usr/bin/env python3
"""
Execute arbitrary Python code in running Isaac Sim instance
"""

import argparse
import asyncio
import websockets
import json
import sys

async def send_execute_code_command(code, host="localhost", port=8765):
    """Send execute code command to Isaac Sim"""
    
    uri = f"ws://{host}:{port}"
    
    command = {
        "type": "execute_python",
        "code": code
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(command))
            print(f"📤 Sent code execution command ({len(code)} characters)")
            
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                print(f"🔍 Check Isaac Sim logs for output")
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
    parser = argparse.ArgumentParser(
        description="Execute Python code in Isaac Sim",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Simple print
  python3 execute_code.py --code "print('Hello Isaac Sim')"
  
  # Create a cube
  python3 execute_code.py --code "
from pxr import UsdGeom, Gf
cube = UsdGeom.Cube.Define(stage, '/World/MyCube')
cube.CreateSizeAttr().Set(1.0)
xformable = UsdGeom.Xformable(cube.GetPrim())
xformable.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
print('Created cube')
  "
  
  # Execute from file
  python3 execute_code.py --file my_script.py
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--code", type=str,
                       help="Python code string to execute")
    group.add_argument("--file", type=str,
                       help="Path to Python file to execute")
    
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    
    args = parser.parse_args()
    
    # Get code from file or command line
    if args.file:
        try:
            with open(args.file, 'r') as f:
                code = f.read()
            print(f"📄 Read code from file: {args.file}")
        except FileNotFoundError:
            print(f"❌ ERROR: File not found: {args.file}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ ERROR reading file: {e}")
            sys.exit(1)
    else:
        code = args.code
    
    # Show code preview if it's long
    if len(code) > 200:
        preview = code[:200] + "..."
        print(f"Code preview:\n{preview}\n")
    else:
        print(f"Code to execute:\n{code}\n")
    
    success = asyncio.run(
        send_execute_code_command(code, args.host, args.port)
    )
    
    sys.exit(0 if success else 1)