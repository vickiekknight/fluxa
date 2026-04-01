#!/usr/bin/env python3
"""
Clear scene in Isaac Sim - remove objects while optionally keeping ground and physics
"""

import argparse
import asyncio
import websockets
import json
import sys

# Clear scene code template
CLEAR_SCENE_CODE_TEMPLATE = """
import omni.usd
import omni.kit.commands

keep_ground = {keep_ground}
keep_physics = {keep_physics}

stage = omni.usd.get_context().get_stage()

# Get the /World prim
world_prim = stage.GetPrimAtPath("/World")
if not world_prim:
    print("❌ No /World prim found")
else:
    # Collect prims to delete
    prims_to_delete = []
    for prim in world_prim.GetChildren():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        
        # Skip ground plane if requested
        if keep_ground and ("GroundPlane" in prim_name or "Ground" in prim_name or "ground" in prim_name.lower()):
            print(f"Keeping ground plane: {{prim_path}}")
            continue
            
        # Skip physics scene if requested  
        if keep_physics and ("PhysicsScene" in prim_name or "physics" in prim_name.lower()):
            print(f"Keeping physics scene: {{prim_path}}")
            continue
        
        # Skip default lights if they exist
        if "DefaultDomeLight" in prim_name or "DefaultDistantLight" in prim_name:
            print(f"Keeping default light: {{prim_path}}")
            continue
            
        # Skip environment if it has "Environment" in the name (optional - can be removed if you want to clear envs too)
        # Commented out - uncomment if you want to preserve environments
        # if "Environment" in prim_name:
        #     print(f"Keeping environment: {{prim_path}}")
        #     continue
        
        prims_to_delete.append(prim_path)
    
    # Delete the prims
    deleted_count = 0
    for prim_path in prims_to_delete:
        try:
            omni.kit.commands.execute("DeletePrims", paths=[prim_path])
            deleted_count += 1
            print(f"Deleted: {{prim_path}}")
        except Exception as e:
            print(f"Failed to delete {{prim_path}}: {{e}}")
    
    print(f"✅ SUCCESS: Cleared {{deleted_count}} objects from scene")
    print(f"   Kept ground: {{keep_ground}}")
    print(f"   Kept physics: {{keep_physics}}")
"""

async def send_clear_scene_command(keep_ground, keep_physics, host="localhost", port=8765):
    """Send clear scene command to Isaac Sim"""
    
    uri = f"ws://{host}:{port}"
    
    # Generate Python code
    code = CLEAR_SCENE_CODE_TEMPLATE.format(
        keep_ground=keep_ground,
        keep_physics=keep_physics
    )
    
    command = {
        "type": "execute_python",
        "code": code
    }
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(command))
            print(f"📤 Sent clear scene command")
            print(f"   Keep ground: {keep_ground}")
            print(f"   Keep physics: {keep_physics}")
            
            response = await websocket.recv()
            result = json.loads(response)
            
            if result.get("status") == "queued":
                print(f"✅ {result.get('message')}")
                print(f"🔍 Check WebRTC stream to see the cleared scene")
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
    parser = argparse.ArgumentParser(description="Clear scene in Isaac Sim")
    parser.add_argument("--keep-ground", dest='keep_ground', action='store_true',
                       help="Keep ground plane (default)")
    parser.add_argument("--no-keep-ground", dest='keep_ground', action='store_false',
                       help="Remove ground plane")
    parser.set_defaults(keep_ground=True)
    
    parser.add_argument("--keep-physics", dest='keep_physics', action='store_true',
                       help="Keep physics scene (default)")
    parser.add_argument("--no-keep-physics", dest='keep_physics', action='store_false',
                       help="Remove physics scene")
    parser.set_defaults(keep_physics=True)
    
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8765)
    
    args = parser.parse_args()
    
    success = asyncio.run(
        send_clear_scene_command(args.keep_ground, args.keep_physics, args.host, args.port)
    )
    
    sys.exit(0 if success else 1)