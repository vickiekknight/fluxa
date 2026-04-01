#!/usr/bin/env python3
"""
List scene objects in Isaac Sim
"""

import asyncio
import websockets
import json
import sys

LIST_SCENE_CODE = """
import omni.usd

stage = omni.usd.get_context().get_stage()
world_prim = stage.GetPrimAtPath("/World")

if not world_prim:
    print("❌ No /World prim found")
else:
    print(f"📦 Objects under /World:")
    for prim in world_prim.GetChildren():
        prim_path = str(prim.GetPath())
        prim_name = prim.GetName()
        prim_type = prim.GetTypeName()
        print(f" - {prim_name} ({prim_type}) at {prim_path}")
"""

async def list_scene(host="localhost", port=8765):
    uri = f"ws://{host}:{port}"
    command = {"type": "execute_python", "code": LIST_SCENE_CODE}
    
    try:
        async with websockets.connect(uri) as websocket:
            await websocket.send(json.dumps(command))
            response = await websocket.recv()
            result = json.loads(response)
            if result.get("status") == "queued":
                print(result.get("message"))
            else:
                print(f"Error: {result.get('message')}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(list_scene())
