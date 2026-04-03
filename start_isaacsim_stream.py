#!/usr/bin/env python3
"""
Isaac Sim with WebRTC Streaming + Generic Command Server
Skills can connect and send Python code to execute in the Isaac Sim context
"""

from dotenv import load_dotenv
import os
if os.path.exists("/isaac-sim/fluxa-ws/.env"):
    load_dotenv("/isaac-sim/fluxa-ws/.env")

# ============================================================================
# Import and initialize Isaac Sim FIRST
# ============================================================================
from isaacsim.simulation_app import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,  
    "hide_ui": False,
    "renderer": "RaytracedLighting",
    "display_options": 3094,
}

simulation_app = SimulationApp(launch_config=CONFIG)

# ============================================================================
# Import other Isaac Sim modules
# ============================================================================
import omni.ui as ui
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.api import World
import carb

# ============================================================================
# Viewport Setup
# ============================================================================
print("🔧 Configuring viewport...")
panels_to_close = ["Stage", "Render Settings", "Property", "Content", "Console"]

for panel_name in panels_to_close:
    win = ui.Workspace.get_window(panel_name)
    if win:
        win.visible = False

viewport = ui.Workspace.get_window("Viewport")
if viewport:
    dock_id = viewport.dock_id
    width = ui.Workspace.get_main_window_width()
    height = ui.Workspace.get_main_window_height()
    ui.Workspace.set_dock_id_width(dock_id, width)
    ui.Workspace.set_dock_id_height(dock_id, height)

# ============================================================================
# Enable WebRTC Livestream 
# ============================================================================
print("🔧 Configuring WebRTC settings...")
settings = carb.settings.get_settings()

settings.set("/exts/omni.services.transport.server.http/https/enabled", False)
settings.set("/exts/omni.services.transport.server.http/port", 8211)
settings.set("/app/livestream/port", 49100)
settings.set("/app/livestream/host", "0.0.0.0")

simulation_app.set_setting("/app/window/drawMouse", True)

print("🔧 Enabling WebRTC livestream...")
enable_extension("omni.kit.livestream.webrtc")
simulation_app.update()

# ============================================================================
# Create World
# ============================================================================
print("🔧 Initializing Isaac Sim world...")
my_world = World(
    stage_units_in_meters=1.0,
    backend="torch",
    device="cuda",
    physics_dt=1.0 / 120.0,
)

stage = simulation_app.context.get_stage()
my_world.scene.add_default_ground_plane()

# =================================================
# ===========================
# Generic Command Server
# ============================================================================
import asyncio
import websockets
import json
import threading

# Command queue for thread-safe communication
command_queue = []
command_lock = threading.Lock()

async def handle_command(websocket):
    """Handle incoming WebSocket commands"""
    async for message in websocket:
        try:
            command = json.loads(message)
            command_type = command.get("type")
            print(f"📨 Received command: {command_type}")
            
            # Add to queue for main thread to process
            with command_lock:
                command_queue.append(command)
            
            # Send acknowledgment
            await websocket.send(json.dumps({
                "status": "queued",
                "message": f"Command '{command_type}' queued for execution"
            }))
            
        except json.JSONDecodeError:
            await websocket.send(json.dumps({
                "status": "error",
                "message": "Invalid JSON"
            }))
        except Exception as e:
            await websocket.send(json.dumps({
                "status": "error",
                "message": str(e)
            }))

async def run_server():
    """Run the WebSocket server"""
    server = await websockets.serve(handle_command, "0.0.0.0", 8765)
    print("🔌 Command server listening on port 8765")
    await server.wait_closed()

def start_command_server():
    """Start WebSocket server in a separate thread"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_server())

# Start command server in background thread
server_thread = threading.Thread(target=start_command_server, daemon=True)
server_thread.start()

# ============================================================================
# Generic Command Executor
# ============================================================================
def execute_python_code(code):
    """Execute Python code in Isaac Sim context"""
    try:
        # Create execution context with Isaac Sim modules available
        exec_globals = {
            'omni': __import__('omni'),
            'stage': stage,
            'my_world': my_world,
            'simulation_app': simulation_app,
        }
        
        # Execute the code
        exec(code, exec_globals)
        print(f"✅ Code executed successfully")
        return True
    except Exception as e:
        print(f"❌ Execution error: {e}")
        import traceback
        traceback.print_exc()
        return False

def process_commands():
    """Process queued commands (called from main thread)"""
    with command_lock:
        while command_queue:
            command = command_queue.pop(0)
            command_type = command.get("type")
            
            if command_type == "execute_python":
                code = command.get("code", "")
                execute_python_code(code)
            else:
                print(f"⚠️  Unknown command type: {command_type}")

# ============================================================================
# Print Startup Info
# ============================================================================
print("=" * 80)
print("✅ Isaac Sim streaming server is ready!")
print("=" * 80)
print("🌐 WebRTC Stream: http://localhost:49100/streaming/webrtc-client")
print("🔌 Command Server: ws://localhost:8765")
print("=" * 80)
print("\n📋 Command Format:")
print('  {"type": "execute_python", "code": "print(\'Hello from Isaac Sim\')"}')
print("=" * 80)

# ============================================================================
# Main Simulation Loop
# ============================================================================
print("\n🎮 Starting simulation loop... (Press Ctrl+C to exit)")
try:
    while simulation_app.is_running():
        # Process any queued commands
        process_commands()
        
        # Step simulation
        my_world.step(render=True)
        
except KeyboardInterrupt:
    print("\n⛔ Shutting down...")
finally:
    simulation_app.close()
    print("✅ Closed successfully")