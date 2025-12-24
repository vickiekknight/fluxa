# fluxa_scripts/agent_server.py

# ============================================================================
# Imports and Flask Setup
# ============================================================================
from flask import Flask, request, jsonify
from flask_cors import CORS
import threading
import time
from dotenv import load_dotenv
from dedalus_labs import AsyncDedalus, DedalusRunner
from examples import scene_tools
import os
import threading
import queue

load_dotenv('/isaac-sim/dedalus.env')

# Verify Dedalus API key is loaded
api_key = os.getenv("DEDALUS_API_KEY")
if api_key:
    print(f"✅ API Key loaded: {api_key[:10]}...")
else:
    print("❌ API Key NOT found! Check dedalus.env")

from dedalus_labs import Omit
import httpx._content

_original_encode_json = httpx._content.encode_json

def patched_encode_json(json_data):
    """Remove Omit sentinel values before encoding."""
    def clean_omit(obj):
        if isinstance(obj, Omit):
            return None
        elif isinstance(obj, dict):
            return {k: clean_omit(v) for k, v in obj.items() if not isinstance(v, Omit)}
        elif isinstance(obj, list):
            return [clean_omit(item) for item in obj if not isinstance(item, Omit)]
        return obj
    
    cleaned_data = clean_omit(json_data)
    return _original_encode_json(cleaned_data)

httpx._content.encode_json = patched_encode_json

# ============================================================================
# Import and initialize Isaac Sim 
# ============================================================================
from isaacsim.simulation_app import SimulationApp

CONFIG = {
    "width": 1280,
    "height": 720,
    "window_width": 1920,
    "window_height": 1080,
    "headless": True,
    "hide_ui": False,  # Show the GUI
    "renderer": "RaytracedLighting",
    "display_options": 3094,
}

simulation_app = SimulationApp(launch_config=CONFIG)

import omni.ui as ui
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.api import World

# ============================================================================
# Viewport Setup
# ============================================================================
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
simulation_app.set_setting("/app/window/drawMouse", True)
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

print("✅ Isaac Sim initialized and streaming enabled!")

# ============================================================================
# Command Queue
#               for executing commands to main thread to prevent 
#               concurrent modifications from multiple threads
# ============================================================================
command_queue = queue.Queue()

# ============================================================================
# Dedalus Tool Wrappers 
# ============================================================================
async def add_franka(x: float = 0.0, y: float = 0.0, z: float = 0.0, name: str = "my_franka") -> str:
    """Add a Franka robot to the scene."""
    print(f"🔧 [TOOL CALLED] add_franka(x={x}, y={y}, z={z}, name={name})")
    
    # Queue the command instead of executing directly
    result_event = threading.Event()
    result_container = {}
    
    def execute():
        position = [x, y, z]
        scene_tools.add_franka_robot(my_world, position, name)
        result_container['result'] = f"Added Franka robot '{name}' at position {position}"
        result_event.set()
    
    command_queue.put(execute)
    result_event.wait(timeout=5.0)  # Wait for execution
    
    result = result_container.get('result', 'Command queued but not executed')
    print(f"✅ [TOOL RESULT] {result}")
    return result

async def add_cube(
    x: float = 0.5,
    y: float = 0.0,
    z: float = 0.5,
    size: float = 0.1,
    r: float = 1.0,
    g: float = 0.0,
    b: float = 0.0,
    dynamic: bool = True,
    name: str = "cube"
) -> str:
    """Add a cube to the scene."""
    print(f"🔧 [TOOL CALLED] add_cube(x={x}, y={y}, z={z}, size={size}, color=[{r},{g},{b}], dynamic={dynamic}, name={name})")
    
    result_event = threading.Event()
    result_container = {}
    
    def execute():
        position = [x, y, z]
        color = [r, g, b]
        scene_tools.add_cube(my_world, position, size, color, dynamic, name)
        result_container['result'] = f"Added cube '{name}' at position {position}"
        result_event.set()
    
    command_queue.put(execute)
    result_event.wait(timeout=5.0)
    
    result = result_container.get('result', 'Command queued but not executed')
    print(f"✅ [TOOL RESULT] {result}")
    return result

async def add_table(
    x: float = 0.5,
    y: float = 0.0,
    z: float = 0.0,
    width: float = 0.6,
    depth: float = 0.6,
    height: float = 0.4,
    name: str = "table"
) -> str:
    """Add a table to the scene."""
    print(f"🔧 [TOOL CALLED] add_table(x={x}, y={y}, z={z}, width={width}, depth={depth}, height={height}, name={name})")
    
    result_event = threading.Event()
    result_container = {}
    
    def execute():
        position = [x, y, z]
        scene_tools.add_table(my_world, position, width, depth, height, name)
        result_container['result'] = f"Added table '{name}' at position {position}"
        result_event.set()
    
    command_queue.put(execute)
    result_event.wait(timeout=5.0)
    
    result = result_container.get('result', 'Command queued but not executed')
    print(f"✅ [TOOL RESULT] {result}")
    return result

async def clear_scene() -> str:
    """Remove all objects from the scene."""
    print(f"🔧 [TOOL CALLED] clear_scene()")
    
    result_event = threading.Event()
    result_container = {}
    
    def execute():
        scene_tools.clear_scene(my_world)
        result_container['result'] = "Scene cleared"
        result_event.set()
    
    command_queue.put(execute)
    result_event.wait(timeout=5.0)
    
    result = result_container.get('result', 'Command queued but not executed')
    print(f"✅ [TOOL RESULT] {result}")
    return result

async def reset_simulation() -> str:
    """Reset the physics simulation."""
    print(f"🔧 [TOOL CALLED] reset_simulation()")
    
    result_event = threading.Event()
    result_container = {}
    
    def execute():
        scene_tools.reset_simulation(my_world)
        result_container['result'] = "Simulation reset"
        result_event.set()
    
    command_queue.put(execute)
    result_event.wait(timeout=5.0)
    
    result = result_container.get('result', 'Command queued but not executed')
    print(f"✅ [TOOL RESULT] {result}")
    return result


# ----------------------------------------------------------------------------
# Flask App Setup
# ----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app, 
     supports_credentials=True, 
     resources={
         r"/*": {
             "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
             "methods": ["GET", "POST", "OPTIONS"],
             "allow_headers": ["Content-Type"]
         }
     })

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', 'http://localhost:5173')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    return response

def run_flask_app():
    """Run Flask in a separate thread"""
    print("🌐 Starting Agent API server on port 4373...")
    print(f"Flask started on Thread: {threading.current_thread().name} ({threading.get_ident()})")
    app.run(host="0.0.0.0", port=4373, threaded=True, use_reloader=False, debug=False)

# Start Flask immediately in a background thread
flask_thread = threading.Thread(target=run_flask_app, daemon=True)
flask_thread.start()

# ============================================================================
# Simulation Loop
# ============================================================================
def simulation_loop():
    """Keep the simulation running"""
    print("🎮 Simulation loop started")
    print(f"Simulation loop started on Thread: {threading.current_thread().name} ({threading.get_ident()})")
    try:
        while True:
            # Process any queued commands
            while not command_queue.empty():
                try:
                    command = command_queue.get_nowait()
                    print(f"⚙️ Executing queued command from main thread")
                    command()
                except queue.Empty:
                    break
                except Exception as e:
                    print(f"❌ Error executing command: {e}")
                    import traceback
                    traceback.print_exc()
            my_world.step(render=True)
    except KeyboardInterrupt:
        print("\n⛔ Shutting down...")
        simulation_app.close()

# ============================================================================
# Flask Routes
# ============================================================================
@app.route("/agent/create_scene", methods=["POST"])
def create_scene():
    """
    Use AI agent to interpret user's natural language and build a scene.
    """
    data = request.get_json()
    user_message = data.get("message", "")
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    print(f"🤖 Agent processing: {user_message}")

    # System prompt for agent reasoning
    system_prompt = """You are an Isaac Sim scene builder assistant. 
    You help users create robotics training environments by interpreting their requests 
    and calling the appropriate functions.

    Available objects:
    - Franka robot (robotic arm)
    - Cubes (can be dynamic or fixed, any color)
    - Tables

    When the user asks to create a scene:
    1. Understand what objects they want
    2. Determine appropriate positions (consider robot workspace is roughly 0.3-1.0m in front)
    3. Call the tools to add objects in a logical order
    4. Always reset the simulation after adding objects

    Position guidelines:
    - Robot is typically at [0, 0, 0]
    - Objects in front of robot: x=0.5-0.9, y=-0.3 to 0.3, z=0.0-1.0
    - Tables should support objects above them

    Be helpful and confirm what you created!
    """

    try:
        import asyncio

        async def run_agent():
            # runner = DedalusRunner(dedalus_client)
            local_client = AsyncDedalus()
            runner = DedalusRunner(local_client)
            print(f"Dedalus started on Thread: {threading.current_thread().name} ({threading.get_ident()})")
            
            # original_build_request = local_client._build_request
            
            # def debug_build_request(options, **kwargs):
            #     print("\n=== DEBUG: API Request Body ===")
            #     import json
            #     try:
            #         body = options.json_data
            #         print("Request body keys:", body.keys() if isinstance(body, dict) else "N/A")
                    
            #         # Check each part for Omit
            #         for key, value in body.items():
            #             try:
            #                 json.dumps({key: value})
            #                 print(f"✓ {key}: serializable")
            #             except TypeError as e:
            #                 print(f"✗ {key}: NOT serializable - {e}")
            #                 print(f"   Type: {type(value)}")
            #                 print(f"   Value preview: {str(value)[:200]}")
            #     except Exception as e:
            #         print(f"Error inspecting request: {e}")
            #     print("=== END DEBUG ===\n")
                
            #     return original_build_request(options, **kwargs)
            
            # local_client._build_request = debug_build_request
            # print("=== END DEBUG ===\n")
                
            result = await runner.run(
                input=f"{system_prompt}\n\nUser request: {user_message}",
                model=["openai/gpt-4.1"],
                tools=[
                    add_franka,
                    add_cube,
                    add_table,
                    clear_scene,
                    reset_simulation
                ],
                stream=False,
                verbose=True 
            )
            return result.final_output

        # Create a new event loop for Dedalus thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response_text = loop.run_until_complete(run_agent())
        finally:
            loop.close()

        print(f"✅ Agent response: {response_text}")

        return jsonify({
            "success": True,
            "message": response_text,
            "user_input": user_message
        })

    except Exception as e:
        print(f"❌ Agent error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok",
        "simulation": "running",
        "agent": "ready"
    })

# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    print("✅ Starting simulation + Agent API server...")
    time.sleep(2)  # Give Flask time to start
    print("✅ Agent API server ready!")
    simulation_loop()