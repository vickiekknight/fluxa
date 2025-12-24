#!/usr/bin/env python3
"""
Simple Isaac Sim with WebRTC Streaming
Just streaming - MCP server handles the tools separately
"""

# Load NVIDA API from .env file
from dotenv import load_dotenv
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

# Create SimulationApp instance - this must come BEFORE other Isaac imports
simulation_app = SimulationApp(launch_config=CONFIG)

# ============================================================================
# Import other Isaac Sim modules
# ============================================================================
import omni.ui as ui
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.api import World
import carb
import omni.kit.app

# ============================================================================
# Add MCP Extension Path
# ============================================================================
print("🔧 Adding MCP extension path...")
extension_manager = omni.kit.app.get_app().get_extension_manager()
extension_manager.add_path("/isaac-sim/fluxa-ws/isaac-sim-mcp")
enable_extension("isaac_sim_mcp_extension")

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
print("🔧 Configuring WebRTC settings...")
settings = carb.settings.get_settings()

settings.set("/exts/omni.services.transport.server.http/https/enabled", False)
settings.set("/exts/omni.services.transport.server.http/port", 8211)
settings.set("/app/livestream/port", 49100)
settings.set("/app/livestream/host", "0.0.0.0")

# Enable mouse in headless mode
simulation_app.set_setting("/app/window/drawMouse", True)

# Enable WebRTC extension
print("🔧 Enabling WebRTC livestream...")
enable_extension("omni.kit.livestream.webrtc")
simulation_app.update()

# ============================================================================
# Enable MCP Extension (Socket server on port 8766)
# ============================================================================
print("🔧 Configuring MCP extension settings...")
settings.set("/exts/isaac_sim_mcp_extension/server.socket", 8766)
settings.set("/exts/isaac_sim_mcp_extension/server.host", "0.0.0.0")  # Bind to all interfaces

print("🔧 Enabling MCP extension...")
enable_extension("isaac_sim_mcp_extension")

import time
time.sleep(3)
simulation_app.update()

print("✅ Extensions enabled!")
print("=" * 60)
print("🌐 WebRTC Signaling: port 49100")
print("🔌 MCP Socket Server: port 8766")
print("=" * 60)


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
print("🌐 WebRTC Stream should be available at: http://localhost:49100/streaming/webrtc-client")

# ============================================================================
# Main Simulation Loop
# ============================================================================
print("🎮 Starting simulation loop...")
try:
    while simulation_app.is_running():
        my_world.step(render=True)
except KeyboardInterrupt:
    print("\n⛔ Shutting down...")
finally:
    simulation_app.close()
    print("✅ Closed successfully")