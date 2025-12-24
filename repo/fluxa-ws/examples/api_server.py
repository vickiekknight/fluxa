# fluxa_scripts/api_server.py
from flask import Flask, send_file, jsonify
from flask_cors import CORS
import threading
import os
import time
import queue
from fluxa_scripts import robot_actions
from fluxa_scripts.simulation_server import init_simulation

# Initialize sim
print("🔧 Initializing simulation...")
simulation_app, my_world, my_franka, left_sensor, right_sensor = init_simulation()

# Create Flask app
app = Flask(__name__)
CORS(app)

# Task queue for trajectory execution
task_queue = queue.Queue()
trajectory_status = {"running": False, "error": None, "complete": False}

def simulation_loop_with_tasks():
    """Main simulation loop that processes trajectory tasks"""
    global trajectory_status
    
    print("🎮 Simulation loop started")
    
    try:
        while True:
            # Check if there's a trajectory to run
            if not task_queue.empty():
                task = task_queue.get()
                
                if task["type"] == "run_trajectory":
                    trajectory_status = {"running": True, "error": None, "complete": False}
                    
                    try:
                        print("🎯 Executing trajectory...")
                        save_path = task["save_path"]
                        
                        robot_actions.run_pickup_trajectory(
                            my_world=my_world,
                            my_franka=my_franka,
                            left_sensor=left_sensor,
                            right_sensor=right_sensor,
                            save_path=save_path
                        )
                        
                        trajectory_status = {"running": False, "error": None, "complete": True}
                        print("✅ Trajectory complete!")
                        
                    except Exception as e:
                        trajectory_status = {"running": False, "error": str(e), "complete": False}
                        print(f"❌ Trajectory failed: {e}")
                        import traceback
                        traceback.print_exc()
            
            # Step the simulation
            my_world.step(render=True)
            
    except KeyboardInterrupt:
        print("\n⛔ Shutting down...")
        simulation_app.close()

@app.route("/run_trajectory", methods=["POST"])
def run_trajectory():
    """Queue a trajectory to run in the simulation loop"""
    
    if trajectory_status["running"]:
        return jsonify({"error": "Trajectory already in progress"}), 409
    
    save_path = "/isaac-sim/fluxa_scripts/tactile_data/tactile_data.json"
    
    # Queue the task
    task_queue.put({
        "type": "run_trajectory",
        "save_path": save_path
    })
    
    # Reset completion status
    trajectory_status["complete"] = False
    
    # Wait for completion (with timeout)
    print("⏳ Waiting for trajectory to complete...")
    timeout = 30  # 30 second timeout
    start_time = time.time()
    
    while not trajectory_status["complete"] and not trajectory_status["error"]:
        if time.time() - start_time > timeout:
            return jsonify({"error": "Trajectory execution timeout"}), 500
        time.sleep(0.1)
    
    # Check for errors
    if trajectory_status["error"]:
        return jsonify({"error": trajectory_status["error"]}), 500
    
    # Return the file
    if os.path.exists(save_path):
        print(f"📦 Sending file: {save_path}")
        return send_file(save_path, as_attachment=True)
    else:
        return jsonify({"error": "Tactile data file not created"}), 500

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "ok", 
        "simulation": "running",
        "trajectory_running": trajectory_status["running"]
    })

@app.route("/status", methods=["GET"])
def status():
    """Get current trajectory status"""
    return jsonify(trajectory_status)

def run_flask_app():
    """Run Flask in a separate thread"""
    print("🌐 Starting Flask API server on port 49101...")
    app.run(host="0.0.0.0", port=49101, threaded=True, use_reloader=False, debug=False)

if __name__ == "__main__":
    print("✅ Starting simulation + API server...")
    
    # Start Flask in a background thread
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    
    # Give Flask a moment to start
    time.sleep(2)
    print("✅ API server ready!")
    
    # Run simulation loop in the MAIN thread with task processing
    simulation_loop_with_tasks()