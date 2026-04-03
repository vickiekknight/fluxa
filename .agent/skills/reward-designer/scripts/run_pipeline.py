#!/usr/bin/env python3
"""
run_pipeline.py — Orchestrates the full DrEureka pipeline.

Manages the single-GPU constraint:
  1. Stops the streaming Isaac Sim instance (if running)
  2. Runs reward-designer stages headless (each eval = fresh subprocess)
  3. Restarts the streaming Isaac Sim instance
  4. Optionally launches the manipulation task for the user to watch

Usage:
    # Full pipeline
    python3 scripts/run_pipeline.py --task franka-reach

    # Fast mode (fewer iterations)
    python3 scripts/run_pipeline.py --task franka-reach --fast

    # Skip streaming restart (if you just want the reward file)
    python3 scripts/run_pipeline.py --task franka-reach --no-restart-stream

    # Only run specific stages
    python3 scripts/run_pipeline.py --task franka-reach --stages eureka
"""

import argparse
import subprocess
import time
import sys
from pathlib import Path


DOCKER_CONTAINER = "fluxa-isaacsim"
STREAM_SCRIPT = "/isaac-sim/fluxa-ws/start_isaacsim_stream.py"
STREAM_STARTUP_WAIT = 20  # seconds to wait for Isaac Sim to boot


def run_command(cmd, description, check=True, timeout=None):
    """Run a shell command with logging."""
    print(f"\n{'─'*50}")
    print(f"{description}")
    print(f"   $ {' '.join(cmd)}")
    print(f"{'─'*50}")
    try:
        result = subprocess.run(cmd, check=check, timeout=timeout,
                                capture_output=True, text=True)
        if result.stdout.strip():
            print(result.stdout.strip())
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode})")
        if e.stderr:
            print(e.stderr[-500:])  # last 500 chars of stderr
        if check:
            raise
        return e
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s")
        return None


def is_stream_running():
    """Check if the streaming Isaac Sim instance is running inside Docker."""
    result = subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, "pgrep", "-f", "start_isaacsim_stream"],
        capture_output=True, text=True
    )
    return result.returncode == 0


def stop_streaming():
    """Stop the streaming Isaac Sim instance to free GPU."""
    if not is_stream_running():
        print("Streaming instance not running, nothing to stop.")
        return

    print("Stopping streaming Isaac Sim to free GPU for optimization...")

    # Kill the Python process running the stream script
    subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, "pkill", "-f", "start_isaacsim_stream"],
        capture_output=True
    )

    # Wait for it to fully shut down
    for i in range(15):
        if not is_stream_running():
            print("Streaming instance stopped.")
            return
        time.sleep(1)

    # Force kill if still running
    subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, "pkill", "-9", "-f", "start_isaacsim_stream"],
        capture_output=True
    )
    time.sleep(2)
    print("Streaming instance force-stopped.")


def start_streaming():
    """Start the streaming Isaac Sim instance."""
    if is_stream_running():
        print("Streaming instance already running.")
        return

    print("Starting streaming Isaac Sim...")
    subprocess.Popen(
        ["docker", "exec", "-d", DOCKER_CONTAINER,
         "/isaac-sim/python.sh", STREAM_SCRIPT],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )

    print(f"Waiting {STREAM_STARTUP_WAIT}s for Isaac Sim to initialize...")
    time.sleep(STREAM_STARTUP_WAIT)

    if is_stream_running():
        print("Streaming instance is ready.")
        print("WebRTC Stream: http://localhost:49100/streaming/webrtc-client")
    else:
        print("Streaming instance may not have started correctly.")


def run_eureka(task, fast=False):
    """Run Stage 1: Eureka reward generation."""
    cmd = [sys.executable, "scripts/1_eureka.py", "--task", task]
    # In fast mode, you could override iterations via env var or config
    # For now, fast mode is handled by reach.yaml having fewer iterations
    run_command(cmd, f"Stage 1: Eureka reward generation ({task})",
                timeout=1800)  # 30 min timeout


def run_rapp(task):
    """Run Stage 2: RAPP physics prior."""
    cmd = [sys.executable, "scripts/2_rapp.py", "--task", task]
    run_command(cmd, f"Stage 2: RAPP physics prior ({task})",
                timeout=1200)  # 20 min timeout


def run_dr_eureka(task):
    """Run Stage 3: DR config generation."""
    cmd = [sys.executable, "scripts/3_dr_eureka.py", "--task", task]
    run_command(cmd, f"Stage 3: DR Eureka config generation ({task})",
                timeout=1800)


def main():
    parser = argparse.ArgumentParser(description="Run the full DrEureka pipeline")
    parser.add_argument("--task", default="franka-reach",
                        help="Task name (default: franka-reach)")
    parser.add_argument("--stages", nargs="+",
                        choices=["eureka", "rapp", "dr_eureka"],
                        default=["eureka", "rapp", "dr_eureka"],
                        help="Which stages to run (default: all)")
    parser.add_argument("--fast", action="store_true",
                        help="Use reduced iterations/envs for quick testing")
    parser.add_argument("--no-restart-stream", action="store_true",
                        help="Don't restart streaming after optimization")
    parser.add_argument("--run-task-after", action="store_true",
                        help="Launch the manipulation task after pipeline completes")
    parser.add_argument("--task-duration", type=int, default=120,
                        help="Duration for post-pipeline task demo (default: 120s)")
    parser.add_argument("--task-num-envs", type=int, default=16,
                        help="Num envs for post-pipeline task demo (default: 16)")
    args = parser.parse_args()

    print(f"""
{'='*60}
  Fluxa Reward Designer Pipeline
{'='*60}
  Task:    {args.task}
  Stages:  {', '.join(args.stages)}
  Fast:    {args.fast}
{'='*60}
""")

    #  Phase 1: Stop streaming to free GPU 
    stop_streaming()
    time.sleep(3)  # let GPU memory fully release

    #  Phase 2: Run optimization stages (headless) 
    try:
        if "eureka" in args.stages:
            run_eureka(args.task, fast=args.fast)

        if "rapp" in args.stages:
            run_rapp(args.task)

        if "dr_eureka" in args.stages:
            run_dr_eureka(args.task)

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        # Still restart streaming even if pipeline fails
        if not args.no_restart_stream:
            start_streaming()
        sys.exit(1)

    #  Phase 3: Restart streaming 
    if not args.no_restart_stream:
        start_streaming()

    #  Phase 4 (optional): Launch the manipulation task for viewing 
    if args.run_task_after and not args.no_restart_stream:
        print(f"\n🤖 Launching {args.task} with generated reward...")
        time.sleep(5)  # extra buffer for stream to be fully ready

        manipulation_script = Path(__file__).parent.parent.parent / \
            "manipulation-tasks" / "scripts" / "reach_task.py"

        if manipulation_script.exists():
            cmd = [
                sys.executable, str(manipulation_script),
                "--task", args.task,
                "--num-envs", str(args.task_num_envs),
                "--duration", str(args.task_duration),
            ]
            run_command(cmd, f"Running {args.task} with optimized reward",
                        check=False)
        else:
            print(f"manipulation-tasks script not found at {manipulation_script}")

    #  Summary 
    outputs_dir = Path("outputs")
    print(f"""
{'='*60}
  Pipeline Complete
{'='*60}
  Outputs:
    Reward:  {outputs_dir / 'reward_fn.py'} {'success' if (outputs_dir / 'reward_fn.py').exists() else 'failed'}
    RAPP:    {outputs_dir / 'rapp_bounds.json'} {'success' if (outputs_dir / 'rapp_bounds.json').exists() else 'skipped'}
    DR:      {outputs_dir / 'dr_config.py'} {'success' if (outputs_dir / 'dr_config.py').exists() else 'skipped'}
{'='*60}
""")


if __name__ == "__main__":
    main()