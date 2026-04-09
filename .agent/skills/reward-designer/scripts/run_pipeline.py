#!/usr/bin/env python3
"""
run_pipeline.py — Orchestrates the full DrEureka pipeline.

Manages the single-GPU constraint:
  1. Stops the streaming Isaac Sim instance (if running)
  2. Runs reward-designer stages headless (each eval = fresh subprocess)
  3. Restarts the streaming Isaac Sim instance
  4. Optionally launches the manipulation task for the user to watch

Pipeline stages:
  1. Eureka         — LLM generates reward function, trains policy, saves checkpoint
  2. RAPP           — Sweeps physics parameters to find feasible DR bounds
  3. DR Eureka      — LLM generates DR configurations from RAPP bounds
  4. Train with DR  — Trains a policy per DR config, ranks by performance

Usage:
    # Full pipeline
    python3 scripts/run_pipeline.py

    # Only run specific stages
    python3 scripts/run_pipeline.py --stages eureka rapp

    # Skip streaming restart (if you just want the artifacts)
    python3 scripts/run_pipeline.py --no-restart-stream

    # Use placeholder bounds for Stage 3 (skip real RAPP)
    python3 scripts/run_pipeline.py --stages dr_eureka train_dr --use-placeholders

    # Scale Stage 4 training
    python3 scripts/run_pipeline.py --stage4-num-envs 256 --stage4-train-iterations 2000
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
    """Run a shell command with live output streamed to the terminal."""
    print(f"\n{'─'*60}")
    print(f"{description}")
    print(f"   $ {' '.join(cmd)}")
    print(f"{'─'*60}")
    try:
        # Stream output directly so user sees progress in real time
        result = subprocess.run(cmd, check=check, timeout=timeout)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed (exit {e.returncode})")
        if check:
            raise
        return e
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout}s")
        return None


# --- Streaming Isaac Sim management -----------------------------------------

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
    subprocess.run(
        ["docker", "exec", DOCKER_CONTAINER, "pkill", "-f", "start_isaacsim_stream"],
        capture_output=True
    )

    for i in range(15):
        if not is_stream_running():
            print("Streaming instance stopped.")
            return
        time.sleep(1)

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


# --- Stage runners ---------------------------------------------------------

def run_eureka():
    """Stage 1: Eureka reward generation + long training run to produce checkpoint."""
    cmd = [sys.executable, "scripts/1_eureka.py"]
    run_command(cmd, "Stage 1: Eureka reward generation", timeout=3600)


def run_rapp():
    """Stage 2: RAPP physics prior sweep."""
    checkpoint = "/isaac-sim/fluxa-agent-pack/.agent/skills/reward-designer/outputs/eureka_policy.pt"
    output = "/isaac-sim/fluxa-agent-pack/.agent/skills/reward-designer/outputs/rapp_bounds.json"

    cmd = [
        "docker", "exec", DOCKER_CONTAINER,
        "/isaac-sim/python.sh",
        "/isaac-sim/fluxa-agent-pack/.agent/skills/reward-designer/scripts/2_rapp.py",
        "--checkpoint", checkpoint,
        "--output", output,
    ]
    run_command(cmd, "Stage 2: RAPP physics prior", timeout=1800)


def run_dr_eureka(use_placeholders=False):
    """Stage 3: LLM generates DR configurations from RAPP bounds."""
    cmd = [sys.executable, "scripts/3_dr_eureka.py"]
    if use_placeholders:
        cmd.append("--use-placeholders")
    run_command(cmd, "Stage 3: DR Eureka config generation", timeout=600)


def run_train_with_dr(num_envs, train_iterations):
    """Stage 4: Train a policy per DR config and rank them."""
    cmd = [
        sys.executable, "scripts/4_train_with_dr.py",
        "--num-envs", str(num_envs),
        "--train-iterations", str(train_iterations),
    ]
    run_command(cmd, "Stage 4: Train with DR configs", timeout=7200)


# --- Main ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run the full DrEureka pipeline")
    parser.add_argument("--stages", nargs="+",
                        choices=["eureka", "rapp", "dr_eureka", "train_dr"],
                        default=["eureka", "rapp", "dr_eureka", "train_dr"],
                        help="Which stages to run (default: all four)")
    parser.add_argument("--use-placeholders", action="store_true",
                        help="Use placeholder DR bounds in Stage 3 (skips real RAPP)")
    parser.add_argument("--no-restart-stream", action="store_true",
                        help="Don't restart streaming after optimization")
    parser.add_argument("--stage4-num-envs", type=int, default=16,
                        help="Num parallel envs for Stage 4 (default: 16)")
    parser.add_argument("--stage4-train-iterations", type=int, default=500,
                        help="PPO iterations per DR config in Stage 4 (default: 500)")
    args = parser.parse_args()

    print(f"""
{'='*60}
  Fluxa Reward Designer Pipeline
{'='*60}
  Stages:          {', '.join(args.stages)}
  Stage 4 envs:    {args.stage4_num_envs}
  Stage 4 iters:   {args.stage4_train_iterations}
  Placeholders:    {args.use_placeholders}
{'='*60}
""")

    # Phase 1: Stop streaming to free GPU 
    stop_streaming()
    time.sleep(3)  # let GPU memory fully release

    # Phase 2: Run pipeline stages 
    try:
        if "eureka" in args.stages:
            run_eureka()

        if "rapp" in args.stages:
            run_rapp()

        if "dr_eureka" in args.stages:
            run_dr_eureka(use_placeholders=args.use_placeholders)

        if "train_dr" in args.stages:
            run_train_with_dr(args.stage4_num_envs, args.stage4_train_iterations)

    except Exception as e:
        print(f"\n Pipeline failed: {e}")
        if not args.no_restart_stream:
            start_streaming()
        sys.exit(1)

    # Phase 3: Restart streaming 
    if not args.no_restart_stream:
        start_streaming()

    # Summary 
    outputs = Path("outputs")
    files = {
        "Reward function":   outputs / "reward_fn.py",
        "Policy checkpoint": outputs / "eureka_policy.pt",
        "RAPP bounds":       outputs / "rapp_bounds.json",
        "DR configs":        outputs / "dr_configs.json",
        "DR training":       outputs / "dr_training_results.json",
    }

    print(f"\n{'='*60}")
    print("  Pipeline Complete")
    print(f"{'='*60}")
    for name, path in files.items():
        marker = "✓" if path.exists() else "✗"
        print(f"  {marker} {name}: {path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()