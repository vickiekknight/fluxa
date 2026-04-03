#!/usr/bin/env python3
"""
2_rapp.py — Reward-Aware Physics Prior (RAPP) orchestrator.

Sweeps each physics parameter across test values, evaluating the
Stage 1 trained policy under each modification. Outputs rapp_bounds.json
containing the min/max feasible values per parameter.

Success criterion: task-specific binary check.
  For Franka reach: "Is mean position error < threshold?"

Usage (from inside Docker):
    /isaac-sim/python.sh 2_rapp.py \
        --checkpoint /tmp/eureka_policy.pt \
        --output /tmp/rapp_bounds.json

Usage (from host):
    docker exec fluxa-isaacsim /isaac-sim/python.sh \
        /isaac-sim/fluxa-agent-pack/.agent/skills/reward-designer/scripts/2_rapp.py \
        --checkpoint /tmp/eureka_policy.pt \
        --output /tmp/rapp_bounds.json
"""

import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

# --- Parameter definitions --------------------------------------------
# Following DrEureka's convention:
#   min_0      = values >= 0 at varying magnitudes
#   centered_1 = scale factors centered around 1.0 (the default)

MIN_0 = [0.0, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0]
CENTERED_1 = [0.0, 0.5, 0.9, 1.0, 1.1, 1.5, 2.0]

PARAMETERS = {
    "joint_friction": {
        "test_values": MIN_0,
        "hint": "Friction coefficient applied uniformly to all robot joints.",
    },
    "joint_armature": {
        "test_values": MIN_0,
        "hint": "Value added to the diagonal of the joint inertia matrix.",
    },
    "joint_stiffness_scale": {
        "test_values": CENTERED_1,
        "hint": "Multiplicative scale on the default actuator stiffness (Kp). 1.0 = default.",
    },
    "joint_damping_scale": {
        "test_values": CENTERED_1,
        "hint": "Multiplicative scale on the default actuator damping (Kd). 1.0 = default.",
    },
}

# --- Helpers ----------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EVAL_RAPP_SCRIPT = os.path.join(SCRIPT_DIR, "scripts", "eval_rapp.py")

# Fallback: if this script is in the same dir as eval_rapp.py
if not os.path.exists(EVAL_RAPP_SCRIPT):
    EVAL_RAPP_SCRIPT = os.path.join(SCRIPT_DIR, "eval_rapp.py")


def run_eval(checkpoint, param_name, param_value, num_envs, eval_steps,
             success_threshold, output_path):
    """Launch eval_rapp.py as a subprocess and return the result dict."""
    cmd = [
        "/isaac-sim/python.sh",
        EVAL_RAPP_SCRIPT,
        "--checkpoint", checkpoint,
        "--param-name", param_name,
        "--param-value", str(param_value),
        "--num-envs", str(num_envs),
        "--eval-steps", str(eval_steps),
        "--success-threshold", str(success_threshold),
        "--output", output_path,
    ]

    print(f"\n{'='*60}")
    print(f"Testing {param_name} = {param_value}")
    print(f"{'='*60}")

    process = subprocess.run(cmd, capture_output=False)

    if process.returncode != 0:
        print(f"WARNING: eval_rapp.py exited with code {process.returncode}")
        return None

    if not os.path.exists(output_path):
        print(f"WARNING: output file not found at {output_path}")
        return None

    with open(output_path, 'r') as f:
        return json.load(f)


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAPP: Reward-Aware Physics Prior")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 1 trained policy checkpoint")
    parser.add_argument("--output", type=str, default="/tmp/rapp_bounds.json",
                        help="Path to write RAPP bounds JSON")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--eval-steps", type=int, default=300,
                        help="Inference steps per evaluation")
    parser.add_argument("--success-threshold", type=float, default=0.10,
                        help="Position error threshold in meters (default: 0.10m)")
    args = parser.parse_args()

    tmp_dir = f"/tmp/rapp_{os.getpid()}"
    os.makedirs(tmp_dir, exist_ok=True)

    # ── Step 1: Baseline evaluation (all default parameters) ────────────
    print("\n" + "#"*60)
    print("STEP 1: Baseline evaluation (default physics)")
    print(f"Success threshold: position_error < {args.success_threshold}m")
    print("#"*60)

    baseline_output = os.path.join(tmp_dir, "baseline.json")
    baseline_result = run_eval(
        args.checkpoint, "default", 0.0,
        args.num_envs, args.eval_steps, args.success_threshold,
        baseline_output
    )

    if baseline_result is None or baseline_result.get("status") != "success":
        print("ERROR: Baseline evaluation failed. Cannot proceed.")
        sys.exit(1)

    baseline_pos_error = baseline_result["mean_position_error"]
    baseline_success = baseline_result["success"]

    print(f"\nBaseline position error: {baseline_pos_error:.4f}m")
    print(f"Baseline success: {baseline_success}")

    if not baseline_success:
        print(f"\nWARNING: Baseline FAILS the success threshold "
              f"({baseline_pos_error:.4f}m >= {args.success_threshold}m).")
        print("The Stage 1 policy may need more training iterations,")
        print("or increase --success-threshold to be more lenient.")
        print("Proceeding anyway — bounds may be empty.\n")

    # --- Step 2: Sweep each parameter ------------------------------------
    print("\n" + "#"*60)
    print("STEP 2: Sweeping physics parameters")
    print("#"*60)

    rapp_bounds = {}
    all_results = {}

    for param_name, param_cfg in PARAMETERS.items():
        test_values = param_cfg["test_values"]
        print(f"\n--- Parameter: {param_name} ({len(test_values)} values to test) ---")

        lowest_ok = float("inf")
        highest_ok = float("-inf")
        param_results = []

        for val in test_values:
            output_path = os.path.join(tmp_dir, f"{param_name}_{val}.json")
            result = run_eval(
                args.checkpoint, param_name, val,
                args.num_envs, args.eval_steps, args.success_threshold,
                output_path
            )

            if result is None or result.get("status") != "success":
                ok = False
                pos_err = None
            else:
                ok = result["success"]
                pos_err = result["mean_position_error"]

            param_results.append({
                "value": val,
                "mean_position_error": pos_err,
                "success": ok,
            })

            status = "PASS" if ok else "FAIL"
            err_str = f"{pos_err:.4f}m" if pos_err is not None else "N/A"
            print(f"  {param_name} = {val:>8} → pos_error = {err_str}  [{status}]")

            if ok:
                lowest_ok = min(lowest_ok, val)
                highest_ok = max(highest_ok, val)

        all_results[param_name] = param_results

        if lowest_ok == float("inf"):
            print(f"  WARNING: No successful values for {param_name}!")
            rapp_bounds[param_name] = {
                "min": None,
                "max": None,
                "hint": param_cfg.get("hint", ""),
                "status": "no_feasible_range",
            }
        else:
            print(f"  RAPP bounds for {param_name}: [{lowest_ok}, {highest_ok}]")
            rapp_bounds[param_name] = {
                "min": lowest_ok,
                "max": highest_ok,
                "hint": param_cfg.get("hint", ""),
                "status": "ok",
            }

    # --- Step 3: Write output --------------------------------------------
    output = {
        "baseline_position_error": baseline_pos_error,
        "success_threshold": args.success_threshold,
        "bounds": rapp_bounds,
        "detailed_results": all_results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output, f, indent=2)

    # --- Summary ------------------------------------------------------------
    print("\n" + "="*60)
    print("RAPP RESULTS SUMMARY")
    print("="*60)
    print(f"Baseline position error: {baseline_pos_error:.4f}m")
    print(f"Success threshold: {args.success_threshold}m")
    print()
    for param_name, bounds in rapp_bounds.items():
        if bounds["status"] == "ok":
            print(f"  {param_name}: [{bounds['min']}, {bounds['max']}]")
        else:
            print(f"  {param_name}: NO FEASIBLE RANGE")
    print(f"\nFull results written to {args.output}")

    # Clean up temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()