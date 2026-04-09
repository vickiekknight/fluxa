#!/usr/bin/env python3
"""
4_train_with_dr.py — Stage 4: Train a policy per DR configuration.

Reads the DR configs produced by 3_dr_eureka.py, then for each config:
  1. Launches eval_headless.py as a subprocess
  2. Trains a policy with the Eureka reward + that specific DR config
  3. Collects final metrics
  4. Ranks all configs by mean_reward

Output: outputs/dr_training_results.json with metrics for all configs.

Usage:
    python3 scripts/4_train_with_dr.py
    python3 scripts/4_train_with_dr.py --train-iterations 1000
"""

import os
import json
import yaml
import subprocess
import argparse
from pathlib import Path
import wandb
import re

def parse_training_log(log_path):
    """Extract (timesteps, mean_reward, position_error) tuples from training log."""
    if not log_path.exists():
        return []

    with open(log_path, 'r') as f:
        content = f.read()

    # Each iteration block has these fields we care about
    iter_pattern = re.compile(
        r'Learning iteration (\d+)/\d+.*?'
        r'Mean reward:\s*(-?\d+\.\d+).*?'
        r'Metrics/ee_pose/position_error:\s*(-?\d+\.\d+).*?'
        r'Metrics/ee_pose/orientation_error:\s*(-?\d+\.\d+).*?'
        r'Total timesteps:\s*(\d+)',
        re.DOTALL
    )

    curve = []
    for match in iter_pattern.finditer(content):
        curve.append({
            "iteration": int(match.group(1)),
            "mean_reward": float(match.group(2)),
            "position_error": float(match.group(3)),
            "orientation_error": float(match.group(4)),
            "timesteps": int(match.group(5)),
        })
    return curve


def run_training(cmd, log_file, timeout):
    """Launch eval_headless.py as a subprocess."""
    try:
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)
        returncode = proc.wait(timeout=timeout)
        return returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        return -1


def main():
    parser = argparse.ArgumentParser(description="Stage 4: Train policies with DR configs")
    parser.add_argument("--config", default="cfg/reach.yaml")
    parser.add_argument("--train-iterations", type=int, default=None,
                        help="PPO iterations per DR config (overrides config)")
    parser.add_argument("--num-envs", type=int, default=None,
                        help="Parallel envs (overrides config)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    wandb.init(
        project="Fluxa-Reward-Designer",
        name=f"stage4-{cfg['task_name']}",
        config={
            "task": cfg['task_name'],
            "num_envs": args.num_envs or cfg['eureka'].get('eval_num_envs', 16),
            "train_iterations": args.train_iterations or cfg['eureka'].get('final_train_iterations', 2000),
            "num_seeds": 3,
        },
    )

    designer_root = Path(cfg['designer_root']).expanduser()
    docker = cfg['docker']
    train_iters = args.train_iterations or cfg['eureka'].get('final_train_iterations', 2000)
    num_envs = args.num_envs or cfg['eureka'].get('eval_num_envs', 16)

    # Paths
    candidates_dir = designer_root / "outputs" / "dr_candidates"
    reward_file_host = designer_root / cfg['reward_output_file']
    results_path = designer_root / "outputs" / "dr_training_results.json"

    if not candidates_dir.exists():
        print(f"ERROR: No DR candidates dir at {candidates_dir}")
        print("Run 3_dr_eureka.py first.")
        return

    # Find all dr_config_*.py files
    dr_py_files = sorted(candidates_dir.glob("dr_config_*.py"))
    if not dr_py_files:
        print(f"ERROR: No dr_config_*.py files found in {candidates_dir}")
        return

    if not reward_file_host.exists():
        print(f"ERROR: Reward file not found at {reward_file_host}")
        print("Run 1_eureka.py first.")
        return

    print(f"Found {len(dr_py_files)} DR configs")
    print(f"Reward file: {reward_file_host}")
    print(f"Training iterations per config: {train_iters}")
    print()

    # Container-side paths
    shared_dir = docker['shared_dir']
    reward_file_container = f"{shared_dir}/{cfg['reward_output_file']}"

    # ---- Train one policy per DR config ----
    all_results = []
    NUM_SEEDS = 3 # DrEureka trains 3 random seeds per config

    for i, dr_py in enumerate(dr_py_files):
        print(f"\n{'#'*60}")
        print(f"  Training config {i+1}/{len(dr_py_files)}: {dr_py.name}")
        print(f"{'#'*60}")

        # Container paths
        rel = dr_py.relative_to(designer_root)
        dr_config_container = f"{shared_dir}/{rel}"
        policy_path_container = f"{shared_dir}/outputs/dr_candidates/policy_{i}.pt"
        metrics_path_container = f"{shared_dir}/outputs/dr_candidates/metrics_{i}.json"

        # Host paths (for reading results)
        metrics_path_host = designer_root / "outputs" / "dr_candidates" / f"metrics_{i}.json"
        log_file = designer_root / "outputs" / "dr_candidates" / f"train_{i}.log"

        seed_rewards = []
        failed_seeds = 0

        # Loop through 3 random seeds for the current DR configuration
        for seed in range(NUM_SEEDS):
            print(f"  --> Running Seed {seed + 1}/{NUM_SEEDS}")
            policy_path_container = f"{shared_dir}/outputs/dr_candidates/policy_{i}_seed_{seed}.pt"
            metrics_path_container = f"{shared_dir}/outputs/dr_candidates/metrics_{i}_seed_{seed}.json"
            
            metrics_path_host = designer_root / "outputs" / "dr_candidates" / f"metrics_{i}_seed_{seed}.json"
            log_file = designer_root / "outputs" / "dr_candidates" / f"train_{i}_seed_{seed}.log"

            cmd = [
                "docker", "exec", docker['container'],
                docker['python'], docker['eval_script'],
                "--reward-file", reward_file_container,
                "--dr-config", dr_config_container,
                "--num-envs", str(num_envs),
                "--train-iterations", str(train_iters),
                "--save-policy", policy_path_container,
                "--output", metrics_path_container,
            ]

            print(f"Launching training (timeout: {train_iters + 300}s)...")
            returncode = run_training(cmd, log_file, timeout=train_iters + 300)

            curve = parse_training_log(log_file)

            if returncode == 0 and curve:
                final_reward = curve[-1]["mean_reward"]
                final_pos_error = curve[-1]["position_error"]
                seed_rewards.append(final_reward)
                print(f"      Seed {seed} final_reward={final_reward:.4f} "
                      f"pos_error={final_pos_error:.4f}")

                # Push entire training curve to wandb
                for point in curve:
                    wandb.log({
                        f"config_{i}/seed_{seed}/mean_reward": point["mean_reward"],
                        f"config_{i}/seed_{seed}/position_error": point["position_error"],
                        f"config_{i}/seed_{seed}/orientation_error": point["orientation_error"],
                        f"config_{i}/seed_{seed}/iteration": point["iteration"],
                        f"config_{i}/seed_{seed}/timesteps": point["timesteps"],
                    })
            else:
                print(f"      Seed {seed} failed (code {returncode}).")
                failed_seeds += 1

        # Calculate averages across the 3 seeds
        if len(seed_rewards) > 0:
            avg_reward = sum(seed_rewards) / len(seed_rewards)
            print(f"  Average for Config {i}: mean_reward={avg_reward:.4f} over {len(seed_rewards)} successful seeds")

            wandb.log({
                "config_id": i,
                "config_avg_reward": avg_reward,
                "config_num_successful_seeds": len(seed_rewards),
            })
            
            all_results.append({
                "config_id": i,
                "config_file": str(dr_py),
                "status": "success",
                "metrics": {
                    "mean_reward": avg_reward,
                    "seed_rewards": seed_rewards
                },
            })
        else:
            all_results.append({
                "config_id": i,
                "config_file": str(dr_py),
                "status": "failed",
                "failed_seeds": failed_seeds
            })
    # ---- Rank and save ----
    successful = [r for r in all_results if r["status"] == "success"]

    if successful:
        successful.sort(key=lambda r: r["metrics"]["mean_reward"], reverse=True)
        best = successful[0]
    else:
        best = None

    output = {
        "task": cfg['task_name'],
        "num_configs": len(dr_py_files),
        "num_successful": len(successful),
        "results": all_results,
        "best_config_id": best["config_id"] if best else None,
        "best_mean_reward": best["metrics"]["mean_reward"] if best else None,
    }

    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2)

    # ---- Summary ----
    print(f"\n{'='*60}")
    print("STAGE 4 RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Configs trained: {len(successful)}/{len(dr_py_files)}")
    print()
    for r in all_results:
        if r["status"] == "success":
            mr = r["metrics"]["mean_reward"]
            print(f"  Config {r['config_id']}: mean_reward={mr:.4f}")
        else:
            print(f"  Config {r['config_id']}: {r['status']}")

    if best:
        print(f"\n Best: Config {best['config_id']} "
              f"(mean_reward={best['metrics']['mean_reward']:.4f})")
        print(f" Policy: outputs/dr_candidates/policy_{best['config_id']}.pt")

        wandb.log({
            "best_config_id": best["config_id"],
            "best_mean_reward": best["metrics"]["mean_reward"],
        })
        wandb.summary["best_config_id"] = best["config_id"]
        wandb.summary["best_mean_reward"] = best["metrics"]["mean_reward"]

    wandb.finish()

    print(f"\nFull results: {results_path}")


if __name__ == "__main__":
    main()