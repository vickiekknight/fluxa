#!/usr/bin/env python3
"""
1_eureka.py — Stage 1: Iterative reward generation using Eureka method.

Each candidate evaluation launches a fresh headless subprocess inside the Docker container. 
Metrics are passed back via a JSON file on the shared filesystem.
"""

import os
import json
import yaml
import re
import subprocess
import time
from pathlib import Path

from google import genai
import wandb


class EurekaManager:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.cfg = yaml.safe_load(f)

        # Gemini client
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.model_name = self.cfg['eureka'].get('model', 'gemini-2.5-flash-lite')

        self.designer_root = Path(self.cfg['designer_root']).expanduser()

        # W&B
        wandb.init(
            project="Fluxa-Reward-Designer",
            name=f"eureka-{self.cfg['task_name']}",
            config=self.cfg,
        )

        # Paths
        self.candidates_dir = self.designer_root / "outputs" / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)

        # Docker config
        self.docker_container = self.cfg.get('docker', {}).get('container', 'fluxa-isaacsim')
        self.docker_python = self.cfg.get('docker', {}).get('python', '/isaac-sim/python.sh')
        # Path to eval_headless.py INSIDE the container
        self.eval_script = self.cfg.get('docker', {}).get(
            'eval_script', '/isaac-sim/fluxa-ws/reward-designer/scripts/eval_headless.py'
        )
        # Shared directory visible to both host and container
        self.shared_dir = self.cfg.get('docker', {}).get(
            'shared_dir', '/isaac-sim/fluxa-ws/reward-designer'
        )

    # --- Prompt Building ---

    def build_prompt(self, task_name, iteration=0, feedback=""):
        """Build the LLM prompt from YAML config + reward signature."""
        robot_cfg = self.cfg['robots'][task_name]

        with open(self.designer_root / self.cfg['reward_signature_file'], 'r') as f:
            signature = f.read()

        prompt = f"""
            Task Context: {self.cfg['task_description']}
            Target Robot: {task_name}
            End-Effector Body Name: {robot_cfg['ee_body_name']}

            Formatting Rules:
            {signature}

            Instructions:
            Provide your response as a single Python code block.
            1. Write the reward functions (one per term).
            2. Define a dictionary called 'reward_dict' that contains 'RewTerm' objects
            pointing to those functions.

            IMPORTANT: Do not include imports. Use the standard Isaac Lab MDP naming conventions.
            IMPORTANT: SceneEntityCfg's constructor is SceneEntityCfg(name: str, body_names: list).
            The first argument is `name`, NOT `asset_name`.
            Example: SceneEntityCfg("robot", body_names=["panda_hand"])
            """

        if iteration > 0 and feedback:
            prompt += f"""

            Optimization Feedback from Previous Run:
            {feedback}
            Modify the weights or logic to improve performance.
            """
        return prompt

    def build_reward_reflection(self, metrics):
        """
        Build per-component reward reflection feedback for the LLM.
        
        Mirrors eureka.py's construct_run_log feedback format:
        each reward component gets its trajectory + summary stats,
        so the LLM can diagnose saturating, dead, or dominating terms.
        """
        lines = []

        # Header with overall metrics
        lines.append(f"Overall: success_rate={metrics.get('success_rate', 0):.4f}, "
                    f"mean_reward={metrics.get('mean_reward', 0):.4f}")

        # Per-component reward trajectories
        reward_components = metrics.get("reward_components", {})
        if reward_components:
            lines.append("\nPer-component reward trajectories during training:")
            for term_name in sorted(reward_components.keys()):
                values = reward_components[term_name]
                if not values:
                    continue
                trajectory = [f"{v:.4f}" for v in values]
                term_max = max(values)
                term_min = min(values)
                term_mean = sum(values) / len(values)
                lines.append(
                    f"  {term_name}: {trajectory}, "
                    f"Max: {term_max:.4f}, Mean: {term_mean:.4f}, Min: {term_min:.4f}"
                )
        else:
            lines.append("\nNo per-component reward data available.")

        # Task-specific metric trajectory
        position_errors = metrics.get("position_error", [])
        if position_errors:
            trajectory = [f"{v:.4f}" for v in position_errors]
            lines.append(
                f"\n  position_error (task metric): {trajectory}, "
                f"Final: {position_errors[-1]:.4f}, Min: {min(position_errors):.4f}"
            )

        # Diagnostic hints based on the data
        lines.append("\nThe reward function needs improvement.")

        if metrics.get("success_rate", 0) == 0:
            lines.append("The success rate is 0 — the robot is not reaching targets at all.")

        # Detect common failure modes from the component trajectories
        for term_name, values in reward_components.items():
            if len(values) < 3:
                continue

            # Saturating: value barely changes in the last 70% of training
            late_values = values[len(values) // 3:]
            if len(late_values) > 1:
                late_range = max(late_values) - min(late_values)
                full_range = max(values) - min(values)
                if full_range > 0 and late_range / full_range < 0.1:
                    lines.append(
                        f"  Note: '{term_name}' appears SATURATED — it stopped changing "
                        f"early in training (final value: {values[-1]:.4f}). "
                        f"Consider tightening its threshold or increasing its weight."
                    )

            # Dead: value is essentially zero throughout
            if all(abs(v) < 1e-6 for v in values):
                lines.append(
                    f"  Note: '{term_name}' is DEAD — it was zero throughout training. "
                    f"The condition may be too hard to trigger. "
                    f"Consider relaxing its threshold or adding intermediate shaping."
                )

            # Dominating: magnitude much larger than other terms
            if reward_components:
                term_magnitude = max(abs(v) for v in values)
                other_magnitudes = [
                    max(abs(v) for v in vals)
                    for name, vals in reward_components.items()
                    if name != term_name and vals
                ]
                if other_magnitudes:
                    max_other = max(other_magnitudes)
                    if max_other > 0 and term_magnitude / max_other > 10:
                        lines.append(
                            f"  Note: '{term_name}' is DOMINATING — its magnitude is "
                            f"{term_magnitude / max_other:.1f}x larger than other terms. "
                            f"Consider reducing its weight."
                        )

        return "\n".join(lines)

    # --- Code Extraction & Injection ---

    def inject_code(self, raw_llm_output, candidate_id=""):
        """Extract code from Gemini's response, inject into template, save."""

        # Find python block
        match = re.search(r"```python\n(.*?)\n```", raw_llm_output, re.DOTALL)
        if not match:
            print(f"Error: LLM did not return a valid Python block.")
            return None

        full_code = match.group(1)

        # Sanitize: fix common Gemini mistakes
        full_code = re.sub(r'(def\s+\w+)\s+(\w+)', r'\1\2', full_code)  # space in func names
        full_code = full_code.replace("asset_name=", "name=")  # wrong SceneEntityCfg kwarg

        # Split functions from reward_dict
        lines = full_code.split('\n')
        dict_start_idx = next(
            (i for i, l in enumerate(lines) if l.strip().startswith('reward_dict')),
            None
        )
        if dict_start_idx is None:
            print(f"Error: 'reward_dict' not found in LLM output.")
            return None

        functions = "\n".join(lines[:dict_start_idx])
        dictionary_block = "\n".join(lines[dict_start_idx:])

        # Load template
        template_path = self.designer_root / "templates" / "reward_template.py"
        with open(template_path, 'r') as f:
            template = f.read()

        # Inject
        final_code = template.replace("# INSERT_REWARD_FUNCTIONS_HERE", functions)
        final_code = final_code.replace("# INSERT_REWARD_DICTIONARY_HERE", dictionary_block)

        # Save to outputs
        output_path = self.designer_root / self.cfg['reward_output_file']
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(final_code)

        # Also save candidate copy for bookkeeping
        if candidate_id:
            candidate_path = self.candidates_dir / f"{candidate_id}_reward.py"
            with open(candidate_path, 'w') as f:
                f.write(final_code)

        return output_path

    # --- Evaluation via Subprocess ---

    def run_evaluation(self, task_name, candidate_id=""):
        """
        Launch eval_headless.py as a subprocess inside the Docker container.
        Read metrics from a shared JSON file when it finishes.
        """
        num_envs = self.cfg['eureka'].get('eval_num_envs', 8)
        rollout_dur = self.cfg['eureka'].get('rollout_duration', 30)

        # Paths as seen from INSIDE the container
        reward_file_container = f"{self.shared_dir}/{self.cfg['reward_output_file']}"
        metrics_file_container = f"{self.shared_dir}/outputs/candidates/{candidate_id}_metrics.json"

        # Path as seen from the HOST (for reading results)
        metrics_file_host = self.candidates_dir / f"{candidate_id}_metrics.json"

        # Log file on host
        log_file = self.candidates_dir / f"{candidate_id}.log"

        # Build docker exec command
        cmd = [
            "docker", "exec", self.docker_container,
            self.docker_python, self.eval_script,
            "--reward-file", reward_file_container,
            "--num-envs", str(num_envs),
            "--train-iterations", str(self.cfg['eureka'].get('train_iterations', 300)),
            "--output", metrics_file_container,
        ]

        print(f"Launching headless eval: {candidate_id}")
        print(f"Command: {' '.join(cmd[-6:])}")  # show just the script args

        try:
            with open(log_file, 'w') as f:
                proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

            # Wait for completion with timeout
            timeout = rollout_dur + 120  # generous buffer for Isaac Sim startup
            returncode = proc.wait(timeout=timeout)

            if returncode != 0:
                print(f"Eval process exited with code {returncode}")
                # Print last 20 lines of log for debugging
                self._print_log_tail(log_file, lines=20)
                return {"mean_reward": 0.0, "success_rate": 0.0, "status": "process_error"}

            # Read metrics from JSON file
            if metrics_file_host.exists():
                with open(metrics_file_host, 'r') as f:
                    metrics = json.load(f)
                print(f"{candidate_id}: reward={metrics['mean_reward']:.4f}, "
                      f"success={metrics['success_rate']:.4f}")
                return metrics
            else:
                print(f"Metrics file not found: {metrics_file_host}")
                self._print_log_tail(log_file, lines=20)
                return {"mean_reward": 0.0, "success_rate": 0.0, "status": "no_metrics"}

        except subprocess.TimeoutExpired:
            print(f"Eval timed out after {timeout}s, killing process...")
            proc.kill()
            proc.wait()
            return {"mean_reward": 0.0, "success_rate": 0.0, "status": "timeout"}

        except Exception as e:
            print(f"Eval error: {e}")
            return {"mean_reward": 0.0, "success_rate": 0.0, "status": "error"}

    def _print_log_tail(self, log_file, lines=20):
        """Print the last N lines of a log file for debugging."""
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
            tail = all_lines[-lines:]
            print(f"--- Last {lines} lines of {log_file.name} ---")
            for line in tail:
                print(f"  {line.rstrip()}")
            print("---")
        except Exception:
            pass

    # --- Main Eureka Loop -------------------------------------

    def main_loop(self, task):
        print(f"Starting Fluxa Stage 1 (Eureka) for {task}...")

        best_metrics = None
        best_reward_path = None
        feedback = ""

        for i in range(self.cfg['eureka']['iterations']):
            print(f"\n{'='*60}")
            print(f"  Iteration {i+1}/{self.cfg['eureka']['iterations']}  (K={K} candidates)")
            print(f"{'='*60}")

            prompt = self.build_prompt(task, iteration=i, feedback=feedback)

            # --- Sample K candidates from LLM ---
            raw_responses = []
            for k in range(K):
                print(f"  Querying Gemini for candidate {k+1}/{K}...")
                time.sleep(2)  # rate limit buffer
                try:
                    response = self.client.models.generate_content(
                        model=self.model_name,
                        contents=prompt,
                    )
                    raw_responses.append(response.text)
                except Exception as e:
                    print(f"  Gemini call failed for candidate {k}: {e}")
                    raw_responses.append(None)

            # --- Inject and evaluate each candidate ---
            candidate_results = []  # list of (metrics, reward_path, raw_text)

            for k, raw_text in enumerate(raw_responses):
                candidate_id = f"iter{i}_k{k}"

                if raw_text is None:
                    print(f"  {candidate_id}: skipped (LLM call failed)")
                    candidate_results.append(None)
                    continue

                # Save raw LLM output
                raw_output_path = self.candidates_dir / f"{candidate_id}_raw_llm.txt"
                with open(raw_output_path, 'w') as f:
                    f.write(raw_text)

                # Inject into template
                output_file = self.inject_code(raw_text, candidate_id=candidate_id)
                if not output_file:
                    print(f"  {candidate_id}: code injection failed")
                    candidate_results.append(None)
                    continue

                # Evaluate
                metrics = self.run_evaluation(task, candidate_id=candidate_id)
                candidate_results.append({
                    "metrics": metrics,
                    "reward_path": output_file,
                    "raw_text": raw_text,
                    "candidate_id": candidate_id,
                })

            # --- Select best candidate by task success ---
            valid_results = [r for r in candidate_results if r is not None]

            if not valid_results:
                print(f"  All {K} candidates failed this iteration, retrying...")
                feedback = ("Previous iteration: all candidates failed to produce valid code. "
                            "Please provide a valid Python code block with reward functions and a reward_dict.")
                continue

            # Sort by success_rate (task success), not mean_reward
            best = max(valid_results, key=lambda r: r["metrics"].get("success_rate", 0.0))

            best_metrics = best["metrics"]
            best_reward_path = best["reward_path"]
            best_candidate_id = best["candidate_id"]

            print(f"\n  Best candidate: {best_candidate_id}")
            print(f"    success_rate={best_metrics.get('success_rate', 0):.4f}, "
                  f"mean_reward={best_metrics.get('mean_reward', 0):.4f}")

            # --- Update global best ---
            if (best_metrics_overall is None or
                    best_metrics.get("success_rate", 0) > best_metrics_overall.get("success_rate", 0)):
                best_metrics_overall = best_metrics
                best_reward_path_overall = best_reward_path

            # --- Build feedback for next iteration ---
            feedback = self.build_reward_reflection(best["metrics"])

            # --- W&B logging ---
            exec_rate = len(valid_results) / K
            wandb.log({
                "iteration": i,
                "best_success_rate": best_metrics["success_rate"],
                "best_mean_reward": best_metrics["mean_reward"],
                "execute_rate": exec_rate,
                "num_valid_candidates": len(valid_results),
                "generated_code": wandb.Html(f"<pre><code>{best['raw_text']}</code></pre>"),
            })

        # --- Final long training with best reward ---
        if best_reward_path_overall:
            import shutil
            final_output = self.designer_root / self.cfg['reward_output_file']
            if best_reward_path_overall != final_output:
                shutil.copy2(best_reward_path_overall, final_output)
            print(f"\n Best reward saved to: {final_output}")
            print(f"   Best metrics: {best_metrics_overall}")

            # Long training run to produce checkpoint for Stage 2
            print(f"\n{'='*60}")
            print(f"   Final training with best reward")
            print(f"{'='*60}")

            final_train_iters = self.cfg['eureka'].get('final_train_iterations', 2000)
            reward_file_container = f"{self.shared_dir}/{self.cfg['reward_output_file']}"
            policy_path = f"{self.shared_dir}/outputs/eureka_policy.pt"
            final_metrics_path = f"{self.shared_dir}/outputs/final_metrics.json"

            cmd = [
                "docker", "exec", self.docker_container,
                self.docker_python, self.eval_script,
                "--reward-file", reward_file_container,
                "--num-envs", str(self.cfg['eureka'].get('eval_num_envs', 16)),
                "--train-iterations", str(final_train_iters),
                "--save-policy", policy_path,
                "--output", final_metrics_path,
            ]

            print(f"Training for {final_train_iters} iterations...")
            log_file = self.candidates_dir / "final_train.log"

            try:
                with open(log_file, 'w') as f:
                    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT)

                timeout = final_train_iters + 300
                returncode = proc.wait(timeout=timeout)

                if returncode == 0:
                    final_metrics_host = self.designer_root / "outputs" / "final_metrics.json"
                    if final_metrics_host.exists():
                        with open(final_metrics_host, 'r') as f:
                            final_metrics = json.load(f)
                        print(f"  Final training complete!")
                        print(f"  mean_reward={final_metrics['mean_reward']:.4f}")
                        print(f"  Policy saved to: {policy_path}")
                        wandb.log({"final_mean_reward": final_metrics["mean_reward"]})
                    else:
                        print(f"  Training finished but metrics file not found")
                        self._print_log_tail(log_file)
                else:
                    print(f"  Final training failed with code {returncode}")
                    self._print_log_tail(log_file)

            except subprocess.TimeoutExpired:
                print(f"  Final training timed out, killing...")
                proc.kill()
                proc.wait()
        else:
            print("\n All iterations failed to produce valid reward code.")

        wandb.finish()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="franka-reach")
    args = parser.parse_args()

    manager = EurekaManager("cfg/reach.yaml")
    manager.main_loop(args.task)