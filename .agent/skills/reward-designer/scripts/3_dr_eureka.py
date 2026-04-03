#!/usr/bin/env python3
"""
3_dr_eureka.py — Stage 3: LLM-guided Domain Randomization configuration.

Given RAPP bounds from Stage 2 and the best reward function from Stage 1,
this script prompts the LLM to generate domain randomization configurations.

The LLM receives:
  1. Task description
  2. RAPP bounds (feasible parameter ranges)
  3. Instructions to select which params to randomize and their ranges

Output: outputs/dr_configs.json containing N DR configuration candidates.

Usage:
    python3 scripts/3_dr_eureka.py
    python3 scripts/3_dr_eureka.py --rapp-bounds outputs/rapp_bounds.json
"""

import os
import json
import yaml
import re
import argparse
from pathlib import Path

from google import genai
import wandb


# ── Default placeholder bounds (used when RAPP hasn't produced real ones) ───

PLACEHOLDER_BOUNDS = {
    "joint_friction": {"min": 0.0, "max": 5.0, "hint": "Friction coefficient applied uniformly to all robot joints."},
    "joint_armature": {"min": 0.0, "max": 1.0, "hint": "Value added to the diagonal of the joint inertia matrix."},
    "joint_stiffness_scale": {"min": 0.5, "max": 1.5, "hint": "Multiplicative scale on default actuator stiffness (Kp). 1.0 = default."},
    "joint_damping_scale": {"min": 0.5, "max": 1.5, "hint": "Multiplicative scale on default actuator damping (Kd). 1.0 = default."},
}


# ── Prompt templates (following DrEureka paper, Prompts 6-7) ───────────────

SYSTEM_PROMPT = """\
You are a reinforcement learning engineer. Your goal is to design a set of \
domain randomization parameters for the given task to facilitate successful \
deployment of the trained policy in the real world.

To do so, you will be given valid parameters as well as a range for each \
parameter that indicates the maximum and minimum values that parameter can take. \
Please note that your randomization ranges do not need to cover most of the range.

Also, you should keep in mind that the more you randomize, the more difficult \
it will be for the policy to learn the task within our fixed compute budget. \
A good policy should be trained only on randomization ranges that will help it \
adapt to the real world.

You should first reason over each parameter and determine if it's useful for \
domain randomization. Then, you should output a range of values for each \
parameter that you think will be useful for the task in a real-world deployment. \
Please explain your reasoning for each parameter.

Output your response in the form of Python code that sets the parameters as \
variables, e.g.:
```python
friction_range = [0.0, 1.0]
```

Please make your variable names match the parameter names provided. Each \
variable should be assigned a range formatted as a Python list with two elements. \
Write everything else as Python comments.\
"""


def build_user_prompt(task_description, bounds_dict):
    """Build the user prompt from task description and RAPP bounds."""
    prompt = f"""\
The task is: {task_description}

The robot will be trained in simulation and then deployed in the real world.

Our parameters and valid ranges are the following:
"""
    for param_name, info in bounds_dict.items():
        low = info["min"]
        high = info["max"]
        hint = info.get("hint", "")
        prompt += f"{param_name}_range = [{low}, {high}]"
        if hint:
            prompt += f"  ({hint})"
        prompt += "\n"

    return prompt


# --- Parsing LLM output ---

def parse_dr_config(llm_text):
    """Extract parameter ranges from LLM-generated Python code.

    Looks for lines like: param_name_range = [low, high]
    Returns a dict of {param_name: [low, high]}.
    """
    # Try to find a python code block first
    match = re.search(r"```python\n(.*?)\n```", llm_text, re.DOTALL)
    code_text = match.group(1) if match else llm_text

    config = {}
    # Match lines like: some_param_range = [0.1, 2.0]
    pattern = r"(\w+_range)\s*=\s*\[([^]]+)\]"
    for m in re.finditer(pattern, code_text):
        var_name = m.group(1)
        values = m.group(2).split(",")
        try:
            low = float(values[0].strip())
            high = float(values[1].strip())
            # Strip the _range suffix for the param name
            param_name = var_name.replace("_range", "")
            config[param_name] = [low, high]
        except (ValueError, IndexError):
            print(f"  Warning: Could not parse {var_name} = [{m.group(2)}]")
            continue

    return config


# --- Main -----------------------

def main():
    parser = argparse.ArgumentParser(description="Stage 3: DR Eureka")
    parser.add_argument("--config", default="cfg/reach.yaml",
                        help="Path to config YAML")
    parser.add_argument("--rapp-bounds", default=None,
                        help="Path to rapp_bounds.json (overrides config)")
    parser.add_argument("--use-placeholders", action="store_true",
                        help="Use placeholder bounds instead of RAPP output")
    parser.add_argument("--num-samples", type=int, default=None,
                        help="Number of DR configs to generate (overrides config)")
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    designer_root = Path(cfg['designer_root']).expanduser()
    dr_cfg = cfg.get('dr_eureka', {})
    num_samples = args.num_samples or dr_cfg.get('num_samples', 3)
    model_name = dr_cfg.get('model', cfg['eureka'].get('model', 'gemini-2.5-flash-lite'))

    # W&B
    wandb.init(
        project="Fluxa-Reward-Designer",
        name=f"dr-eureka-{cfg['task_name']}",
        config=cfg,
    )

    # --- Load RAPP bounds ---
    bounds_dict = None

    if args.use_placeholders:
        print("Using placeholder bounds (--use-placeholders)")
        bounds_dict = PLACEHOLDER_BOUNDS
    else:
        # Try to load from file
        rapp_path = args.rapp_bounds or (designer_root / cfg.get('rapp_bounds_file', 'outputs/rapp_bounds.json'))
        rapp_path = Path(rapp_path)

        if rapp_path.exists():
            with open(rapp_path, 'r') as f:
                rapp_data = json.load(f)

            raw_bounds = rapp_data.get("bounds", {})

            # Filter to only params with feasible ranges
            bounds_dict = {}
            for param_name, info in raw_bounds.items():
                if info.get("status") == "ok" and info["min"] is not None:
                    bounds_dict[param_name] = info
                else:
                    print(f"  Skipping {param_name}: no feasible range from RAPP")

            if not bounds_dict:
                print("\nNo feasible RAPP bounds found. Falling back to placeholders.")
                bounds_dict = PLACEHOLDER_BOUNDS
        else:
            print(f"\nRAPP bounds file not found at {rapp_path}. Using placeholders.")
            bounds_dict = PLACEHOLDER_BOUNDS

    print(f"\nDR parameters ({len(bounds_dict)} total):")
    for name, info in bounds_dict.items():
        print(f"  {name}: [{info['min']}, {info['max']}]")

    # --- Query LLM for DR configurations ---
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    task_description = cfg['task_description']
    user_prompt = build_user_prompt(task_description, bounds_dict)

    print(f"\nGenerating {num_samples} DR configurations...")

    all_configs = []
    output_dir = designer_root / "outputs" / "dr_candidates"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        print(f"\n{'='*60}")
        print(f"  DR Sample {i+1}/{num_samples}")
        print(f"{'='*60}")

        response = client.models.generate_content(
            model=model_name,
            contents=[
                {"role": "user", "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_prompt}]}
            ],
        )

        raw_text = response.text

        # Save raw output
        raw_path = output_dir / f"dr_sample_{i}_raw.txt"
        with open(raw_path, 'w') as f:
            f.write(raw_text)

        # Parse
        dr_config = parse_dr_config(raw_text)

        if not dr_config:
            print(f"  Warning: Failed to parse DR config from sample {i}")
            continue

        print(f"  Parsed {len(dr_config)} parameter ranges:")
        for name, rng in dr_config.items():
            print(f"    {name}: {rng}")

        all_configs.append({
            "sample_id": i,
            "config": dr_config,
        })

        # Log to W&B
        wandb.log({
            "dr_sample": i,
            "num_params_randomized": len(dr_config),
            "raw_output": wandb.Html(f"<pre><code>{raw_text}</code></pre>"),
        })

    # --- Save all configs -----------------------
    output = {
        "task": cfg['task_name'],
        "num_samples": len(all_configs),
        "bounds_used": {k: {"min": v["min"], "max": v["max"]} for k, v in bounds_dict.items()},
        "configs": all_configs,
    }

    output_path = designer_root / "outputs" / "dr_configs.json"
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    # --- Summary -----------------------
    print(f"\n{'='*60}")
    print("DR EUREKA RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Generated {len(all_configs)} DR configurations")
    print(f"Bounds source: {'placeholders' if args.use_placeholders else 'RAPP'}")
    print()

    for cfg_entry in all_configs:
        sid = cfg_entry["sample_id"]
        config = cfg_entry["config"]
        print(f"  Sample {sid}:")
        for name, rng in config.items():
            print(f"    {name}: [{rng[0]}, {rng[1]}]")
        print()

    print(f"Full output saved to: {output_path}")

    wandb.finish()


if __name__ == "__main__":
    main()