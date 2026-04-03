# reward-designer

Automatically generates optimized reward functions and domain randomization configs
for manipulation tasks in Isaac Sim using the DrEureka pipeline. Runs before every
manipulation task execution to improve policy quality and enable sim-to-real transfer.

## When to use this skill

This skill acts as an optimization "shadow" for robot learning. **By default, always invoke this skill before `manipulation-tasks`**, but use the following logic to choose the execution mode:

### Decision Tree: Choose your Stage
1. **FULL PIPELINE (Planning Mode):** Use when starting a NEW task from scratch (e.g., "Learn to open a drawer").
   - Use if the previous training failed to converge or the robot is "jittery."
   - *Agent Action:* Run `run_pipeline.py` without flags.

2. **FAST OPTIMIZATION (`--fast`):** Use for quick iterative tests or small tweaks (e.g., "Make it a bit smoother").
   - Use during live demos where long waits are not feasible.
   - *Agent Action:* Run `run_pipeline.py --fast`.

3. **SKIP GENERATION (Direct Execution):** Use if `outputs/reward_fn.py` was generated < 30 minutes ago for the same task.
   - Use if the user explicitly says: "use existing reward" or "just run it."
   - *Agent Action:* Skip `reward-designer` and proceed directly to `manipulation-tasks`.

## Skill Shadowing Rules

- **Pre-emptive Check:** Before running any manipulation task, check `outputs/` for existing configs. If they exist, ask: *"I have a recent reward config for this task. Should I re-optimize or use the current one?"*
- **Mode Switching:** If the user is in **Antigravity Fast Mode**, always default to the `--fast` flag or skip Stage 1 (Eureka) to save time.
- **Context Awareness:** If the robot's physical mass or joint friction is changed in the prompt, you **MUST** re-run Stage 2 (RAPP) to update the DR bounds.

## Architecture

### Single-GPU Lifecycle Management

The system runs on a single GPU (e.g., RTX 3090). Isaac Sim cannot share the GPU
between a streaming instance and headless training. The pipeline manages this automatically:

```
User says "do the reach task"
       │
       ▼
┌─────────────────────────────────────┐
│  run_pipeline.py stops streaming    │
│  Isaac Sim (frees GPU)              │
└──────────────┬──────────────────────┘
               │
       ┌───────▼────────┐
       │  Optimization   │  Each eval = fresh headless subprocess
       │  (Stages 1-3)   │  via: docker exec ... /isaac-sim/python.sh eval_headless.py
       │  No streaming   │  Metrics returned via JSON files on shared filesystem
       └───────┬─────────┘
               │
┌──────────────▼──────────────────────┐
│  run_pipeline.py restarts streaming │
│  Isaac Sim for visualization        │
└──────────────┬──────────────────────┘
               │
       ┌───────▼────────┐
       │  manipulation-  │  WebSocket to streaming Isaac Sim
       │  tasks runs     │  User watches robot in WebRTC stream
       │  with optimized │
       │  reward + DR    │
       └────────────────┘
```

### Why Subprocesses Instead of WebSocket

The original approach sent eval code to a running Isaac Sim via WebSocket and `exec()`.
This caused cascading failures: asyncio conflicts, GPU state leaks between iterations,
WebSocket ping timeouts during long rollouts, and `sim.stop()` killing the command server.

The current architecture follows the original Eureka paper: each candidate evaluation
is a **fresh subprocess** (`docker exec ... python.sh eval_headless.py`). This gives:
- Clean GPU context per evaluation (no state leaks)
- No asyncio conflicts (no event loop sharing)
- Automatic resource cleanup when process exits
- Metrics passed via JSON files (no WebSocket/TCP needed)

### Evaluation Architecture

```
Host (1_eureka.py)                    Docker Container
       │                                    │
       │  subprocess.Popen(                 │
       │    "docker exec ...                │
       │     eval_headless.py               │
       │     --reward-file reward_fn.py     │
       │     --train-iterations 300         │
       │     --output metrics.json")        │
       │ ──────────────────────────────────►│
       │                                    │  AppLauncher(headless=True)
       │                                    │  Create env with reward
       │                                    │  PPO training (300 iters)
       │                                    │  Evaluate trained policy
       │                                    │  Write metrics.json
       │                                    │  Save policy checkpoint
       │  proc.wait()                       │  Process exits (GPU freed)
       │◄──────────────────────────────────│
       │                                    
       │  json.load("metrics.json")         
       │  Feed metrics to LLM              
       │  Generate next candidate           
```

## Pipeline Overview

```
Stage 1: 1_eureka.py    → reward_fn.py + policy checkpoint
                           (LLM generates reward, PPO trains + evaluates)

Stage 2: 2_rapp.py      → rapp_bounds.json
                           (sweep physics params, find safe ranges for trained policy)

Stage 3: 3_dr_eureka.py → dr_config.py
                           (LLM generates DR config within RAPP bounds)

run_pipeline.py          → orchestrates all three + Isaac Sim lifecycle
```

Outputs are consumed by the `manipulation-tasks` skill at env creation time.

## Prerequisites

- **Docker:** Isaac Sim runs inside the `fluxa-isaacsim` Docker container with GPU access.
- **Simulator:** Isaac Sim 5.0+ with IsaacLab installed. RSL-RL must be installed
  (`cd /isaac-sim/IsaacLab && ./isaaclab.sh -i rsl_rl`).
- **Skill Dependencies:** This skill shadows and requires the `manipulation-tasks` skill.
- **LLM API Key:** `GEMINI_API_KEY` must be set in the environment.
- **Python (host):** Python 3.10+ with `google-genai`, `pyyaml`, and `wandb` installed.

## Directory Structure

```
reward-designer/
├── SKILL.md                        ← this file
├── cfg/
│   └── reach.yaml                  ← env + docker + training config
├── prompts/
│   ├── reward_signature_reach.txt  ← LLM reward function format spec
│   └── initial_users/
│       └── reach_rapp.txt          ← RAPP bounds for DR LLM prompt (generated by Stage 2)
├── templates/
│   ├── reward_template.py          ← boilerplate wrapping generated reward fn
│   └── dr_template.py              ← boilerplate wrapping generated DR config
├── scripts/
│   ├── eval_headless.py            ← runs INSIDE Docker: PPO train + eval
│   ├── eval_rapp.py                ← runs INSIDE Docker: policy inference under modified physics
│   ├── 1_eureka.py                 ← Stage 1: iterative reward generation (host-side orchestrator)
│   ├── 2_rapp.py                   ← Stage 2: physics parameter sweep (host-side orchestrator)
│   ├── 3_dr_eureka.py              ← Stage 3: DR config generation (host-side orchestrator)
│   └── run_pipeline.py             ← runs all stages + manages Isaac Sim lifecycle
└── outputs/
    ├── reward_fn.py                ← CONSUMED BY manipulation-tasks
    ├── best_policy.pt              ← trained policy from Stage 1 (used by Stage 2)
    ├── rapp_bounds.json            ← intermediate: Stage 2 → Stage 3
    ├── dr_config.py                ← CONSUMED BY manipulation-tasks
    └── candidates/                 ← per-iteration logs, metrics, raw LLM output
        ├── iter0.log
        ├── iter0_metrics.json
        ├── iter0_reward.py
        └── iter0_raw_llm.txt
```

## Stage 1: Reward Generation (`1_eureka.py`)

Implements the Eureka loop with actual PPO training:

1. Prompts LLM (Gemini) to generate candidate reward functions
2. Injects generated code into `reward_template.py`
3. For each candidate: launches `eval_headless.py` as a headless subprocess inside Docker
4. `eval_headless.py` trains a PPO policy for N iterations using RSL-RL, evaluates it,
   writes metrics to a JSON file, and saves the policy checkpoint
5. Host reads metrics, feeds them back to LLM for iterative improvement
6. Best reward function and policy checkpoint are saved to `outputs/`

### Key Design Decision: PPO Training per Candidate

The original eval used random actions for 30 seconds. This couldn't differentiate
reward quality — every reward scored roughly the same noise (~-0.007). Now each
candidate gets 300 PPO iterations (~26s on RTX 3090 with 16 envs), producing
meaningful reward signal that the LLM can use for improvement.

### Usage
```bash
cd ~/fluxa-agent-pack/.agent/skills/reward-designer
python3 scripts/1_eureka.py --task franka-reach
```

### Config (`reach.yaml`)
```yaml
eureka:
  iterations: 3           # Eureka feedback iterations
  candidates: 4           # reward candidates per iteration (future)
  model: "gemini-2.5-flash-lite"
  eval_num_envs: 16
  train_iterations: 300   # PPO iterations per candidate (~26s on RTX 3090)

docker:
  container: "fluxa-isaacsim"
  python: "/isaac-sim/python.sh"
  eval_script: "/isaac-sim/fluxa-agent-pack/.agent/skills/reward-designer/scripts/eval_headless.py"
  shared_dir: "/isaac-sim/fluxa-agent-pack/.agent/skills/reward-designer"
```

### Scaling for Paper Experiments
- **Development/testing:** 3 iterations × 1 candidate × 300 PPO iters (~5 min)
- **Paper experiments:** 3-5 iterations × 4 candidates × 500 PPO iters (~2-3 hours)
- **Full Eureka reproduction:** 5 iterations × 16 candidates × 1000+ PPO iters (~8+ hours)

## Stage 2: RAPP (`2_rapp.py`)

Computes Reward-Aware Physics Prior bounds by sweeping physics parameters:

1. Loads best policy checkpoint from Stage 1 (`outputs/best_policy.pt`)
2. For each randomizable physics parameter:
   - For each test value in a predefined range:
     - Launches `eval_rapp.py` as a headless subprocess
     - `eval_rapp.py` loads the policy, modifies that one physics parameter,
       runs inference (no training), measures success
   - Records the min and max values where the policy still succeeds
3. Writes bounds to `outputs/rapp_bounds.json`

### Usage
```bash
python3 scripts/2_rapp.py --task franka-reach
```

### Randomizable Parameters (from `reach.yaml`)
```yaml
rapp:
  num_envs: 8
  rollout_duration: 20
  success_threshold: 0.05   # 5cm position error threshold
  parameter_test_vals:
    joint_damping_scale:  [0.5, 0.75, 1.0, 1.25, 1.5]
    joint_friction_scale: [0.5, 0.75, 1.0, 1.25, 1.5]
    mass_scale:           [0.7, 0.85, 1.0, 1.15, 1.3]
    action_delay_steps:   [0, 1, 2, 3]
```

### Success Criteria
For the reach task, success means the trained policy still gets the end-effector
within 5cm of the target under modified physics conditions.

### Output Format (`rapp_bounds.json`)
```json
{
    "joint_damping_scale":  {"min": 0.75, "max": 1.25, "nominal": 1.0},
    "joint_friction_scale": {"min": 0.5,  "max": 1.5,  "nominal": 1.0},
    "mass_scale":           {"min": 0.85, "max": 1.15, "nominal": 1.0},
    "action_delay_steps":   {"min": 0,    "max": 2,    "nominal": 0}
}
```

## Stage 3: DR Generation (`3_dr_eureka.py`)

LLM generates domain randomization config using RAPP bounds as guardrails:

1. Reads `outputs/rapp_bounds.json` + `outputs/reward_fn.py`
2. Formats RAPP bounds into `prompts/initial_users/reach_rapp.txt`
3. Prompts LLM to select which parameters to randomize and their ranges
   (must stay within RAPP bounds)
4. Optionally trains with DR applied and checks for performance collapse
5. Writes `outputs/dr_config.py`

### Usage
```bash
python3 scripts/3_dr_eureka.py --task franka-reach
```

### Output Format (`dr_config.py`)
```python
# Generated by 3_dr_eureka.py — DO NOT EDIT MANUALLY
domain_randomization = {
    "joint_damping_scale":  {"distribution": "uniform", "range": [0.8, 1.2]},
    "joint_friction_scale": {"distribution": "uniform", "range": [0.6, 1.4]},
    "mass_scale":           {"distribution": "uniform", "range": [0.9, 1.1]},
    "action_delay_steps":   {"distribution": "choice",  "values": [0, 1]},
}
```

## Full Pipeline (`run_pipeline.py`)

Orchestrates all three stages and manages the Isaac Sim lifecycle.

### What it does:
1. **Stops** the streaming Isaac Sim instance (frees GPU)
2. **Runs** optimization stages headless (each eval = fresh subprocess)
3. **Restarts** the streaming Isaac Sim instance
4. **Optionally** launches the manipulation task for the user to watch

### Usage
```bash
# Full pipeline
python3 scripts/run_pipeline.py --task franka-reach

# Fast mode (fewer iterations)
python3 scripts/run_pipeline.py --task franka-reach --fast

# Run specific stages only
python3 scripts/run_pipeline.py --task franka-reach --stages eureka
python3 scripts/run_pipeline.py --task franka-reach --stages rapp dr_eureka

# Launch demo task after pipeline completes
python3 scripts/run_pipeline.py --task franka-reach --run-task-after --task-duration 120

# Skip streaming restart (just want the output files)
python3 scripts/run_pipeline.py --task franka-reach --no-restart-stream
```

### Arguments
- `--task`: Task name (default: franka-reach)
- `--stages`: Which stages to run — eureka, rapp, dr_eureka (default: all three)
- `--fast`: Reduce iterations/envs for quick testing
- `--run-task-after`: Launch manipulation task with optimized reward after pipeline
- `--task-duration`: Duration for post-pipeline demo (default: 120s)
- `--task-num-envs`: Num envs for post-pipeline demo (default: 16)
- `--no-restart-stream`: Don't restart streaming after optimization

## Integration with manipulation-tasks

After running this skill, `manipulation-tasks` picks up the outputs automatically.
The `reach_task.py` script accepts `--reward-file` and `--dr-config-file` flags
to inject the generated reward and DR config into the streaming Isaac Sim instance.

The `manipulation-tasks` skill falls back to the default Isaac Lab reward if
`reward_fn.py` does not exist — so this skill is purely additive.

## User Experience in Antigravity

The user sees the pipeline progress as chat messages from the Fluxa agent:

```
User: "do the reach task"

Agent: "Starting reward optimization for Franka Reach.
        This takes a few minutes — the stream will pause while I optimize,
        then restart so you can watch the result."

Agent: "Iteration 1/3: mean_reward=-0.0073 — robot isn't reaching targets yet.
        Generating improved reward..."

Agent: "Iteration 2/3: mean_reward=-0.0034 — getting closer."

Agent: "Iteration 3/3: mean_reward=-0.0018 ✅ best result.
        Running RAPP physics sweep...
        Generating DR config...
        Restarting stream now."

Agent: "The optimized reach task is running — check the WebRTC stream
        to watch the robot."
```

The Isaac Sim viewer can be opened as a side panel in Antigravity via the
Simple Browser or a custom extension.

## Standard Execution Order

```bash
# Step 1: Run reward-designer pipeline (stops/restarts streaming automatically)
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/run_pipeline.py \
    --task franka-reach --run-task-after

# That's it — run_pipeline.py handles everything:
#   - Stops streaming Isaac Sim
#   - Runs Stage 1 (Eureka) → reward_fn.py + policy checkpoint
#   - Runs Stage 2 (RAPP) → rapp_bounds.json
#   - Runs Stage 3 (DR Eureka) → dr_config.py
#   - Restarts streaming Isaac Sim
#   - Launches manipulation task with optimized reward + DR
```

## Troubleshooting

### GPU not available / CUDA errors
- The streaming Isaac Sim may still be running and holding the GPU
- `run_pipeline.py` handles this automatically, but if running stages manually:
  `docker exec fluxa-isaacsim pkill -f start_isaacsim_stream && sleep 5`

### LLM generates syntactically invalid reward code
- `inject_code()` sanitizes common Gemini mistakes (e.g., `asset_name=` → `name=`)
- After injection failure, the LLM is told its output was malformed and retries

### Reward function causes NaN rewards
- Eureka detects NaN in metrics and discards that candidate
- Usually caused by division by zero in the generated reward function

### `RslRlOnPolicyRunnerCfg` not defined
- RSL-RL imports must happen *inside* `run_evaluation()`, not at module level
- The `pxr` module is only available after `AppLauncher` initializes the Kit runtime
- This is handled correctly in `eval_headless.py`

### RAPP all values fail success criteria
- Success threshold may be too tight — loosen `success_threshold` in `reach.yaml`
- Or run Stage 1 with more training iterations before Stage 2
- Check that `outputs/best_policy.pt` exists and is valid

### DR config causes training instability
- DR ranges from LLM may be too wide even within RAPP bounds
- `3_dr_eureka.py` detects collapse (reward drops >50%) and re-prompts

### Inference mode tensor errors during evaluation
- `env.step()` and `env.reset()` must stay OUTSIDE `torch.no_grad()` context
- Only the policy forward pass should be wrapped in `torch.no_grad()`
- RSL-RL's `get_inference_policy()` handles this correctly

## Notes

- Each evaluation subprocess takes ~30-60s (Isaac Sim startup + PPO training)
- Candidate reward functions are saved to `outputs/candidates/` for inspection
- The skill is task-agnostic — adding `ur10-reach` support requires only a new
  reward signature prompt and yaml config entry
- W&B logging tracks reward improvement across iterations
- The DrEureka paper used 5 iterations × 16 candidates; for a single RTX 3090,
  start with 3 iterations × 1 candidate and scale up for paper experiments