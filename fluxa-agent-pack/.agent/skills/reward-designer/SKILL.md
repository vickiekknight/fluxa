# reward-designer

Automatically generates optimized reward functions and domain randomization configs
for manipulation tasks in Isaac Sim. Runs before every manipulation task execution
to improve policy quality.

## When to use this skill

This skill acts as an optimization "shadow" for robot learning. **By default, always invoke this skill before `manipulation-tasks`**, but use the following logic to choose the execution mode:

### Decision Tree: Choose your Stage
1. **FULL PIPELINE (Planning Mode):** - Use when starting a NEW task from scratch (e.g., "Learn to open a drawer").
   - Use if the previous training failed to converge or the robot is "jittery."
   - *Agent Action:* Run `run_pipeline.py` without flags.

2. **FAST OPTIMIZATION (`--fast`):**
   - Use for quick iterative tests or small tweaks (e.g., "Make it a bit smoother").
   - Use during live demos where 10+ minute waits are not feasible.
   - *Agent Action:* Run `run_pipeline.py --fast`.

3. **SKIP GENERATION (Direct Execution):**
   - Use if `outputs/reward_fn.py` was generated < 30 minutes ago for the same task.
   - Use if the user explicitly says: "use existing reward" or "just run it."
   - *Agent Action:* Skip `reward-designer` and proceed directly to `manipulation-tasks`.

---

## Skill Shadowing Rules

- **Pre-emptive Check:** Before running any manipulation task, check `outputs/` for existing configs. If they exist, ask: *"I have a recent reward config for this task. Should I re-optimize or use the current one?"*
- **Mode Switching:** If the user is in **Antigravity Fast Mode**, always default to the `--fast` flag or skip Stage 1 (Eureka) to save time.
- **Context Awareness:** If the robot's physical mass or joint friction is changed in the prompt, you **MUST** re-run Stage 2 (RAPP) to update the DR bounds.

## Pipeline Overview

```
1_eureka.py      → reward_fn.py         (LLM generates + iterates reward code)
2_rapp.py        → rapp_bounds.json     (physics rollouts define safe DR ranges)
3_dr_eureka.py   → dr_config.py         (LLM generates DR config from RAPP bounds)
run_pipeline.py  → runs all three in sequence
```

Outputs are consumed by the `manipulation-tasks` skill at env creation time.

## Prerequisites

- **Simulator:** Isaac Sim must be active with the WebSocket command server enabled (default port: 8765).
- **Skill Dependencies:** This skill shadows and requires the `manipulation-tasks` skill to be present in the same workspace.
- **Connectivity:** - `ANTHROPIC_API_KEY` must be configured in the Antigravity environment settings or a local `.env`.
  - Internet access is required for Eureka LLM synthesis calls.
- **Environment:** Python 3.10+ with `anthropic`, `torch`, and `websockets` installed in the Isaac Sim Python context.

## Directory Structure

```
reward-designer/
├── SKILL.md                        ← this file
├── cfg/
│   └── reach.yaml                  ← env-specific config (paths, params)
├── prompts/
│   ├── reward_signature_reach.txt  ← LLM reward function format spec
│   └── initial_users/
│       └── reach_rapp.txt          ← RAPP bounds formatted for DR LLM prompt (generated)
├── templates/
│   ├── reward_template.py          ← boilerplate wrapping generated reward fn
│   └── dr_template.py             ← boilerplate wrapping generated DR config
├── scripts/
│   ├── 1_eureka.py                 ← Stage 1: iterative reward generation
│   ├── 2_rapp.py                   ← Stage 2: physics perturbation + RAPP bounds
│   ├── 3_dr_eureka.py              ← Stage 3: DR config generation
│   └── run_pipeline.py             ← Runs all three stages in sequence
└── outputs/
    ├── reward_fn.py                ← CONSUMED BY manipulation-tasks
    ├── rapp_bounds.json            ← intermediate: Stage 2 → Stage 3
    └── dr_config.py               ← CONSUMED BY manipulation-tasks
```

## Stage 1: Reward Generation (`1_eureka.py`)

Implements the Eureka loop:
1. Reads `reach_env_cfg.py` (env source) + `reward_signature_reach.txt` (format spec)
2. Prompts LLM to generate N candidate reward functions
3. For each candidate: injects into reward template → runs short Isaac Sim rollout → collects metrics
4. Feeds metrics back to LLM for iterative improvement
5. Writes best reward function to `outputs/reward_fn.py`

### Usage
```bash
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/1_eureka.py \
    --task franka-reach \
    --num-envs 16 \
    --rollout-duration 30 \
    --iterations 3 \
    --candidates 4
```

### Arguments
- `--task`: Task name passed to manipulation-tasks (default: franka-reach)
- `--num-envs`: Environments per rollout (default: 16, fewer = faster iteration)
- `--rollout-duration`: Seconds per candidate evaluation (default: 30)
- `--iterations`: LLM feedback iterations (default: 3)
- `--candidates`: Reward candidates per iteration (default: 4)
- `--host`: Isaac Sim host (default: localhost)
- `--port`: Command server port (default: 8765)

## Stage 2: RAPP (`2_rapp.py`)

Computes Reward-Aware Physics Prior bounds:
1. Loads best `outputs/reward_fn.py` from Stage 1
2. Defines list of randomizable physics parameters (joint damping, friction, mass scale, etc.)
3. For each parameter: sweeps across test values, runs short rollout, measures task success
4. Identifies the range where the policy still succeeds → RAPP bounds
5. Writes bounds to `outputs/rapp_bounds.json`

### Usage
```bash
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/2_rapp.py \
    --task franka-reach \
    --num-envs 8 \
    --rollout-duration 20
```

### Arguments
- `--task`: Task name (default: franka-reach)
- `--num-envs`: Environments per physics sweep rollout (default: 8)
- `--rollout-duration`: Seconds per parameter sweep evaluation (default: 20)

### Randomizable Parameters
Defined in `parameter_test_vals` inside `2_rapp.py`:
```python
parameter_test_vals = {
    "joint_damping_scale":   [0.5, 0.75, 1.0, 1.25, 1.5],
    "joint_friction_scale":  [0.5, 0.75, 1.0, 1.25, 1.5],
    "mass_scale":            [0.7, 0.85, 1.0, 1.15, 1.3],
    "action_delay_steps":    [0, 1, 2, 3],
}
```

### Success Criteria
Defined as `reach_success()` in `2_rapp.py`:
```python
def reach_success(metrics: dict) -> bool:
    # Returns True if policy is performing adequately under this DR value
    return metrics.get("mean_position_error", float("inf")) < 0.05  # 5cm threshold
```

### Output Format (`rapp_bounds.json`)
```json
{
    "joint_damping_scale":  {"min": 0.75, "max": 1.25, "nominal": 1.0},
    "joint_friction_scale": {"min": 0.5,  "max": 1.5,  "nominal": 1.0},
    "mass_scale":           {"min": 0.85, "max": 1.15, "nominal": 1.0},
    "action_delay_steps":   {"min": 0,   "max": 2,    "nominal": 0}
}
```

## Stage 3: DR Generation (`3_dr_eureka.py`)

LLM generates domain randomization config using RAPP bounds:
1. Reads `outputs/rapp_bounds.json` + `outputs/reward_fn.py`
2. Formats RAPP bounds into `prompts/initial_users/reach_rapp.txt`
3. Prompts LLM to generate DR configuration within those bounds
4. Runs training rollout with DR applied, collects metrics
5. Iterates if performance degrades significantly
6. Writes `outputs/dr_config.py`

### Usage
```bash
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/3_dr_eureka.py \
    --task franka-reach \
    --num-envs 16 \
    --rollout-duration 60
```

### Output Format (`dr_config.py`)
```python
# Generated by dr_eureka.py — DO NOT EDIT MANUALLY
domain_randomization = {
    "joint_damping_scale":  {"distribution": "uniform", "range": [0.8, 1.2]},
    "joint_friction_scale": {"distribution": "uniform", "range": [0.6, 1.4]},
    "mass_scale":           {"distribution": "uniform", "range": [0.9, 1.1]},
    "action_delay_steps":   {"distribution": "choice",  "values": [0, 1]},
}
```

## Full Pipeline (`run_pipeline.py`)

Runs all three stages in sequence with consistent config.

### Usage
```bash
# Full pipeline with defaults
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/run_pipeline.py \
    --task franka-reach

# Full pipeline, faster (fewer envs, shorter rollouts, fewer iterations)
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/run_pipeline.py \
    --task franka-reach \
    --fast

# Run only specific stages
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/run_pipeline.py \
    --task franka-reach \
    --stages eureka dr_eureka      # skip RAPP, use existing rapp_bounds.json

# Skip reward generation, use existing reward_fn.py
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/run_pipeline.py \
    --task franka-reach \
    --stages rapp dr_eureka
```

### Arguments
- `--task`: Task name (default: franka-reach)
- `--stages`: Which stages to run — eureka, rapp, dr_eureka (default: all three)
- `--fast`: Reduce iterations/envs/duration for quick testing
- `--host`: Isaac Sim host (default: localhost)
- `--port`: Command server port (default: 8765)

## Integration with manipulation-tasks

After running this skill, `manipulation-tasks` picks up the outputs automatically
if they exist at the expected paths:

```python
# In reach_task.py — injected before ManagerBasedRLEnv creation:
reward_fn_path = "~/fluxa-agent-pack/.agent/skills/reward-designer/outputs/reward_fn.py"
dr_config_path = "~/fluxa-agent-pack/.agent/skills/reward-designer/outputs/dr_config.py"
```

The `manipulation-tasks` skill falls back to the default Isaac Lab reward if
`reward_fn.py` does not exist — so this skill is purely additive.

## Standard Execution Order

This skill always runs before `manipulation-tasks`. The full sequence for any
manipulation task is:

```bash
# Step 1: Start Isaac Sim (if not running)
docker exec -d fluxa-isaacsim /isaac-sim/python.sh /isaac-sim/fluxa-ws/start_isaacsim_stream.py
sleep 15

# Step 2: Clear scene
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground --no-keep-physics

# Step 3: Run reward-designer pipeline (always first)
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/run_pipeline.py \
    --task franka-reach

# Step 4: Run manipulation task — picks up generated reward + DR automatically
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py \
    --task franka-reach --num-envs 32 --duration 120
```

When the user asks something like "run the Franka reach task with 32 environments",
the agent should execute all four steps above, not just Step 4.

### Fast mode (for quick demos or testing)
```bash
python3 ~/fluxa-agent-pack/.agent/skills/reward-designer/scripts/run_pipeline.py \
    --task franka-reach --fast
```
Use `--fast` when the user wants results quickly, is testing, or has already
run the pipeline recently and just wants to re-run the task.

## Troubleshooting

### LLM generates syntactically invalid reward code
- Eureka catches `SyntaxError` and `NameError` and retries automatically
- After 3 consecutive failures, script exits with error
- Check `outputs/reward_candidates/` for raw LLM output

### Reward function causes NaN rewards
- Eureka detects NaN in metrics and discards that candidate
- Usually caused by division by zero — prompt includes guard advice
- LLM is shown the NaN result as feedback in next iteration

### RAPP all values fail success criteria
- Success threshold in `reach_success()` may be too tight
- Loosen the threshold or run Stage 1 longer before Stage 2
- Check that `reward_fn.py` exists and is valid before running Stage 2

### DR config causes training instability
- DR ranges from LLM may be too wide even within RAPP bounds
- `3_dr_eureka.py` will detect collapse (reward drops >50%) and re-prompt

## Notes
- Rollout metrics are extracted from Isaac Sim WebSocket stdout responses
- Candidate reward functions are saved to `outputs/reward_candidates/` for inspection
- The skill is task-agnostic — adding `ur10-reach` support requires only a new
  reward signature prompt and yaml config entry