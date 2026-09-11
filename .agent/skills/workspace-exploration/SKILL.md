---
name: workspace-exploration
description: Discover a robot's reachable workspace, collision-free joint configurations, achievable success threshold, and PD controller gains, from a natural-language task description. Use this skill FIRST, before any manipulation or training skills, whenever a user wants to set up a new training scenario. Produces discovered_config.json that downstream skills consume. Triggers on phrases like "train the franka to reach", "set up a reach task", "discover where the robot can reach", "explore the franka's workspace", or any new task setup that requires knowing the robot's reachable region or controller settings.
---

# workspace-exploration

Characterizes a robot for a task described in natural language, and writes
what it measures to `outputs/discovered_config.json`, which downstream skills
(`manipulation-tasks`, `reward-designer`) consume.

## Scope
- Task types: **reach** (the parser understands others; the probes don't — see
  Capability gate)
- Robots: **franka** (via Isaac Lab's built-in asset config)
- Probes: all four are implemented and orchestrated
  - `workspace_probe` — reachable EE region via batched random-config FK
  - `joint_limits_probe` — self-collision-free configs (Sobol + PhysX contacts)
  - `success_threshold_probe` — how close the EE actually settles to a target
    under gravity, i.e. what "success" can mean for this robot
  - `controller_gains_probe` — PD gains (Kp/Kd) that settle a step command
    smoothly

## How to invoke

Everything runs **inside the container** — the host has no Isaac Sim (and no
`pydantic` / `google-genai`). From the repo root on the host:

```bash
docker compose exec fluxa ./python.sh \
  fluxa/.agent/skills/workspace-exploration/scripts/run_skill.py \
  "train the franka to reach random targets on a table"
```

Or, already inside the container:

```bash
./python.sh fluxa/.agent/skills/workspace-exploration/scripts/run_skill.py \
  "train the franka to reach random targets on a table"
```

`run_skill.py` is the orchestrator and runs **all four probes**. Useful flags:
`--no-llm-parser` (force keyword parsing), `--skip-success-threshold`,
`--skip-controller-gains` (fast iteration), `--num_envs`, `--n_targets`.

Single probe, for debugging:
```bash
./python.sh .../scripts/run_probe.py --success-threshold --n_targets 1000
./python.sh .../scripts/run_probe.py --controller-gains
```

Convergence analysis (one-off experiment to pick N):
```bash
./python.sh .../scripts/convergence_analysis.py
```

## Why the probes run as separate processes

Gravity is fixed when the `SimulationContext` is constructed, and the probes
disagree about it:

| probe | gravity | why |
|---|---|---|
| `workspace_probe` | either | pure FK, gravity-independent |
| `joint_limits_probe` | **off** | labels self-collisions from PhysX contact forces; with no ground and no gravity the only contacts are link-link |
| `success_threshold_probe` | **on** | measures settled error *under gravity* |
| `controller_gains_probe` | **on** | measures step response under load |

One process gets one gravity setting, so `run_skill.py` sequences them as
subprocesses and each merges only its own section into
`discovered_config.json` (read-modify-write), letting them compose in any
order. **Do not try to collapse these into one process.** `run_skill.py`
itself runs no sim, so it never holds the GPU while spawning them.

Pass order matters in one place: **controller-gains runs before
success-threshold**, because success-threshold applies the discovered gains
before measuring.

## Capability gate

The parser (Gemini structured output, keyword fallback) emits an **open**
`task_type` — it will happily classify `"stack the red block on the blue
block"` as `stack`. Whether that can be *run* is decided separately, in
`run_skill.py`, against `SUPPORTED_TASK_TYPES`.

An unsupported task prints what it parsed and why the probes can't
characterize it, then exits **2** without spawning a sim. Exit codes: `0` ok,
`1` error, `2` understood-but-unsupported.

This split is deliberate: adding a task family needs new probes that measure
the right thing (grasp feasibility for lift/stack), not just parser changes.

## Architecture

```
workspace-exploration/
├── parser/           NL → TaskSpec (Gemini structured output + keyword fallback)
├── probes/           Sim experiments that discover parameters
├── scripts/          run_skill (orchestrator), run_probe (single probe), convergence_analysis
├── helpers/          Shared JSON I/O + matplotlib plotting
├── tests/            Unit tests and CuRobo cross-validation
└── outputs/          Generated artifacts (configs, diagnostics)
```

Flow: `run_skill.py` parses → gates → runs `run_probe.py` three times
(gravity-off pass, then the two gravity-on passes) → reports which sections
landed.

## Outputs
- `outputs/task_spec.json` — parsed description (audit trail)
- `outputs/discovered_config.json` — the handoff artifact
- `outputs/diagnostics/` — workspace scatter, error histogram, gains sweep /
  cost / transient plots, `safe_configs.npy`

### `task_spec.json`
```json
{
  "task_type": "reach",
  "robot_name": "franka",
  "robot_urdf_path": null,
  "ee_body_name": "panda_hand",
  "objects": [],
  "constraints": {"surface": "table"},
  "notes": [],
  "raw_description": "train the franka to reach targets on a table",
  "parsed_by": "gemini"
}
```

### `discovered_config.json`
Validated against `common/schemas.py` (`DiscoveredConfig`). Sections are
`null` until their probe runs.
```json
{
  "robot": {"name": "franka"},
  "probes": {
    "workspace":        {"x": [...], "y": [...], "z": [...]},
    "joint_limits":     {"n_safe": 1772, "collision_rate": 0.114,
                         "joint_lower": [...], "joint_upper": [...],
                         "safe_config_path": "/abs/path/safe_configs.npy"},
    "success_threshold": {"threshold_m": 1.0e-5, "statistic": "p90",
                          "convergence_rate": 0.949,
                          "arm_stiffness": 2459.7, "arm_damping": 104.3},
    "controller_gains":  {"kp": 2459.7, "kd": 104.3,
                          "n_feasible": 316, "settling_steps": 12}
  }
}
```

`success_threshold` records the gains it was measured under: the threshold is
a property of the controller as much as the arm (the same arm at weak gains
droops under gravity and measures far worse), so the two are only meaningful
together.

## Downstream consumers
- `manipulation-tasks` reads `discovered_config.json` via `reach_task.py
  --config`, overriding target sampling ranges, the joint-reset distribution,
  the fine-grained reward `std`, and the actuator gains.
- `reward-designer` is intended to consume the same file (not yet wired).

## Defaults and assumptions
- Robot base at z=0 on the table surface (when the `table` constraint applies).
- Bounds are robot-base-relative, in meters.
- `num_envs=1000` parallel envs; `n_samples=2000`; `n_targets=1000`.
- Controller-gains tolerances (`pos_tol=0.02 rad`, `vel_tol=0.2 rad/s`) are
  empirical for this setup — tighter values yield 0/1000 feasible because
  pure PD can't hold tighter against gravity droop, and the velocity floor
  sits above the sim's residual jitter.

## What this skill does not do
- Non-Franka robots — the probes are robot-agnostic by construction (gain
  sweep ranges derive from the robot's own loaded gains), but
  `ROBOT_REGISTRY` needs an entry and this is untested on anything else.
- Task families beyond reach — see Capability gate.
- Self-collision filtering inside `workspace_probe` — the point cloud may
  include self-colliding configs; `joint_limits_probe` covers that separately.
- Automatic N selection — N is fixed; `convergence_analysis.py` is a one-off
  justification, not runtime calibration.

## Known gaps
- ~5% of success-threshold targets don't converge, and a 6x stiffness
  increase didn't move the worst-case error — so it isn't a controller
  authority problem. Unresolved.
- `controller_gains_probe` has no CuRobo cross-validation. Unlike the other
  probes it's pure joint-space (no FK to compare), and CuRobo's franka config
  exposes only a scalar `max_acceleration`, so there's little to validate
  against.
- `SUPPORTED_TASK_TYPES` gates on the probes, but `reach_task.py` separately
  hardcodes its own `--task` choices; both need updating for a new family.
