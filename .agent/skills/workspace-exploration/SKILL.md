# workspace-exploration

Discovers the reachable end-effector workspace for a given robot, given a
natural-language task description. Output is a structured JSON file that
downstream skills (manipulation-tasks, reward-designer) consume.

## v1 scope
- Task types: reach
- Robots: franka (via Isaac Lab built-in asset config)
- Probes: workspace_probe (kinematic reachability via random-config FK)

## Inputs
- Natural-language task description, e.g. "train the franka to reach
  random targets on a table"

## Outputs
- `outputs/task_spec.json` — parsed task description (audit trail)
- `outputs/discovered_config.json` — discovered parameters (handoff)
- `outputs/diagnostics/workspace_scatter.png` — 3D point cloud + bbox
- `outputs/diagnostics/convergence_curve.png` — N-sweep (one-off)

## How to invoke

Full pipeline (parser + probe + outputs):
```bash
python scripts/run_skill.py "train the franka to reach random targets on a table"
```

Probe only (skips parser, hardcoded args, for debugging):
```bash
python scripts/run_probe.py --num_envs 1000 --n_samples 10000
```

Convergence analysis (one-off experiment to pick N):
```bash
python scripts/convergence_analysis.py
```

## Architecture

```
workspace-exploration/
├── parser/           Deterministic NL → TaskSpec
├── probes/           Sim experiments that discover parameters
├── scripts/          Entry points (run_skill, run_probe, convergence_analysis)
├── utils/            Shared helpers (JSON I/O, matplotlib plotting)
└── outputs/          Generated artifacts (configs, diagnostics)
```

Pipeline flow:
1. `parser/task_parser.py` — keyword-based parser produces a `TaskSpec`
   (task type, robot name, EE body name, constraints).
2. `probes/workspace_probe.py` — samples random joint configurations,
   batched FK on parallel envs, fits axis-aligned bounding box.
3. `scripts/run_skill.py` — orchestrates the above and writes outputs.

## Output schemas

### `task_spec.json`
```json
{
  "task_type": "reach",
  "robot_name": "franka",
  "robot_urdf_path": null,
  "ee_body_name": "panda_hand",
  "objects": [],
  "constraints": {"surface": "table"},
  "raw_description": "train the franka to reach on a table"
}
```

### `discovered_config.json`
```json
{
  "robot": {
    "name": "franka",
    "workspace_bounds": {
      "x": [0.18, 0.82],
      "y": [-0.55, 0.55],
      "z": [0.00, 1.18]
    }
  }
}
```

The schema is intentionally minimal in v1. As more probes are added
(joint_limits_probe, success_threshold_probe, etc.), they extend this same
file rather than producing separate JSONs.

## Downstream consumers
- `manipulation-tasks` reads `discovered_config.json` to instantiate the
  Isaac Lab task config (target pose distribution, episode length, etc.).
- `reward-designer` indirectly consumes the same file via manipulation-tasks.

## Defaults and assumptions
- Robot base is at z=0 on the table surface (when "table" constraint applies).
- All bounds are reported in robot-base-relative coordinates, in meters.
- N=10000 is the default sample count for the workspace probe; see
  `convergence_curve.png` for the empirical justification.
- num_envs=1000 parallel envs is the default for batched FK.

## What this skill does not do (v1)
- LLM-based parsing — v1 uses deterministic keyword matching.
- Self-collision filtering — point cloud may include self-colliding configs.
- Non-Franka robots — SO-100 and others require URDF integration in v2.
- Other task types (open drawer, lift cube) — handled by future probes.
- Automatic N selection per robot — N is hardcoded; convergence analysis
  is a one-off justification, not a runtime calibration.

## Roadmap (v2+)
- Replace deterministic parser with LLM-based parser
- Add self-collision filtering to workspace_probe
- Add additional probes: effective_joint_limits, success_threshold,
  controller_gain, articulation, handle_identification
- Support non-Franka robots via URDF loading
- Auto-select N via runtime convergence check (instead of hardcoded value)