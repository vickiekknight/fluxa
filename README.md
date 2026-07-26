# Fluxa: Natural Language-Driven Automation of Robot Simulation Pipelines
Despite their promise for scalable robot skill acquisition, sim-to-real approaches remain slow and expert-dependent, requiring manual workspace boundary specification, manipulation task set up, reward function design, and deep familiarity with complex simulation platforms. In this paper, we present FLUXA, an agentic system that takes a natural language prompt and independently orchestrates the full sim-to-real pipeline in Isaac Lab. FLUXA uses Agent Skills that encapsulate Isaac Lab API knowledge, task-specific examples, and each pipeline component to dynamically generate reward functions and simulation configurations. We first demonstrate that FLUXA produces a policy that outperforms standard simulation-based training approaches. Then, we showcase that FLUXA is capable of solving novel robot tasks and handling new embodiments not included in the agent skill examples.

# Installation
This repository contains the FLUXA agent pipeline for automated sim-to-real robotics workflows, built on NVIDIA Isaac Lab / Isaac Sim. FLUXA runs inside a Docker container with Isaac Lab, cuRobo, and FLUXA's own dependencies layered on top of the Isaac Sim base image.

The following instructions will set up everything inside one Docker container. We have tested with NVIDIA driver 580 and CUDA 12.8.

1. Make sure you have an NVIDIA GPU with driver >= 535.129.03, [Docker](https://docs.docker.com/get-docker/), and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed.

2. Get an NGC API key from NVIDIA, used to pull the Isaac Sim base image: https://ngc.nvidia.com/setup/api-key


3. Log in to the NGC registry with your API key:

```
docker login nvcr.io -u '$oauthtoken' -p <YOUR_NGC_API_KEY>

```

4. Get a Gemini API key, used for reward function generation: https://ai.google.dev/gemini-api/docs/get-started


5. Create a `.env` file in the project root with your Gemini API key:

```
echo "GEMINI_API_KEY=<your-key-here>" > .env

```

6. Build the Docker image:

```
docker compose build

```

This installs Isaac Lab v2.3.2, compiles cuRobo v0.7.7 against CUDA 12.8, and installs FLUXA's Python dependencies on top of the Isaac Sim 5.0.0 base image.

7. Start the container:

```
docker compose up -d

```

8. Shell into the container:

```
docker compose exec fluxa bash

```

From here, use `python` (aliased to `/isaac-sim/python.sh`) to run FLUXA scripts with Isaac Sim's bundled Python environment.

To stop the container:

```
docker compose down

```

# Usage
### workspace-exploration
This skill explores the reachable workspace of the robot to identify safe and valid regions for manipulation. It uses a series of probes to validate different aspects of the workspace, such as:
1. `workspace_probe.py`: where can the arm actually reach
2. `joint_limits_probe.py`: what are the valid joint configs
3. `success_threshold_probe.py`: how close does the gripper need to be to count as success
4. `controller_gains_probe.py`: find the kp and kd of what would make the arm move smoothly

#### Running the full skill
`run_skill.py` is the main entry point. It takes a natural-language task description and runs the workspace-exploration pipeline end to end: parsing the prompt into a task spec, spawning the sim, running the probes, applying task constraints, and writing a `discovered_config.json` that downstream stages consume.

```
./python.sh /isaac-sim/fluxa/.agent/skills/workspace-exploration/scripts/run_skill.py "train the franka to reach random targets on a table"
```

Optional flags:
- `--num_envs` — parallel envs (default `1000`)
- `--n_samples` — configs to sample per probe (default `2000`)
- `--seed` — RNG seed (default `42`)
- `--skip-validation` — skip the cuRobo FK check that runs before probing (default: validation runs)
- `--n_validate` — configs used for FK validation (default `50`)

Constraints are read from the prompt: for example, "on a table" clamps the workspace's lower z bound to the table height. This runs the workspace and joint-limits probes (gravity off) and writes:
- `outputs/task_spec.json` — the parsed task specification
- `outputs/discovered_config.json` — workspace bounds + joint-limits summary (the main artifact downstream stages consume)
- `outputs/diagnostics/safe_configs.npy` — collision-free joint configs, referenced by `discovered_config.json` via `safe_config_path`
- `outputs/diagnostics/workspace_scatter.png` — scatter plot of the reachable workspace

> Note: v1 supports `franka` + `reach` only. `run_skill.py` currently wires in the workspace and joint-limits probes; the success-threshold and controller-gains probes aren't part of the orchestrated run yet, so use `run_probe.py` below to exercise those on their own.

#### Debugging individual probes
Each probe can also be tested independently with `run_probe.py` using hardcoded args to debug. Note that the workspace probe always runs (the success-threshold probe reuses its point cloud), and the gravity state determines which of the remaining probes runs with it: gravity-off (the default) runs joint-limits, while `--success-threshold` flips gravity on and runs the success-threshold probe instead.

#### 1. Workspace probe
Always runs. The plain invocation runs the workspace probe (gravity off):

```
./python.sh /isaac-sim/fluxa/.agent/skills/workspace-exploration/scripts/run_probe.py
```

Optional flags:
- `--num_envs` — parallel envs (default `1000`)
- `--n_samples` — configs to sample (default `2000`)
- `--seed` — RNG seed (default `42`)
- `--validate-fk` — validate FK against cuRobo before probing
- `--n_validate` — configs used for validation (default `50`)

```
./python.sh /isaac-sim/fluxa/.agent/skills/workspace-exploration/scripts/run_probe.py --n_samples 10000 --validate-fk
```

#### 2. Joint-limits probe
Runs automatically alongside the workspace probe whenever gravity is off (the default), so the plain command above already runs it. To also validate against cuRobo's self-collision API:

```
./python.sh /isaac-sim/fluxa/.agent/skills/workspace-exploration/scripts/run_probe.py --validate-collision
```

Uses the shared `--n_samples`, `--seed`, and `--n_validate` flags.

#### 3. Success-threshold probe
Requires gravity ON, so `--success-threshold` flips gravity on and skips the joint-limits probe:

```
./python.sh /isaac-sim/fluxa/.agent/skills/workspace-exploration/scripts/run_probe.py --success-threshold
```

Optional flags:
- `--n_targets` — targets to measure, padded to a multiple of `num_envs` (default `500`)
- `--st_n_steps` — oracle rollout horizon (default `200`)
- `--st_statistic` — error percentile used as the threshold (default `p90`)
- `--gravity_z` — override gravity z (default `-9.81` under `--success-threshold`)

```
./python.sh /isaac-sim/fluxa/.agent/skills/workspace-exploration/scripts/run_probe.py --success-threshold --n_targets 1000 --st_n_steps 250
```

#### 4. Controller-gains probe
Not yet implemented.


### manipulation-tasks
This skill runs a manipulation task in Isaac Lab, optionally consuming the `discovered_config.json` produced by `workspace-exploration`. When a config is passed, its workspace bounds override Isaac Lab's default target-sampling ranges and its safe joint configs override the default reset behavior; without one, Isaac Lab defaults are used.

Unlike the workspace-exploration scripts, `reach_task.py` doesn't launch its own Isaac Sim instance — it connects to a running Isaac Sim streaming server over WebSocket (default port `8765`) and sends it the task to execute. So you need **two terminals**, both inside the container: one running the stream server, one running the task.

#### 1. Start the Isaac Sim streaming server
In the first terminal, start the stream server and leave it running:

```
./python.sh /isaac-sim/fluxa/start_isaacsim_stream.py
```

Wait until it reports the command server is listening before moving on — `reach_task.py` fails with a connection-refused error if this isn't up.

#### 2. Run the reach task
In a second terminal, run the task. To use the workspace bounds and safe configs discovered by workspace-exploration, pass its `discovered_config.json` (written to `outputs/discovered_config.json` by `run_skill.py`) via `--config`:

```
./python.sh /isaac-sim/fluxa/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --config outputs/discovered_config.json
```

Without `--config`, the task runs with Isaac Lab's built-in target ranges and joint reset:

```
./python.sh /isaac-sim/fluxa/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach
```

Available tasks:

| Task | Description | Robot |
|------|-------------|-------|
| `franka-reach` | Franka reaching random target poses | Franka Panda |
| `franka-reach-play` | Demo/play mode for the Franka reach policy | Franka Panda |
| `ur10-reach` | UR10 reaching random target poses | Universal Robots UR10 |
| `ur10-reach-play` | Demo/play mode for the UR10 reach policy | Universal Robots UR10 |

Optional flags:
- `--task` — which task to run (required; one of the four above)
- `--num-envs` — parallel envs (default `16`)
- `--env-spacing` — spacing between envs in meters (default `2.0`)
- `--duration` — how long to run, in seconds (default `60`)
- `--config` — path to `discovered_config.json` from workspace-exploration; overrides target ranges + joint reset
- `--reward-file` — path to a Python file with reward modifications (future: produced by reward-designer)
- `--dr-config-file` — path to a Python file with DR config modifications (future: produced by reward-designer)
- `--host` — Isaac Sim host (default `localhost`)
- `--port` — command server port (default `8765`)

```
./python.sh /isaac-sim/fluxa/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --num-envs 64 --duration 300 --config outputs/discovered_config.json
```

### reward-designer
> **Status:** This skill works end to end in its own right, but it predates the workspace-exploration skill and has not yet been reconciled with it. It currently uses hardcoded values (Isaac Lab's built-in target ranges and a hardcoded RAPP success threshold) instead of the `discovered_config.json` produced by workspace-exploration. See [Known issues / what needs work](#known-issues--what-needs-work) below before relying on this section.
 
This skill generates an optimized reward function and domain randomization (DR) config for a task using a DrEureka-style pipeline: an LLM (Gemini) proposes reward functions, each is trained with PPO and scored, the best policy is swept across physics parameters to find safe DR bounds, and the LLM then proposes DR configs within those bounds. The outputs (`reward_fn.py`, a `dr_config_*.py`) are consumed by `manipulation-tasks` via `reach_task.py`'s `--reward-file` / `--dr-config-file` flags.
 
Two things make this skill different from the others:
 
1. **It runs from the host, not inside the container.** The pipeline manages a single-GPU constraint — it stops the streaming Isaac Sim instance to free the GPU, runs optimization headless (each evaluation is a fresh `docker exec` subprocess), then restarts streaming. So these commands run on the host from the repo, and reach into the container themselves.
2. **It does not yet consume workspace-exploration's output.** `reward-designer` currently uses Isaac Lab's built-in target ranges and a hardcoded RAPP success threshold rather than the `discovered_config.json` produced by `workspace-exploration`. Wiring the discovered workspace bounds and the success-threshold probe's output into this skill is future work.
#### Prerequisites
- **RSL-RL installed in the container** (used for PPO training/inference):
```
docker exec fluxa-isaacsim bash -c "cd /isaac-sim/IsaacLab && ./isaaclab.sh -i rsl_rl"
```
- **Host-side Python 3.10+** with the orchestrator's dependencies:
```
pip install google-genai pyyaml wandb
```
- `GEMINI_API_KEY` set in the environment (same key from Installation).
- Optional: `wandb login`, or set `WANDB_MODE=offline` to skip W&B logging prompts.
#### Running the full pipeline
`run_pipeline.py` is the main entry point. Run it from the host, from the reward-designer directory:
 
```
cd .agent/skills/reward-designer
python3 scripts/run_pipeline.py
```
 
This runs all four stages, handling the GPU lifecycle around them (stop stream → optimize → restart stream).
 
Optional flags:
- `--stages` — which stages to run, any subset of `eureka rapp dr_eureka train_dr` (default: all four)
- `--use-placeholders` — use placeholder DR bounds in Stage 3, skipping real RAPP output
- `--no-restart-stream` — don't restart streaming afterward (just produce the artifacts)
- `--stage4-num-envs` — parallel envs for Stage 4 training (default `16`)
- `--stage4-train-iterations` — PPO iterations per DR config in Stage 4 (default `500`)
```
python3 scripts/run_pipeline.py --stages dr_eureka train_dr --use-placeholders --stage4-train-iterations 1000
```
 
#### Pipeline stages
| Stage | Script | Produces | Runs on |
|-------|--------|----------|---------|
| 1. Eureka | `1_eureka.py` | `outputs/reward_fn.py`, `outputs/eureka_policy.pt` | host (evals via `docker exec`) |
| 2. RAPP | `2_rapp.py` | `outputs/rapp_bounds.json` | container |
| 3. DR Eureka | `3_dr_eureka.py` | `outputs/dr_configs.json`, `outputs/dr_candidates/dr_config_*.py` | host |
| 4. Train w/ DR | `4_train_with_dr.py` | `outputs/dr_training_results.json` (ranked) | host (training via `docker exec`) |
 
`run_pipeline.py` is the recommended way to run these, since it sequences them and manages the stream lifecycle and container paths. If you run a stage on its own for debugging: Stages 1, 3, and 4 are host scripts (`python3 scripts/1_eureka.py --task franka-reach`, `python3 scripts/3_dr_eureka.py`, `python3 scripts/4_train_with_dr.py`), while Stage 2 runs inside the container and needs the Stage 1 checkpoint:
 
```
docker exec fluxa-isaacsim /isaac-sim/python.sh \
  /isaac-sim/fluxa/.agent/skills/reward-designer/scripts/2_rapp.py \
  --checkpoint /isaac-sim/fluxa/.agent/skills/reward-designer/outputs/eureka_policy.pt \
  --output /isaac-sim/fluxa/.agent/skills/reward-designer/outputs/rapp_bounds.json
```
 
(Note `3_dr_eureka.py` and `4_train_with_dr.py` also take a `--config` flag, but it points to `cfg/reach.yaml`, not `discovered_config.json`.)
 
#### Feeding results into manipulation-tasks
Once the pipeline has produced a reward function and DR config, pass them to a manipulation task:
 
```
./python.sh /isaac-sim/fluxa/.agent/skills/manipulation-tasks/scripts/reach_task.py \
  --task franka-reach \
  --reward-file /isaac-sim/fluxa/.agent/skills/reward-designer/outputs/reward_fn.py \
  --dr-config-file /isaac-sim/fluxa/.agent/skills/reward-designer/outputs/dr_candidates/dr_config_0.py
```
 
If neither flag is given, `reach_task.py` falls back to Isaac Lab's default reward — so this skill is purely additive.
 
#### Known issues / what needs work
This skill is not yet reconciled with the rest of the pipeline. The following are known inconsistencies to fix before it can be relied on end to end:
 
1. **SKILL.md is stale — three stages vs. four.** The current `SKILL.md` describes a three-stage pipeline and documents `run_pipeline.py` flags (`--fast`, `--run-task-after`, `--task-duration`) that no longer exist. The actual `run_pipeline.py` has **four** stages (adds Stage 4, `train_dr`) and a different flag set (`--stages`, `--use-placeholders`, `--no-restart-stream`, `--stage4-num-envs`, `--stage4-train-iterations`). This README documents the code; the SKILL.md needs to be brought in line.
2. **Hardcoded container paths don't match the mount.** The compose file mounts the repo to `/isaac-sim/fluxa`, and this README uses `/isaac-sim/fluxa/.agent/skills/...`. But `run_pipeline.py` hardcodes `/isaac-sim/fluxa-agent-pack/...` (in `run_rapp`) and `/isaac-sim/fluxa-ws/start_isaacsim_stream.py` (as `STREAM_SCRIPT`), and `reach.yaml`'s `eval_script` / `shared_dir` also point at `fluxa-agent-pack`. As written these paths won't resolve under the current mount — the scripts and config need updating to `/isaac-sim/fluxa/...` (or the mount needs to change) before the documented commands will run.
3. **Stage 1 (`1_eureka.py`) is only half-wired for K>1 sampling.** The candidate loop references a `K` variable (and `best_metrics_overall` / `best_reward_path_overall`) that are never defined in the script — `K` is presumably meant to be `cfg['eureka']['candidates']`. As-is, a run hits a `NameError` before producing anything, which also breaks a full `run_pipeline.py` run. This is the same "missing K>1 parallel sampling" gap noted below; those names need binding before Stage 1 runs.
4. **The reward/DR injection contract may not line up between scripts.** The pipeline emits `reward_fn.py` (from `reward_template.py`, defining a `reward_dict`) and `dr_config_*.py` files, and `reach_task.py` exposes `--reward-file` / `--dr-config-file` to receive them. But `reach_task.py` `exec`s the injected code expecting it to operate on `env_cfg`, whereas `eval_headless.py` does `env_cfg.rewards = reward_dict` after exec. Because reward-designer predates the current `reach_task.py` injection design, this handoff needs to be verified end to end before it can be advertised as automatic.
5. **Does not consume `discovered_config.json`.** As noted above, the intended integration is for reward-designer to pick up workspace bounds and the success-threshold from workspace-exploration's `discovered_config.json` (replacing the hardcoded RAPP `success_threshold: 0.05`). That wiring does not exist yet.

# Code Structure
FLUXA is organized as three Agent Skills under `.agent/skills/`, plus a small shared package and the Docker/stream entry points at the repo root. Each skill produces artifacts that the next one consumes. Below is explained how the components interact, using `franka-reach` as the running example. Note that the pipeline is mid-integration: workspace-exploration and reward-designer are currently separate islands that both feed `manipulation-tasks`, and the intended `discovered_config.json` → reward-designer link does not exist yet (see [reward-designer Known issues](#known-issues--what-needs-work)).
 
```
fluxa/
├── Dockerfile                    # Isaac Sim 5.0 + Isaac Lab + cuRobo + Fluxa deps
├── docker-compose.yml            # build/run, GPU reservations, repo mount, GEMINI_API_KEY
├── requirements.txt              # Fluxa Python deps (installed into Isaac Sim's python)
├── .env.example                  # template for the GEMINI_API_KEY .env file
├── start_isaacsim_stream.py      # launches the streaming Isaac Sim + command server (port 8765)
└── .agent/skills/
    ├── common/
    │   └── schemas.py            # Pydantic schemas shared across skills (DiscoveredConfig, ...)
    ├── workspace-exploration/
    ├── manipulation-tasks/
    └── reward-designer/
```
 
## Shared: `.agent/skills/common/`
`common/schemas.py` defines the Pydantic models that form the contract between skills — most importantly `DiscoveredConfig` (with nested `RobotConfig`, `ProbeResults`, `WorkspaceProbeResult`, `JointLimitsProbeResult`). workspace-exploration writes these; manipulation-tasks reads them back. Keeping the schema in one place is what lets the two skills stay decoupled but interoperable.
 
## workspace-exploration
This skill discovers the reachable workspace and safe joint configs for a robot and writes them to `discovered_config.json`.
 
`scripts/run_skill.py` is the top-level entry point (NL description → `discovered_config.json`). It uses:
- **A task parser**, `parser/task_parser.py` (`parse_task_description`), which turns the natural-language prompt into a `task_spec` (task type, robot name, EE body name, constraints). This is the current hardcoded/keyword parser slated to be replaced with an LLM parser.
- **The probes**, in `probes/`: `workspace_probe.py` (Monte-Carlo FK sampling of the reachable envelope) and `joint_limits_probe.py` (rejection sampling of collision-free configs via PhysX contact labels). Both are validated against cuRobo before use.
- **FK validation**, `tests/test_workspace_integration.py` (`run_integration_test`), run before probing unless `--skip-validation` is passed, to catch kinematics regressions from URDF/Isaac Lab drift.
- **I/O helpers**, `helpers/io.py` (`save_json`, `save_scatter_plot`), which serialize the results and diagnostics.
- **The shared schema**, `common/schemas.py`, to assemble and validate the final `DiscoveredConfig` before writing `outputs/discovered_config.json` (plus `outputs/task_spec.json`, `outputs/diagnostics/safe_configs.npy`, and a workspace scatter plot).
`scripts/run_probe.py` is a debug entry point that runs probes individually with hardcoded args. It also drives `probes/success_threshold_probe.py` (the third probe, gravity-on), which is still in progress and not yet wired into `run_skill.py` or the schema. `scripts/convergence_analysis.py` and the `tests/` unit/validation scripts support probe development.
 
## manipulation-tasks
This skill runs an Isaac Lab manipulation task, consuming the artifacts from the other two skills. It is the point where the pipeline currently converges.
 
`scripts/reach_task.py` is the entry point. It does not launch its own sim — it connects over WebSocket (port 8765) to the streaming Isaac Sim started by the repo-root `start_isaacsim_stream.py`, and sends it a task to execute. It uses three optional injection points, all no-ops if their path is absent:
- **`--config`** → a `discovered_config.json` from workspace-exploration. `load_discovered_config` validates it against `common/schemas.py`, then `extract_template_overrides` pulls out the workspace bounds and `safe_config_path`. These override Isaac Lab's built-in target-sampling ranges (`commands.ee_pose.ranges`) and the default joint reset (via a safe-set reset event). **This link is wired.**
- **`--reward-file`** → a Python file operating on `env_cfg`, intended to be the `reward_fn.py` produced by reward-designer Stage 1. **This injection point exists but the handoff contract is unverified** (see reward-designer Known issues #4).
- **`--dr-config-file`** → a Python file with DR modifications, intended to be a `dr_config_*.py` from reward-designer Stage 3. Same status as `--reward-file`.
The task config itself is built from `RUN_TASK_CODE_TEMPLATE` — the injected values are formatted into this template string, which is `exec`'d inside the Isaac Sim process. `FrankaReachEnvCfg` / `UR10ReachEnvCfg` subclass Isaac Lab's `ReachEnvCfg`. A `reach_task_status.json` marker is written recording what actually applied, so you can confirm whether the config/reward/DR overrides took effect.
 
## reward-designer
This skill generates an optimized reward function and DR config via a four-stage DrEureka-style pipeline. Unlike the other skills it runs from the **host** and manages the single-GPU lifecycle (stop stream → optimize headless → restart stream). It currently uses hardcoded Isaac Lab ranges and a hardcoded RAPP success threshold rather than `discovered_config.json` — wiring that in is the main intended integration still outstanding.
 
`scripts/run_pipeline.py` orchestrates everything and manages the streaming instance. It sequences the four stage scripts:
 
**Stage 1 — `scripts/1_eureka.py`** runs iterative reward generation. It uses:
- **The config**, `cfg/reach.yaml`, which defines the task, the Gemini model, Eureka iteration/candidate counts, PPO settings, and the container paths (`eval_script`, `shared_dir`) and output file names (`reward_output_file`).
- **A reward signature**, `prompts/reward_signature_reach.txt`, given to the LLM as the required output format. Other prompt fragments live alongside it in `prompts/` (`policy_feedback.txt`, `code_feedback.txt`, `execution_error_feedback.txt`, etc.).
- **A reward template**, `templates/reward_template.py`, whose boilerplate is filled with the LLM's generated functions and `reward_dict` and written to `outputs/reward_fn.py` (plus per-candidate copies under `outputs/candidates/`).
- **A headless evaluator**, `scripts/eval_headless.py`, launched as a fresh `docker exec` subprocess per candidate. It trains a PPO policy (RSL-RL) for N iterations, evaluates it, writes metrics to a JSON file on the shared filesystem, and saves a checkpoint (`outputs/eureka_policy.pt`). Metrics are read back on the host and fed to the LLM as reward-reflection feedback.

**Stage 2 — `scripts/2_rapp.py`** computes Reward-Aware Physics Prior bounds. It uses:
- **The Stage 1 checkpoint** (`outputs/eureka_policy.pt`) as the fixed policy to stress-test.
- **The randomizable parameters and their test values**, defined in the `PARAMETERS` dict in `2_rapp.py` itself.
- **A per-value evaluator**, `scripts/eval_rapp.py`, which loads the policy, modifies one physics parameter, runs inference (no training), and reports whether the EE stays within the (hardcoded) success threshold. The min/max feasible values per parameter are written to `outputs/rapp_bounds.json`.

**Stage 3 — `scripts/3_dr_eureka.py`** generates DR configs. It uses:
- **The RAPP bounds** (`outputs/rapp_bounds.json`) as guardrails in the LLM prompt (falling back to `PLACEHOLDER_BOUNDS` if absent or `--use-placeholders` is set).
- **The best reward function** from Stage 1 (`outputs/reward_fn.py`) for context.
- **A DR template**, `templates/dr_template.py`, filled with the LLM's chosen ranges and written to `outputs/dr_candidates/dr_config_*.py` (plus `outputs/dr_configs.json` summarizing all samples).

**Stage 4 — `scripts/4_train_with_dr.py`** trains a policy per DR config (3 seeds each) via `eval_headless.py`, ranks them by mean reward, and writes `outputs/dr_training_results.json`.
 
The final `outputs/reward_fn.py` and `outputs/dr_candidates/dr_config_*.py` are the artifacts intended for `manipulation-tasks` (via `reach_task.py`'s `--reward-file` / `--dr-config-file`).

# Notes
This project is currently ongoing. 

What's left for the initial basic pipeline:

- The reward-designer skill is still in progress, so the results are not where we would like yet. The Erueka and DrEureka algorithms need to be updated to match their paper implementations in Isaac Lab. The scripts also need to take in the discovered outputs from workspace exploration.
- The SKILL.md files need to be updated to include the latest changes in the skills. 
- In the workspace-exploration skill, Joint Limit Probe validation test needs to be updated to pass a p70 threshold instead of p95.
- In the workspace-exploration skill, the success-threshold probe is still in progress.
- The workspace-exploration skill still needs the remaining controller gains probe and replace current parser with LLM parser.

Next steps:
- Add more complex manipulation tasks in manipulation-tasks skill.