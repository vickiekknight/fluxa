---
name: manipulation-tasks
description: Execute or evaluate a manipulation task in Isaac Sim using Isaac Lab, GIVEN an existing discovered_config.json produced by workspace-exploration. Use AFTER workspace-exploration has run. Do NOT use this skill to set up new tasks from scratch — its job is to run training or play-mode given an existing config. Triggers on phrases like "run the franka reach task", "execute training with this config", "play back the trained policy", "start the manipulation training run".
---

# Manipulation Tasks Skill

This skill enables manipulation tasks and reinforcement learning scenarios using Isaac Lab in Isaac Sim.

## When to use this skill

Use AFTER workspace-exploration has produced `discovered_config.json`. This skill assumes the workspace bounds and other parameters are already known.

- User asks to "run the franka reach training"
- User says "execute the manipulation task" or "start training"
- User asks to "play back" or "demo" a trained policy
- User mentions running an Isaac Lab task with specific parameters already in hand

Do NOT use this skill if the user is describing a NEW task ("train the franka to do X") — that goes to workspace-exploration first.

## Available Tasks

| Task | Description | Robot |
|------|-------------|-------|
| `franka-reach` | Train Franka to reach random target poses | Franka Panda |
| `franka-reach-play` | Demo trained Franka reach policy (50 envs) | Franka Panda |
| `ur10-reach` | Train UR10 to reach random target poses | Universal Robots UR10 |
| `ur10-reach-play` | Demo trained UR10 reach policy (50 envs) | Universal Robots UR10 |

## How to use it

### Step 1: Ensure Isaac Sim is running

Check if Isaac Sim streaming server is running:
```bash
docker exec fluxa-isaacsim pgrep -f "start_isaacsim_stream.py"
```

If not running, start it:
```bash
docker exec -d fluxa-isaacsim /isaac-sim/python.sh /isaac-sim/fluxa-ws/start_isaacsim_stream.py
sleep 15
```

### Step 2: Clear scene (recommended)
```bash
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground --no-keep-physics
```

### Step 3: Run manipulation task
```bash
# Run Franka reach training task (default: 16 environments)
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach

# Run with more environments
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --num-envs 32

# Run demo/play mode (50 environments, no randomization)
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach-play

# Custom environment spacing
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --env-spacing 3.0
```

## Script Arguments

### reach_task.py
- `--task`: Task name (required: franka-reach, franka-reach-play)
- `--num-envs`: Number of parallel environments (default: 16)
- `--env-spacing`: Spacing between environments in meters (default: 2.0)
- `--duration`: How long to run in seconds (default: 60)
- `--headless`: Run without rendering (default: False, since we're streaming)
- `--host`: Isaac Sim host (default: localhost)
- `--port`: Command server port (default: 8765)

## Task Details

### franka-reach

**Objective:** Train the Franka Panda robot to move its end-effector to random target poses.

**Observations:**
- Joint positions and velocities
- End-effector position and orientation
- Target pose (position + orientation)

**Actions:**
- Joint position control (7-DOF arm)
- Scaled by 0.5 for smooth motion
- Uses default joint offsets

**Rewards:**
- Position tracking: Reward for moving closer to target position
- Fine-grained position tracking: Additional reward when very close
- Orientation tracking: Reward for matching target orientation
- Action smoothness: Penalty for large joint movements

**Reset Conditions:**
- Episode timeout
- Target reached successfully
- Robot collision or singularity

**Environment:**
- Multiple parallel environments for efficient learning
- Random target generation within reachable workspace
- Ground plane and physics enabled

### franka-reach-play

**Objective:** Demo mode for trained reach policy.

**Differences from training:**
- 50 environments (smaller for visualization)
- Larger spacing (2.5m) for better viewing
- No observation corruption/noise
- Pre-trained policy evaluation

### ur10-reach

**Objective:** Train the UR10 robot to move its end-effector to random target poses.

**Key Differences from Franka:**
- **End-effector:** ee_link (instead of panda_hand)
- **Target orientation:** Pitch = π/2 (pointing sideways, not down)
- **Joint configuration:** All joints controlled (6-DOF)
- **Reset range:** Joints randomized between 0.75-1.25x default

**Observations:**
- Joint positions and velocities (6 joints)
- End-effector position and orientation
- Target pose (position + orientation)

**Actions:**
- Joint position control (6-DOF arm)
- Scaled by 0.5 for smooth motion
- Uses default joint offsets

**Rewards:** Same structure as Franka reach

### ur10-reach-play

**Objective:** Demo mode for trained UR10 reach policy.

**Configuration:** Same as franka-reach-play but with UR10 robot.

## Common Patterns

### Quick test with few environments
```bash
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --num-envs 4 --duration 30
```

### Training run with many environments
```bash
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --num-envs 64 --duration 300
```

### Demo/visualization mode
```bash
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach-play --duration 120
```

### Compact environment spacing
```bash
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --num-envs 25 --env-spacing 1.5
```

## Task Configuration

The Franka reach task uses these key parameters:

**Robot Configuration:**
- Asset: Franka Panda (7-DOF manipulator)
- End-effector: panda_hand
- Control: Joint position control

**Target Generation:**
- Position range: Within reachable workspace
- Orientation: Fixed pitch (π radians, pointing down)
- Random sampling each episode

**Action Space:**
- 7 continuous values (one per joint)
- Range: [-1, 1] scaled by 0.5
- Applied as joint position deltas

**Observation Space:**
- Robot state: Joint positions, velocities
- Goal: Target position and orientation
- Proprioceptive information

## Rewards Breakdown

| Reward Component | Weight | Description |
|-----------------|--------|-------------|
| Position tracking | High | Distance to target position |
| Fine-grained tracking | Medium | Bonus when very close (<0.1m) |
| Orientation tracking | Medium | Alignment with target orientation |
| Action rate | Negative | Penalty for jerky movements |

## Viewing the Task

**In WebRTC Stream:**
- Multiple Franka robots in grid layout
- Green spheres indicate target positions
- Robots move to reach targets
- Environments reset independently

**Camera Position:**
- Automatically positioned to view all environments
- Adjusts based on number of environments
- Can manually adjust in WebRTC viewer

## Troubleshooting

### "Task not found"
- **Cause**: Invalid task name
- **Fix**: Use exact task name: `franka-reach` or `franka-reach-play`

### Robots spawn on top of each other
- **Cause**: env-spacing too small for num-envs
- **Fix**: Increase `--env-spacing` or decrease `--num-envs`

### Scene too crowded
- **Cause**: Too many environments
- **Fix**: Reduce `--num-envs` or increase `--env-spacing`

### Robots fall through floor
- **Cause**: Physics scene not properly initialized
- **Fix**: Clear scene completely before running task

### Task runs but nothing happens
- **Cause**: Duration too short
- **Fix**: Increase `--duration` parameter

### Performance issues / lag
- **Cause**: Too many environments for GPU
- **Fix**: Reduce `--num-envs` (try 8, 16, or 32)

## Technical Notes

**Physics Timestep:** 1/120 seconds (120 Hz)

**Control Frequency:** Depends on task configuration

**Parallel Environments:** Uses GPU parallelization for efficient training

**Scene Structure:**
```
/World
├── /envs
│   ├── /env_0
│   │   ├── Robot (Franka)
│   │   └── Target (visual marker)
│   ├── /env_1
│   │   └── ...
│   └── /env_N
├── GroundPlane
└── PhysicsScene
```

**Namespace Pattern:** Each environment uses `{ENV_REGEX_NS}` pattern for unique paths

## Integration with Other Skills

### Before manipulation-tasks
```bash
# 1. Clear scene completely
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground --no-keep-physics

# 2. Run manipulation task
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach
```

### After manipulation-tasks
```bash
# Task will run for specified duration, then stop
# Scene remains with all environments
# Can clear and run different task:
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach-play
```

## Understanding the Reach Task

**What happens:**

1. **Initialization:**
   - Multiple Franka robots spawn in grid layout
   - Each robot gets a random target pose
   - Targets visualized as spheres/markers

2. **Episode Loop:**
   - Robot observes current state and target
   - Policy outputs joint position commands
   - Robot moves toward target
   - Rewards calculated based on progress
   - Episode resets when target reached or timeout

3. **Training:**
   - Multiple environments run in parallel
   - Each environment trains independently
   - Efficient GPU-accelerated simulation
   - Experiences collected for learning

**Success Criteria:**
- End-effector reaches within threshold of target position
- End-effector orientation matches target orientation
- Smooth, efficient motion trajectory

## Performance Guidelines

| Num Envs | GPU Memory | Recommended Hardware |
|----------|------------|---------------------|
| 4-8 | ~2GB | Minimum (testing) |
| 16-32 | ~4GB | Standard (training) |
| 64-128 | ~8GB | High-performance |
| 256+ | ~16GB+ | Research/large-scale |

## Advanced Usage

### Monitor during execution
```bash
# Run task in background
python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --duration 600 &

# Watch WebRTC stream to see progress
# Open: http://localhost:49100/streaming/webrtc-client
```

### Quick iteration testing
```bash
# Fast test cycle
for i in {1..5}; do
  echo "Test run $i"
  python3 ~/fluxa-agent-pack/.agent/skills/manipulation-tasks/scripts/reach_task.py --task franka-reach --num-envs 4 --duration 20
  python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground
done
```

## Future Tasks

This skill is designed to be extended with additional manipulation tasks:

**Planned additions:**
- `franka-pick-place` - Pick and place objects
- `franka-cabinet` - Open/close cabinet doors
- `franka-stack` - Stack blocks
- `franka-push` - Push objects to targets
- Multi-robot tasks
- Bimanual manipulation

## Notes

- Task uses Isaac Lab framework (built on Isaac Sim)
- Requires Isaac Lab assets to be installed
- Compatible with Isaac Sim 5.0+
- Uses GPU for parallel simulation
- Results visible in WebRTC stream