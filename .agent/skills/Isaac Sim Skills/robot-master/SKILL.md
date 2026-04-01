---
name: robot-master
description: Spawns and manages robots in NVIDIA Isaac Sim. Use when adding robots (humanoids, quadrupeds, manipulators, mobile bases) to a simulation scene or when users ask to spawn/add/place robots.
---

# Robot Master Skill

This skill spawns robots in a persistent Isaac Sim instance by sending commands via WebSocket.

## When to use this skill

- User asks to add a robot to the scene
- User mentions spawning humanoids (H1, G1), quadrupeds (Go1, Go2), manipulators (Franka), or mobile bases (Carter, Jetbot)
- User needs to place multiple robots in a scene
- User wants to build a complex simulation with multiple robots

## Architecture

This skill connects to a **running Isaac Sim streaming server** via WebSocket (port 8765). Commands are sent as Python code and executed in the Isaac Sim context.

## Available Robots

| Robot Type | Use Case |
|------------|----------|
| `franka` | Industrial manipulation, pick-and-place |
| `h1` | Humanoid research, bipedal locomotion |
| `g1` | Humanoid manipulation tasks |
| `go2` | Quadruped navigation, inspection (preferred) |
| `go1` | Quadruped tasks, legacy support |
| `carter` | Wheeled navigation, warehouse robots |
| `jetbot` | Small-scale navigation, education |

## How to use it

### Step 1: Ensure Isaac Sim streaming server is running

**CRITICAL**: The Isaac Sim streaming server with command server must be running!

Check if running:
```bash
docker exec fluxa-isaacsim pgrep -f "start_isaacsim_stream.py"
```

If not running, start it:
```bash
docker exec -d fluxa-isaacsim /isaac-sim/python.sh /isaac-sim/fluxa-ws/start_isaacsim_stream.py
sleep 15  # Wait for initialization
```

**Verify services are ready:**
- 🌐 WebRTC Stream: User can view at WebRTC client app
- 🔌 Command Server: ws://localhost:8765 (internal)

### Step 2: Select appropriate robot

Choose based on task:
- **Manipulation**: `franka`
- **Humanoid**: `h1` or `g1`
- **Quadruped**: `go2` (preferred) or `go1`
- **Wheeled**: `carter` or `jetbot`

### Step 3: Validate placement

Before spawning:
- ✅ Ground plane exists (created automatically by streaming server)
- ✅ Z position > 0 (avoid spawning inside floor)
- ✅ Use unique `--path` for multiple robots of same type

### Step 4: Spawn the robot

Execute the spawn script **from the host machine**:
```bash
# Basic spawn at origin
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka

# Custom position (X, Y, Z in meters)
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type go2 --pos 1.5 0.0 0.3

# Multiple robots with unique paths
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --path /World/Franka_Left --pos -1.0 0.0 0.0
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --path /World/Franka_Right --pos 1.0 0.0 0.0
```

**Prerequisites:**
- Python 3 with `websockets` library installed
- Isaac Sim running with command server on port 8765

### Step 5: Verify in stream

The robot appears immediately in the WebRTC stream. Tell user to check their WebRTC viewer.

## Script Arguments

- `--type`: Robot type (required, choices: franka, h1, g1, go2, go1, carter, jetbot)
- `--pos X Y Z`: Position in meters (default: 0 0 0)
- `--path`: Custom USD prim path (required for duplicate robots)
- `--host`: Isaac Sim host (default: localhost)
- `--port`: Command server port (default: 8765)

## Common Patterns

### Single robot
```bash
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type h1 --pos 0.0 0.0 1.0
```

### Multi-robot assembly line
```bash
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --path /World/Station_1 --pos -2.0 0.0 0.0
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --path /World/Station_2 --pos 0.0 0.0 0.0
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --path /World/Station_3 --pos 2.0 0.0 0.0
```

### Mixed robot fleet
```bash
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type go2 --path /World/Scout --pos 3.0 0.0 0.3
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type carter --path /World/Transporter --pos -3.0 0.0 0.0
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type h1 --path /World/Operator --pos 0.0 2.0 1.0
```

## Troubleshooting

### "Cannot connect to Isaac Sim command server"
- **Cause**: Streaming server not running or port 8765 not accessible
- **Fix**: Start Isaac Sim with the command in Step 1, ensure Docker port 8765 is mapped

### "Prim already exists"
- **Cause**: Robot already spawned at that path
- **Fix**: Use `--path` with unique name (e.g., `/World/Franka_02`)

### Robot falls through floor
- **Cause**: Z coordinate too low
- **Fix**: Use appropriate heights:
  - Humanoids (H1, G1): Z = 1.0 - 1.2
  - Quadrupeds (Go2, Go1): Z = 0.3 - 0.5
  - Manipulators (Franka): Z = 0.0

### Command queued but robot doesn't appear
- **Cause**: Isaac Sim may have crashed or frozen
- **Fix**: Check Isaac Sim logs in container, restart if needed

## Technical Notes

**Connection**: Skills connect via WebSocket to Isaac Sim command server (port 8765)

**Execution**: Python code is sent over WebSocket and executed in Isaac Sim's main thread

**Persistence**: Scene persists until you restart Isaac Sim - all robots accumulate

**Thread Safety**: Commands are queued and executed sequentially in Isaac Sim's main loop

**Multi-Skill Support**: Other skills can connect to the same command server to add props, lights, cameras, etc.

**Performance**: Minimal overhead - WebSocket messages are ~1-5ms latency