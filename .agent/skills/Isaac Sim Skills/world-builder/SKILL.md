---
name: world-builder
description: Create environments and physics scenes in Isaac Sim. Use when the user wants to load pre-built environments (warehouse, hospital, office, grid rooms) or create custom physics scenes with gravity and ground planes.
---

# World Builder Skill

This skill loads pre-built Isaac Sim environments and creates physics scenes with proper lighting and camera framing.

## When to use this skill

- User asks to load an environment (warehouse, hospital, office, room, grid)
- User wants to create a physics scene with gravity
- User needs a ground plane or floor
- User wants to set up a simulation environment
- **Use BEFORE spawning robots** - environments should be loaded first

## Available Environments

### Warehouse Environments
- `warehouse` / `simple_warehouse` - Basic warehouse
- `warehouse_forklifts` - Warehouse with forklifts
- `warehouse_shelves` - Warehouse with multiple shelves
- `full_warehouse` - Large complete warehouse

### Indoor Environments
- `simple_room` / `room` - Small indoor room
- `hospital` - Hospital environment
- `office` - Office environment

### Grid/Testing Environments
- `simple_grid` / `grid` / `default_environment` - Basic testing grid
- `black_grid` / `gridroom_black` - Black grid room
- `curved_grid` / `gridroom_curved` - Curved grid room

### Other
- `jetracer` / `jetracer_track` - Jetracer racing track
- `digital_twin` / `small_warehouse_digital_twin` - Small warehouse digital twin

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

### Step 2: Load environment or create physics scene

#### Load a pre-built environment:
```bash
# Basic warehouse
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment warehouse

# Custom position
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment hospital --pos 0 0 0

# With custom prim path
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment office --path /World/MyOffice
```

#### Create a custom physics scene:
```bash
# Basic physics scene with ground plane
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/create_physics_scene.py --floor

# Custom gravity
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/create_physics_scene.py --floor --gravity 0 0 -9.81

# No floor (for environments that include their own)
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/create_physics_scene.py --no-floor
```

## Script Arguments

### load_environment.py
- `--environment`: Environment name (required)
- `--pos X Y Z`: Position offset (default: 0 0 0)
- `--path`: Custom prim path (default: /World/Environment)

### create_physics_scene.py
- `--floor` / `--no-floor`: Include ground plane (default: True)
- `--gravity X Y Z`: Gravity vector (default: 0 0 -9.81)
- `--name`: Scene name (default: physics_scene)

## Common Patterns

### Setup warehouse with physics
```bash
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment warehouse
```

### Testing environment
```bash
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment simple_grid
```

### Custom physics scene for outdoor simulation
```bash
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/create_physics_scene.py --floor --gravity 0 0 -9.81
```

## Troubleshooting

### "Unknown environment"
- **Cause**: Environment name not recognized
- **Fix**: Use one of the available environments listed above

### Environment appears but no lighting
- **Cause**: Some environments don't include lights
- **Fix**: The script automatically adds default lighting

### Camera not framing environment properly
- **Cause**: Environment is very large or very small
- **Fix**: The script sets camera position based on environment type

## Technical Notes

**Auto-lighting**: Script adds dome light and distant light if not present

**Camera Setup**: Automatically positions camera to frame the environment

**Physics**: Environments include physics by default, custom scenes create physics scene

**Multi-Environment**: Can load multiple environments at different positions using `--path`

## Examples

### Create warehouse simulation
```bash
# Load warehouse
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment warehouse

# Then spawn robots (use robot-master skill)
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type carter --pos -5 0 0
```

### Create testing environment
```bash
# Load grid
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment simple_grid

# Then add props (use props-spawner skill)
```