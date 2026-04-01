---
name: scene-manager
description: Manage Isaac Sim scene state - clear scene, reset simulation, or remove specific objects. Use when the user wants to start fresh, clean up the scene, or remove objects while keeping the environment.
---

# Scene Manager Skill

This skill manages the Isaac Sim scene state, allowing you to clear objects, reset the simulation, or selectively remove elements.

## When to use this skill

- User asks to "clear the scene" or "start fresh"
- User wants to "remove all robots" or "delete everything"
- User needs to "reset the simulation"
- User wants to keep environment but remove objects
- User asks to "clean up" the scene

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

### Step 2: Clear the scene
```bash
# Clear everything except ground plane and physics
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py

# Clear everything including ground plane
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground

# Clear everything including physics scene
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-physics

# Clear absolutely everything
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground --no-keep-physics
```

## Script Arguments

### clear_scene.py
- `--keep-ground` / `--no-keep-ground`: Keep ground plane (default: True)
- `--keep-physics` / `--no-keep-physics`: Keep physics scene (default: True)
- `--host`: Isaac Sim host (default: localhost)
- `--port`: Command server port (default: 8765)

## Common Patterns

### Start fresh but keep environment
```bash
# Clear all objects but keep ground and physics
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py
```

### Complete reset
```bash
# Remove everything
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground --no-keep-physics

# Then rebuild environment
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment warehouse
```

### Clear objects but keep physics setup
```bash
# Useful when you want to respawn different robots in same environment
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --keep-ground --keep-physics
```

## What Gets Cleared

### Default behavior (keep-ground=True, keep-physics=True):
- ✅ Robots (Franka, Carter, Go2, etc.)
- ✅ Props (cubes, spheres, cylinders)
- ✅ Generated 3D models
- ✅ Custom objects
- ❌ Ground plane (kept)
- ❌ Physics scene (kept)
- ❌ Lights (kept)
- ❌ Cameras (kept)

### With --no-keep-ground:
- ✅ Everything above +
- ✅ Ground plane
- ❌ Physics scene (kept if --keep-physics)

### With --no-keep-physics:
- ✅ Everything +
- ✅ Physics scene
- ❌ Ground plane (kept if --keep-ground)

### With both flags:
- ✅ Removes ALL children of /World
- Results in completely empty scene

## Troubleshooting

### "No /World prim found"
- **Cause**: Scene not properly initialized
- **Fix**: Load an environment first

### Objects not being removed
- **Cause**: Objects may be in unexpected locations
- **Fix**: Use --no-keep-ground --no-keep-physics to remove everything

### Scene cleared but physics broken
- **Cause**: Removed physics scene
- **Fix**: Create physics scene again:
```bash
  python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/create_physics_scene.py
```

### Ground plane removed accidentally
- **Cause**: Used --no-keep-ground
- **Fix**: Create physics scene with ground:
```bash
  python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/create_physics_scene.py --floor
```

## Technical Notes

**Robot Counter Reset**: Clearing the scene resets robot instance counters, so next robot will use default names

**Performance**: Clearing is fast - typically completes in <1 second

**Persistent State**: Scene state is cleared, but Isaac Sim continues running

**Safe Operation**: Uses proper USD commands to delete prims, maintaining scene integrity

## Workflow Examples

### Iterative scene building
```bash
# 1. Load environment
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment warehouse

# 2. Add some robots
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --pos 0 0 0

# 3. Not satisfied? Clear and try again
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py

# 4. Add different robots
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type carter --pos 0 0 0
```

### Switch environments
```bash
# 1. Clear current scene completely
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground --no-keep-physics

# 2. Load new environment
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment hospital

# 3. Add appropriate robots
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type g1 --pos 0 0 1.0
```

### Testing different configurations
```bash
# Test configuration 1
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --path /World/Test1 --pos 0 0 0

# Clear for next test (keep environment)
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py

# Test configuration 2
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --path /World/Test2 --pos 1 0 0
```

## Integration with Other Skills

**Before world-builder**: Clear scene before loading new environment
```bash
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py --no-keep-ground
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment warehouse
```

**Before robot-master**: Clear old robots before spawning new ones
```bash
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type go2
```

**After props-spawner**: Remove all spawned props
```bash
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py
```