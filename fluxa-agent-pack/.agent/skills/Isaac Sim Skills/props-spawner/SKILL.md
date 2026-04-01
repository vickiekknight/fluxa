---
name: props-spawner
description: Create basic geometric objects (cubes, spheres, cylinders, cones) in Isaac Sim with physics, colors, and custom properties. Use when the user wants to add simple objects, shapes, or props to test physics, build environments, or create custom scenes.
---

# Props Spawner Skill

This skill creates basic geometric objects in Isaac Sim with physics properties, colors, and transforms.

## When to use this skill

- User asks to "add a cube" or "create a sphere"
- User wants to test physics with simple objects
- User needs placeholders or props for a scene
- User wants to create obstacles or targets
- User asks to "spawn boxes" or "add objects"
- User wants to build custom environments from primitives

## Available Object Types

| Type | Description | Use Case |
|------|-------------|----------|
| `cube` | Box shape | General purpose, containers, obstacles |
| `sphere` | Ball shape | Rolling objects, targets, planets |
| `cylinder` | Tube shape | Pillars, wheels, containers |
| `cone` | Cone shape | Markers, traffic cones, pointers |

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

### Step 2: Spawn objects
```bash
# Basic cube at origin
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube

# Colored sphere at custom position
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type sphere --pos 1 0 2 --color 1 0 0 --size 0.5

# Cylinder with custom size
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cylinder --pos 0 2 1 --size 0.3 --height 2.0

# Cone without physics (static)
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cone --pos -1 0 0.5 --no-physics

# Custom name and mass
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --path /World/HeavyBox --mass 10.0 --size 1.0
```

## Script Arguments

### spawn_prop.py
- `--type`: Object type (required: cube, sphere, cylinder, cone)
- `--pos X Y Z`: Position in meters (default: 0 0 1)
- `--size`: Size/radius in meters (default: 0.5)
- `--height`: Height for cylinder/cone (default: 1.0, only used for cylinder/cone)
- `--color R G B`: RGB color 0-1 range (default: random color)
- `--mass`: Mass in kg (default: 1.0)
- `--path`: Custom USD prim path (default: auto-generated)
- `--physics` / `--no-physics`: Enable/disable physics (default: True)
- `--host`: Isaac Sim host (default: localhost)
- `--port`: Command server port (default: 8765)

## Common Patterns

### Create a stack of boxes
```bash
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --pos 0 0 0.5 --size 1.0 --path /World/Box1
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --pos 0 0 1.5 --size 1.0 --path /World/Box2
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --pos 0 0 2.5 --size 1.0 --path /World/Box3
```

### Create colored targets
```bash
# Red sphere
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type sphere --pos 5 0 1 --color 1 0 0 --size 0.3

# Green sphere
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type sphere --pos 5 3 1 --color 0 1 0 --size 0.3

# Blue sphere
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type sphere --pos 5 -3 1 --color 0 0 1 --size 0.3
```

### Create obstacle course
```bash
# Walls (static cylinders)
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cylinder --pos 2 0 0 --size 0.2 --height 2.0 --no-physics
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cylinder --pos 4 0 0 --size 0.2 --height 2.0 --no-physics

# Rolling balls (with physics)
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type sphere --pos 0 5 2 --size 0.5 --mass 2.0
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type sphere --pos 0 -5 2 --size 0.5 --mass 2.0
```

### Create grid of objects
```bash
# 3x3 grid of cubes
for i in {0..2}; do
  for j in {0..2}; do
    python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py \
      --type cube \
      --pos $i $j 0.5 \
      --size 0.4 \
      --path /World/Grid_${i}_${j}
  done
done
```

### Create markers (static cones)
```bash
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cone --pos 0 0 0 --size 0.3 --height 0.8 --color 1 0.5 0 --no-physics
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cone --pos 3 0 0 --size 0.3 --height 0.8 --color 1 0.5 0 --no-physics
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cone --pos 6 0 0 --size 0.3 --height 0.8 --color 1 0.5 0 --no-physics
```

## Physics Behavior

### With Physics (default)
- Objects have rigid body dynamics
- Objects collide with each other
- Objects are affected by gravity
- Objects have mass and inertia
- Good for: Testing, simulation, interaction

### Without Physics (--no-physics)
- Objects are static/kinematic
- Objects don't move or collide
- Not affected by gravity
- Good for: Walls, markers, decorations, environment props

## Color System

Colors use RGB values from 0.0 to 1.0:

| Color | RGB Values | Example |
|-------|------------|---------|
| Red | `1 0 0` | `--color 1 0 0` |
| Green | `0 1 0` | `--color 0 1 0` |
| Blue | `0 0 1` | `--color 0 0 1` |
| Yellow | `1 1 0` | `--color 1 1 0` |
| Cyan | `0 1 1` | `--color 0 1 1` |
| Magenta | `1 0 1` | `--color 1 0 1` |
| White | `1 1 1` | `--color 1 1 1` |
| Black | `0 0 0` | `--color 0 0 0` |
| Orange | `1 0.5 0` | `--color 1 0.5 0` |
| Purple | `0.5 0 1` | `--color 0.5 0 1` |

If no color is specified, a random color is assigned.

## Troubleshooting

### Object falls through floor
- **Cause**: No ground plane or spawned below ground (Z < 0)
- **Fix**: Ensure Z position > 0, create ground plane if needed:
```bash
  python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/create_physics_scene.py --floor
```

### Object doesn't move
- **Cause**: Physics disabled with --no-physics
- **Fix**: Remove --no-physics flag or spawn a new object with physics

### Objects overlap/intersect
- **Cause**: Spawned at same position
- **Fix**: Use unique positions or unique --path names

### "Prim already exists"
- **Cause**: Object already exists at that path
- **Fix**: Use custom --path with unique name

### Object too small/large
- **Cause**: Size parameter too small/large
- **Fix**: Adjust --size parameter (typical range: 0.1 - 2.0)

### Wrong object height
- **Cause**: --height only works for cylinders and cones
- **Fix**: Use --height only with cylinder/cone types

## Technical Notes

**Collision**: All objects with physics have collision enabled automatically

**Mass**: Default mass is 1.0 kg, can be adjusted with --mass

**Materials**: Objects use simple colored materials (no textures)

**Naming**: Auto-generated names follow pattern: /World/Cube_1, /World/Sphere_1, etc.

**Z-axis**: Remember Z is UP in Isaac Sim (not Y)

## Size Guidelines

| Object Type | Recommended Size | Notes |
|-------------|-----------------|-------|
| Cube | 0.3 - 1.0 | Default: 0.5 |
| Sphere | 0.2 - 1.0 | Default: 0.5 |
| Cylinder | 0.2 - 0.5 | Default: 0.5, also set --height |
| Cone | 0.2 - 0.5 | Default: 0.5, also set --height |

## Integration with Other Skills

### After world-builder
```bash
# Load environment first
python3 ~/fluxa-agent-pack/.agent/skills/world-builder/scripts/load_environment.py --environment warehouse

# Then add props
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --pos 5 5 1
```

### With robot-master
```bash
# Spawn robot
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka --pos 0 0 0

# Add object for robot to interact with
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --pos 0.5 0 0.8 --size 0.1 --color 1 0 0
```

### Before scene-manager
```bash
# Add many test objects
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type sphere --pos 0 0 2
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --pos 1 0 2

# Clear when done testing
python3 ~/fluxa-agent-pack/.agent/skills/scene-manager/scripts/clear_scene.py
```

## Advanced Examples

### Physics stress test
```bash
# Drop 10 spheres from height
for i in {0..9}; do
  x=$((i % 5))
  y=$((i / 5))
  python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py \
    --type sphere \
    --pos $x $y 5 \
    --size 0.3 \
    --mass 0.5 \
    --path /World/Ball_$i
done
```

### Domino effect setup
```bash
# Create line of cubes standing upright
for i in {0..10}; do
  python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py \
    --type cube \
    --pos $((i * 0.3)) 0 0.5 \
    --size 0.2 \
    --mass 0.1 \
    --path /World/Domino_$i
done
```

### Target practice
```bash
# Create targets at different distances
for dist in 5 10 15 20; do
  python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py \
    --type sphere \
    --pos $dist 0 1.5 \
    --size 0.4 \
    --color 1 0 0 \
    --no-physics \
    --path /World/Target_${dist}m
done
```