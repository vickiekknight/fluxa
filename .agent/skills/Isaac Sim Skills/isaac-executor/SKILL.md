---
name: isaac-executor
description: Execute arbitrary Python code in the running Isaac Sim instance. Use when the user asks for Isaac Sim operations not covered by specific skills, or for novel/experimental tasks. ONLY use this skill when no specific skill exists for the task - always prefer robot-master, world-builder, props-spawner, or scene-manager when applicable.
---

# Isaac Executor Skill

Execute custom Python code directly in the Isaac Sim environment. This is a **fallback skill** for operations not covered by specialized skills.

## When to use this skill

**USE this skill when:**
- User requests Isaac Sim operations not covered by specific skills
- Novel or experimental tasks (e.g., "make the robot green", "rotate all objects")
- Custom scene modifications
- Advanced USD manipulations
- Testing new Isaac Sim features
- One-off custom operations

**DO NOT use this skill when:**
- ❌ Spawning robots → Use `robot-master` skill
- ❌ Loading environments → Use `world-builder` skill
- ❌ Creating cubes/spheres → Use `props-spawner` skill
- ❌ Clearing scene → Use `scene-manager` skill

## Skill Priority

Always check if a specific skill exists first:

1. **robot-master** - For spawning any robot
2. **world-builder** - For environments and physics scenes
3. **props-spawner** - For basic geometric objects
4. **scene-manager** - For clearing/managing scene
5. **isaac-executor** - For everything else (THIS SKILL)

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

### Step 2: Write Python code

The code has access to Isaac Sim's full Python environment:

**Available modules:**
- `omni.usd` - USD stage manipulation
- `pxr` - USD primitives (UsdGeom, Gf, Sdf, UsdPhysics, etc.)
- `isaacsim.core.utils.stage` - Stage utilities
- `isaacsim.core.utils.prims` - Prim utilities
- `isaacsim.core.api` - World, Physics APIs
- `omni.kit.commands` - Kit commands
- Standard Python libraries (numpy, math, etc.)

**Pre-defined variables:**
- `stage` - Current USD stage
- `omni` - Omni module
- `carb` - Carb settings module

### Step 3: Execute code
```bash
# Execute Python code from command line
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "print('Hello from Isaac Sim')"

# Execute code from file
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --file my_script.py

# Execute multi-line code
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdGeom, Gf
cube = UsdGeom.Cube.Define(stage, '/World/MyCube')
cube.CreateSizeAttr().Set(1.0)
xformable = UsdGeom.Xformable(cube.GetPrim())
xformable.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
print('Created cube at /World/MyCube')
"
```

## Script Arguments

### execute_code.py
- `--code`: Python code string to execute (use quotes)
- `--file`: Path to Python file to execute
- `--host`: Isaac Sim host (default: localhost)
- `--port`: Command server port (default: 8765)

**Note:** Use either `--code` or `--file`, not both.

## Common Use Cases

### 1. Change Object Colors
```python
# Change robot color to green
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdGeom, Gf
prim = stage.GetPrimAtPath('/World/Franka')
if prim:
    # Find mesh prims recursively
    for child in prim.GetAllChildren():
        if child.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(child)
            mesh.CreateDisplayColorAttr().Set([Gf.Vec3f(0.0, 1.0, 0.0)])
    print('Changed robot color to green')
"
```

### 2. Rotate Objects
```python
# Rotate a cube 45 degrees around Z axis
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdGeom, Gf
prim = stage.GetPrimAtPath('/World/Cube')
if prim:
    xformable = UsdGeom.Xformable(prim)
    xformable.AddRotateZOp().Set(45.0)
    print('Rotated cube 45 degrees')
"
```

### 3. Add Lights
```python
# Add a dome light
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdLux
dome_light = UsdLux.DomeLight.Define(stage, '/World/MyDomeLight')
dome_light.GetIntensityAttr().Set(1500.0)
print('Added dome light')
"
```

### 4. Query Scene Information
```python
# List all prims in scene
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
prims = []
for prim in stage.Traverse():
    prims.append(str(prim.GetPath()))
print(f'Scene contains {len(prims)} prims:')
for p in prims[:10]:  # Show first 10
    print(f'  {p}')
"
```

### 5. Modify Physics Properties
```python
# Change object mass
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdPhysics
prim = stage.GetPrimAtPath('/World/Cube')
if prim:
    mass_api = UsdPhysics.MassAPI.Apply(prim)
    mass_api.CreateMassAttr().Set(10.0)
    print('Changed cube mass to 10 kg')
"
```

### 6. Scale Objects
```python
# Scale an object
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdGeom, Gf
prim = stage.GetPrimAtPath('/World/Robot')
if prim:
    xformable = UsdGeom.Xformable(prim)
    xformable.AddScaleOp().Set(Gf.Vec3d(2.0, 2.0, 2.0))
    print('Scaled robot to 2x size')
"
```

### 7. Create Custom Materials
```python
# Create and apply a material
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdShade, Sdf
material = UsdShade.Material.Define(stage, '/World/Materials/RedMaterial')
shader = UsdShade.Shader.Define(stage, '/World/Materials/RedMaterial/Shader')
shader.CreateIdAttr('UsdPreviewSurface')
shader.CreateInput('diffuseColor', Sdf.ValueTypeNames.Color3f).Set((1.0, 0.0, 0.0))
material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), 'surface')
print('Created red material')
"
```

### 8. Animate Objects
```python
# Move object over time (simple animation)
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdGeom, Gf
import math
prim = stage.GetPrimAtPath('/World/Sphere')
if prim:
    xformable = UsdGeom.Xformable(prim)
    translate_op = xformable.AddTranslateOp()
    # Animate in a circle (this sets keyframes)
    for i in range(0, 360, 10):
        angle = math.radians(i)
        x = math.cos(angle) * 2.0
        y = math.sin(angle) * 2.0
        translate_op.Set(Gf.Vec3d(x, y, 1.0), i)
    print('Created circular animation for sphere')
"
```

## Working with Files

### Create a Python file

Create `my_custom_script.py`:
```python
# my_custom_script.py
from pxr import UsdGeom, Gf

print("Running custom script...")

# Create multiple colored cubes
colors = [
    (1, 0, 0),  # Red
    (0, 1, 0),  # Green
    (0, 0, 1),  # Blue
]

for i, color in enumerate(colors):
    path = f"/World/ColoredCube_{i}"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr().Set(0.5)
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    
    xformable = UsdGeom.Xformable(cube.GetPrim())
    xformable.AddTranslateOp().Set(Gf.Vec3d(i * 1.5, 0, 0.5))
    
    print(f"Created {path} with color {color}")

print("Script complete!")
```

### Execute the file
```bash
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --file my_custom_script.py
```

## Advanced Examples

### Example 1: Find and Modify All Objects of Type
```python
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdGeom, Gf

# Find all cubes and make them red
count = 0
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Cube):
        cube = UsdGeom.Cube(prim)
        cube.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
        count += 1

print(f'Made {count} cubes red')
"
```

### Example 2: Create Grid of Objects with Code
```python
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdGeom, Gf

# Create 5x5 grid of spheres
for i in range(5):
    for j in range(5):
        path = f'/World/GridSphere_{i}_{j}'
        sphere = UsdGeom.Sphere.Define(stage, path)
        sphere.CreateRadiusAttr().Set(0.2)
        
        # Rainbow colors
        hue = (i * 5 + j) / 25.0
        color = [
            abs((hue * 6) % 1),
            abs(((hue * 6) - 2) % 1),
            abs(((hue * 6) - 4) % 1)
        ]
        sphere.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
        
        xformable = UsdGeom.Xformable(sphere.GetPrim())
        xformable.AddTranslateOp().Set(Gf.Vec3d(i * 0.5, j * 0.5, 0.2))

print('Created 5x5 grid of rainbow spheres')
"
```

### Example 3: Apply Physics to All Objects
```python
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdPhysics

# Add physics to all geometric primitives
count = 0
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Gprim):  # Generic geometric primitive
        # Add collision
        UsdPhysics.CollisionAPI.Apply(prim)
        # Add rigid body
        UsdPhysics.RigidBodyAPI.Apply(prim)
        # Add mass
        mass_api = UsdPhysics.MassAPI.Apply(prim)
        mass_api.CreateMassAttr().Set(1.0)
        count += 1

print(f'Added physics to {count} objects')
"
```

## Error Handling

The executor will show:
- ✅ Success messages for completed operations
- ❌ Error messages with Python tracebacks
- ⚠️ Warnings for potential issues

Example error output:
```
❌ ERROR: name 'invalid_function' is not defined
Traceback:
  File "<string>", line 2, in <module>
NameError: name 'invalid_function' is not defined
```

## Best Practices

### 1. Always Check if Prims Exist
```python
prim = stage.GetPrimAtPath('/World/MyObject')
if prim and prim.IsValid():
    # Do something
    pass
else:
    print("Prim not found!")
```

### 2. Use Try-Except for Safety
```python
try:
    # Your code here
    pass
except Exception as e:
    print(f"Error: {e}")
```

### 3. Print Status Messages
```python
print("Starting operation...")
# Do work
print("Operation complete!")
```

### 4. Test Small Changes First

Test with simple code before complex operations:
```bash
# Test basic access
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "print('Isaac Sim is accessible')"

# Then try more complex code
```

## Limitations

**Cannot do:**
- Create new SimulationApp instances (Isaac Sim already running)
- Access external files without absolute paths
- Make blocking calls (use async if needed)
- Access user's local filesystem (runs in container)

**Can do:**
- Modify USD stage
- Create/delete prims
- Change properties
- Query scene state
- Execute Isaac Sim APIs
- Import Python libraries available in Isaac Sim

## Troubleshooting

### "Module not found"
- **Cause**: Module not available in Isaac Sim's Python
- **Fix**: Check if module is part of Isaac Sim installation

### "Prim not found"
- **Cause**: Path doesn't exist or typo in path
- **Fix**: Use correct path, check with scene query first

### Code executes but nothing happens
- **Cause**: Code may have logic error or wrong path
- **Fix**: Add print statements to debug

### "Cannot connect to Isaac Sim"
- **Cause**: Isaac Sim not running or WebSocket server down
- **Fix**: Restart Isaac Sim streaming server

## Integration with Other Skills

### After robot-master
```bash
# Spawn robot
python3 ~/fluxa-agent-pack/.agent/skills/robot-master/scripts/spawn_robot.py --type franka

# Then customize with executor
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
from pxr import UsdGeom, Gf
# Make robot red
for prim in stage.GetPrimAtPath('/World/Franka').GetAllChildren():
    if prim.IsA(UsdGeom.Mesh):
        UsdGeom.Mesh(prim).CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.0, 0.0)])
"
```

### With props-spawner
```bash
# Create cubes with props-spawner
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --path /World/Cube1
python3 ~/fluxa-agent-pack/.agent/skills/props-spawner/scripts/spawn_prop.py --type cube --path /World/Cube2

# Then animate them with executor
python3 ~/fluxa-agent-pack/.agent/skills/isaac-executor/scripts/execute_code.py --code "
# Animate both cubes
# (animation code here)
"
```

## Security Note

This skill executes arbitrary Python code in Isaac Sim. Only use with **trusted code**.

## Examples Library

Common code snippets for reference:

### List all prims
```python
for prim in stage.Traverse():
    print(prim.GetPath())
```

### Get prim properties
```python
prim = stage.GetPrimAtPath('/World/Cube')
for attr in prim.GetAttributes():
    print(f"{attr.GetName()}: {attr.Get()}")
```

### Create camera
```python
from pxr import UsdGeom, Gf
camera = UsdGeom.Camera.Define(stage, '/World/MyCamera')
xformable = UsdGeom.Xformable(camera.GetPrim())
xformable.AddTranslateOp().Set(Gf.Vec3d(5, 5, 5))
```

### Add point light
```python
from pxr import UsdLux
light = UsdLux.SphereLight.Define(stage, '/World/PointLight')
light.GetIntensityAttr().Set(5000.0)
light.GetRadiusAttr().Set(0.5)
```