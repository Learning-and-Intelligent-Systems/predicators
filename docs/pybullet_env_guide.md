# PyBulletEnv Developer Guide

This guide explains how to create new PyBullet-based robotic manipulation environments by extending the `PyBulletEnv` base class.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Required Class Variables](#required-class-variables)
- [Required Methods](#required-methods)
  - [get_name](#1-get_name)
  - [initialize_pybullet](#2-initialize_pybullet)
  - [_store_pybullet_bodies](#3-_store_pybullet_bodies)
  - [_get_object_ids_for_held_check](#4-_get_object_ids_for_held_check)
  - [_create_task_specific_objects](#5-_create_task_specific_objects)
  - [_reset_custom_env_state](#6-_reset_custom_env_state)
  - [_extract_feature](#7-_extract_feature)
- [Optional Methods to Override](#optional-methods-to-override)
- [State Management](#state-management)
- [Object Handling](#object-handling)
- [Grasping System](#grasping-system)
- [Utility Functions](#utility-functions)
- [Implementation Checklist](#implementation-checklist)
- [Examples](#examples)

---

## Overview

`PyBulletEnv` is an abstract base class in `predicators/envs/pybullet_env.py` that provides common functionality for PyBullet-based environments, including:

- Robot initialization and control
- State synchronization between abstract `State` objects and PyBullet simulation
- Automatic grasp detection and constraint management
- Rendering and camera configuration
- Task generation with PyBullet state conversion

## Architecture

To create a new PyBullet environment, use multiple inheritance combining `PyBulletEnv` with a corresponding abstract environment class:

```python
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.envs.my_abstract_env import MyAbstractEnv

class PyBulletMyEnv(PyBulletEnv, MyAbstractEnv):
    """PyBullet version of MyAbstractEnv."""

    # Class variables
    robot_init_x: ClassVar[float] = 0.5
    robot_init_y: ClassVar[float] = 0.5
    robot_init_z: ClassVar[float] = 0.5

    # ... implement required methods
```

The inheritance order matters: `PyBulletEnv` should come first to ensure proper method resolution.

---

## Required Class Variables

| Variable | Type | Description |
|----------|------|-------------|
| `robot_init_x` | `float` | Initial robot end-effector X position |
| `robot_init_y` | `float` | Initial robot end-effector Y position |
| `robot_init_z` | `float` | Initial robot end-effector Z position |
| `robot_base_pos` | `Optional[Tuple[float, float, float]]` | Robot base position (or `None` for default) |
| `robot_base_orn` | `Optional[Tuple[float, float, float, float]]` | Robot base orientation as quaternion |

### Optional Class Variables

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `grasp_tol` | `float` | `5e-2` | Distance tolerance for grasp detection |
| `grasp_tol_small` | `float` | `5e-4` | Smaller tolerance for precise grasping |
| `open_fingers` | `float` | `0.04` | Finger state value when open |
| `closed_fingers` | `float` | `0.01` | Finger state value when closed |
| `_camera_distance` | `float` | `0.8` | Camera distance for rendering |
| `_camera_yaw` | `float` | `90.0` | Camera yaw angle |
| `_camera_pitch` | `float` | `-24` | Camera pitch angle |
| `_camera_target` | `Pose3D` | `(1.65, 0.75, 0.42)` | Camera target position |
| `_obj_colors` | `Sequence[Tuple[float, ...]]` | (see source) | Available RGBA colors for objects |

---

## Required Methods

### 1. `get_name`

```python
@classmethod
def get_name(cls) -> str:
    """Returns the unique string identifier for this environment."""
    return "pybullet_my_env"
```

This identifier is used for configuration lookup (e.g., `CFG.pybullet_robot_ee_orns`).

---

### 2. `initialize_pybullet`

```python
@classmethod
def initialize_pybullet(
    cls, using_gui: bool
) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
    """Initialize PyBullet simulation and load environment assets."""
```

**Purpose**: Set up the physics simulation, load the robot, and create persistent objects (objects that exist across all tasks).

**Must do**:
1. Call `super().initialize_pybullet(using_gui)` first
2. Load environment-specific assets (tables, fixtures, etc.)
3. Create objects that persist across tasks (e.g., maximum number of blocks)
4. Return `(physics_client_id, pybullet_robot, bodies_dict)`

**Example**:

```python
@classmethod
def initialize_pybullet(cls, using_gui: bool) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
    # Call parent to set up physics, plane, and robot
    physics_client_id, pybullet_robot, bodies = super().initialize_pybullet(using_gui)

    # Load table
    table_id = p.loadURDF(
        utils.get_env_asset_path("urdf/table.urdf"),
        useFixedBase=True,
        physicsClientId=physics_client_id
    )
    p.resetBasePositionAndOrientation(
        table_id, cls._table_pose, cls._table_orientation,
        physicsClientId=physics_client_id
    )
    bodies["table_id"] = table_id

    # Pre-create maximum number of blocks (reused across tasks)
    block_ids = []
    num_blocks = max(CFG.blocks_num_blocks_train + CFG.blocks_num_blocks_test)
    for i in range(num_blocks):
        color = cls._obj_colors[i % len(cls._obj_colors)]
        block_id = create_pybullet_block(
            color=color,
            half_extents=(0.02, 0.02, 0.02),
            mass=cls._obj_mass,
            friction=cls._obj_friction,
            physics_client_id=physics_client_id
        )
        block_ids.append(block_id)
    bodies["block_ids"] = block_ids

    return physics_client_id, pybullet_robot, bodies
```

---

### 3. `_store_pybullet_bodies`

```python
def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
    """Store references to PyBullet body IDs from initialization."""
```

**Purpose**: Save the body IDs returned by `initialize_pybullet()` to instance variables. Called once during `__init__`.

**Example**:

```python
def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
    self._table_id = pybullet_bodies["table_id"]
    self._block_ids = pybullet_bodies["block_ids"]

    # Assign IDs to Object instances
    for block, block_id in zip(self._blocks, self._block_ids):
        block.id = block_id
```

---

### 4. `_get_object_ids_for_held_check`

```python
def _get_object_ids_for_held_check(self) -> List[int]:
    """Return PyBullet IDs of objects that can be grasped."""
```

**Purpose**: Tell the grasping system which objects to check when detecting what the robot is holding.

**Example**:

```python
def _get_object_ids_for_held_check(self) -> List[int]:
    # Only blocks can be grasped, not the table
    return list(self._block_id_to_block.keys())
```

---

### 5. `_create_task_specific_objects`

```python
def _create_task_specific_objects(self, state: State) -> None:
    """Create or recreate objects that vary between tasks."""
```

**Purpose**: Handle objects that need to be created fresh for each task (e.g., cups with varying sizes, liquids, dynamic elements). Called during `_reset_state()` before objects are positioned.

**When to use**:
- Objects with task-specific sizes (can't change size after creation in PyBullet)
- Visual elements that depend on task state (liquids, indicators)
- Objects whose count varies between tasks

**Example**:

```python
def _create_task_specific_objects(self, state: State) -> None:
    # Remove old cups
    for cup in self._cups:
        if cup.id is not None:
            p.removeBody(cup.id, physicsClientId=self._physics_client_id)

    # Create new cups with task-specific capacities
    cup_objs = state.get_objects(self._cup_type)
    for cup_obj in cup_objs:
        capacity = state.get(cup_obj, "capacity")
        scale = capacity / self.max_capacity
        cup_id = create_object(
            "urdf/cup.urdf",
            scale=scale,
            physics_client_id=self._physics_client_id
        )
        cup_obj.id = cup_id
```

If your environment has no task-specific objects, simply pass:

```python
def _create_task_specific_objects(self, state: State) -> None:
    pass  # All objects are created in initialize_pybullet
```

---

### 6. `_reset_custom_env_state`

```python
def _reset_custom_env_state(self, state: State) -> None:
    """Perform environment-specific reset operations."""
```

**Purpose**: Handle reset operations not covered by the base class. Called after robot and standard object positions have been reset.

**Common uses**:
- Setting object colors based on state
- Creating/updating visual elements (liquids, lights)
- Moving unused objects out of view
- Updating UI elements (button colors)

**Example**:

```python
def _reset_custom_env_state(self, state: State) -> None:
    block_objs = state.get_objects(self._block_type)
    self._block_id_to_block.clear()

    # Position and color each block
    for i, block_obj in enumerate(block_objs):
        block_id = self._block_ids[i]
        self._block_id_to_block[block_id] = block_obj

        # Update color from state
        r = state.get(block_obj, "color_r")
        g = state.get(block_obj, "color_g")
        b = state.get(block_obj, "color_b")
        p.changeVisualShape(
            block_id, linkIndex=-1,
            rgbaColor=(r, g, b, 1.0),
            physicsClientId=self._physics_client_id
        )

    # Move unused blocks out of view
    for i in range(len(block_objs), len(self._block_ids)):
        block_id = self._block_ids[i]
        p.resetBasePositionAndOrientation(
            block_id, [10.0, 10.0, i * 0.1],  # Out of view
            self._default_orn,
            physicsClientId=self._physics_client_id
        )
```

---

### 7. `_extract_feature`

```python
def _extract_feature(self, obj: Object, feature: str) -> float:
    """Extract custom feature values from PyBullet state."""
```

**Purpose**: Called by `_get_state()` for features not handled by the base class. The base class automatically handles: `x`, `y`, `z`, `rot`, `yaw`, `roll`, `is_held`, `r`, `g`, `b`.

**Example**:

```python
def _extract_feature(self, obj: Object, feature: str) -> float:
    if obj.type == self._block_type:
        block_id = self._get_block_id(obj)

        if feature == "color_r":
            visual_data = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id
            )[0]
            return visual_data[7][0]  # RGBA tuple, index 0 is R
        elif feature == "color_g":
            visual_data = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id
            )[0]
            return visual_data[7][1]
        elif feature == "color_b":
            visual_data = p.getVisualShapeData(
                block_id, physicsClientId=self._physics_client_id
            )[0]
            return visual_data[7][2]

    elif obj.type == self._machine_type:
        if feature == "is_on":
            button_color = p.getVisualShapeData(
                self._button_id,
                physicsClientId=self._physics_client_id
            )[0][-1]
            return 1.0 if button_color == self.button_color_on else 0.0

    raise ValueError(f"Unknown feature '{feature}' for object type '{obj.type}'")
```

---

## Optional Methods to Override

### `step`

Override to add domain-specific physics handling:

```python
def step(self, action: Action, render_obs: bool = False) -> Observation:
    # Call parent step first
    state = super().step(action, render_obs=render_obs)

    # Domain-specific logic
    self._handle_button_press(state)
    self._handle_liquid_pouring(state)

    # Refresh observation after modifications
    self._current_observation = self._get_state()
    return self._current_observation.copy()
```

### `_extract_robot_state`

Override if your environment uses non-standard robot features:

```python
def _extract_robot_state(self, state: State) -> np.ndarray:
    """Returns 8D array: [x, y, z, qx, qy, qz, qw, finger_joint]."""
    robot = state.get_objects(self._robot_type)[0]
    rx = state.get(robot, "pose_x")
    ry = state.get(robot, "pose_y")
    rz = state.get(robot, "pose_z")
    f = state.get(robot, "fingers")
    f = self._fingers_state_to_joint(self._pybullet_robot, f)
    qx, qy, qz, qw = self.get_robot_ee_home_orn()
    return np.array([rx, ry, rz, qx, qy, qz, qw, f], dtype=np.float32)
```

### `_get_tasks` and `_load_task_from_json`

Override to convert abstract tasks to PyBullet tasks:

```python
def _get_tasks(self, num_tasks: int, ...) -> List[EnvironmentTask]:
    tasks = super()._get_tasks(num_tasks, ...)
    return self._add_pybullet_state_to_tasks(tasks)

def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
    task = super()._load_task_from_json(json_file)
    return self._add_pybullet_state_to_tasks([task])[0]
```

---

## State Management

The base class handles synchronization between abstract `State` objects and PyBullet:

| Method | Description |
|--------|-------------|
| `_reset_state(state)` | Resets PyBullet to match a given State. Handles robot pose, object positions, and held object constraints. |
| `_get_state()` | Reads current PyBullet simulation into a `PyBulletState`. Extracts robot pose, object poses, and calls `_extract_feature()` for custom features. |
| `simulate(state, action)` | Convenience method that resets to state if needed, then steps. Used by option models. |

### State Flow Diagram

```
Task Generation:
    _get_tasks() → abstract State → _add_pybullet_state_to_tasks() → PyBulletState

Reset:
    reset() → _reset_state(state) → [robot reset, object reset, _create_task_specific_objects, _reset_custom_env_state]

Step:
    step(action) → [physics simulation] → _get_state() → PyBulletState
```

---

## Object Handling

Objects are tracked via `self._objects`, populated during reset from `state.data`. Each `Object` has an `id` attribute storing its PyBullet body ID.

**Base class handles automatically**:
- Standard features: `x`, `y`, `z`, `rot`/`yaw`, `is_held`, `r`, `g`, `b`
- Position/orientation reset via `_reset_single_object()`
- Grasp constraint management for held objects

**Non-physical objects** (e.g., abstract locations, angles) should have their type names listed in skip lists:

```python
# In _reset_state and _get_state
if obj.type.name in ["robot", "loc", "angle", "human", "side", "direction"]:
    continue  # Skip PyBullet operations
```

---

## Grasping System

The base class provides automatic grasp detection and constraint management:

| Method | Description |
|--------|-------------|
| `_detect_held_object()` | Checks finger contact points against objects from `_get_object_ids_for_held_check()` |
| `_create_grasp_constraint()` | Creates a fixed constraint between gripper and held object |
| `_fingers_closing(action)` | Returns `True` if action is closing fingers |
| `_fingers_opening(action)` | Returns `True` if action is opening fingers |

Constraints are automatically removed when fingers open. The held object ID is tracked in `self._held_obj_id`.

---

## Utility Functions

The module provides helper functions for creating common objects:

### `create_pybullet_block`

```python
def create_pybullet_block(
    color: Tuple[float, float, float, float],
    half_extents: Tuple[float, float, float],
    mass: float,
    friction: float,
    position: Pose3D = (0.0, 0.0, 0.0),
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    physics_client_id: int = 0,
    add_top_triangle: bool = False,  # Adds directional marker
) -> int:
    """Creates a box-shaped body. Returns PyBullet body ID."""
```

### `create_pybullet_sphere`

```python
def create_pybullet_sphere(
    color: Tuple[float, float, float, float],
    radius: float,
    mass: float,
    friction: float,
    position: Pose3D = (0.0, 0.0, 0.0),
    orientation: Quaternion = (0.0, 0.0, 0.0, 1.0),
    physics_client_id: int = 0,
) -> int:
    """Creates a sphere-shaped body. Returns PyBullet body ID."""
```

---

## Implementation Checklist

Use this checklist when creating a new PyBullet environment:

- [ ] Define class with multiple inheritance: `class PyBulletMyEnv(PyBulletEnv, MyAbstractEnv)`
- [ ] Set required class variables (`robot_init_x/y/z`, `robot_base_pos/orn`)
- [ ] Implement `get_name()` → unique environment identifier
- [ ] Implement `initialize_pybullet()` → load assets, create persistent objects
- [ ] Implement `_store_pybullet_bodies()` → save body IDs to instance vars
- [ ] Implement `_get_object_ids_for_held_check()` → list graspable object IDs
- [ ] Implement `_create_task_specific_objects()` → per-task object creation (or `pass`)
- [ ] Implement `_reset_custom_env_state()` → colors, visuals, unused objects
- [ ] Implement `_extract_feature()` → custom feature extraction
- [ ] Override `step()` if domain needs custom physics handling
- [ ] Override `_get_tasks()` to call `_add_pybullet_state_to_tasks()`
- [ ] Register environment in `predicators/envs/__init__.py`

---

## Examples

### Minimal Example: PyBulletBlocksEnv

See `predicators/envs/pybullet_blocks.py` for a straightforward implementation with:
- Pre-created blocks moved in/out of view per task
- Simple feature extraction (position, color)
- No task-specific object creation

### Complex Example: PyBulletCoffeeEnv

See `predicators/envs/pybullet_coffee.py` for a more complex implementation with:
- Task-specific cups with varying sizes
- Visual elements (liquids) created/destroyed dynamically
- Custom step logic (button press detection, pouring simulation)
- Multiple object types with different features

---

## Related Files

- `predicators/envs/pybullet_env.py` - Base class implementation
- `predicators/pybullet_helpers/` - Robot, geometry, and object utilities
- `predicators/settings.py` - Configuration options (`CFG.pybullet_*`)
- `predicators/structs.py` - `State`, `Object`, `Action` definitions
