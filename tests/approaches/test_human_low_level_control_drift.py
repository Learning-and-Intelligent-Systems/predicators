"""Test to verify robot doesn't drift when no keys pressed."""

import numpy as np

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv
from predicators.structs import Action

# Configure
utils.reset_config({
    'env': 'pybullet_circuit',
    'approach': 'human_low_level_control',
    'seed': 0,
    'human_control_move_speed': 0.1,
    'human_control_rot_speed': 0.2,
    'pybullet_max_vel_norm': 0.15,
    'pybullet_sim_steps_per_action': 5,
})

print("Creating environment (no GUI)...")
env = PyBulletCircuitEnv(use_gui=False)

print("Resetting environment...")
obs = env.reset("test", 0)
state = obs

# Get robot object
robot_obj = None
for obj in state.data.keys():
    if obj.type.name == "robot":
        robot_obj = obj
        break

initial_x = state.get(robot_obj, "x")
initial_y = state.get(robot_obj, "y")
initial_z = state.get(robot_obj, "z")
print(
    f"\nInitial position: ({initial_x:.4f}, {initial_y:.4f}, {initial_z:.4f})")

# Test: Apply NO movement (zero deltas) for 20 steps
print("\n" + "=" * 60)
print("TEST: No-op actions (robot should stay still)")
print("=" * 60)

print("\nApplying 20 no-op actions (maintaining current joint positions)...")
current_state = state
positions = []

for i in range(20):
    current_x = current_state.get(robot_obj, "x")
    current_y = current_state.get(robot_obj, "y")
    current_z = current_state.get(robot_obj, "z")
    positions.append((current_x, current_y, current_z))

    # NO MOVEMENT - use current joint positions directly (like the fixed
    # approach)
    action_arr = np.array(current_state.joint_positions, dtype=np.float32)
    action = Action(action_arr)

    if i % 5 == 0:
        print(f"  [Step {i+1}] Pos: "
              f"({current_x:.4f}, {current_y:.4f}, "
              f"{current_z:.4f})")

    # Apply action
    current_state = env.step(action)

# Final position
final_x = current_state.get(robot_obj, "x")
final_y = current_state.get(robot_obj, "y")
final_z = current_state.get(robot_obj, "z")

print("\n" + "=" * 60)
print("RESULTS (with no-op fix):")
print("=" * 60)
print(f"Initial position: ({initial_x:.4f}, "
      f"{initial_y:.4f}, {initial_z:.4f})")
print(f"Final position:   ({final_x:.4f}, "
      f"{final_y:.4f}, {final_z:.4f})")

drift_x = final_x - initial_x
drift_y = final_y - initial_y
drift_z = final_z - initial_z
total_drift = np.sqrt(drift_x**2 + drift_y**2 + drift_z**2)

print(f"\nDrift: dx={drift_x:.4f}, dy={drift_y:.4f}, dz={drift_z:.4f}")
print(f"Total drift magnitude: {total_drift:.4f}")

# Check if drift is acceptable (less than 1mm)
DRIFT_THRESHOLD = 0.001  # 1mm
if total_drift < DRIFT_THRESHOLD:
    print(f"\n✓ PASS: Robot drift ({total_drift:.6f}) "
          f"is below threshold ({DRIFT_THRESHOLD})")
else:
    print(f"\n✗ FAIL: Robot drift ({total_drift:.4f}) "
          f"exceeds threshold ({DRIFT_THRESHOLD})")

# Now test with the actual approach
print("\n" + "=" * 60)
print("TEST: Using actual approach policy")
print("=" * 60)

# Reset environment
obs = env.reset("test", 0)
state = obs

initial_x = state.get(robot_obj, "x")
initial_y = state.get(robot_obj, "y")
initial_z = state.get(robot_obj, "z")
print(
    f"\nInitial position: ({initial_x:.4f}, {initial_y:.4f}, {initial_z:.4f})")

# Create approach (use empty sets for predicates/options since not needed)
approach = create_approach(
    'human_low_level_control',
    set(),  # predicates
    set(),  # options
    env.types,
    env.action_space,
    [])

# Get task and create policy
task = env.get_task("test", 0)

# We need to test the approach's _get_action_from_keyboard method
# but without actual keyboard input. Let's directly test the no-op behavior.

print("\nTesting approach's no-op behavior (simulating no key press)...")

# Access the approach's internal method
# pylint: disable=protected-access
approach._setup_terminal = lambda: None  # type: ignore
approach._restore_terminal = lambda: None  # type: ignore
approach._get_pressed_key = lambda: None  # type: ignore
# pylint: enable=protected-access

policy = approach.solve(task, timeout=10)  # type: ignore[arg-type]

current_state = state
for i in range(20):
    current_x = current_state.get(robot_obj, "x")
    current_y = current_state.get(robot_obj, "y")
    current_z = current_state.get(robot_obj, "z")

    if i % 5 == 0:
        print(f"  [Step {i+1}] Pos: "
              f"({current_x:.4f}, {current_y:.4f}, "
              f"{current_z:.4f})")

    # Get action from approach (should be no-op since no key pressed)
    action = policy(current_state)

    # Apply action
    current_state = env.step(action)

# Final position
final_x = current_state.get(robot_obj, "x")
final_y = current_state.get(robot_obj, "y")
final_z = current_state.get(robot_obj, "z")

drift_x = final_x - initial_x
drift_y = final_y - initial_y
drift_z = final_z - initial_z
total_drift = np.sqrt(drift_x**2 + drift_y**2 + drift_z**2)

print(f"\nFinal position:   ({final_x:.4f}, {final_y:.4f}, {final_z:.4f})")
print(f"Drift: dx={drift_x:.4f}, dy={drift_y:.4f}, dz={drift_z:.4f}")
print(f"Total drift magnitude: {total_drift:.6f}")

if total_drift < DRIFT_THRESHOLD:
    print(f"\n✓ PASS: Approach no-op drift "
          f"({total_drift:.6f}) is below "
          f"threshold ({DRIFT_THRESHOLD})")
else:
    print(f"\n✗ FAIL: Approach no-op drift "
          f"({total_drift:.4f}) exceeds "
          f"threshold ({DRIFT_THRESHOLD})")

print("\nTest complete!")
