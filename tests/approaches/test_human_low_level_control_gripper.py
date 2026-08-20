"""Test to verify gripper open/close is instant."""

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv

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

print("\n" + "=" * 60)
print("TEST: Gripper toggle should be instant (1 step)")
print("=" * 60)

# Create approach
approach = create_approach(
    'human_low_level_control',
    set(),  # predicates
    set(),  # options
    env.types,
    env.action_space,
    [])

# Get task and create policy
task = env.get_task("test", 0)

# Disable terminal setup for testing
# pylint: disable=protected-access
approach._setup_terminal = lambda: None  # type: ignore
approach._restore_terminal = lambda: None  # type: ignore

policy = approach.solve(task, timeout=10)  # type: ignore

# Get robot for checking finger positions
_, shadow_robot, _ = (
    PyBulletCircuitEnv.initialize_pybullet(  # type: ignore
        using_gui=False))

print(f"\nFully open position: {shadow_robot.open_fingers}")
print(f"Fully closed position: {shadow_robot.closed_fingers}")

# Test 1: Toggle to closed
print("\n--- Test 1: Toggle gripper to CLOSED (spacebar) ---")
initial_finger_pos = state.joint_positions[shadow_robot.left_finger_joint_idx]
print(f"Initial finger position: {initial_finger_pos:.4f}")

approach._get_pressed_key = lambda: ' '  # type: ignore
action = policy(state)
state = env.step(action)

final_finger_pos = state.joint_positions[shadow_robot.left_finger_joint_idx]
print(f"After 1 step: {final_finger_pos:.4f}")

# Check if finger moved significantly toward closed in 1 step
closed_dist = abs(final_finger_pos - shadow_robot.closed_fingers)
if closed_dist < 0.001:
    print("✓ PASS: Gripper closed instantly in 1 step")
    test1_pass = True
else:
    print(f"✗ FAIL: Gripper not fully closed. "
          f"Distance from target: {closed_dist:.4f}")
    test1_pass = False

# Test 2: Toggle back to open
print("\n--- Test 2: Toggle gripper to OPEN (spacebar) ---")
approach._step_count += 10  # type: ignore
approach._get_pressed_key = lambda: ' '  # type: ignore
action = policy(state)
state = env.step(action)

final_finger_pos = state.joint_positions[shadow_robot.left_finger_joint_idx]
print(f"After 1 step: {final_finger_pos:.4f}")

open_dist = abs(final_finger_pos - shadow_robot.open_fingers)
if open_dist < 0.001:
    print("✓ PASS: Gripper opened instantly in 1 step")
    test2_pass = True
else:
    print(f"✗ FAIL: Gripper not fully open. "
          f"Distance from target: {open_dist:.4f}")
    test2_pass = False

# Test 3: No drift when not pressing keys after toggle
print("\n--- Test 3: No drift after gripper toggle ---")
approach._get_pressed_key = lambda: None  # type: ignore
for i in range(5):
    action = policy(state)
    state = env.step(action)

drift_finger_pos = state.joint_positions[shadow_robot.left_finger_joint_idx]
print(f"Finger pos after 5 no-op steps: "
      f"{drift_finger_pos:.4f}")

finger_drift = abs(drift_finger_pos - final_finger_pos)
if finger_drift < 0.001:
    print("✓ PASS: No finger drift after gripper toggle")
    test3_pass = True
else:
    print(f"✗ FAIL: Finger drifted by "
          f"{finger_drift:.4f}")
    test3_pass = False
# pylint: enable=protected-access

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
if test1_pass and test2_pass and test3_pass:
    print("✓ ALL TESTS PASSED: Gripper is instant and doesn't drift")
else:
    print("✗ SOME TESTS FAILED:")
    print(f"  - Instant close: {'PASS' if test1_pass else 'FAIL'}")
    print(f"  - Instant open: {'PASS' if test2_pass else 'FAIL'}")
    print(f"  - No drift: {'PASS' if test3_pass else 'FAIL'}")

print("\nTest complete!")
