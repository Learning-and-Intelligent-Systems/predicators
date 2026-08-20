"""Test to verify robot handles out-of-bounds actions gracefully."""
# pylint: disable=protected-access

from predicators import utils
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv

# Configure
utils.reset_config({
    'env': 'pybullet_circuit',
    'approach': 'human_low_level_control',
    'seed': 0,
    'human_control_move_speed': 10.0,  # Extremely large to force out-of-bounds
    'human_control_rot_speed': 10.0,  # Extremely large to force out-of-bounds
    'pybullet_max_vel_norm': 10.0,  # Allow large movements
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

print("\n" + "=" * 60)
print("TEST: Extreme movements that may exceed action space bounds")
print("=" * 60)

from predicators.approaches import \
    create_approach  # pylint: disable=wrong-import-position

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
approach._setup_terminal = lambda: None  # type: ignore[attr-defined]
approach._restore_terminal = lambda: None  # type: ignore[attr-defined]

# Simulate extreme key presses that should trigger bounds violations
test_keys = [
    ('w', "Forward (extreme)"),
    ('s', "Backward (extreme)"),
    ('q', "Up (extreme)"),
    ('e', "Down (extreme)"),
    ('a', "Left (extreme)"),
    ('d', "Right (extreme)"),
]

policy = approach.solve(task, timeout=10)  # type: ignore[arg-type]

current_state = state
success_count = 0
bounds_warning_count = 0

for key, description in test_keys:
    print(f"\nTesting: {description} (key='{key}')")

    # Simulate key press
    approach._get_pressed_key = lambda k=key: k  # type: ignore[attr-defined]

    try:
        # Get action from approach
        action = policy(current_state)

        # Verify action is within bounds
        if env.action_space.contains(action.arr):
            print("  ✓ Action generated and within bounds")
            success_count += 1
        else:
            print("  ✗ ERROR: Action still out of bounds after validation!")

        # Apply action (should not crash)
        current_state = env.step(action)

    except Exception as e:  # pylint: disable=broad-except
        print(f"  ✗ CRASH: {e}")
        import traceback
        traceback.print_exc()
        break

# Final position
final_x = current_state.get(robot_obj, "x")
final_y = current_state.get(robot_obj, "y")
final_z = current_state.get(robot_obj, "z")

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Initial position: ({initial_x:.4f}, {initial_y:.4f}, {initial_z:.4f})")
print(f"Final position:   ({final_x:.4f}, {final_y:.4f}, {final_z:.4f})")
print(f"\nSuccessful actions: {success_count}/{len(test_keys)}")

if success_count == len(test_keys):
    print("\n✓ PASS: All extreme movements handled gracefully without crashes")
else:
    print("\n✗ FAIL: Some movements caused crashes")

print("\nTest complete!")
