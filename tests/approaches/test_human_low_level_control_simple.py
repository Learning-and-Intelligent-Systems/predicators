"""Simple test to debug human control issues."""

import sys
import traceback

import numpy as np
import pybullet as p

from predicators import utils
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv
from predicators.pybullet_helpers.controllers import \
    get_move_end_effector_to_pose_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.settings import CFG

# Configure with simpler circuit environment
utils.reset_config({
    'env': 'pybullet_circuit',
    'approach': 'human_low_level_control',
    'seed': 0,
    'human_control_move_speed': 0.1,
    'human_control_rot_speed': 0.2,
    'pybullet_max_vel_norm': 0.15,
    'pybullet_sim_steps_per_action': 5,
})

print("Creating environment (no GUI for testing)...")
env = PyBulletCircuitEnv(use_gui=False)

print("Resetting environment...")
obs = env.reset("test", 0)
state = obs

print(f"State type: {type(state)}")
print(f"Has simulator_state: {hasattr(state, 'simulator_state')}")

# Get robot object
robot_obj = None
for obj in state.data.keys():
    if obj.type.name == "robot":
        robot_obj = obj
        break

if robot_obj is None:
    print("ERROR: No robot found!")
    sys.exit(1)

print(f"\nRobot object: {robot_obj}")
print(f"Robot features: {robot_obj.type.feature_names}")

initial_x = state.get(robot_obj, "x")
initial_y = state.get(robot_obj, "y")
initial_z = state.get(robot_obj, "z")
print(f"Initial position: ({initial_x:.3f}, {initial_y:.3f}, {initial_z:.3f})")

# Check if state has joint_positions
if hasattr(state, 'joint_positions'):
    print(f"Joint positions: {state.joint_positions[:5]}... (first 5)")
else:
    print("WARNING: State doesn't have joint_positions attribute")

# Test: Apply actions manually and see robot move
print("\n" + "=" * 60)
print("TEST: Manual action application")
print("=" * 60)

# Get robot for IK - need to use the same env's robot or create a shadow one
print("\nGetting shadow robot for IK...")
_, shadow_robot, _ = (
    PyBulletCircuitEnv.initialize_pybullet(  # type: ignore
        using_gui=False))
print(f"Shadow robot: {shadow_robot}")
print(f"Shadow robot action space: {shadow_robot.action_space}")

print("\nApplying 5 forward movements (dx=+0.1 each)...")
current_state = state

for i in range(5):
    current_x = current_state.get(robot_obj, "x")
    current_y = current_state.get(robot_obj, "y")
    current_z = current_state.get(robot_obj, "z")

    # Get tilt and wrist if available
    current_tilt = current_state.get(
        robot_obj, "tilt") if "tilt" in robot_obj.type.feature_names else 0.0
    current_wrist = current_state.get(
        robot_obj, "wrist") if "wrist" in robot_obj.type.feature_names else 0.0

    # Create target pose (move forward by 0.1)
    target_x = current_x + 0.1
    target_y = current_y
    target_z = current_z

    current_orn = p.getQuaternionFromEuler([0, current_tilt, current_wrist])
    target_orn = p.getQuaternionFromEuler([0, current_tilt, current_wrist])

    current_pose = Pose((current_x, current_y, current_z), current_orn)
    target_pose = Pose((target_x, target_y, target_z), target_orn)

    print(f"\n[Step {i+1}]")
    print(f"  Current: ({current_x:.3f}, {current_y:.3f}, {current_z:.3f})")
    print(f"  Target:  ({target_x:.3f}, {target_y:.3f}, {target_z:.3f})")

    # Generate action using IK
    try:
        action = get_move_end_effector_to_pose_action(
            robot=shadow_robot,
            current_joint_positions=current_state.joint_positions,
            current_pose=current_pose,
            target_pose=target_pose,
            finger_status="open",
            max_vel_norm=CFG.pybullet_max_vel_norm,
            finger_action_nudge_magnitude=1e-3,
            validate=CFG.pybullet_ik_validate,
        )
        print("  ✓ IK succeeded")
        joint_delta = action.arr - np.array(current_state.joint_positions)
        print(f"  Joint delta (first 5): {joint_delta[:5]}")
        print(f"  Max joint delta: {np.max(np.abs(joint_delta)):.4f}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"  ✗ IK failed: {e}")
        traceback.print_exc()
        break

    # Apply action to environment
    current_state = env.step(action)

# Final position
final_x = current_state.get(robot_obj, "x")
final_y = current_state.get(robot_obj, "y")
final_z = current_state.get(robot_obj, "z")

print("\n" + "=" * 60)
print("RESULTS:")
print("=" * 60)
print(f"Initial position: ({initial_x:.3f}, {initial_y:.3f}, {initial_z:.3f})")
print(f"Final position:   ({final_x:.3f}, {final_y:.3f}, {final_z:.3f})")

total_dx = final_x - initial_x
total_dy = final_y - initial_y
total_dz = final_z - initial_z
print(
    f"Total movement:   dx={total_dx:.3f}, dy={total_dy:.3f}, dz={total_dz:.3f}"
)
print("Expected movement: dx~0.5 (5 steps × 0.1, limited by max_vel_norm)")

if abs(total_dx) < 0.01 and abs(total_dy) < 0.01 and abs(total_dz) < 0.01:
    print("\n✗ ERROR: Robot barely moved! Possible issues:")
    print("  - IK is failing silently")
    print("  - Actions are not being applied to environment")
    print("  - max_vel_norm constraint too small")
elif total_dx > 0.1:
    print("\n✓ Robot moved forward as expected!")
else:
    print("\n⚠ Robot moved but not in expected direction")

print("\nTest complete!")
