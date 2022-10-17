"""Script for converting a saved eval trajectory into a LISDFPlan json file.

First run the pipeline to collect a trajectory. E.g.:

    python predicators/main.py --approach oracle --env pybullet_blocks \
        --seed 0 --num_test_tasks 1 --pybullet_robot panda

Then run this script:

    python scripts/eval_trajectory_to_lisdf.py \
        --input eval_trajectories/pybullet_blocks__oracle__0________task1.traj \
        --output /tmp/pybullet_blocks__oracle__0________task1.json
"""
import argparse
import itertools
from typing import List, Tuple

import dill as pkl
import numpy as np
import pybullet as p
from lisdf.planner_output.command import ActuateGripper, Command, \
    GripperPosition, JointSpacePath
from lisdf.planner_output.plan import LISDFPlan
from numpy.typing import NDArray

from predicators import utils
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot


def _main() -> None:
    utils.reset_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        type=str,
                        help="File path with a saved eval trajectory",
                        required=True)
    parser.add_argument("--output",
                        type=str,
                        help="File path to save the JointSpacePath",
                        required=True)
    parser.add_argument("--time_per_conf",
                        type=float,
                        help="Time in seconds per joint command",
                        default=0.5)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        traj_data = pkl.load(f)

    ll_traj = traj_data["trajectory"]
    joint_arr = np.array([a.arr for a in ll_traj.actions], dtype=np.float32)
    robot_name = traj_data["pybullet_robot"]
    # Create an instance of the robot so that we can extract the joint names.
    physics_client_id = p.connect(p.DIRECT)
    robot = create_single_arm_pybullet_robot(robot_name, physics_client_id)
    joint_names = robot.arm_joint_names
    assert len(joint_names) == joint_arr.shape[1]

    # Convert the joint array into a list of (arm joint state, gripper state)
    # where the gripper state is GripperPosition.open or GripperPosition.close
    # and the arm joint state is the original joints with grippers removed.
    gripper_idxs = [robot.left_finger_joint_idx, robot.right_finger_joint_idx]
    arm_joint_names = list(np.delete(joint_names, gripper_idxs))

    def _joints_to_arm_gripper_vals(
        joints: NDArray[np.float32]
    ) -> Tuple[NDArray[np.float32], GripperPosition]:
        # Determine if gripper is currently closed or open.
        # Use arbitrary gripper idx.
        gripper_val = joints[gripper_idxs[0]]
        dist_to_open = abs(gripper_val - robot.open_fingers)
        dist_to_closed = abs(gripper_val - robot.closed_fingers)
        if dist_to_closed < dist_to_open:
            gripper_state = GripperPosition.close
        else:
            gripper_state = GripperPosition.open
        # Extract the arm joint state.
        arm_state = np.delete(joints, gripper_idxs)
        return (arm_state, gripper_state)

    arm_gripper_states = list(map(_joints_to_arm_gripper_vals, joint_arr))

    # List of LISDF commands, alternating JointSpacePath and ActuateGripper.
    commands: List[Command] = []
    path_command_count = itertools.count()
    gripper_command_count = itertools.count()

    # Accumulate the arm-only actions until a change in the gripper state
    # is observed, at which point we finalize the JointSpacePath command,
    # add an ActuateGripper command, and then start a new JointSpacePath.
    accum_arm_states: List[NDArray[np.float32]] = []

    # Helper function for converting an accum_arm_states to a JointSpacePath.
    def _create_path_command(
            arm_states: List[NDArray[np.float32]]) -> JointSpacePath:
        duration = args.time_per_conf * len(arm_states)
        arm_states_np = np.array(arm_states)
        label = f"path_command{next(path_command_count)}"
        path_command = JointSpacePath.from_waypoints_np_array(
            arm_states_np,
            arm_joint_names,
            duration=duration,
            label=label,
        )
        return path_command

    # Track changes in the gripper state.
    _, prev_gripper_state = arm_gripper_states[0]

    for arm_state, current_gripper_state in arm_gripper_states:
        accum_arm_states.append(arm_state)
        # Check if a change in the gripper state has occurred.
        if current_gripper_state != prev_gripper_state:
            # Finalize the JointSpacePath until this point.
            path_command = _create_path_command(accum_arm_states)
            commands.append(path_command)
            accum_arm_states = [arm_state]
            # Create the ActuateGripper command.
            gripper_command = ActuateGripper(
                configurations={"panda_gripper": current_gripper_state},
                label=f"gripper_command{next(gripper_command_count)}")
            commands.append(gripper_command)
            prev_gripper_state = current_gripper_state

    # Finish the last JointSpacePath.
    if len(accum_arm_states) > 1:
        path_command = _create_path_command(accum_arm_states)
        commands.append(path_command)

    # Write out the commands to JSON.
    lisdf_plan = LISDFPlan(commands=commands, lisdf_problem=args.output)
    lisdf_plan.write_json(args.output)
    print(f"Wrote out LISDFPlan to {args.output}")


if __name__ == "__main__":
    _main()
