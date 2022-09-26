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

import dill as pkl
import numpy as np
import pybullet as p
from lisdf.planner_output.command import JointSpacePath, ActuateGripper, GripperPosition
from lisdf.planner_output.plan import LISDFPlan

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
                        default=0.1)
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        traj_data = pkl.load(f)

    ll_traj = traj_data["trajectory"]
    joint_arr = np.array([a.arr for a in ll_traj.actions], dtype=float)
    robot_name = traj_data["pybullet_robot"]
    # Create an instance of the robot with default position values so that we
    # can extract the joint names.
    physics_client_id = p.connect(p.DIRECT)
    robot = create_single_arm_pybullet_robot(robot_name, (1.35, 0.6, 0.7),
                                             (0.0, 0.0, 0.0, 1.0),
                                             physics_client_id)
    joint_names = robot.arm_joint_names
    assert len(joint_names) == joint_arr.shape[1]

    assert robot.left_finger_joint_idx < robot.right_finger_joint_idx
    del joint_names[robot.left_finger_joint_idx]
    del joint_names[robot.right_finger_joint_idx - 1]

    commands = []
    accum_action = []

    prev_gripper_position = GripperPosition.open

    for action in joint_arr:
        action = action.tolist()
        # Use right (left should be the same).
        gripper_val = action[robot.right_finger_joint_idx]
        dist_to_open = abs(gripper_val - robot.open_fingers)
        dist_to_closed = abs(gripper_val - robot.closed_fingers)

        assert robot.left_finger_joint_idx < robot.right_finger_joint_idx
        del action[robot.left_finger_joint_idx]
        del action[robot.right_finger_joint_idx - 1]
        accum_action.append(action)

        # Determine if gripper is currently closed or open
        if dist_to_closed < dist_to_open:
            current_gripper_position = GripperPosition.close
        else:
            current_gripper_position = GripperPosition.open

        # Convert gripper joint actions to open / close.
        if current_gripper_position != prev_gripper_position:
            duration = args.time_per_conf * len(accum_action)

            accum_action_np = np.array(accum_action)
            path = JointSpacePath.from_waypoints_np_array(
                accum_action_np,
                joint_names,
                duration=duration,
                label=args.input,
            )
            commands.append(path)

            accum_action = [action]

            gripper_command = ActuateGripper(configurations={
                "panda_gripper": current_gripper_position
            }, label="gripper_action")
            commands.append(gripper_command)
            prev_gripper_position = current_gripper_position

    # handle this case
    assert len(accum_action) == 1

    assert isinstance(commands[-1], ActuateGripper)
    last_path = commands[-2]
    reversed_pos = []
    for i in range(1, 5 + 1):
        reversed_pos.append(list(last_path.waypoint(-1 * i).values()))

    final_path = JointSpacePath.from_waypoints_np_array(
        np.array(reversed_pos),
        joint_names,
    )
    commands.append(final_path)
    # For now, we just convert the trajectory into one big JointSpacePath. In
    # the future, we may want to chunk it up.
    lisdf_plan = LISDFPlan(commands=commands, lisdf_problem="no_problem")
    lisdf_plan.write_json(args.output)
    print(f"Wrote out LISDFPlan to {args.output}")


if __name__ == "__main__":
    _main()
