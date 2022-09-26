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
from lisdf.planner_output.command import JointSpacePath
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

    # For now, we just convert the trajectory into one big JointSpacePath. In
    # the future, we may want to chunk it up.
    duration = args.time_per_conf * len(joint_arr)
    command = JointSpacePath.from_waypoints_np_array(
        joint_arr,
        joint_names,
        duration=duration,
        label=args.input,
    )
    lisdf_plan = LISDFPlan(commands=[command], lisdf_problem="no_problem")
    lisdf_plan.write_json(args.output)
    print(f"Wrote out LISDFPlan to {args.output}")


if __name__ == "__main__":
    _main()
