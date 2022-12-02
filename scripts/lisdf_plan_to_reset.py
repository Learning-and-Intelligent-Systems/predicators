"""Create a command that resets the robot from its current state to the initial
state of another LISDF plan.

Prepend that "reset" command with the other LISDF plan to create a final plan.
"""
import argparse
from pathlib import Path
from typing import Dict, cast

import numpy as np
import pybullet as p
from lisdf.planner_output.command import JointSpacePath
from lisdf.planner_output.plan import LISDFPlan
from panda_robot_client import PandaClient  # pylint: disable=import-error

from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot
from scripts.eval_trajectory_to_lisdf import create_path_command, \
    joints_to_arm_gripper_vals


def _get_first_joints_from_lisdf(plan: LISDFPlan) -> Dict[str, float]:
    path_commands = list(
        filter(lambda command: isinstance(command, JointSpacePath),
               plan.commands))
    assert len(path_commands) > 0
    first_path_command = cast(JointSpacePath, path_commands[0])
    return first_path_command.waypoint(0)


def _main(lisdf_filepath: Path, output_filepath: Path, seed: int = 0) -> None:
    utils.reset_config({"seed": seed})
    # Load the LISDF plan.
    with open(lisdf_filepath, "r", encoding="utf-8") as f:
        plan_json = f.read()
        plan = cast(LISDFPlan, LISDFPlan.from_json(plan_json))
    # Set up PyBullet.
    physics_client_id = p.connect(p.DIRECT)
    table_pose = PyBulletBlocksEnv._table_pose  # pylint: disable=protected-access
    table_orientation = PyBulletBlocksEnv._table_orientation  # pylint: disable=protected-access
    table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(table_id,
                                      table_pose,
                                      table_orientation,
                                      physicsClientId=physics_client_id)
    ee_home = (PyBulletBlocksEnv.robot_init_x, PyBulletBlocksEnv.robot_init_y,
               PyBulletBlocksEnv.robot_init_z)
    robot = create_single_arm_pybullet_robot("panda", physics_client_id,
                                             ee_home)
    # Get the initial robot state in the plan.
    gripper_idxs = [robot.left_finger_joint_idx, robot.right_finger_joint_idx]
    joint_names = robot.arm_joint_names
    arm_joint_names = list(np.delete(joint_names, gripper_idxs))
    plan_init_joints = _get_first_joints_from_lisdf(plan)
    # Set up the Panda client.
    client = PandaClient()
    # Get the current robot state.
    current_joints = client.get_joint_positions()
    # Create a motion plan from the current state to the first state in the
    # LISDF plan. For now, ignore possible collisions with other objects.
    # Assume grippers are open, as that makes motion planning more difficult
    # and hence safer.
    current_joints[robot.left_finger_joint_name] = robot.open_fingers
    current_joints[robot.right_finger_joint_name] = robot.open_fingers
    plan_init_joints[robot.left_finger_joint_name] = robot.open_fingers
    plan_init_joints[robot.right_finger_joint_name] = robot.open_fingers
    current_joint_lst = [
        current_joints[joint_name] for joint_name in joint_names
    ]
    plan_init_joint_lst = [
        plan_init_joints[joint_name] for joint_name in joint_names
    ]
    motion_plan = run_motion_planning(robot,
                                      current_joint_lst,
                                      plan_init_joint_lst,
                                      collision_bodies={table_id},
                                      seed=seed,
                                      physics_client_id=physics_client_id)
    assert motion_plan is not None
    # Concatenate the motion plan with the original plan.
    arm_states = [
        joints_to_arm_gripper_vals(robot, np.array(joints), gripper_idxs)[0]
        for joints in motion_plan
    ]
    path_to_plan_init = create_path_command(arm_states, arm_joint_names,
                                            "path_to_plan_init", 0.5)
    plan.commands.insert(0, path_to_plan_init)

    # Write out the combined plan.
    plan.write_json(str(output_filepath))
    print(f"Wrote out LISDFPlan to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lisdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _main(args.lisdf, args.output)
