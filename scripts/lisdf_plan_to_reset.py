"""Create a plan that resets the robot from its current state to the initial
state of another LISDF plan. Concatenate that "reset" plan with the other
LISDF plan to create a final plan.
"""
import argparse
from pathlib import Path
from typing import Sequence, cast

import numpy as np
import pybullet as p
from lisdf.planner_output.command import JointSpacePath
from lisdf.planner_output.plan import LISDFPlan
from panda_robot_client import PandaClient  # pylint: disable=import-error

from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.robots import create_single_arm_pybullet_robot


def _get_first_joints_from_lisdf(
        plan: LISDFPlan, joint_name_ordering: Sequence[str]) -> JointPositions:
    path_commands = list(filter(lambda command: isinstance(command, JointSpacePath), plan.commands))
    assert len(path_commands) > 0
    first_path_command = path_commands[0]
    return first_path_command.waypoint_as_np_array(
        -1, joint_name_ordering).tolist()


def _concatenate_lisdf_plans(plans: Sequence[LISDFPlan]) -> LISDFPlan:
    # TODO
    import ipdb; ipdb.set_trace()


def _main(lisdf_filepath: Path,
          config_filepath: Path,
          output_filepath: Path,
          seed: int = 0) -> None:
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
    plan_init_joints = _get_first_joints_from_lisdf(plan, arm_joint_names)
    # Set up the Panda client.
    client = PandaClient(config_filepath)
    # Get the current robot state.
    current_joints = client.get_joint_positions()
    # Create a motion plan from the current state to the first state in the
    # LISDF plan. For now, ignore possible collisions with other objects.
    motion_plan = run_motion_planning(robot,
                                      current_joints,
                                      plan_init_joints,
                                      collision_bodies={table_id},
                                      seed=seed,
                                      physics_client_id=physics_client_id)
    assert motion_plan is not None
    # Concatenate the motion plan with the original plan.
    combined_plan = _concatenate_lisdf_plans([motion_plan, current_joints])
    # Write out the combined plan.
    combined_plan.write_json(output_filepath)
    print(f"Wrote out LISDFPlan to {output_filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lisdf", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    _main(args.lisdf, args.config, args.output)
