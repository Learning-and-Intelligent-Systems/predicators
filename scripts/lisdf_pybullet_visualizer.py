"""Visualize an LISDF plan in real time.

Currently specific to the blocks environment.

For safety, this script should be run before executing an LISDF plan on the
real robot. If the plan looks jerky, fast, etc., don't execute it.

Quit early with the "q" key.
"""
import argparse
import time
from pathlib import Path
from typing import cast

import numpy as np
import pybullet as p
from lisdf.plan_executor.interpolator import LinearInterpolator
from lisdf.plan_executor.lisdf_executor import LISDFPlanExecutor
from lisdf.plan_executor.robots.panda import Panda as LISDFPanda
from lisdf.planner_output.plan import LISDFPlan

from predicators import utils
from predicators.envs.pybullet_blocks import PyBulletBlocksEnv
from predicators.pybullet_helpers.camera import create_gui_connection
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot


def _main(lisdf_filepath: Path) -> None:
    """Execute LISDF Plan in Pybullet."""
    # Load the LISDF plan.
    with open(lisdf_filepath, "r", encoding="utf-8") as f:
        plan_json = f.read()
        plan = cast(LISDFPlan, LISDFPlan.from_json(plan_json))
    # Set up PyBullet.
    utils.reset_config({"blocks_block_size": 0.0505})
    physics_client_id = create_gui_connection()
    # Load table.
    table_pose = PyBulletBlocksEnv._table_pose  # pylint: disable=protected-access
    table_orientation = PyBulletBlocksEnv._table_orientation  # pylint: disable=protected-access
    table_id = p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                          useFixedBase=True,
                          physicsClientId=physics_client_id)
    p.resetBasePositionAndOrientation(table_id,
                                      table_pose,
                                      table_orientation,
                                      physicsClientId=physics_client_id)
    # Create the robot.
    robot = create_single_arm_pybullet_robot("panda", physics_client_id)
    # Set up the LISDF structs.
    initial_conf = np.array(robot.get_joints())
    lisdf_robot = LISDFPanda(initial_conf)
    plan_executor = LISDFPlanExecutor(
        robot=lisdf_robot,
        plan=plan,
        path_interpolator_cls=LinearInterpolator,
    )
    # Execute the plan.
    start_time = time.perf_counter()
    current_time = 0.0
    count = 0
    q_key = ord('q')
    while current_time < plan_executor.end_time:
        # Quit early if "q" is pressed.
        p.getKeyboardEvents(physicsClientId=physics_client_id)
        keys = p.getKeyboardEvents()
        if q_key in keys:
            break
        plan_executor.execute(current_time)
        robot.set_joints(lisdf_robot.configuration.tolist())
        if count % 5000 == 0:
            x, y, z, _ = robot.get_state()
            p.addUserDebugPoints([[x, y, z]], [[255.0, 0.0, 0.0]],
                                 pointSize=5,
                                 physicsClientId=physics_client_id)
        count += 1
        current_time = time.perf_counter() - start_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lisdf", required=True, type=Path)
    args = parser.parse_args()
    _main(args.lisdf)
