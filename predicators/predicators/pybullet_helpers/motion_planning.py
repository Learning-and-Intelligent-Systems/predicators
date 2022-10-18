"""Motion Planning in PyBullet."""
from __future__ import annotations

from typing import Collection, Iterator, Optional, Sequence

import numpy as np
import pybullet as p

from predicators import utils
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG


def run_motion_planning(
        robot: SingleArmPyBulletRobot, initial_positions: JointPositions,
        target_positions: JointPositions, collision_bodies: Collection[int],
        seed: int,
        physics_client_id: int) -> Optional[Sequence[JointPositions]]:
    """Run BiRRT to find a collision-free sequence of joint positions.

    Note that this function changes the state of the robot.
    """
    rng = np.random.default_rng(seed)
    joint_space = robot.action_space
    joint_space.seed(seed)
    _sample_fn = lambda _: joint_space.sample()
    num_interp = CFG.pybullet_birrt_extend_num_interp

    def _extend_fn(pt1: JointPositions,
                   pt2: JointPositions) -> Iterator[JointPositions]:
        pt1_arr = np.array(pt1)
        pt2_arr = np.array(pt2)
        num = int(np.ceil(max(abs(pt1_arr - pt2_arr)))) * num_interp
        if num == 0:
            yield pt2
        for i in range(1, num + 1):
            yield list(pt1_arr * (1 - i / num) + pt2_arr * i / num)

    def _collision_fn(pt: JointPositions) -> bool:
        robot.set_joints(pt)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        for body in collision_bodies:
            if p.getContactPoints(robot.robot_id,
                                  body,
                                  physicsClientId=physics_client_id):
                return True
        return False

    def _distance_fn(from_pt: JointPositions, to_pt: JointPositions) -> float:
        from_ee = robot.forward_kinematics(from_pt)
        to_ee = robot.forward_kinematics(to_pt)
        return sum(np.subtract(from_ee, to_ee)**2)

    birrt = utils.BiRRT(_sample_fn,
                        _extend_fn,
                        _collision_fn,
                        _distance_fn,
                        rng,
                        num_attempts=CFG.pybullet_birrt_num_attempts,
                        num_iters=CFG.pybullet_birrt_num_iters,
                        smooth_amt=CFG.pybullet_birrt_smooth_amt)

    return birrt.query(initial_positions, target_positions)
