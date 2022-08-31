"""Functions that consume a plan for a BEHAVIOR robot and return functions that
can be used to step each of the actions in the plan.

This requires implementing some closed-loop control to make sure that
the states expected by the plan are actually reached.
"""

import logging
from typing import Callable, List, Tuple

import numpy as np
import pybullet as p

from predicators.behavior_utils.behavior_utils import \
    get_delta_low_level_base_action, get_delta_low_level_hand_action
from predicators.structs import Array, State

try:
    from igibson.envs.behavior_env import \
        BehaviorEnv  # pylint: disable=unused-import
except (ImportError, ModuleNotFoundError) as e:
    pass


def create_dummy_policy(
    plan: List[List[float]], _original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a dummy option policy."""
    del plan

    def dummyOptionPolicy(_state: State,
                          env: "BehaviorEnv") -> Tuple[Array, bool]:
        del env
        raise NotImplementedError

    return dummyOptionPolicy


def create_navigate_policy(
    plan: List[List[float]], original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a navigation option policy given an RRT plan,
    which is a list of 3-element lists each containing a series of (x, y, rot)
    waypoints for the robot to pass through."""

    def navigateToOptionPolicy(_state: State,
                               env: "BehaviorEnv") -> Tuple[Array, bool]:
        atol_xy = 1e-2
        atol_theta = 1e-3
        atol_vel = 1e-4

        # 1. Get current position and orientation
        current_pos = list(env.robots[0].get_position()[0:2])
        current_orn = p.getEulerFromQuaternion(
            env.robots[0].get_orientation())[2]

        expected_pos = np.array(plan[0][0:2])
        expected_orn = np.array(plan[0][2])

        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xy) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: navigation policy completed "
                             "execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = get_delta_low_level_base_action(
                env.robots[0].get_position()[2],
                tuple(original_orientation[0:2]),
                np.array(current_pos + [current_orn]), np.array(plan[0]),
                env.action_space.shape)

            # But if the corrective action is 0, take the next action
            if np.allclose(low_level_action,
                           np.zeros((env.action_space.shape[0], 1)),
                           atol=atol_vel):
                low_level_action = get_delta_low_level_base_action(
                    env.robots[0].get_position()[2],
                    tuple(original_orientation[0:2]),
                    np.array(current_pos + [current_orn]), np.array(plan[1]),
                    env.action_space.shape)
                plan.pop(0)

            return low_level_action, False

        # In this case, we're at the final position we wanted to reach.
        if len(plan) == 1:
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: navigation policy completed execution!")

        else:
            low_level_action = get_delta_low_level_base_action(
                env.robots[0].get_position()[2],
                tuple(original_orientation[0:2]), np.array(plan[0]),
                np.array(plan[1]), env.action_space.shape)
            done_bit = False

        plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return navigateToOptionPolicy


def create_grasp_policy(
    plan: List[List[float]], _original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a grasp option policy given an RRT plan, which
    is a list of 6-element lists containing a series of (x, y, z, roll, pitch,
    yaw) waypoints for the hand to pass through."""
    # Set up two booleans to be used as 'memory', as well as
    # a 'reversed' plan to be used for our option that's
    # defined below. Note that the reversed plan makes a
    # copy of the list instead of just assigning by reference,
    # and this is critical to the functioning of our option. The reversed
    # plan is necessary because RRT just gives us a plan to move our hand
    # to the grasping location, but not to getting back.
    reversed_plan = list(reversed(plan))
    plan_executed_forwards = False
    tried_closing_gripper = False

    def graspObjectOptionPolicy(_state: State,
                                env: "BehaviorEnv") -> Tuple[Array, bool]:
        nonlocal plan
        nonlocal reversed_plan
        nonlocal plan_executed_forwards
        nonlocal tried_closing_gripper
        done_bit = False

        atol_xyz = 1e-4
        atol_theta = 0.1
        atol_vel = 5e-3

        # 1. Get current position and orientation
        current_pos, current_orn_quat = p.multiplyTransforms(
            env.robots[0].parts["right_hand"].parent.parts["body"].new_pos,
            env.robots[0].parts["right_hand"].parent.parts["body"].new_orn,
            env.robots[0].parts["right_hand"].local_pos,
            env.robots[0].parts["right_hand"].local_orn,
        )
        current_orn = p.getEulerFromQuaternion(current_orn_quat)

        if (not plan_executed_forwards and not tried_closing_gripper):
            expected_pos = np.array(plan[0][0:3])
            expected_orn = np.array(plan[0][3:])
            # 2. if error is greater that MAX_ERROR
            if not np.allclose(current_pos, expected_pos,
                               atol=atol_xyz) or not np.allclose(
                                   current_orn, expected_orn, atol=atol_theta):
                # 2.a take a corrective action
                if len(plan) <= 1:
                    done_bit = False
                    plan_executed_forwards = True
                    low_level_action = np.zeros(env.action_space.shape,
                                                dtype=np.float32)
                    return low_level_action, done_bit

                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[0][0:3]),
                    np.array(plan[0][3:]),
                ))

                # But if the corrective action is 0, take the next action
                if np.allclose(
                        low_level_action,
                        np.zeros((env.action_space.shape[0], 1)),
                        atol=atol_vel,
                ):
                    low_level_action = (get_delta_low_level_hand_action(
                        env.robots[0].parts["body"],
                        np.array(current_pos),
                        np.array(current_orn),
                        np.array(plan[1][0:3]),
                        np.array(plan[1][3:]),
                    ))
                    plan.pop(0)

                return low_level_action, False

            if len(plan) <= 1:  # In this case, we're at the final position
                low_level_action = np.zeros(env.action_space.shape,
                                            dtype=float)
                done_bit = False
                plan_executed_forwards = True
            else:
                # Step thru the plan to execute placing
                # phases 1 and 2
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    plan[0][0:3],
                    plan[0][3:],
                    plan[1][0:3],
                    plan[1][3:],
                ))
                if len(plan) == 1:
                    plan_executed_forwards = True

            plan.pop(0)
            return low_level_action, done_bit

        if (plan_executed_forwards and not tried_closing_gripper):
            # Close the gripper to see if you've gotten the
            # object
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            low_level_action[16] = 1.0
            tried_closing_gripper = True
            plan = reversed_plan
            return low_level_action, False

        expected_pos = np.array(plan[0][0:3])
        expected_orn = np.array(plan[0][3:])
        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xyz) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: grasp policy completed execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = (get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                np.array(current_pos),
                np.array(current_orn),
                np.array(plan[0][0:3]),
                np.array(plan[0][3:]),
            ))

            # But if the corrective action is 0, take the next action
            if np.allclose(
                    low_level_action,
                    np.zeros((env.action_space.shape[0], 1)),
                    atol=atol_vel,
            ):
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[1][0:3]),
                    np.array(plan[1][3:]),
                ))
                plan.pop(0)

            return low_level_action, False

        if len(plan) == 1:  # In this case, we're at the final position
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: grasp policy completed execution!")

        else:
            # Grasping Phase 3: getting the hand back to
            # resting position near the robot.
            low_level_action = get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                reversed_plan[0][0:3],  # current pos
                reversed_plan[0][3:],  # current orn
                reversed_plan[1][0:3],  # next pos
                reversed_plan[1][3:],  # next orn
            )
            if len(reversed_plan) == 1:
                done_bit = True
                logging.info("PRIMITIVE: grasp policy completed execution!")

        reversed_plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return graspObjectOptionPolicy


def create_place_policy(
    plan: List[List[float]], _original_orientation: List[List[float]]
) -> Callable[[State, "BehaviorEnv"], Tuple[Array, bool]]:
    """Instantiates and returns a place option policy given an RRT plan, which
    is a list of 6-element lists containing a series of (x, y, z, roll, pitch,
    yaw) waypoints for the hand to pass through."""

    # Note that the reversed plan code below makes a
    # copy of the list instead of just assigning by reference,
    # and this is critical to the functioning of our option. The reversed
    # plan is necessary because RRT just gives us a plan to move our hand
    # to the grasping location, but not to getting back.
    reversed_plan = list(reversed(plan))
    plan_executed_forwards = False
    tried_opening_gripper = False

    def placeOntopObjectOptionPolicy(_state: State,
                                     env: "BehaviorEnv") -> Tuple[Array, bool]:
        nonlocal plan
        nonlocal reversed_plan
        nonlocal plan_executed_forwards
        nonlocal tried_opening_gripper

        done_bit = False
        atol_xyz = 0.1
        atol_theta = 0.1
        atol_vel = 2.5

        # 1. Get current position and orientation
        current_pos, current_orn_quat = p.multiplyTransforms(
            env.robots[0].parts["right_hand"].parent.parts["body"].new_pos,
            env.robots[0].parts["right_hand"].parent.parts["body"].new_orn,
            env.robots[0].parts["right_hand"].local_pos,
            env.robots[0].parts["right_hand"].local_orn,
        )
        current_orn = p.getEulerFromQuaternion(current_orn_quat)

        if (not plan_executed_forwards and not tried_opening_gripper):
            expected_pos = np.array(plan[0][0:3])
            expected_orn = np.array(plan[0][3:])

            # 2. if error is greater that MAX_ERROR
            if not np.allclose(current_pos, expected_pos,
                               atol=atol_xyz) or not np.allclose(
                                   current_orn, expected_orn, atol=atol_theta):
                # 2.a take a corrective action
                if len(plan) <= 1:
                    done_bit = False
                    plan_executed_forwards = True
                    low_level_action = np.zeros(env.action_space.shape,
                                                dtype=np.float32)
                    return low_level_action, done_bit

                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[0][0:3]),
                    np.array(plan[0][3:]),
                ))

                # But if the corrective action is 0, take the next action
                if np.allclose(
                        low_level_action,
                        np.zeros((env.action_space.shape[0], 1)),
                        atol=atol_vel,
                ):
                    low_level_action = (get_delta_low_level_hand_action(
                        env.robots[0].parts["body"],
                        np.array(current_pos),
                        np.array(current_orn),
                        np.array(plan[1][0:3]),
                        np.array(plan[1][3:]),
                    ))
                    plan.pop(0)

                return low_level_action, False

            if len(plan) <= 1:  # In this case, we're at the final position
                low_level_action = np.zeros(env.action_space.shape,
                                            dtype=float)
                done_bit = False
                plan_executed_forwards = True

            else:
                # Step thru the plan to execute placing
                # phases 1 and 2
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    plan[0][0:3],
                    plan[0][3:],
                    plan[1][0:3],
                    plan[1][3:],
                ))
                if len(plan) == 1:
                    plan_executed_forwards = True

            plan.pop(0)
            return low_level_action, done_bit

        if (plan_executed_forwards and not tried_opening_gripper):
            # Open the gripper to see if you've released the
            # object
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            low_level_action[16] = -1.0
            tried_opening_gripper = True
            plan = reversed_plan
            return low_level_action, False

        expected_pos = np.array(plan[0][0:3])
        expected_orn = np.array(plan[0][3:])
        # 2. if error is greater that MAX_ERROR
        if not np.allclose(current_pos, expected_pos,
                           atol=atol_xyz) or not np.allclose(
                               current_orn, expected_orn, atol=atol_theta):
            # 2.a take a corrective action
            if len(plan) <= 1:
                done_bit = True
                logging.info("PRIMITIVE: place policy completed execution!")
                return np.zeros(env.action_space.shape,
                                dtype=np.float32), done_bit
            low_level_action = (get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                np.array(current_pos),
                np.array(current_orn),
                np.array(plan[0][0:3]),
                np.array(plan[0][3:]),
            ))

            # But if the corrective action is 0, take the next action
            if np.allclose(
                    low_level_action,
                    np.zeros((env.action_space.shape[0], 1)),
                    atol=atol_vel,
            ):
                low_level_action = (get_delta_low_level_hand_action(
                    env.robots[0].parts["body"],
                    np.array(current_pos),
                    np.array(current_orn),
                    np.array(plan[1][0:3]),
                    np.array(plan[1][3:]),
                ))
                plan.pop(0)

            return low_level_action, False

        if len(plan) == 1:  # In this case, we're at the final position
            low_level_action = np.zeros(env.action_space.shape, dtype=float)
            done_bit = True
            logging.info("PRIMITIVE: place policy completed execution!")

        else:
            # Placing Phase 3: getting the hand back to
            # resting position near the robot.
            low_level_action = get_delta_low_level_hand_action(
                env.robots[0].parts["body"],
                reversed_plan[0][0:3],  # current pos
                reversed_plan[0][3:],  # current orn
                reversed_plan[1][0:3],  # next pos
                reversed_plan[1][3:],  # next orn
            )
            if len(reversed_plan) == 1:
                done_bit = True
                logging.info("PRIMITIVE: place policy completed execution!")

        reversed_plan.pop(0)

        # Ensure that the action is clipped to stay within the expected
        # range
        low_level_action = np.clip(low_level_action, -1.0, 1.0)
        return low_level_action, done_bit

    return placeOntopObjectOptionPolicy
