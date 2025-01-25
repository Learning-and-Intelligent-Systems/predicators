"""Ground-truth options for the coffee environment."""

import logging
from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.envs.pybullet_float import PyBulletFloatEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


@lru_cache
def _get_pybullet_robot() -> SingleArmPyBulletRobot:
    _, pybullet_robot, _ = \
        PyBulletFloatEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletFloatGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletFloatEnv]] = PyBulletFloatEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.25
    _hand_empty_z: ClassVar[float] = env_cls.z_ub - 0.1
    _offset_z: ClassVar[float] = 0.01

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_float"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the grow environment."""
        del env_name, predicates, action_space  # unused

        _, pybullet_robot, _ = \
            PyBulletFloatEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        block_type = types["block"]
        vessel_type = types["vessel"]

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletFloatEnv._fingers_state_to_joint(
                pybullet_robot, state.get(robot, "fingers"))

        def open_fingers_func(state: State, objects: Sequence[Object],
                              params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.open_fingers
            return current, target

        def close_fingers_func(state: State, objects: Sequence[Object],
                               params: Array) -> Tuple[float, float]:
            del objects, params  # unused
            current = get_current_fingers(state)
            target = pybullet_robot.closed_fingers
            return current, target

        options = set()
        # PickBlock
        option_types = [robot_type, block_type]
        params_space = Box(0, 1, (0, ))
        PickBlock = utils.LinearChainParameterizedOption(
            "PickBlock",
            [
                # Move to far the block which we will grasp.
                cls._create_float_move_to_above_block_option(
                    "MoveToAboveBlock", lambda _: cls._transport_z, "open",
                    option_types, params_space),
                # Move down to grasp.
                cls._create_float_move_to_above_block_option(
                    "MoveToGraspBlock",
                    lambda block_z: block_z + cls._offset_z, 
                    "open",
                    option_types, params_space),
                # Close fingers
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_types, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletEnv.grasp_tol),
                # Move up
                cls._create_float_move_to_above_block_option(
                    "MoveEndEffectorBackUp", lambda _: cls._transport_z,
                    "closed", option_types, params_space),
            ])
        options.add(PickBlock)

        # Drop
        option_types = [robot_type, vessel_type]
        params_space = Box(0, 1, (0, ))
        Drop = utils.LinearChainParameterizedOption(
            "Drop",
            [
                # Move to above the position for connecting.
                cls._create_float_move_to_above_vessel_option(
                    "MoveToAboveVessel", lambda _: cls._transport_z,
                    "closed", option_types, params_space),
                # Open fingers
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletEnv.grasp_tol),
                # Move to above the position for connecting.
                cls._create_float_move_to_above_vessel_option(
                    "MoveHigher", lambda _: cls._hand_empty_z,
                    "open", option_types, params_space),
            ])
        options.add(Drop)

        return options

    @classmethod
    def _create_float_move_to_above_block_option(
            cls, name: str, z_func: Callable[[float], float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        block argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, block = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            target_position = (state.get(block, "x"), state.get(block, "y"),
                               z_func(state.get(block, "z")))
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, cls.env_cls.robot_init_wrist])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            _get_pybullet_robot(), name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate
            )

    @classmethod
    def _create_float_move_to_above_vessel_option(
            cls, name: str, z_func: Callable[[float], float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        block argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, vessel = objects
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            rz = state.get(robot, "z")
            current_position = (rx, ry, rz)
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)

            target_x = state.get(vessel, "x") +\
                cls.env_cls.CONTAINER_OPENING_LEN/2
            target_y = state.get(vessel, "y")
            target_z = z_func(state.get(vessel, "z"))
            target_pos = (target_x, target_y, target_z)
            # Calculate rot from lx, ly, bx, by
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, cls.env_cls.robot_init_wrist])
            target_pose = Pose(target_pos, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            _get_pybullet_robot(), name, option_types, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate
            )
