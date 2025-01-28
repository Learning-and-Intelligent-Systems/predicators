"""Ground-truth options for the coffee environment."""

import logging
from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.coffee.options import \
    PyBulletCoffeeGroundTruthOptionFactory
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
        PyBulletFanEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletFanGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletFanEnv]] = PyBulletFanEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _hand_empty_move_z: ClassVar[float] = env_cls.z_ub - 0.3
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.5
    _z_offset: ClassVar[float] = 0.1
    _y_offset: ClassVar[float] = 0.03

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_fan"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the grow environment."""
        del env_name, predicates, action_space  # unused

        _, pybullet_robot, _ = \
            PyBulletFanEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        switch_type = types["switch"]

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletFanEnv._fingers_state_to_joint(
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

        # SwitchOnAndOff
        option_type = [robot_type, switch_type]
        params_space = Box(0, 1, (0, ))
        behind_factor = 1.8
        push_factor = 0
        push_above_factor = 1.3
        SwitchOn = utils.LinearChainParameterizedOption(
            "SwitchOnOff", [
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_type, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletFanEnv.grasp_tol_small),
                cls._create_fan_move_to_push_switch_option(
                    "MoveToAboveAndBehindSwitch",
                    lambda y: y - cls._y_offset * behind_factor,
                    lambda _: cls._hand_empty_move_z, 
                    "closed", option_type,
                    params_space),
                cls._create_fan_move_to_push_switch_option(
                    "MoveToBehindSwitch", 
                    lambda y: y - cls._y_offset * behind_factor,
                    lambda z: z + cls.env_cls.switch_height * push_above_factor, 
                    "closed",
                    option_type, params_space),
                cls._create_fan_move_to_push_switch_option(
                    "PushSwitchOn", 
                    lambda y: y - cls._y_offset * push_factor,
                    lambda z: z + cls.env_cls.switch_height * push_above_factor, 
                    "closed",
                    option_type, params_space),
                cls._create_fan_move_to_push_switch_option(
                    "MoveToAboveAndInFrontOfSwitch",
                    lambda y: y - cls._y_offset * push_factor,
                    lambda _: cls._hand_empty_move_z, 
                    "closed", option_type,
                    params_space),
                cls._create_fan_move_to_push_switch_option(
                    "MoveToInFrontOfSwitch", 
                    lambda y: y + cls._y_offset * behind_factor,
                    lambda z: z + cls.env_cls.switch_height * push_above_factor, 
                    "closed",
                    option_type, params_space),
                cls._create_fan_move_to_push_switch_option(
                    "PushSwitchOff", 
                    lambda y: y + cls._y_offset * push_factor,
                    lambda z: z + cls.env_cls.switch_height * push_above_factor, 
                    "closed",
                    option_type, params_space),
                cls._create_fan_move_to_push_switch_option(
                    "MoveBack", lambda y: y + cls._y_offset * behind_factor,
                    lambda _: cls._hand_empty_move_z, 
                    "closed", option_type,
                    params_space),
            ])
        options.add(SwitchOn)

        return options

    @classmethod
    def _create_fan_move_to_push_switch_option(
            cls, name: str, y_func: Callable[[float],
                                             float], z_func: Callable[[float],
                                                                      float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Create a move-to-pose option for the switch environment."""

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            assert not params
            robot, switch = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            target_position = (state.get(switch, "x"),
                               y_func(state.get(switch, "y")),
                               z_func(state.get(switch, "z")))
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, 0])
            target_pose = Pose(target_position, target_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            _get_pybullet_robot(),
            name,
            option_types,
            params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol,
            CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude,
            validate=CFG.pybullet_ik_validate)