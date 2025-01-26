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
from predicators.envs.pybullet_laser import PyBulletLaserEnv
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
        PyBulletLaserEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletLaserGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletLaserEnv]] = PyBulletLaserEnv
    _move_to_pose_tol: ClassVar[float] = 1e-3
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.3
    _z_offset: ClassVar[float] = 0.1
    _x_offset: ClassVar[float] = 0.03

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_laser"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the grow environment."""
        del env_name, predicates, action_space  # unused

        _, pybullet_robot, _ = \
            PyBulletLaserEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        mirror_type = types["mirror"]
        target_type = types["target"]
        station_type = types["station"]

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletLaserEnv._fingers_state_to_joint(
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
        # PickMirror
        option_types = [robot_type, mirror_type]
        params_space = Box(0, 1, (0, ))
        PickMirror = utils.LinearChainParameterizedOption(
            "PickMirror",
            [
                # Move to far the mirror which we will grasp.
                cls._create_laser_move_to_above_mirror_option(
                    "MoveToAboveMirror",
                    lambda _: cls._transport_z, "open",
                    option_types, params_space),
                # Move down to grasp.
                cls._create_laser_move_to_above_mirror_option(
                    "MoveToGraspMirror", lambda _: cls.env_cls.piece_height +
                    cls.env_cls.z_lb - 0.02, "open", option_types,
                    params_space),
                # Close fingers
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_types, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletEnv.grasp_tol),
                # Move up
                cls._create_laser_move_to_above_mirror_option(
                    "MoveEndEffectorBackUp",
                    lambda _: cls._transport_z, "closed",
                    option_types, params_space),
            ])
        options.add(PickMirror)

        # Place
        option_types = [robot_type]
        params_space = Box(0, 1, (0, ))
        Place = utils.LinearChainParameterizedOption(
            "Place",
            [
                # Move to above the position for connecting.
                cls._create_laser_move_to_above_position_option(
                    "MoveToAboveTwoSnaps",
                    lambda _: cls._transport_z, "closed",
                    option_types, params_space),
                # Move down to connect.
                cls._create_laser_move_to_above_position_option(
                    "MoveToPlace", 
                    lambda _: cls.env_cls.piece_height + cls.env_cls.z_lb, 
                    "closed", option_types, params_space),
                # Open fingers
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_types, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletEnv.grasp_tol),
                # Move back up
                cls._create_laser_move_to_above_position_option(
                    "MoveEndEffectorBackUp",
                    lambda _: cls._transport_z, "open",
                    option_types, params_space),
            ])
        options.add(Place)

        # SwitchOn
        option_type = [robot_type, station_type]
        params_space = Box(0, 1, (0, ))
        SwitchOn = utils.LinearChainParameterizedOption(
            "SwitchOn", [
                cls._create_laser_move_to_push_switch_option(
                    "MoveToAboveAndBehindSwitch",
                    lambda x: x - cls._x_offset * 5,
                    lambda _: cls._transport_z,
                    "open", option_type,
                    params_space),
                cls._create_laser_move_to_push_switch_option(
                    "MoveToBehindSwitch", 
                    lambda x: x - cls._x_offset * 5,
                    lambda z: z + cls.env_cls.station_height/2, "open",
                    option_type, params_space),
                cls._create_laser_move_to_push_switch_option(
                    "PushSwitch", 
                    lambda x: x - cls._x_offset  ,
                    lambda z: z + cls.env_cls.station_height/2, 
                    "open",
                    option_type, params_space),
                cls._create_laser_move_to_push_switch_option(
                    "MoveBack", 
                    lambda x: x - cls._x_offset * 3,
                    lambda _: cls._transport_z, 
                    "open", option_type,
                    params_space),
            ])
        options.add(SwitchOn)

        return options

    @classmethod
    def _create_laser_move_to_push_switch_option(
            cls, name: str, x_func: Callable[[float],
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

            sx = state.get(switch, "x")
            sy = state.get(switch, "y")
            sz = state.get(switch, "z")
            srot = state.get(switch, "rot") + np.pi/2

            target_position = (x_func(sx),
                               sy,
                               z_func(sz))
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, srot])
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

    @classmethod
    def _create_laser_move_to_above_mirror_option(
            cls, name: str, z_func: Callable[[float], float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        mirror argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, mirror = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            target_position = (state.get(mirror, "x"), state.get(mirror, "y"),
                               z_func(state.get(mirror, "z")))
            mirror_orn = p.getQuaternionFromEuler([0,
                                            cls.env_cls.robot_init_tilt,
                                            state.get(mirror, "rot")+\
                                                cls.env_cls.mirror_rot_offset])
            target_pose = Pose(target_position, mirror_orn)
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

    @classmethod
    def _create_laser_move_to_above_position_option(
            cls, name: str, z_func: Callable[[float], float],
            finger_status: str, option_types: List[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        mirror argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, = objects
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            rz = state.get(robot, "z")
            current_position = (rx, ry, rz)
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)

            # TODO: this is just for demo
            target_pos = (cls.env_cls.robot_init_x,
                          cls.env_cls.y_lb + 4 * cls.env_cls.piece_width,
                          z_func(cls.env_cls.piece_height))

            # Calculate rot from lx, ly, bx, by
            target_orn = p.getQuaternionFromEuler([0,
                                            cls.env_cls.robot_init_tilt,
                                            cls.env_cls.robot_init_wrist-\
                                                cls.env_cls.mirror_rot_offset])
            target_pose = Pose(target_pos, target_orn)
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
