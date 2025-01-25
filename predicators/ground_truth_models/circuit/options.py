"""Ground-truth options for the coffee environment."""

import logging
from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_circuit import PyBulletCircuitEnv
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
        PyBulletCircuitEnv.initialize_pybullet(using_gui=False)
    return pybullet_robot


class PyBulletCircuitGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the grow environment."""

    env_cls: ClassVar[TypingType[PyBulletCircuitEnv]] = PyBulletCircuitEnv
    _move_to_pose_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3
    _hand_empty_move_z: ClassVar[float] = env_cls.z_ub - 0.2
    _transport_z: ClassVar[float] = env_cls.z_ub - 0.5
    _z_offset: ClassVar[float] = 0.1
    _y_offset: ClassVar[float] = 0.03

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_circuit"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        """Get the ground-truth options for the grow environment."""
        del env_name, predicates, action_space  # unused

        _, pybullet_robot, _ = \
            PyBulletCircuitEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        wire_type = types["wire"]
        light_type = types["light"]
        battery_type = types["battery"]

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletCircuitEnv._fingers_state_to_joint(
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
        # PickWire
        option_type = [robot_type, wire_type]
        params_space = Box(0, 1, (0, ))
        PickWire = utils.LinearChainParameterizedOption(
            "PickWire",
            [
                # Move to far the wire which we will grasp.
                cls._create_circuit_move_to_above_wire_option(
                    "MoveToAboveWire",
                    lambda _: cls.env_cls.z_ub - cls._z_offset, "open",
                    option_type, params_space),
                # Move down to grasp.
                cls._create_circuit_move_to_above_wire_option(
                    "MoveToGraspWire", lambda snap_z: snap_z + 0.01, "open",
                    option_type, params_space),
                # Close fingers
                create_change_fingers_option(
                    pybullet_robot, "CloseFingers", option_type, params_space,
                    close_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletCircuitEnv.grasp_tol),
                # Move up
                cls._create_circuit_move_to_above_wire_option(
                    "MoveEndEffectorBackUp", lambda _: cls._transport_z,
                    "closed", option_type, params_space),
            ])
        options.add(PickWire)

        # Connect
        option_type = [robot_type, wire_type, light_type, battery_type]
        params_space = Box(0, 1, (0, ))
        Connect = utils.LinearChainParameterizedOption(
            "Connect",
            [
                # Move to above the position for connecting.
                cls._create_circuit_move_to_above_two_snaps_option(
                    "MoveToAboveTwoSnaps", lambda _: cls._transport_z,
                    "closed", option_type, params_space),
                # Move down to connect.
                cls._create_circuit_move_to_above_two_snaps_option(
                    "MoveToConnect", lambda snap_z: snap_z + 0.01, "closed",
                    option_type, params_space),
                # Open fingers
                create_change_fingers_option(
                    pybullet_robot, "OpenFingers", option_type, params_space,
                    open_fingers_func, CFG.pybullet_max_vel_norm,
                    PyBulletCircuitEnv.grasp_tol),
                # Move back up
                cls._create_circuit_move_to_above_two_snaps_option(
                    "MoveEndEffectorBackUp", lambda _: cls._hand_empty_move_z,
                    "open", option_type, params_space),
            ])
        options.add(Connect)

        # SwitchOn
        option_type = [robot_type, battery_type]
        params_space = Box(0, 1, (0, ))
        SwitchOn = utils.LinearChainParameterizedOption(
            "SwitchOn", [
                cls._create_circuit_move_to_push_switch_option(
                    "MoveToAboveAndBehindSwitch",
                    lambda y: y - cls._y_offset * 5,
                    lambda _: cls._hand_empty_move_z, "open", option_type,
                    params_space),
                cls._create_circuit_move_to_push_switch_option(
                    "MoveToBehindSwitch", lambda y: y - cls._y_offset * 5,
                    lambda z: z + cls.env_cls.switch_height, "open",
                    option_type, params_space),
                cls._create_circuit_move_to_push_switch_option(
                    "PushSwitch", lambda y: y - cls._y_offset * 1.8,
                    lambda z: z + cls.env_cls.switch_height, "open",
                    option_type, params_space),
                cls._create_circuit_move_to_push_switch_option(
                    "MoveBack", lambda y: y - cls._y_offset * 3,
                    lambda _: cls._hand_empty_move_z, "open", option_type,
                    params_space),
            ])
        options.add(SwitchOn)

        return options

    @classmethod
    def _create_circuit_move_to_push_switch_option(
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
            target_position = (state.get(switch, "x") + \
                        cls.env_cls.snap_width/2 + cls.env_cls.switch_width/2,
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

    @classmethod
    def _create_circuit_move_to_above_wire_option(
            cls, name: str, z_func: Callable[[float],
                                             float], finger_status: str,
            option_type: List[Type], params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        wire argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, snap = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            target_position = (state.get(snap, "x"), state.get(snap, "y"),
                               z_func(state.get(snap, "z")))
            snap_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt,
                 state.get(snap, "rot")])
            target_pose = Pose(target_position, snap_orn)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            _get_pybullet_robot(), name, option_type, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)

    @classmethod
    def _create_circuit_move_to_above_two_snaps_option(
            cls, name: str, z_func: Callable[[float],
                                             float], finger_status: str,
            option_type: List[Type], params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        wire argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose, Pose, str]:
            assert not params
            robot, wire, light, battery = objects
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            rz = state.get(robot, "z")
            current_position = (rx, ry, rz)
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)

            wy = state.get(wire, "y")
            ly = state.get(light, "y")
            lx = state.get(light, "x")
            lz = state.get(light, "z")
            bx = state.get(battery, "x")
            at_top = 1 if (wy > ly) else -1
            target_x = (lx + bx) / 2
            y_pad = 0.003 if at_top == 1 else 0
            target_y = ly + at_top * (cls.env_cls.bulb_snap_length / 2 +
                                      cls.env_cls.snap_width / 2 - y_pad)
            target_pos = (target_x, target_y, z_func(lz))
            # Calculate rot from lx, ly, bx, by
            target_orn = p.getQuaternionFromEuler(
                [0, cls.env_cls.robot_init_tilt, 0])
            # np.arctan2(by - ly, bx - lx)])
            target_pose = Pose(target_pos, target_orn)
            # logging.debug(f"Current pos:{current_position}, \ntarget pos:{target_pos}\n")
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            _get_pybullet_robot(), name, option_type, params_space,
            _get_current_and_target_pose_and_finger_status,
            cls._move_to_pose_tol, CFG.pybullet_max_vel_norm,
            cls._finger_action_nudge_magnitude)
