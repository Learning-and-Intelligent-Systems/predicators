"""Legacy option implementations for the fan environment."""

from functools import lru_cache
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_fan import PyBulletFanEnv
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


class _FanLegacyOptionsMixin:
    # Declare attributes provided by the concrete class that uses this mixin.
    env_cls: ClassVar[TypingType[PyBulletFanEnv]]
    _move_to_pose_tol: ClassVar[float]
    _finger_action_nudge_magnitude: ClassVar[float]
    _hand_empty_move_z: ClassVar[float]
    _y_offset: ClassVar[float]
    """Legacy option implementations, mixed into the main factory class."""

    @classmethod
    def _get_options_legacy(cls, env_name: str, types: Dict[str, Type],
                            predicates: Dict[str, Predicate],
                            action_space: Box) -> Set[ParameterizedOption]:
        """Legacy option implementations."""
        del env_name, predicates  # unused

        _, pybullet_robot, _ = \
            PyBulletFanEnv.initialize_pybullet(using_gui=False)

        # Types
        robot_type = types["robot"]
        switch_type = types["switch"]
        fan_type = types["fan"]

        def get_current_fingers(state: State) -> float:
            robot, = state.get_objects(robot_type)
            return PyBulletFanEnv._fingers_state_to_joint(  # pylint: disable=protected-access
                pybullet_robot, state.get(robot, "fingers"))

        def _open_fingers_func(state: State, objects: Sequence[Object],
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

        options: Set[ParameterizedOption] = set()

        # Common parameters for switch options
        if CFG.fan_known_controls_relation:
            control_obj_type = fan_type
        else:
            control_obj_type = switch_type
        option_type = [robot_type, control_obj_type]
        params_space = Box(0, 1, (0, ))
        behind_factor = 1.8
        push_factor = 0
        push_above_factor = 1.3

        if CFG.fan_combine_switch_on_off:
            # Combined SwitchOnOff option (original implementation)
            SwitchOnOff = utils.LinearChainParameterizedOption(
                "SwitchOnOff", [
                    create_change_fingers_option(
                        pybullet_robot, "CloseFingers", option_type,
                        params_space, close_fingers_func,
                        CFG.pybullet_max_vel_norm,
                        PyBulletFanEnv.grasp_tol_small),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveToAboveAndBehindSwitch",
                        lambda y: y - cls._y_offset * behind_factor,
                        lambda _: cls._hand_empty_move_z, "closed",
                        option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveToBehindSwitch",
                        lambda y: y - cls._y_offset * behind_factor, lambda z:
                        z + cls.env_cls.switch_height * push_above_factor,
                        "closed", option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "PushSwitchOn",
                        lambda y: y - cls._y_offset * push_factor, lambda z: z
                        + cls.env_cls.switch_height * push_above_factor,
                        "closed", option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveToAboveAndInFrontOfSwitch",
                        lambda y: y - cls._y_offset * push_factor,
                        lambda _: cls._hand_empty_move_z, "closed",
                        option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveToInFrontOfSwitch",
                        lambda y: y + cls._y_offset * behind_factor, lambda z:
                        z + cls.env_cls.switch_height * push_above_factor,
                        "closed", option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "PushSwitchOff",
                        lambda y: y + cls._y_offset * push_factor, lambda z: z
                        + cls.env_cls.switch_height * push_above_factor,
                        "closed", option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveBack",
                        lambda y: y + cls._y_offset * behind_factor,
                        lambda _: cls._hand_empty_move_z, "closed",
                        option_type, params_space, switch_type),
                ])
            options.add(SwitchOnOff)
        else:
            # Separate SwitchOn and SwitchOff options
            SwitchOn = utils.LinearChainParameterizedOption(
                "SwitchOn",
                [
                    create_change_fingers_option(
                        pybullet_robot, "CloseFingers", option_type,
                        params_space, close_fingers_func,
                        CFG.pybullet_max_vel_norm,
                        PyBulletFanEnv.grasp_tol_small),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveToAboveAndBehindSwitch",
                        lambda y: y - cls._y_offset * behind_factor,
                        lambda _: cls._hand_empty_move_z, "closed",
                        option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveToBehindSwitch",
                        lambda y: y - cls._y_offset * behind_factor, lambda z:
                        z + cls.env_cls.switch_height * push_above_factor,
                        "closed", option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "PushSwitchOn",
                        lambda y: y - cls._y_offset * push_factor, lambda z: z
                        + cls.env_cls.switch_height * push_above_factor,
                        "closed", option_type, params_space, switch_type),
                    # cls._create_fan_move_to_push_switch_option(  # noqa
                    #     "MoveBack",
                    #     lambda y: y + cls._y_offset * behind_factor,
                    #     lambda _: cls._hand_empty_move_z,
                    #     "closed", option_type,
                    #     params_space, switch_type),
                ])
            options.add(SwitchOn)

            SwitchOff = utils.LinearChainParameterizedOption(
                "SwitchOff",
                [
                    create_change_fingers_option(
                        pybullet_robot, "CloseFingers", option_type,
                        params_space, close_fingers_func,
                        CFG.pybullet_max_vel_norm,
                        PyBulletFanEnv.grasp_tol_small),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveToAboveAndInFrontOfSwitch",
                        lambda y: y - cls._y_offset * push_factor,
                        lambda _: cls._hand_empty_move_z, "closed",
                        option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "MoveToInFrontOfSwitch",
                        lambda y: y + cls._y_offset * behind_factor, lambda z:
                        z + cls.env_cls.switch_height * push_above_factor,
                        "closed", option_type, params_space, switch_type),
                    cls._create_fan_move_to_push_switch_option(
                        "PushSwitchOff",
                        lambda y: y + cls._y_offset * push_factor, lambda z: z
                        + cls.env_cls.switch_height * push_above_factor,
                        "closed", option_type, params_space, switch_type),
                    # cls._create_fan_move_to_push_switch_option(  # noqa
                    #     "MoveBack",
                    #     lambda y: y + cls._y_offset * behind_factor,
                    #     lambda _: cls._hand_empty_move_z,
                    #     "closed", option_type,
                    #     params_space, switch_type),
                ])
            options.add(SwitchOff)

        # Wait
        params_space = Box(0, 1, (0, ))

        def _create_wait_policy() -> ParameterizedPolicy:
            nonlocal action_space

            def _policy(state: State, memory: Dict, objects: Sequence[Object],
                        params: Array) -> Action:
                del memory, params
                robot = objects[0]
                nonlocal action_space
                # check finger open or closed
                finger = state.get(robot, "fingers")
                mid_point = (pybullet_robot.open_fingers +
                             pybullet_robot.closed_fingers) / 2
                if finger > mid_point:
                    # currently open
                    finger_delta = cls._finger_action_nudge_magnitude
                else:
                    finger_delta = -cls._finger_action_nudge_magnitude

                # nudge finger to the direction of the current state to counter
                assert isinstance(state, utils.PyBulletState)
                joint_positions = state.joint_positions.copy()
                finger_position = joint_positions[
                    pybullet_robot.left_finger_joint_idx]
                # The finger action is an absolute joint position for the
                # fingers.
                f_action = finger_position + finger_delta
                # Override the meaningless finger values in joint_action.
                joint_positions[
                    pybullet_robot.left_finger_joint_idx] = f_action
                joint_positions[
                    pybullet_robot.right_finger_joint_idx] = f_action
                # slide
                action = np.array(joint_positions, dtype=np.float32)
                action = action.clip(action_space.low,
                                     action_space.high).astype(np.float32)
                return Action(action)

            return _policy

        Wait = ParameterizedOption(
            "Wait",
            types=[robot_type],
            params_space=params_space,
            policy=_create_wait_policy(),
            initiable=lambda _1, _2, _3, _4: True,
            terminal=lambda _1, _2, _3, _4: False,
        )
        options.add(Wait)

        return options

    @classmethod
    def _create_fan_move_to_push_switch_option(
            cls, name: str, y_func: Callable[[float],
                                             float], z_func: Callable[[float],
                                                                      float],
            finger_status: str, option_types: List[Type], params_space: Box,
            switch_type: Type) -> ParameterizedOption:
        """Create a move-to-pose option for the switch environment."""

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object], params: Array) -> \
                Tuple[Pose, Pose, str]:
            assert not params
            if CFG.fan_known_controls_relation:
                robot, fan = objects
                switch = [switch for switch in state.get_objects(switch_type)
                          if state.get(switch, "controls_fan") ==\
                                state.get(fan, "facing_side")][0]
            else:
                robot, switch = objects
            current_position = (state.get(robot, "x"), state.get(robot, "y"),
                                state.get(robot, "z"))
            ee_orn = p.getQuaternionFromEuler(
                [0, state.get(robot, "tilt"),
                 state.get(robot, "wrist")])
            current_pose = Pose(current_position, ee_orn)
            target_position = (state.get(switch,
                                         "x"), y_func(state.get(switch, "y")),
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
