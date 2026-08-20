"""Legacy option implementations for the pybullet_coffee environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Dict, Sequence, Set
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_coffee import PyBulletCoffeeEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type

if TYPE_CHECKING:
    from predicators.ground_truth_models.coffee.options import \
        CoffeeGroundTruthOptionFactory
    _MixinBase = CoffeeGroundTruthOptionFactory
else:
    _MixinBase = object


class _PyBulletCoffeeLegacyOptionsMixin(_MixinBase):
    """Legacy option implementations, mixed into
    PyBulletCoffeeGroundTruthOptionFactory."""

    # Declare attributes provided by the concrete class that uses this mixin.
    env_cls: ClassVar[TypingType[PyBulletCoffeeEnv]]
    pick_policy_tol: ClassVar[float]
    _finger_action_nudge_magnitude: ClassVar[float]

    @classmethod
    def _get_options_legacy(cls, env_name: str, types: Dict[str, Type],
                            predicates: Dict[str, Predicate],
                            action_space: Box) -> Set[ParameterizedOption]:
        """Legacy option implementations."""
        options = super().get_options(  # type: ignore[misc]
            env_name, types, predicates, action_space)

        _, pybullet_robot, _ = \
            PyBulletCoffeeEnv.initialize_pybullet(using_gui=False)

        robot_type = types["robot"]
        jug_type = types["jug"]
        machine_type = types["coffee_machine"]
        plug_type = types["plug"]

        if CFG.coffee_machine_has_plug:
            PluggedIn = predicates["PluggedIn"]

        if not CFG.coffee_use_pixelated_jug:
            # TwistJug
            def _TwistJug_terminal(state: State, memory: Dict,
                                   objects: Sequence[Object],
                                   params: Array) -> bool:
                del memory, params  # unused
                robot, _ = objects
                # return HandEmpty.holds(state, [robot])
                # modify to stop at the beginning state
                robot_pose = [
                    state.get(robot, "x"),
                    state.get(robot, "y"),
                    state.get(robot, "z"),
                ]
                robot_wrist = state.get(robot, "wrist")
                robot_tilt = state.get(robot, "tilt")
                robot_finger = state.get(robot, "fingers")
                return np.allclose(robot_pose, [cls.env_cls.robot_init_x,
                                                cls.env_cls.robot_init_y,
                                                cls.env_cls.robot_init_z],
                                                atol=1e-2) and \
                    np.allclose([robot_wrist, robot_tilt, robot_finger],
                                [cls.env_cls.robot_init_wrist,
                                    cls.env_cls.robot_init_tilt,
                                    cls.env_cls.open_fingers],
                                                atol=1e-2)

            TwistJug = ParameterizedOption(
                "TwistJug",
                types=[robot_type, jug_type],
                # The parameter is a normalized amount to twist by.
                params_space=Box(-1, 1, (1 if CFG.coffee_twist_sampler else
                                         0, )),  # temp; originally 1
                policy=cls._create_twist_jug_policy(),
                initiable=lambda s, m, o, p: True,
                terminal=_TwistJug_terminal,
            )
            # Rewrite by removing and adding
            options.remove(TwistJug)
            options.add(TwistJug)

            if CFG.coffee_combined_move_and_twist_policy:
                # Get from the options MoveToTwistJug
                _MoveToTwistJug = utils.get_parameterized_option_by_name(
                    options, "MoveToTwistJug")
                assert _MoveToTwistJug is not None
                options.remove(_MoveToTwistJug)
                options.remove(TwistJug)

                Twist = utils.LinearChainParameterizedOption(
                    "Twist", [_MoveToTwistJug, TwistJug])
                options.add(Twist)

        if CFG.coffee_move_back_after_place_and_push:

            def _MoveBackAfterPlaceOrPush_terminal(state: State, memory: Dict,
                                                   objects: Sequence[Object],
                                                   params: Array) -> bool:
                del memory, params
                robot = objects[0]
                # y = state.get(robot, "y")
                x = state.get(robot, "x")
                y = state.get(robot, "y")
                z = state.get(robot, "z")
                robot_pos = (x, y, z)
                # target_x = cls.env_cls.robot_init_x
                target_x = x
                target_y = cls.env_cls.y_lb + 0.1
                # target_z = cls.env_cls.robot_init_z
                target_z = z
                target_pos = (target_x, target_y, target_z)
                return np.allclose(robot_pos, target_pos, atol=1e-2)

            MoveBackAfterPlace = ParameterizedOption(
                "MoveBackAfterPlace",
                types=[robot_type, jug_type, machine_type],
                params_space=Box(0, 1, (0, )),
                policy=cls._create_move_back_after_place_or_push_policy(),
                initiable=lambda s, m, o, p: True,
                terminal=_MoveBackAfterPlaceOrPush_terminal,
            )

            MoveBackAfterPush = ParameterizedOption(
                "MoveBackAfterPush",
                types=[robot_type, machine_type],
                params_space=Box(0, 1, (0, )),
                policy=cls._create_move_back_after_place_or_push_policy(),
                initiable=lambda s, m, o, p: True,
                terminal=_MoveBackAfterPlaceOrPush_terminal,
            )

            _TurnMachineOn = utils.get_parameterized_option_by_name(
                options, "TurnMachineOn")
            _PlaceJugInMachine = utils.get_parameterized_option_by_name(
                options, "PlaceJugInMachine")
            assert _TurnMachineOn is not None
            assert _PlaceJugInMachine is not None
            options.remove(_PlaceJugInMachine)
            options.remove(_TurnMachineOn)

            PlaceJugInMachine = utils.LinearChainParameterizedOption(
                "PlaceJugInMachine", [_PlaceJugInMachine, MoveBackAfterPlace])

            TurnMachineOn = utils.LinearChainParameterizedOption(
                "TurnMachineOn", [_TurnMachineOn, MoveBackAfterPush])
            options.add(PlaceJugInMachine)
            options.add(TurnMachineOn)

        if CFG.coffee_machine_has_plug:

            def _Restore_terminal(state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> bool:
                del memory, params
                robot = objects[0]
                robot_pos = (state.get(robot, "x"), state.get(robot, "y"),
                             state.get(robot, "z"))
                robot_init_pos = (cls.env_cls.robot_init_x,
                                  cls.env_cls.robot_init_y,
                                  cls.env_cls.robot_init_z)
                return bool(np.allclose(robot_pos, robot_init_pos, atol=1e-2))

            RestoreForPlugIn = ParameterizedOption(
                "RestoreForPlugIn",
                types=[robot_type, plug_type],
                params_space=Box(0, 1, (0, )),
                policy=cls._create_move_to_initial_position_policy(),
                initiable=lambda s, m, o, p: True,
                terminal=_Restore_terminal)

            # Plug in the plug to the socket
            def _PlugIn_initiable(state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> bool:
                del memory, params
                robot, _ = objects
                finger_open = state.get(robot, "fingers") > 0.03
                return finger_open

            def _PlugIn_terminal(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
                del memory, params
                robot, plug = objects
                finger_open = state.get(robot, "fingers") > 0.03
                return PluggedIn.holds(state, [plug]) and finger_open

            _PlugIn = ParameterizedOption("PlugIn",
                                          types=[robot_type, plug_type],
                                          params_space=Box(0, 1, (0, )),
                                          policy=cls._create_plug_in_policy(),
                                          initiable=_PlugIn_initiable,
                                          terminal=_PlugIn_terminal)

            PlugIn = utils.LinearChainParameterizedOption(
                "PlugIn", [RestoreForPlugIn, _PlugIn, RestoreForPlugIn])
            options.add(PlugIn)

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
    def _create_plug_in_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            """This works by first rotate the gripper by 90 degrees, then move
            the gripper to the plug, then close the fingers to pick 1) Rotate,
            2) Pick up, 3) Rotate back, 4) Plug in, 5) Place."""
            del memory, params

            robot, plug = objects
            target_wrist = cls.env_cls.robot_init_wrist

            # 5) When it has been plugged in, open the finger
            plugged_in = state.get(plug, "plugged_in")
            if plugged_in > 0.5:
                return cls._get_place_action(state)

            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            wrist = state.get(robot, "wrist")
            robot_pos = (x, y, z)
            finger = state.get(robot, "fingers")
            gripper_open = finger > 0.03

            plug_x = state.get(plug, "x")
            plug_y = state.get(plug, "y")
            plug_z = state.get(plug, "z")
            plug_pos = (plug_x, plug_y, plug_z)
            sq_dist_to_plug = np.sum(np.subtract(plug_pos, robot_pos)**2)

            # When it's close, pick it up
            if sq_dist_to_plug < cls.pick_policy_tol:
                # 2) Pick up
                if gripper_open:
                    return cls._get_pick_action(state)
                # 3) Rotate back & 4) Plug in.
                # After grasping, move to the socket
                socket_pos = (cls.env_cls.socket_x, cls.env_cls.socket_y,
                              cls.env_cls.socket_z)
                # Adding a waypoint to avoid collision
                waypoint = (cls.env_cls.plug_x, cls.env_cls.dispense_area_y,
                            cls.env_cls.socket_z)
                # sq_dist_to_way_point = np.sum(np.subtract(waypoint,
                #                                           robot_pos)**2)
                xz_distance = np.sum(
                    np.subtract(
                        (x, z),
                        (cls.env_cls.socket_x, cls.env_cls.socket_z))**2)
                if xz_distance > 0.01:  # and \
                    # sq_dist_to_way_point > cls.env_cls.pick_policy_tol:
                    target_robot_pos = waypoint
                else:
                    target_robot_pos = socket_pos
                # Rotate back to init orientation
                dwrist = np.clip(target_wrist - wrist,
                                -cls.env_cls.max_angular_vel,
                                cls.env_cls.max_angular_vel) /\
                        cls.env_cls.max_angular_vel

                return cls._get_move_action(state,
                                            target_robot_pos,
                                            robot_pos,
                                            finger_status="closed",
                                            dwrist=dwrist)

            # When moving toward the plug, first map to the correct x-y location
            # at the initial height(z), then move down to pick up the plug
            xy_sq_dist = np.sum(np.subtract(plug_pos[:2], robot_pos[:2])**2)
            if xy_sq_dist > 0.01:
                target_robot_pos = (plug_x, plug_y, cls.env_cls.socket_z)
            else:
                target_robot_pos = plug_pos
            # When the gripper is far away from the plug, move to it
            target_wrist = 0
            dwrist = np.clip(target_wrist - wrist,
                             -cls.env_cls.max_angular_vel,
                             cls.env_cls.max_angular_vel) /\
                    cls.env_cls.max_angular_vel

            # 2) Pick up
            return cls._get_move_action(state,
                                        target_robot_pos,
                                        robot_pos,
                                        finger_status="open",
                                        dwrist=dwrist)

        return policy

    @classmethod
    def _create_move_back_after_place_or_push_policy(
            cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params
            robot = objects[0]
            x = state.get(robot, "x")
            y = state.get(robot, "y")
            z = state.get(robot, "z")
            wrist = state.get(robot, "wrist")
            dwrist = cls.env_cls.robot_init_wrist - wrist
            robot_pos = (x, y, z)
            # target_x = cls.env_cls.robot_init_x
            target_x = x
            # target_y = cls.env_cls.robot_init_y
            target_y = cls.env_cls.y_lb + 0.1
            # target_z = cls.env_cls.robot_init_z
            target_z = z
            target_pos = (target_x, target_y, target_z)
            return cls._get_move_action(state,
                                        target_pos,
                                        robot_pos,
                                        dwrist=dwrist)

        return policy
