"""Ground-truth options for the stick button environment."""

from typing import Dict, Sequence, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.stick_button import StickButtonEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class StickButtonGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the stick button environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"stick_button"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        button_type = types["button"]
        stick_type = types["stick"]
        holder_type = types["holder"]

        Pressed = predicates["Pressed"]
        Grasped = predicates["Grasped"]

        # RobotPressButton
        def _RobotPressButton_terminal(state: State, memory: Dict,
                                       objects: Sequence[Object],
                                       params: Array) -> bool:
            del memory, params  # unused
            _, button, _ = objects
            return Pressed.holds(state, [button])

        RobotPressButton = ParameterizedOption(
            "RobotPressButton",
            types=[robot_type, button_type, stick_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_robot_press_button_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_RobotPressButton_terminal,
        )

        # PickStick
        def _PickStick_terminal(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> bool:
            del memory, params  # unused
            return Grasped.holds(state, objects)

        PickStick = ParameterizedOption(
            "PickStick",
            types=[robot_type, stick_type],
            params_space=Box(0, 1, (1, )),  # normalized w.r.t. stick width
            policy=cls._create_pick_stick_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_PickStick_terminal,
        )

        # StickPressButton
        def _StickPressButton_terminal(state: State, memory: Dict,
                                       objects: Sequence[Object],
                                       params: Array) -> bool:
            del memory, params  # unused
            _, _, button = objects
            return Pressed.holds(state, [button])

        StickPressButton = ParameterizedOption(
            "StickPressButton",
            types=[robot_type, stick_type, button_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_stick_press_button_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_StickPressButton_terminal,
        )

        # PlaceStick
        def _PlaceStick_terminal(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
            del memory, params  # unused
            return not Grasped.holds(state, objects)

        PlaceStick = ParameterizedOption(
            "PlaceStick",
            types=[robot_type, stick_type],
            params_space=Box(-1, 1, (1, )),
            policy=cls._create_place_stick_policy(holder_type),
            initiable=lambda s, m, o, p: True,
            terminal=_PlaceStick_terminal,
        )

        return {RobotPressButton, PickStick, StickPressButton, PlaceStick}

    @classmethod
    def _create_robot_press_button_policy(cls) -> ParameterizedPolicy:

        max_speed = StickButtonEnv.max_speed

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            # If the robot and button are already pressing, press.
            if StickButtonEnv.Above_holds(state, objects[:2]):
                return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            # Otherwise, move toward the button.
            robot, button, _ = objects
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            px = state.get(button, "x")
            py = state.get(button, "y")
            dx = np.clip(px - rx, -max_speed, max_speed)
            dy = np.clip(py - ry, -max_speed, max_speed)
            # Normalize.
            dx = dx / max_speed
            dy = dy / max_speed
            # No need to rotate, and we don't want to press until we're there.
            return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))

        return policy

    @classmethod
    def _create_pick_stick_policy(cls) -> ParameterizedPolicy:

        max_speed = StickButtonEnv.max_speed

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            robot, stick = objects
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            tx, ty = cls._get_stick_grasp_loc(state, stick, params)
            # If we're close enough to the grasp button, press.
            if (tx - rx)**2 + (ty - ry)**2 < StickButtonEnv.pick_grasp_tol:
                return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            # Move toward the target.
            dx = np.clip(tx - rx, -max_speed, max_speed)
            dy = np.clip(ty - ry, -max_speed, max_speed)
            # Normalize.
            dx = dx / max_speed
            dy = dy / max_speed
            # No need to rotate or press.
            return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))

        return policy

    @classmethod
    def _create_stick_press_button_policy(cls) -> ParameterizedPolicy:

        max_speed = StickButtonEnv.max_speed

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, stick, button = objects
            button_circ = StickButtonEnv.object_to_geom(button, state)
            stick_rect = StickButtonEnv.object_to_geom(stick, state)
            assert isinstance(stick_rect, utils.Rectangle)
            tip_rect = StickButtonEnv.stick_rect_to_tip_rect(stick_rect)
            # If the stick tip is pressing the button, press.
            if tip_rect.intersects(button_circ):
                return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            # If the stick is vertical, move the tip toward the button.
            stheta = state.get(stick, "theta")
            desired_theta = np.pi / 2
            if abs(stheta - desired_theta) < 1e-3:
                tx = tip_rect.x
                ty = tip_rect.y
                px = state.get(button, "x")
                py = state.get(button, "y")
                dx = np.clip(px - tx, -max_speed, max_speed)
                dy = np.clip(py - ty, -max_speed, max_speed)
                # Normalize.
                dx = dx / max_speed
                dy = dy / max_speed
                # No need to rotate or press.
                return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))
            assert not CFG.stick_button_disable_angles
            # Otherwise, rotate the stick.
            dtheta = np.clip(desired_theta - stheta,
                             -StickButtonEnv.max_angular_speed,
                             StickButtonEnv.max_angular_speed)
            # Normalize.
            dtheta = dtheta / StickButtonEnv.max_angular_speed
            return Action(np.array([0.0, 0.0, dtheta, -1.0], dtype=np.float32))

        return policy

    @classmethod
    def _create_place_stick_policy(cls,
                                   holder_type: Type) -> ParameterizedPolicy:

        max_speed = StickButtonEnv.max_speed

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            robot, _ = objects
            holder = state.get_objects(holder_type)[0]
            norm_offset_y, = params
            offset_y = (StickButtonEnv.stick_width / 2) * norm_offset_y
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            tx = state.get(holder, "x") - StickButtonEnv.holder_height / 2
            ty = state.get(holder, "y") + offset_y
            # If we're close enough, put the stick down.
            if (tx - rx)**2 + (ty - ry)**2 < StickButtonEnv.pick_grasp_tol:
                return Action(np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32))
            # Move toward the target.
            dx = np.clip(tx - rx, -max_speed, max_speed)
            dy = np.clip(ty - ry, -max_speed, max_speed)
            # Normalize.
            dx = dx / max_speed
            dy = dy / max_speed
            # No need to rotate or press.
            return Action(np.array([dx, dy, 0.0, -1.0], dtype=np.float32))

        return policy

    @classmethod
    def _get_stick_grasp_loc(cls, state: State, stick: Object,
                             params: Array) -> Tuple[float, float]:
        stheta = state.get(stick, "theta")
        # We always aim for the center of the shorter dimension. The params
        # selects a position along the longer dimension.
        h = StickButtonEnv.stick_height
        sx = state.get(stick, "x") + (h / 2) * np.cos(stheta + np.pi / 2)
        sy = state.get(stick, "y") + (h / 2) * np.sin(stheta + np.pi / 2)
        # Calculate the target button to reach based on the parameter.
        pick_param, = params
        scale = StickButtonEnv.stick_width * pick_param
        tx = sx + scale * np.cos(stheta)
        ty = sy + scale * np.sin(stheta)
        return (tx, ty)


class StickButtonMovementGroundTruthOptionFactory(
        StickButtonGroundTruthOptionFactory):
    """Ground-truth options for the stick button environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"stick_button_move"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # First, instantiate the original pick and place options,
        # but (1) override the policies for RobootPressButton and
        # StickPressButton to make them no longer move the robot, and (2)
        # redefine RobotPressButton to update its arguments.
        init_options = super().get_options(env_name, types, predicates,
                                           action_space)
        robot_type = types["robot"]
        button_type = types["button"]
        stick_type = types["stick"]
        holder_type = types["holder"]

        RobotAboveButton = predicates["RobotAboveButton"]
        StickAboveButton = predicates["StickAboveButton"]
        Pressed = predicates["Pressed"]
        Grasped = predicates["Grasped"]

        # RobotMoveToButton
        def _RobotMoveToButton_terminal(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> bool:
            del memory, params  # unused
            robot, button = objects
            return RobotAboveButton.holds(state, [robot, button])

        RobotMoveToButton = ParameterizedOption(
            "RobotMoveToButton",
            types=[robot_type, button_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_robot_moveto_button_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_RobotMoveToButton_terminal,
        )

        # StickMoveToButton
        def _StickMoveToButton_terminal(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> bool:
            del memory, params  # unused
            _, button, stick = objects
            return StickAboveButton.holds(state, [stick, button])

        StickMoveToButton = ParameterizedOption(
            "StickMoveToButton",
            types=[robot_type, button_type, stick_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_stick_moveto_button_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_StickMoveToButton_terminal,
        )

        # RobotPressButton
        def _RobotPressButton_terminal(state: State, memory: Dict,
                                       objects: Sequence[Object],
                                       params: Array) -> bool:
            del memory, params  # unused
            _, button = objects
            return Pressed.holds(state, [button])

        RobotPressButton = ParameterizedOption(
            "RobotPressButton",
            types=[robot_type, button_type],
            params_space=Box(0, 1, (0, )),
            policy=cls._create_robot_press_button_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_RobotPressButton_terminal,
        )

        # PlaceStick
        def _PlaceStick_terminal(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
            del memory, params  # unused
            robot, stick, _ = objects
            return not Grasped.holds(state, [robot, stick])

        PlaceStick = ParameterizedOption(
            "PlaceStick",
            types=[robot_type, stick_type, holder_type],
            params_space=Box(-1, 1, (1, )),
            policy=cls._create_place_stick_policy_diff_signature(),
            initiable=lambda s, m, o, p: True,
            terminal=_PlaceStick_terminal,
        )

        unchanged_options = {
            opt
            for opt in init_options
            if opt.name not in ["RobotPressButton", "PlaceStick"]
        }
        changed_options = {RobotPressButton, PlaceStick}
        new_options = {RobotMoveToButton, StickMoveToButton}

        return unchanged_options | changed_options | new_options

    @classmethod
    def _create_robot_moveto_button_policy(cls) -> ParameterizedPolicy:

        max_speed = StickButtonEnv.max_speed

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused.
            # Otherwise, move toward the button.
            robot, button = objects
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            px = state.get(button, "x")
            py = state.get(button, "y")
            dx = np.clip(px - rx, -max_speed, max_speed)
            dy = np.clip(py - ry, -max_speed, max_speed)
            # Normalize.
            dx = dx / max_speed
            dy = dy / max_speed
            # No need to rotate, and we don't want to press until we're there.
            return Action(np.array([dx, dy, 0.0, -1.0, -1.0],
                                   dtype=np.float32))

        return policy

    @classmethod
    def _create_stick_moveto_button_policy(cls) -> ParameterizedPolicy:

        max_speed = StickButtonEnv.max_speed

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, button, stick = objects
            stick_rect = StickButtonEnv.object_to_geom(stick, state)
            assert isinstance(stick_rect, utils.Rectangle)
            tip_rect = StickButtonEnv.stick_rect_to_tip_rect(stick_rect)
            # If the stick is vertical, move the tip toward the button.
            stheta = state.get(stick, "theta")
            desired_theta = np.pi / 2
            if abs(stheta - desired_theta) < 1e-3:
                tx = tip_rect.x
                ty = tip_rect.y
                px = state.get(button, "x")
                py = state.get(button, "y")
                dx = np.clip(px - tx, -max_speed, max_speed)
                dy = np.clip(py - ty, -max_speed, max_speed)
                # Normalize.
                dx = dx / max_speed
                dy = dy / max_speed
                # No need to rotate or press.
                return Action(
                    np.array([dx, dy, 0.0, -1.0, -1.0], dtype=np.float32))
            assert not CFG.stick_button_disable_angles
            # Otherwise, rotate the stick.
            dtheta = np.clip(desired_theta - stheta,
                             -StickButtonEnv.max_angular_speed,
                             StickButtonEnv.max_angular_speed)
            # Normalize.
            dtheta = dtheta / StickButtonEnv.max_angular_speed
            return Action(
                np.array([0.0, 0.0, dtheta, -1.0, -1.0], dtype=np.float32))

        return policy

    @classmethod
    def _create_robot_press_button_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            robot, button = objects
            action = Action(
                np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32))
            # If the robot is above the button, press.
            if StickButtonEnv.Above_holds(state, [robot, button]):
                action = Action(
                    np.array([0.0, 0.0, 0.0, 1.0, -1.0], dtype=np.float32))
            # Else, do nothing.
            return action

        return policy

    @classmethod
    def _create_stick_press_button_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory, params  # unused
            _, stick, button = objects
            button_circ = StickButtonEnv.object_to_geom(button, state)
            stick_rect = StickButtonEnv.object_to_geom(stick, state)
            assert isinstance(stick_rect, utils.Rectangle)
            tip_rect = StickButtonEnv.stick_rect_to_tip_rect(stick_rect)
            # If the stick tip is above the button, press.
            if tip_rect.intersects(button_circ):
                return Action(
                    np.array([0.0, 0.0, 0.0, 1.0, -1.0], dtype=np.float32))
            # Else, do nothing.
            return Action(
                np.array([0.0, 0.0, 0.0, -1.0, -1.0], dtype=np.float32))

        return policy

    @classmethod
    def _create_pick_stick_policy(cls) -> ParameterizedPolicy:

        max_speed = StickButtonEnv.max_speed

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            robot, stick = objects
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            tx, ty = cls._get_stick_grasp_loc(state, stick, params)
            # If we're close enough to the grasp location, pickplace.
            if (tx - rx)**2 + (ty - ry)**2 < StickButtonEnv.pick_grasp_tol:
                return Action(
                    np.array([0.0, 0.0, 0.0, -1.0, 1.0], dtype=np.float32))
            # Move toward the target.
            dx = np.clip(tx - rx, -max_speed, max_speed)
            dy = np.clip(ty - ry, -max_speed, max_speed)
            # Normalize.
            dx = dx / max_speed
            dy = dy / max_speed
            # No need to rotate or press.
            return Action(np.array([dx, dy, 0.0, -1.0, -1.0],
                                   dtype=np.float32))

        return policy

    @classmethod
    def _create_place_stick_policy_diff_signature(cls) -> ParameterizedPolicy:

        max_speed = StickButtonEnv.max_speed

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused
            robot, _, holder = objects
            norm_offset_y, = params
            offset_y = (StickButtonEnv.stick_width / 2) * norm_offset_y
            rx = state.get(robot, "x")
            ry = state.get(robot, "y")
            tx = state.get(holder, "x") - StickButtonEnv.holder_height / 2
            ty = state.get(holder, "y") + offset_y
            # If we're close enough, put the stick down.
            if (tx - rx)**2 + (ty - ry)**2 < StickButtonEnv.pick_grasp_tol:
                return Action(
                    np.array([0.0, 0.0, 0.0, -1.0, 1.0], dtype=np.float32))
            # Move toward the target.
            dx = np.clip(tx - rx, -max_speed, max_speed)
            dy = np.clip(ty - ry, -max_speed, max_speed)
            # Normalize.
            dx = dx / max_speed
            dy = dy / max_speed
            # No need to rotate or press.
            return Action(np.array([dx, dy, 0.0, -1.0, -1.0],
                                   dtype=np.float32))

        return policy
