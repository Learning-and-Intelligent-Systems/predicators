"""Ground-truth options for the Kitchen environment."""

from typing import ClassVar, Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, ParameterizedTerminal, Predicate, State, Type

try:
    from gymnasium_robotics.utils.rotations import euler2quat, quat2euler, \
        subtract_euler
    _MJKITCHEN_IMPORTED = True
except (ImportError, RuntimeError):
    _MJKITCHEN_IMPORTED = False


class KitchenGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the Kitchen environment."""

    moveto_tol: ClassVar[float] = 0.01  # for terminating moving
    max_delta_mag: ClassVar[float] = 1.0  # don't move more than this per step
    max_push_mag: ClassVar[float] = 0.05  # for pushing forward
    # A reasonable home position for the end effector.
    home_pos: ClassVar[Pose3D] = (0.0, 0.37, 2.1)
    # Keep pushing a bit even if the On classifier holds.
    push_lr_thresh_pad: ClassVar[float] = 0.02
    push_microhandle_thresh_pad: ClassVar[float] = 0.02
    turn_knob_tol: ClassVar[float] = 0.02  # for twisting the knob

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        assert _MJKITCHEN_IMPORTED, "See kitchen.py"

        # Need to define these here because users may not have euler2quat.
        down_quat = euler2quat((-np.pi, 0.0, -np.pi / 2))
        # End effector facing forward (e.g., toward the knobs.)
        fwd_quat = euler2quat((-np.pi / 2, 0.0, -np.pi / 2))
        angled_quat = euler2quat((-3 * np.pi / 4, 0.0, -np.pi / 2))

        # Types
        gripper_type = types["gripper"]
        on_off_type = types["on_off"]
        kettle_type = types["kettle"]
        surface_type = types["surface"]
        switch_type = types["switch"]
        knob_type = types["knob"]
        hinge_door_type = types["hinge_door"]

        # Predicates
        OnTop = predicates["OnTop"]

        options: Set[ParameterizedOption] = set()

        # MoveTo
        def _MoveTo_initiable(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> bool:
            # Store the target pose.
            gripper, obj = objects
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            ox = state.get(obj, "x")
            oy = state.get(obj, "y")
            oz = state.get(obj, "z")
            dx, dy, dz = params
            target_pose = (ox + dx, oy + dy, oz + dz)
            # Turn the knobs by pushing from a "forward" position.
            init_quat = angled_quat
            if obj.is_instance(knob_type):
                target_quat = fwd_quat
            elif obj.is_instance(hinge_door_type):
                if obj.name != "hinge1":
                    target_quat = fwd_quat
                elif obj.is_instance(hinge_door_type):
                    target_quat = angled_quat
            else:
                init_quat = down_quat
                target_quat = down_quat
            memory["waypoints"] = [
                (cls.home_pos, init_quat),
                (target_pose, target_quat),
            ]
            # Moves away from handle to prevent collision.
            if obj.name == "microhandle":
                memory["waypoints"] = [
                    ((gx - 0.15, gy - 0.15, gz + 0.2), down_quat)
                ] + memory["waypoints"]
            if obj.name == "slide":
                memory["waypoints"] = [(
                    (gx, gy - 0.15, gz), down_quat)] + memory["waypoints"]
            return True

        def _MoveTo_policy(state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
            del params  # unused
            gripper = objects[0]
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            gqw = state.get(gripper, "qw")
            gqx = state.get(gripper, "qx")
            gqy = state.get(gripper, "qy")
            gqz = state.get(gripper, "qz")
            current_euler = quat2euler([gqw, gqx, gqy, gqz])
            way_pos, way_quat = memory["waypoints"][0]
            if np.allclose((gx, gy, gz), way_pos, atol=cls.moveto_tol):
                memory["waypoints"].pop(0)
                way_pos, way_quat = memory["waypoints"][0]
            dx, dy, dz = np.subtract(way_pos, (gx, gy, gz))
            target_euler = quat2euler(way_quat)
            droll, dpitch, dyaw = subtract_euler(target_euler, current_euler)
            arr = np.array([dx, dy, dz, droll, dpitch, dyaw, 0.0],
                           dtype=np.float32)
            action_mag = np.linalg.norm(arr)
            if action_mag > cls.max_delta_mag:
                scale = cls.max_delta_mag / action_mag
                arr = arr * scale
            return Action(arr)

        def _MoveTo_terminal(state: State, memory: Dict,
                             objects: Sequence[Object], params: Array) -> bool:
            del params  # unused
            gripper = objects[0]
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            return np.allclose((gx, gy, gz),
                               memory["waypoints"][-1][0],
                               atol=cls.moveto_tol)

        # Create copies just to preserve one-to-one-ness with NSRTs.
        for suffix in ["PreTurnOn", "PreTurnOff"]:
            opt = ParameterizedOption(
                f"MoveTo{suffix}",
                types=[gripper_type, on_off_type],
                # Parameter is a position to move to relative to the object.
                params_space=Box(-5, 5, (3, )),
                policy=_MoveTo_policy,
                initiable=_MoveTo_initiable,
                terminal=_MoveTo_terminal)

            options.add(opt)

        # MoveToPrePushOnTop (different type)
        def _MoveToPrePushOnTop_initiable(state: State, memory: Dict,
                                          objects: Sequence[Object],
                                          params: Array) -> bool:
            # Store the target pose.
            gripper, obj = objects
            gx = state.get(gripper, "x")
            gy = state.get(gripper, "y")
            gz = state.get(gripper, "z")
            gripper_pose = (gx, gy - 0.1, gz + 0.1)
            ox = state.get(obj, "x")
            oy = state.get(obj, "y")
            oz = state.get(obj, "z")
            dx, dy, dz = params
            target_pose = (ox + dx, oy + dy, oz + dz)
            # Turn the knobs by pushing from a "forward" position.
            if obj.is_instance(knob_type):
                target_quat = fwd_quat
            else:
                target_quat = down_quat
            memory["waypoints"] = [
                (gripper_pose, fwd_quat),
                (cls.home_pos, angled_quat),
                (target_pose, target_quat),
            ]
            return True

        move_to_pre_push_on_top = ParameterizedOption(
            "MoveToPrePushOnTop",
            types=[gripper_type, kettle_type],
            # Parameter is a position to move to relative to the object.
            params_space=Box(-5, 5, (3, )),
            policy=_MoveTo_policy,
            initiable=_MoveToPrePushOnTop_initiable,
            terminal=_MoveTo_terminal)

        options.add(move_to_pre_push_on_top)

        # MoveToPrePullKettle (requires waypoints to avoid collisions).
        def _MoveToPrePullKettle_initiable(state: State, memory: Dict,
                                           objects: Sequence[Object],
                                           params: Array) -> bool:
            # Store the target pose.
            _, obj = objects
            ox = state.get(obj, "x")
            oy = state.get(obj, "y")
            oz = state.get(obj, "z")
            dx, dy, dz = params
            target_pose = (ox + dx, oy + dy, oz + dz)
            target_quat = down_quat
            offset = 0.25
            entry_pose = (ox + dx + offset, oy + dy, oz + dz)
            memory["waypoints"] = [
                (cls.home_pos, angled_quat),
                (entry_pose, fwd_quat),
                (target_pose, angled_quat),
                (target_pose, target_quat),
            ]
            return True

        move_to_pre_pull_kettle = ParameterizedOption(
            "MoveToPrePullKettle",
            types=[gripper_type, kettle_type],
            # Parameter is a position to move to relative to the object.
            params_space=Box(-5, 5, (3, )),
            policy=_MoveTo_policy,
            initiable=_MoveToPrePullKettle_initiable,
            terminal=_MoveTo_terminal)

        options.add(move_to_pre_pull_kettle)

        # PushObjOnObjForward
        def _PushObjOnObjForward_policy(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> Action:
            del state, memory, objects  # unused
            # The parameter is a push direction angle with respect to y.
            push_angle = params[0]
            unit_y, unit_x = np.cos(push_angle), np.sin(push_angle)
            dx = unit_x * cls.max_push_mag
            dy = unit_y * cls.max_push_mag
            arr = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return Action(arr)

        def _PushObjOnObjForward_terminal(state: State, memory: Dict,
                                          objects: Sequence[Object],
                                          params: Array) -> bool:
            del memory, params  # unused
            gripper, obj, obj2 = objects
            gripper_y = state.get(gripper, "y")
            obj_y = state.get(obj, "y")
            obj2_y = state.get(obj2, "y")
            # Terminate early if the gripper is far past either of the objects.
            if gripper_y - obj_y > 2 * cls.moveto_tol or \
               gripper_y - obj2_y > 2 * cls.moveto_tol:
                return True
            # NOTE: this stronger check was necessary at some point to deal
            # with a subtle case where this action pushes the kettle off
            # the burner when it ends. However, this stronger check often
            # doesn't terminate when the goal is set to pushing the kettle
            # onto a particular burner. So now, we just terminate
            # when the action's symbolic effects hold; we might have to
            # reinstate/incorporate this stronger check later if the issue
            # starts cropping up again.
            # return obj_y > obj2_y - cls.moveto_tol / 4.0
            return GroundAtom(OnTop, [obj, obj2]).holds(state)

        PushObjOnObjForward = ParameterizedOption(
            "PushObjOnObjForward",
            types=[gripper_type, kettle_type, surface_type],
            # Parameter is an angle for pushing forward.
            params_space=Box(-np.pi, np.pi, (1, )),
            policy=_PushObjOnObjForward_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PushObjOnObjForward_terminal)

        options.add(PushObjOnObjForward)

        # PushKettleOntoBurner
        def _PushKettleOntoBurner_initiable(state: State, memory: Dict,
                                            objects: Sequence[Object],
                                            params: Array) -> bool:
            gripper, obj, _ = objects
            memory["gripper_infront_kettle"] = False
            return _MoveTo_initiable(state, memory, [gripper, obj], params[:3])

        def _PushKettleOntoBurner_policy(state: State, memory: Dict,
                                         objects: Sequence[Object],
                                         params: Array) -> Action:
            gripper, obj, _ = objects
            if not memory["gripper_infront_kettle"]:
                # Check if the MoveTo option has terminated.
                if _MoveTo_terminal(state, memory, [gripper, obj], params[:3]):
                    memory["gripper_infront_kettle"] = True
                else:
                    return _MoveTo_policy(state, memory, [gripper, obj],
                                          params[:3])
            return _PushObjOnObjForward_policy(state, memory, objects,
                                               params[3:])

        def _PushKettleOntoBurner_terminal(state: State, memory: Dict,
                                           objects: Sequence[Object],
                                           params: Array) -> bool:
            del memory, params  # unused
            _, obj, obj2 = objects
            return GroundAtom(OnTop, [obj, obj2]).holds(state)

        PushKettleOntoBurner = ParameterizedOption(
            "PushKettleOntoBurner",
            types=[gripper_type, kettle_type, surface_type],
            # Parameter is an angle for pushing forward.
            params_space=Box(np.array([-5.0, -5.0, -5.0, -np.pi]),
                             np.array([5.0, 5.0, 5.0, np.pi]), (4, )),
            policy=_PushKettleOntoBurner_policy,
            initiable=_PushKettleOntoBurner_initiable,
            terminal=_PushKettleOntoBurner_terminal)

        options.add(PushKettleOntoBurner)

        # PullKettle
        def _PullKettle_policy(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> Action:
            del state, memory, objects  # unused
            # The parameter is a push direction angle with respect to y.
            pull_angle = params[0]
            unit_y, unit_x = np.cos(pull_angle), np.sin(pull_angle)
            dx = unit_x * cls.max_push_mag / 4.0
            dy = unit_y * cls.max_push_mag / 4.0
            arr = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return Action(arr)

        def _PullKettle_terminal(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
            del memory, params  # unused
            _, obj, obj2 = objects
            return GroundAtom(OnTop, [obj, obj2]).holds(state)

        PullKettle = ParameterizedOption(
            "PullKettle",
            types=[gripper_type, kettle_type, surface_type],
            # Parameter is an angle for pulling backward.
            params_space=Box(-np.pi, np.pi, (1, )),
            policy=_PullKettle_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PullKettle_terminal)

        options.add(PullKettle)

        # TurnOnSwitch / TurnOffSwitch
        def _TurnSwitch_initiable(state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> bool:
            del params  # unused
            # Memorize whether to push left or right based on the relative
            # position of the gripper and object when pushing starts.
            gripper, obj = objects
            ox = state.get(obj, "x")
            gx = state.get(gripper, "x")
            if gx > ox:
                sign = -1
            else:
                sign = 1
            memory["sign"] = sign
            return True

        def _TurnSwitch_policy(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> Action:
            del state, objects  # unused
            sign = memory["sign"]
            # The parameter is a push direction angle with respect to x, with
            # the sign possibly flipping the x direction.
            push_angle = params[0]
            unit_x, unit_y = np.cos(push_angle), np.sin(push_angle)
            dx = sign * unit_x * cls.max_push_mag
            dy = unit_y * cls.max_push_mag
            arr = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return Action(arr)

        def _create_TurnSwitch_terminal(
                on_or_off: str) -> ParameterizedTerminal:

            def _terminal(state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> bool:
                del params  # unused
                gripper, obj = objects
                gripper_x = state.get(gripper, "x")
                obj_x = state.get(obj, "x")
                # Terminate early if the gripper is far past the object.
                if memory["sign"] * (gripper_x - obj_x) > 5 * cls.moveto_tol:
                    return True
                # Use a more stringent threshold to avoid numerical issues.
                if on_or_off == "on":
                    return KitchenEnv.On_holds(
                        state, [obj], thresh_pad=cls.push_lr_thresh_pad)
                assert on_or_off == "off"
                return KitchenEnv.Off_holds(state, [obj],
                                            thresh_pad=cls.push_lr_thresh_pad)

            return _terminal

        # Create copies to preserve one-to-one-ness with NSRTs.
        for on_or_off in ["on", "off"]:
            name = f"Turn{on_or_off.capitalize()}Switch"
            terminal = _create_TurnSwitch_terminal(on_or_off)
            option = ParameterizedOption(
                name,
                types=[gripper_type, switch_type],
                # The parameter is a push direction angle with respect to x,
                # with the sign possibly flipping the x direction.
                params_space=Box(-np.pi, np.pi, (1, )),
                policy=_TurnSwitch_policy,
                initiable=_TurnSwitch_initiable,
                terminal=terminal)
            options.add(option)

        # TurnOnKnob
        def _TurnOnKnob_policy(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> Action:
            del state, memory, objects  # unused
            # The parameter is a push direction angle with respect to x.
            push_angle = params[0]
            unit_x, unit_y = np.cos(push_angle), np.sin(push_angle)
            dx = unit_x * cls.max_push_mag
            dy = unit_y * cls.max_push_mag
            arr = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return Action(arr)

        def _TurnOnKnob_terminal(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
            del memory, params  # unused
            gripper, obj = objects
            gripper_x = state.get(gripper, "x")
            obj_x = state.get(obj, "x")
            # Terminate early if the gripper is far past the object.
            if (gripper_x - obj_x) > 5 * cls.moveto_tol:
                return True
            # Use a more stringent threshold to avoid numerical issues.
            return KitchenEnv.On_holds(state, [obj],
                                       thresh_pad=cls.turn_knob_tol)

        TurnOnKnob = ParameterizedOption(
            "TurnOnKnob",
            types=[gripper_type, knob_type],
            # The parameter is a push direction angle with respect to x.
            params_space=Box(-np.pi, np.pi, (1, )),
            policy=_TurnOnKnob_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_TurnOnKnob_terminal)
        options.add(TurnOnKnob)

        # MoveAndTurnOnKnob
        def _MoveAndTurnOnKnob_initiable(state: State, memory: Dict,
                                         objects: Sequence[Object],
                                         params: Array) -> bool:
            del params  # unused
            gripper, obj = objects
            memory["gripper_infront_knob"] = False
            movement_params = np.array(KitchenEnv.get_pre_push_delta_pos(
                obj, "on"),
                                       dtype=np.float32)
            memory["movement_params"] = movement_params

            return _MoveTo_initiable(state, memory, [gripper, obj],
                                     movement_params)

        def _MoveAndTurnOnKnob_policy(state: State, memory: Dict,
                                      objects: Sequence[Object],
                                      params: Array) -> Action:
            gripper, obj = objects
            if not memory["gripper_infront_knob"]:
                # Check if the MoveTo option has terminated.
                if _MoveTo_terminal(state, memory, [gripper, obj], params[:3]):
                    memory["gripper_infront_knob"] = True
                else:
                    return _MoveTo_policy(state, memory, [gripper, obj],
                                          memory["movement_params"])
            return _TurnOnKnob_policy(state, memory, objects, params)

        def _MoveAndTurnOnKnob_terminal(state: State, memory: Dict,
                                        objects: Sequence[Object],
                                        params: Array) -> bool:
            del memory, params  # unused
            _, obj = objects
            # Use a more stringent threshold to avoid numerical issues.
            return KitchenEnv.On_holds(state, [obj],
                                       thresh_pad=cls.turn_knob_tol)

        MoveAndTurnOnKnob = ParameterizedOption(
            "MoveAndTurnOnKnob",
            types=[gripper_type, knob_type],
            # The parameter is a push direction angle with respect to x.
            params_space=Box(low=np.array([-np.pi]),
                             high=np.array([np.pi]),
                             shape=(1, )),
            policy=_MoveAndTurnOnKnob_policy,
            initiable=_MoveAndTurnOnKnob_initiable,
            terminal=_MoveAndTurnOnKnob_terminal)
        options.add(MoveAndTurnOnKnob)

        # TurnOffKnob
        def _TurnOffKnob_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
            del state, memory, objects  # unused
            # Push in the zy plane.
            push_angle = params[0]
            unit_z, unit_y = np.cos(push_angle), np.sin(push_angle)
            dy = unit_y * cls.max_push_mag
            dz = unit_z * cls.max_push_mag
            arr = np.array([0.0, dy, dz, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return Action(arr)

        def _TurnOffKnob_terminal(state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> bool:
            del memory, params  # unused
            gripper, obj = objects
            gripper_z = state.get(gripper, "z")
            obj_z = state.get(obj, "z")
            # Terminate early if the gripper is far past the object.
            if (gripper_z - obj_z) > 5 * cls.moveto_tol:
                return True
            # Use a more stringent threshold to avoid numerical issues.
            return KitchenEnv.Off_holds(state, [obj],
                                        thresh_pad=cls.turn_knob_tol)

        TurnOffKnob = ParameterizedOption(
            "TurnOffKnob",
            types=[gripper_type, knob_type],
            # The parameter is a push direction angle with respect to x.
            params_space=Box(-np.pi, np.pi, (1, )),
            policy=_TurnOffKnob_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_TurnOffKnob_terminal)
        options.add(TurnOffKnob)

        # PushOpen
        def _PushOpen_policy(state: State, memory: Dict,
                             objects: Sequence[Object],
                             params: Array) -> Action:
            del state, memory, objects  # unused
            # The parameter is a push direction angle with respect to x.
            push_angle = params[0]
            unit_x, unit_y = np.cos(push_angle), np.sin(push_angle)
            dx = unit_x * cls.max_push_mag / 2.0
            dy = unit_y * cls.max_push_mag / 2.0
            arr = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, -1.0],
                           dtype=np.float32)
            return Action(arr)

        def _PushOpen_terminal(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> bool:
            del memory, params  # unused
            _, obj = objects
            # Use a more stringent threshold to avoid numerical issues.
            return KitchenEnv.Open_holds(
                state, [obj], thresh_pad=cls.push_microhandle_thresh_pad)

        PushOpen = ParameterizedOption(
            "PushOpen",
            types=[gripper_type, on_off_type],
            # The parameter is a push direction angle with respect to x.
            params_space=Box(-np.pi, np.pi, (1, )),
            policy=_PushOpen_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PushOpen_terminal)
        options.add(PushOpen)

        # PushClose
        def _PushClose_policy(state: State, memory: Dict,
                              objects: Sequence[Object],
                              params: Array) -> Action:
            del state, memory, objects  # unused
            # The parameter is a push direction angle with respect to x.
            push_angle = params[0]
            unit_x, unit_y = np.cos(push_angle), np.sin(push_angle)
            dx = unit_x * cls.max_push_mag
            dy = unit_y * cls.max_push_mag
            arr = np.array([dx, dy, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            return Action(arr)

        def _PushClose_terminal(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> bool:
            del memory, params  # unused
            _, obj = objects
            # Use a more stringent threshold to avoid numerical issues.
            return KitchenEnv.Closed_holds(
                state, [obj], thresh_pad=cls.push_microhandle_thresh_pad)

        PushClose = ParameterizedOption(
            "PushClose",
            types=[gripper_type, hinge_door_type],
            # The parameter is a push direction angle with respect to x.
            params_space=Box(-np.pi, np.pi, (1, )),
            policy=_PushClose_policy,
            initiable=lambda _1, _2, _3, _4: True,
            terminal=_PushClose_terminal)
        options.add(PushClose)

        return options
