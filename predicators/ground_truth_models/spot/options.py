"""Ground-truth options for PDDL environments."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot
from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv, get_robot
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.spot_utils.perception.object_detection import \
    ObjectDetectionID, detect_objects, \
    get_object_center_pixel_from_artifacts
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, \
    get_relative_se2_from_se3
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, STRIPSOperator, Type


def _get_se3_pose_from_state(state: State,
                             obj: Object) -> math_helpers.SE3Pose:
    # TODO consider moving this function.
    return math_helpers.SE3Pose(
        state.get(obj, "x"), state.get(obj, "y"), state.get(obj, "z"),
        math_helpers.Quat(state.get(obj, "W_quat"), state.get(obj, "X_quat"),
                          state.get(obj, "Y_quat"), state.get(obj, "Z_quat")))


def _create_action(action_name: str, operator_objects: Tuple[Object],
                   fn: Callable, fn_args: Tuple) -> Action:
    return Action(np.array([], dtype=np.float32),
                  extra_info=(action_name, operator_objects, fn, fn_args))


def _create_navigate_parameterized_policy(
        robot_obj_idx: int, target_obj_idx: int, distance_param_idx: int,
        yaw_param_idx: int) -> ParameterizedPolicy:

    robot, _, _ = get_robot()

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory  # not used

        distance = params[distance_param_idx]
        yaw = params[yaw_param_idx]

        robot_obj = objects[robot_obj_idx]
        robot_pose = _get_se3_pose_from_state(state, robot_obj)

        target_obj = objects[target_obj_idx]
        target_pose = _get_se3_pose_from_state(state, target_obj)

        rel_pose = get_relative_se2_from_se3(robot_pose, target_pose, distance,
                                             yaw)

        return _create_action("execute", objects, navigate_to_relative_pose,
                              (robot, rel_pose))

    return _policy


def _create_open_hand_parameterized_policy() -> ParameterizedPolicy:

    robot, _, _ = get_robot()

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del state, memory, params  # not used
        return _create_action("execute", objects, open_gripper, (robot, ))

    return _policy


def _create_move_hand_parameterized_policy(
        hand_pose: math_helpers.SE3Pose) -> ParameterizedPolicy:

    robot, _, _ = get_robot()

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del state, memory, params  # not used
        return _create_action("execute", objects, move_hand_to_relative_pose,
                              (robot, hand_pose))

    return _policy


def _create_operator_finish_parameterized_policy(
        operator_name: str) -> ParameterizedPolicy:

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del state, memory, params  # not used
        return _create_action(operator_name, objects, None, tuple())

    return _policy


class _MoveToToolOnSurfaceParameterizedOption(
        utils.LinearChainParameterizedOption):
    """Navigate to the surface and then open the hand.

    The types are [robot, tool, surface].
    """

    def __init__(self, name: str, types: List[Type]) -> None:

        # Parameters are relative distance, dyaw.
        params_space = Box(-np.inf, np.inf, (2, ))

        # Navigate to the surface.
        navigate = utils.SingletonParameterizedOption(
            "MoveToToolOnSurface-Navigate",
            _create_navigate_parameterized_policy(robot_obj_idx=0,
                                                  target_obj_idx=2,
                                                  distance_param_idx=0,
                                                  yaw_param_idx=1),
            types=types,
            params_space=params_space,
        )

        # Look down at the surface.
        move_hand = utils.SingletonParameterizedOption(
            "MoveToToolOnSurface-MoveHand",
            _create_move_hand_parameterized_policy(
                DEFAULT_HAND_LOOK_DOWN_POSE),
            types=types,
            params_space=params_space,
        )

        # Open the hand.
        open_hand = utils.SingletonParameterizedOption(
            "MoveToToolOnSurface-OpenHand",
            _create_open_hand_parameterized_policy(),
            types=types,
            params_space=params_space,
        )

        # Finish the action.
        finish = utils.SingletonParameterizedOption(
            "MoveToToolOnSurface-Finish",
            _create_operator_finish_parameterized_policy(name),
            types=types,
            params_space=params_space,
        )

        # Create the linear chain.
        children = [navigate, move_hand, open_hand, finish]

        super().__init__(name, children)


class _MoveToToolOnFloorParameterizedOption(ParameterizedOption):

    def __init__(self, name: str, types: List[Type]) -> None:
        super().__init__(name,
                         types,
                         self._params_space,
                         policy=self._policy,
                         initiable=lambda _1, _2, _3, _4: True,
                         terminal=self._terminal)

    @property
    def _params_space(self) -> Box:
        # Parameters are relative distance, dyaw.
        return Box(-np.inf, np.inf, (2, ))

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        # TODO
        import ipdb
        ipdb.set_trace()

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        # TODO
        import ipdb
        ipdb.set_trace()


class _MoveToSurfaceParameterizedOption(ParameterizedOption):

    def __init__(self, name: str, types: List[Type]) -> None:
        super().__init__(name,
                         types,
                         self._params_space,
                         policy=self._policy,
                         initiable=lambda _1, _2, _3, _4: True,
                         terminal=self._terminal)

    @property
    def _params_space(self) -> Box:
        # Parameters are relative distance, dyaw.
        return Box(-np.inf, np.inf, (2, ))

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        # TODO
        import ipdb
        ipdb.set_trace()

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        # TODO
        import ipdb
        ipdb.set_trace()


class _GraspToolFromSurfaceParameterizedOption(ParameterizedOption):

    def __init__(self, name: str, types: List[Type]) -> None:
        super().__init__(name,
                         types,
                         self._params_space,
                         policy=self._policy,
                         initiable=lambda _1, _2, _3, _4: True,
                         terminal=self._terminal)

    @property
    def _params_space(self) -> Box:
        # Parameters are relative distance, dyaw.
        return Box(-np.inf, np.inf, (2, ))

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        # TODO
        import ipdb
        ipdb.set_trace()

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        # TODO
        import ipdb
        ipdb.set_trace()


class _GraspToolFromFloorParameterizedOption(ParameterizedOption):

    def __init__(self, name: str, types: List[Type]) -> None:
        super().__init__(name,
                         types,
                         self._params_space,
                         policy=self._policy,
                         initiable=lambda _1, _2, _3, _4: True,
                         terminal=self._terminal)

    @property
    def _params_space(self) -> Box:
        # Parameters are relative distance, dyaw.
        return Box(-np.inf, np.inf, (2, ))

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        # TODO
        import ipdb
        ipdb.set_trace()

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        # TODO
        import ipdb
        ipdb.set_trace()


class _PlaceToolOnSurfaceParameterizedOption(ParameterizedOption):

    def __init__(self, name: str, types: List[Type]) -> None:
        super().__init__(name,
                         types,
                         self._params_space,
                         policy=self._policy,
                         initiable=lambda _1, _2, _3, _4: True,
                         terminal=self._terminal)

    @property
    def _params_space(self) -> Box:
        # Parameters are relative distance, dyaw.
        return Box(-np.inf, np.inf, (2, ))

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        # TODO
        import ipdb
        ipdb.set_trace()

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        # TODO
        import ipdb
        ipdb.set_trace()


class _PlaceToolOnFloorParameterizedOption(ParameterizedOption):

    def __init__(self, name: str, types: List[Type]) -> None:
        super().__init__(name,
                         types,
                         self._params_space,
                         policy=self._policy,
                         initiable=lambda _1, _2, _3, _4: True,
                         terminal=self._terminal)

    @property
    def _params_space(self) -> Box:
        # Parameters are relative distance, dyaw.
        return Box(-np.inf, np.inf, (2, ))

    def _policy(self, state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        # TODO
        import ipdb
        ipdb.set_trace()

    def _terminal(self, state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        # TODO
        import ipdb
        ipdb.set_trace()


class SpotCubeEnvGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for Spot environments."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"spot_cube_env"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # Note that these are 1:1 with the operators.
        env = get_or_create_env(env_name)
        assert isinstance(env, SpotEnv)

        op_name_to_option_cls: Dict[str, TypingType[ParameterizedOption]] = {
            "MoveToToolOnSurface": _MoveToToolOnSurfaceParameterizedOption,
            "MoveToToolOnFloor": _MoveToToolOnFloorParameterizedOption,
            "MoveToSurface": _MoveToSurfaceParameterizedOption,
            "GraspToolFromSurface": _GraspToolFromSurfaceParameterizedOption,
            "GraspToolFromFloor": _GraspToolFromFloorParameterizedOption,
            "PlaceToolOnSurface": _PlaceToolOnSurfaceParameterizedOption,
            "PlaceToolOnFloor": _PlaceToolOnFloorParameterizedOption,
        }

        options: Set[ParameterizedOption] = set()
        for operator in env.strips_operators:
            option_cls = op_name_to_option_cls[operator.name]
            operator_types = [p.type for p in operator.parameters]
            option = option_cls(operator.name, operator_types)
            options.add(option)

        return options


# TODO reincorporate

# def _navigate_to(robot: Robot, pose_to_nav_to: math_helpers.SE3Pose,
#                  hand_pose: Optional[math_helpers.SE3Pose]) -> None:
#     """Convenience function that wraps navigation and hand movement calls."""
#     navigate_to_relative_pose(robot, pose_to_nav_to)
#     if hand_pose is not None:
#         move_hand_to_relative_pose(robot, hand_pose)

# def _grasp_obj(robot: Robot, localizer: SpotLocalizer,
#                manipuland_id: ObjectDetectionID) -> None:
#     """Convenience function that wraps object detection and grasping calls."""
#     # Capture an image from the hand camera.
#     hand_camera = "hand_color_image"
#     rgbds = capture_images(robot, localizer, [hand_camera])
#     # Run detection to get a pixel for grasping.
#     _, artifacts = detect_objects([manipuland_id], rgbds)
#     pixel = get_object_center_pixel_from_artifacts(artifacts, manipuland_id,
#                                                    hand_camera)
#     grasp_at_pixel(robot, rgbds[hand_camera], pixel)
#     stow_arm(robot)

# def _place_obj(robot: Robot, localizer: SpotLocalizer,
#                target_surface_pose: math_helpers.SE3Pose,
#                params: Array, downward_angle: float = np.pi/2) -> None:
#     robot_pose = localizer.get_last_robot_pose()
#     surface_rel_pose = robot_pose.inverse() * target_surface_pose
#     place_rel_pos = math_helpers.Vec3(x=surface_rel_pose.x + params[0],
#                                       y=surface_rel_pose.y + params[1],
#                                       z=surface_rel_pose.z + params[2])
#     place_at_relative_position(robot, place_rel_pos, downward_angle)
#     # Finally, stow the arm.
#     stow_arm(robot)

# class _SpotEnvOption(utils.SingletonParameterizedOption):
#     """An option defined by an operator in the spot environment.

#     Defined in this way to avoid pickling anything bosdyn related. The
#     key thing to note is that the args to __init__ are just strings, and
#     the magic happens in __getnewargs__().
#     """

#     def __init__(self, operator_name: str, env_name: str) -> None:
#         self._operator_name = operator_name
#         self._env_name = env_name
#         types = [p.type for p in self._get_operator().parameters]
#         policy = self._policy_from_operator
#         params_space = self._create_params_space()
#         super().__init__(self._operator_name,
#                          policy,
#                          types,
#                          params_space=params_space)

#     def __getnewargs__(self) -> Tuple:
#         return (self._operator_name, self._env_name)

#     def _get_env(self) -> SpotEnv:
#         env = get_or_create_env(self._env_name)
#         assert isinstance(env, SpotEnv)
#         return env

#     def _get_operator(self) -> STRIPSOperator:
#         matches = [
#             op for op in self._get_env().strips_operators
#             if op.name == self._operator_name
#         ]
#         assert len(matches) == 1
#         return matches[0]

#     def _policy_from_operator(self, s: State, m: Dict, o: Sequence[Object],
#                               p: Array) -> Action:
#         del m  # unused
#         curr_env = self._get_env()
#         # Get the operator that's been invoked, and use
#         # this to find the name of the controller function
#         # that we will use.
#         operator = self._get_operator()
#         controller_name = curr_env.operator_to_controller_name(operator)
#         # Based on this controller name, invoke the correct function
#         # with the right objects and params.
#         func_to_invoke: Optional[Callable] = None
#         func_args: List[Any] = []

#         if controller_name == "navigate":
#             assert len(o) in [2, 3, 4]
#             robot = o[0]
#             if o[-1].type.name != "floor":
#                 obj = o[-1]
#             else:
#                 assert len(o) == 3
#                 obj = o[1]
#             robot_pose = math_helpers.SE3Pose(
#                 s.get(robot, "x"), s.get(robot, "y"), s.get(robot, "z"),
#                 math_helpers.Quat(s.get(robot,
#                                         "W_quat"), s.get(robot, "X_quat"),
#                                   s.get(robot, "Y_quat"),
#                                   s.get(robot, "Z_quat")))
#             obj_pose = math_helpers.SE3Pose(
#                 s.get(obj, "x"), s.get(obj, "y"), s.get(obj, "z"),
#                 math_helpers.Quat(s.get(obj, "W_quat"), s.get(obj, "X_quat"),
#                                   s.get(obj, "Y_quat"), s.get(obj, "Z_quat")))
#             pose_to_nav_to = get_relative_se2_from_se3(robot_pose, obj_pose,
#                                                        p[0], p[1])

#             # Move the hand only if we're trying to move to then pick up an
#             # object.
#             hand_pose = None
#             if len(o) == 3 and o[-2].type.name == "tool":
#                 hand_pose = DEFAULT_HAND_LOOK_DOWN_POSE
#             func_to_invoke = _navigate_to
#             func_args = [curr_env.robot, pose_to_nav_to, hand_pose]
#         elif controller_name == "grasp":
#             obj_to_grasp = o[1]
#             obj_to_grasp_id = curr_env.obj_to_detection_id(obj_to_grasp)
#             func_to_invoke = _grasp_obj
#             func_args = [curr_env.robot, curr_env.localizer, obj_to_grasp_id]
#         elif controller_name == "place":
#             surface = o[-1]
#             target_surface_pose = math_helpers.SE3Pose(
#                 s.get(surface, "x"), s.get(surface, "y"), s.get(surface, "z"),
#                 math_helpers.Quat(s.get(surface, "W_quat"),
#                                   s.get(surface, "X_quat"),
#                                   s.get(surface, "Y_quat"),
#                                   s.get(surface, "Z_quat")))
#             func_to_invoke = _place_obj
#             func_args = [
#                 curr_env.robot, curr_env.localizer, target_surface_pose, p
#             ]
#         else:
#             raise NotImplementedError(
#                 f"Controller {controller_name} not implemented.")

#         assert func_to_invoke is not None

#         # We return an Action whose array contains an arbitrary, unused
#         # value, but whose extra info field contains a tuple of the
#         # controller name, function to invoke, and its arguments.
#         return Action(curr_env.action_space.low,
#                       extra_info=(controller_name, o, func_to_invoke,
#                                   func_args))

#     def _types_from_operator(self) -> List[Type]:
#         return [p.type for p in self._get_operator().parameters]

#     def _create_params_space(self) -> Box:
#         env = self._get_env()
#         operator = self._get_operator()
#         controller_name = env.operator_to_controller_name(operator)
#         return env.controller_name_to_param_space(controller_name)
