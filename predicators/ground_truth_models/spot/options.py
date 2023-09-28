"""Ground-truth options for PDDL environments."""

from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot
from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv, get_detection_id_for_object, \
    get_robot
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.spot_utils.perception.object_detection import \
    get_last_detected_objects, get_object_center_pixel_from_artifacts
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import \
    get_last_captured_images
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, \
    DEFAULT_HAND_LOOK_FLOOR_POSE, get_relative_se2_from_se3
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, Type

###############################################################################
#            Helper functions for chaining multiple spot skills               #
###############################################################################


def _navigate_to_relative_pose_and_move_hand(
        robot: Robot, rel_pose: math_helpers.SE2Pose,
        hand_pose: math_helpers.SE3Pose) -> None:
    # First navigate to the pose.
    navigate_to_relative_pose(robot, rel_pose)
    # Then look down.
    move_hand_to_relative_pose(robot, hand_pose)


def _grasp_at_pixel_and_stow(robot: Robot, img: RGBDImageWithContext,
                             pixel: Tuple[int, int]) -> None:
    # Grasp.
    grasp_at_pixel(robot, img, pixel)
    # Stow.
    stow_arm(robot)


def _place_at_relative_position_and_stow(
        robot: Robot, rel_pose: math_helpers.SE3Pose) -> None:
    # Place.
    place_at_relative_position(robot, rel_pose)
    # Stow.
    stow_arm(robot)


def _drop_and_stow(robot: Robot) -> None:
    # First, move the arm to a position from which the object will drop.
    move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_DOWN_POSE)
    # Open the hand.
    open_gripper(robot)
    # Stow.
    stow_arm(robot)


###############################################################################
#                    Helper parameterized option policies                     #
###############################################################################


def _move_to_target_policy(name: str, distance_param_idx: int,
                           yaw_param_idx: int, robot_obj_idx: int,
                           target_obj_idx: int,
                           hand_pose: Optional[math_helpers.SE3Pose],
                           state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:

    del memory  # not used

    robot, _, _ = get_robot()

    distance = params[distance_param_idx]
    yaw = params[yaw_param_idx]

    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    target_obj = objects[target_obj_idx]
    target_pose = utils.get_se3_pose_from_state(state, target_obj)

    rel_pose = get_relative_se2_from_se3(robot_pose, target_pose, distance,
                                         yaw)

    if hand_pose is None:
        fn: Callable = navigate_to_relative_pose
        fn_args: Tuple = (robot, rel_pose)
    else:
        fn = _navigate_to_relative_pose_and_move_hand
        fn_args = (robot, rel_pose, hand_pose)

    return utils.create_spot_env_action(name, objects, fn, fn_args)


def _grasp_policy(name: str, target_obj_idx: int, state: State, memory: Dict,
                  objects: Sequence[Object], params: Array) -> Action:
    del state, memory, params  # not used

    robot, _, _ = get_robot()

    target_obj = objects[target_obj_idx]
    target_detection_id = get_detection_id_for_object(target_obj)
    rgbds = get_last_captured_images()
    _, artifacts = get_last_detected_objects()
    hand_camera = "hand_color_image"
    img = rgbds[hand_camera]
    pixel = get_object_center_pixel_from_artifacts(artifacts,
                                                   target_detection_id,
                                                   hand_camera)

    return utils.create_spot_env_action(name, objects,
                                        _grasp_at_pixel_and_stow,
                                        (robot, img, pixel))


###############################################################################
#                   Concrete parameterized option policies                    #
###############################################################################


def _move_to_tool_on_surface_policy(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> Action:
    name = "MoveToToolOnSurface"
    distance_param_idx = 0
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 1
    hand_pose = DEFAULT_HAND_LOOK_DOWN_POSE
    return _move_to_target_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, hand_pose,
                                  state, memory, objects, params)


def _move_to_tool_on_floor_policy(state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> Action:
    name = "MoveToToolOnFloor"
    distance_param_idx = 0
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 1
    hand_pose = DEFAULT_HAND_LOOK_FLOOR_POSE
    return _move_to_target_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, hand_pose,
                                  state, memory, objects, params)


def _move_to_surface_policy(state: State, memory: Dict,
                            objects: Sequence[Object],
                            params: Array) -> Action:
    name = "MoveToSurface"
    distance_param_idx = 0
    yaw_param_idx = 1
    robot_obj_idx = 0
    target_obj_idx = 1
    hand_pose = None
    return _move_to_target_policy(name, distance_param_idx, yaw_param_idx,
                                  robot_obj_idx, target_obj_idx, hand_pose,
                                  state, memory, objects, params)


def _grasp_tool_from_surface_policy(state: State, memory: Dict,
                                    objects: Sequence[Object],
                                    params: Array) -> Action:
    name = "GraspToolFromSurface"
    target_obj_idx = 1
    return _grasp_policy(name, target_obj_idx, state, memory, objects, params)


def _grasp_tool_from_floor_policy(state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> Action:
    name = "GraspToolFromFloor"
    target_obj_idx = 1
    return _grasp_policy(name, target_obj_idx, state, memory, objects, params)


def _place_tool_on_surface_policy(state: State, memory: Dict,
                                  objects: Sequence[Object],
                                  params: Array) -> Action:
    del memory  # not used

    name = "PlaceToolOnSurface"
    robot_obj_idx = 0
    surface_obj_idx = 2

    robot, _, _ = get_robot()

    dx, dy, dz = params

    robot_obj = objects[robot_obj_idx]
    robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

    surface_obj = objects[surface_obj_idx]
    surface_pose = utils.get_se3_pose_from_state(state, surface_obj)

    surface_rel_pose = robot_pose.inverse() * surface_pose
    place_rel_pos = math_helpers.Vec3(x=surface_rel_pose.x + dx,
                                      y=surface_rel_pose.y + dy,
                                      z=surface_rel_pose.z + dz)

    return utils.create_spot_env_action(name, objects,
                                        _place_at_relative_position_and_stow,
                                        (robot, place_rel_pos))


def _place_tool_on_floor_policy(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> Action:
    del state, memory, params  # not used
    name = "PlaceToolOnFloor"
    robot, _, _ = get_robot()
    return utils.create_spot_env_action(name, objects, _drop_and_stow,
                                        (robot, ))


###############################################################################
#                       Parameterized option factory                          #
###############################################################################

_OPERATOR_NAME_TO_PARAM_SPACE = {
    "MoveToToolOnSurface": Box(-np.inf, np.inf, (2, )),  # rel dist, dyaw
    "MoveToToolOnFloor": Box(-np.inf, np.inf, (2, )),  # rel dist, dyaw
    "MoveToSurface": Box(-np.inf, np.inf, (2, )),  # rel dist, dyaw
    "GraspToolFromSurface": Box(0, 1, (0, )),
    "GraspToolFromFloor": Box(0, 1, (0, )),
    "PlaceToolOnSurface": Box(-np.inf, np.inf, (3, )),  # rel dx, dy, dz
    "PlaceToolOnFloor": Box(0, 1, (0, )),
}

_OPERATOR_NAME_TO_POLICY = {
    "MoveToToolOnSurface": _move_to_tool_on_surface_policy,
    "MoveToToolOnFloor": _move_to_tool_on_floor_policy,
    "MoveToSurface": _move_to_surface_policy,
    "GraspToolFromSurface": _grasp_tool_from_surface_policy,
    "GraspToolFromFloor": _grasp_tool_from_floor_policy,
    "PlaceToolOnSurface": _place_tool_on_surface_policy,
    "PlaceToolOnFloor": _place_tool_on_floor_policy,
}


class _SpotParameterizedOption(utils.SingletonParameterizedOption):
    """A parameterized option for spot.

    NOTE: parameterized options MUST be singletons in order to avoid nasty
    issues with the expected atoms monitoring.

    Also note that we need to define the policies outside the class, rather
    than pass the policies into the class, to avoid pickling issues via bosdyn.
    """

    def __init__(self, operator_name: str, types: List[Type]) -> None:
        params_space = _OPERATOR_NAME_TO_PARAM_SPACE[operator_name]
        policy = _OPERATOR_NAME_TO_POLICY[operator_name]
        super().__init__(operator_name, policy, types, params_space)

    def __getnewargs__(self) -> Tuple:
        """Avoid pickling issues with bosdyn functions."""
        return (self.name, self.types)

    def __getstate__(self) -> Dict:
        """Avoid pickling issues with bosdyn functions."""
        return {"name": self.name}


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

        options: Set[ParameterizedOption] = set()
        for operator in env.strips_operators:
            operator_types = [p.type for p in operator.parameters]
            option = _SpotParameterizedOption(operator.name, operator_types)
            options.add(option)

        return options
