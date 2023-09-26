"""Ground-truth options for PDDL environments."""

from typing import Dict, List, Sequence, Set, Tuple
from typing import Type as TypingType

import numpy as np
from bosdyn.client import math_helpers
from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv, get_robot
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    get_last_detected_objects, get_object_center_pixel_from_artifacts
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
    get_relative_se2_from_se3
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type
from predicators.utils import LinearChainParameterizedOption


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
        robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

        target_obj = objects[target_obj_idx]
        target_pose = utils.get_se3_pose_from_state(state, target_obj)

        rel_pose = get_relative_se2_from_se3(robot_pose, target_pose, distance,
                                             yaw)

        return utils.create_spot_env_action("execute", objects,
                                            navigate_to_relative_pose,
                                            (robot, rel_pose))

    return _policy


def _create_grasp_parameterized_policy(
        target_obj_idx: int) -> ParameterizedPolicy:

    robot, _, _ = get_robot()
    env = get_or_create_env(CFG.env)
    assert isinstance(env, SpotEnv)
    detection_id_to_obj = env._detection_id_to_obj  # pylint: disable=protected-access
    obj_to_detection_id = {o: d for d, o in detection_id_to_obj.items()}

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory, params, state  # not used
        target_obj = objects[target_obj_idx]
        target_detection_id = obj_to_detection_id[target_obj]
        rgbds = get_last_captured_images()
        _, artifacts = get_last_detected_objects()
        hand_camera = "hand_color_image"
        img = rgbds[hand_camera]
        pixel = get_object_center_pixel_from_artifacts(artifacts,
                                                       target_detection_id,
                                                       hand_camera)
        return utils.create_spot_env_action("execute", objects, grasp_at_pixel,
                                            (robot, img, pixel))

    return _policy


def _create_place_parameterized_policy(
        robot_obj_idx: int, surface_obj_idx: int) -> ParameterizedPolicy:

    robot, _, _ = get_robot()

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory  # not used

        dx, dy, dz = params

        robot_obj = objects[robot_obj_idx]
        robot_pose = utils.get_se3_pose_from_state(state, robot_obj)

        surface_obj = objects[surface_obj_idx]
        surface_pose = utils.get_se3_pose_from_state(state, surface_obj)

        surface_rel_pose = robot_pose.inverse() * surface_pose
        place_rel_pos = math_helpers.Vec3(x=surface_rel_pose.x + dx,
                                          y=surface_rel_pose.y + dy,
                                          z=surface_rel_pose.z + dz)

        return utils.create_spot_env_action("execute", objects,
                                            place_at_relative_position,
                                            (robot, place_rel_pos))

    return _policy


def _create_stow_arm_parameterized_policy() -> ParameterizedPolicy:

    robot, _, _ = get_robot()

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del state, memory, params  # not used
        return utils.create_spot_env_action("execute", objects, stow_arm,
                                            (robot, ))

    return _policy


def _create_move_hand_parameterized_policy(
        hand_pose: math_helpers.SE3Pose) -> ParameterizedPolicy:

    robot, _, _ = get_robot()

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del state, memory, params  # not used
        return utils.create_spot_env_action("execute", objects,
                                            move_hand_to_relative_pose,
                                            (robot, hand_pose))

    return _policy


def _create_open_hand_parameterized_policy() -> ParameterizedPolicy:

    robot, _, _ = get_robot()

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del state, memory, params  # not used
        return utils.create_spot_env_action("execute", objects, open_gripper,
                                            (robot, ))

    return _policy


def _create_operator_finish_parameterized_policy(
        operator_name: str) -> ParameterizedPolicy:

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del state, memory, params  # not used
        return utils.create_spot_env_action(operator_name, objects, None,
                                            tuple())

    return _policy


class _MoveToToolOnSurfaceParameterizedOption(LinearChainParameterizedOption):
    """Navigate to the surface and look down at the object so it's in view.

    The types are [robot, tool, surface].

    The parameters are relative distance and relative yaw between the robot
    and the surface.
    """

    def __init__(self, name: str, types: List[Type]) -> None:

        # Parameters are relative distance, dyaw.
        params_space = Box(-np.inf, np.inf, (2, ))

        # Navigate to the tool.
        navigate = utils.SingletonParameterizedOption(
            "MoveToToolOnSurface-Navigate",
            _create_navigate_parameterized_policy(robot_obj_idx=0,
                                                  target_obj_idx=1,
                                                  distance_param_idx=0,
                                                  yaw_param_idx=1),
            types=types,
            params_space=params_space,
        )

        # Look down at the surface. Note that we can't open the hand because
        # that would mess up the HandEmpty detector.
        move_hand = utils.SingletonParameterizedOption(
            "MoveToToolOnSurface-MoveHand",
            _create_move_hand_parameterized_policy(
                DEFAULT_HAND_LOOK_DOWN_POSE),
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
        children = [navigate, move_hand, finish]

        super().__init__(name, children)

    def __getnewargs__(self) -> Tuple:
        """Avoid pickling issues with bosdyn functions."""
        return (self.name, self.types)

    def __getstate__(self) -> Dict:
        """Avoid pickling issues with bosdyn functions."""
        return {"name": self.name}


class _MoveToToolOnFloorParameterizedOption(LinearChainParameterizedOption):
    """Navigate to the object and look down at the object so it's in view.

    The types are [robot, tool].

    The parameters are relative distance and relative yaw between the robot
    and the tool.
    """

    def __init__(self, name: str, types: List[Type]) -> None:

        # Parameters are relative distance, dyaw.
        params_space = Box(-np.inf, np.inf, (2, ))

        # Navigate to the floor.
        navigate = utils.SingletonParameterizedOption(
            "MoveToToolOnFloor-Navigate",
            _create_navigate_parameterized_policy(robot_obj_idx=0,
                                                  target_obj_idx=1,
                                                  distance_param_idx=0,
                                                  yaw_param_idx=1),
            types=types,
            params_space=params_space,
        )

        # Finish the action.
        finish = utils.SingletonParameterizedOption(
            "MoveToToolOnFloor-Finish",
            _create_operator_finish_parameterized_policy(name),
            types=types,
            params_space=params_space,
        )

        # Create the linear chain.
        children = [navigate, finish]

        super().__init__(name, children)

    def __getnewargs__(self) -> Tuple:
        """Avoid pickling issues with bosdyn functions."""
        return (self.name, self.types)

    def __getstate__(self) -> Dict:
        """Avoid pickling issues with bosdyn functions."""
        return {"name": self.name}


class _MoveToSurfaceParameterizedOption(LinearChainParameterizedOption):
    """Navigate to the surface.

    The types are [robot, surface].

    The parameters are relative distance and relative yaw between the robot
    and the surface.
    """

    def __init__(self, name: str, types: List[Type]) -> None:

        # Parameters are relative distance, dyaw.
        params_space = Box(-np.inf, np.inf, (2, ))

        # Navigate to the floor.
        navigate = utils.SingletonParameterizedOption(
            "MoveToSurface-Navigate",
            _create_navigate_parameterized_policy(robot_obj_idx=0,
                                                  target_obj_idx=1,
                                                  distance_param_idx=0,
                                                  yaw_param_idx=1),
            types=types,
            params_space=params_space,
        )

        # Finish the action.
        finish = utils.SingletonParameterizedOption(
            "MoveToSurface-Finish",
            _create_operator_finish_parameterized_policy(name),
            types=types,
            params_space=params_space,
        )

        # Create the linear chain.
        children = [navigate, finish]

        super().__init__(name, children)

    def __getnewargs__(self) -> Tuple:
        """Avoid pickling issues with bosdyn functions."""
        return (self.name, self.types)

    def __getstate__(self) -> Dict:
        """Avoid pickling issues with bosdyn functions."""
        return {"name": self.name}


class _GraspToolFromSurfaceParameterizedOption(LinearChainParameterizedOption):
    """Grasp a tool on a surface.

    The types are [robot, tool, surface].

    There are currently no parameters.
    """

    def __init__(self, name: str, types: List[Type]) -> None:

        # Currently no parameters.
        params_space = Box(0, 1, (0, ))

        # Pick the tool.
        grasp = utils.SingletonParameterizedOption(
            "GraspToolFromSurface-Grasp",
            _create_grasp_parameterized_policy(target_obj_idx=1),
            types=types,
            params_space=params_space,
        )

        # Stow the arm.
        stow_arm = utils.SingletonParameterizedOption(
            "GraspToolFromSurface-Stow",
            _create_stow_arm_parameterized_policy(),
            types=types,
            params_space=params_space,
        )

        # Finish the action.
        finish = utils.SingletonParameterizedOption(
            "GraspToolFromSurface-Finish",
            _create_operator_finish_parameterized_policy(name),
            types=types,
            params_space=params_space,
        )

        # Create the linear chain.
        children = [grasp, stow_arm, finish]

        super().__init__(name, children)

    def __getnewargs__(self) -> Tuple:
        """Avoid pickling issues with bosdyn functions."""
        return (self.name, self.types)

    def __getstate__(self) -> Dict:
        """Avoid pickling issues with bosdyn functions."""
        return {"name": self.name}


class _GraspToolFromFloorParameterizedOption(LinearChainParameterizedOption):
    """Grasp a tool from thje floor.

    The types are [robot, tool].

    There are currently no parameters.
    """

    def __init__(self, name: str, types: List[Type]) -> None:

        # Currently no parameters.
        params_space = Box(0, 1, (0, ))

        # Pick the tool.
        grasp = utils.SingletonParameterizedOption(
            "GraspToolFromSurface-Grasp",
            _create_grasp_parameterized_policy(target_obj_idx=1),
            types=types,
            params_space=params_space,
        )

        # Stow the arm.
        stow_arm = utils.SingletonParameterizedOption(
            "GraspToolFromSurface-Stow",
            _create_stow_arm_parameterized_policy(),
            types=types,
            params_space=params_space,
        )

        # Finish the action.
        finish = utils.SingletonParameterizedOption(
            "GraspToolFromFloor-Finish",
            _create_operator_finish_parameterized_policy(name),
            types=types,
            params_space=params_space,
        )

        # Create the linear chain.
        children = [grasp, stow_arm, finish]

        super().__init__(name, children)

    def __getnewargs__(self) -> Tuple:
        """Avoid pickling issues with bosdyn functions."""
        return (self.name, self.types)

    def __getstate__(self) -> Dict:
        """Avoid pickling issues with bosdyn functions."""
        return {"name": self.name}


class _PlaceToolOnSurfaceParameterizedOption(LinearChainParameterizedOption):
    """Place a tool on a surface.

    The types are [robot, tool, surface].

    Parameters are relative dx, dy, dz (to surface objects center).
    """

    def __init__(self, name: str, types: List[Type]) -> None:

        # Parameters are relative dx, dy, dz (to surface objects center).
        params_space = Box(-np.inf, np.inf, (3, ))

        # Place the tool.
        place = utils.SingletonParameterizedOption(
            "PlaceToolOnSurface-Place",
            _create_place_parameterized_policy(robot_obj_idx=0,
                                               surface_obj_idx=2),
            types=types,
            params_space=params_space,
        )

        # Stow the arm.
        stow_arm = utils.SingletonParameterizedOption(
            "GraspToolFromSurface-Stow",
            _create_stow_arm_parameterized_policy(),
            types=types,
            params_space=params_space,
        )

        # Finish the action.
        finish = utils.SingletonParameterizedOption(
            "PlaceToolOnSurface-Finish",
            _create_operator_finish_parameterized_policy(name),
            types=types,
            params_space=params_space,
        )

        # Create the linear chain.
        children = [place, stow_arm, finish]

        super().__init__(name, children)

    def __getnewargs__(self) -> Tuple:
        """Avoid pickling issues with bosdyn functions."""
        return (self.name, self.types)

    def __getstate__(self) -> Dict:
        """Avoid pickling issues with bosdyn functions."""
        return {"name": self.name}


class _PlaceToolOnFloorParameterizedOption(LinearChainParameterizedOption):
    """Place a tool on a surface.

    The types are [robot, tool].

    There are currently no parameters.
    """

    def __init__(self, name: str, types: List[Type]) -> None:

        # There are currently no parameters.
        params_space = Box(0, 1, (0, ))

        # Just naively open the hand and assume the object will drop.
        open_hand = utils.SingletonParameterizedOption(
            "PlaceToolOnFloor-OpenHand",
            _create_open_hand_parameterized_policy(),
            types=types,
            params_space=params_space,
        )

        # Finish the action.
        finish = utils.SingletonParameterizedOption(
            "PlaceToolOnFloor-Finish",
            _create_operator_finish_parameterized_policy(name),
            types=types,
            params_space=params_space,
        )

        # Create the linear chain.
        children = [open_hand, finish]

        super().__init__(name, children)

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
            option = option_cls(operator.name, operator_types)  # type: ignore
            options.add(option)

        return options
