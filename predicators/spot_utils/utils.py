"""Small utility functions for spot."""

import functools
import sys
from enum import Enum
from pathlib import Path
from typing import Collection, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pbrspot
import scipy
import yaml
from bosdyn.api import estop_pb2, robot_state_pb2
from bosdyn.client import math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.exceptions import ProxyConnectionError, TimedOutError
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot
from numpy.typing import NDArray
from scipy.spatial import Delaunay  # pylint: disable=no-name-in-module

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Object, State, Type
from predicators.utils import Rectangle, _Geom2D

# Pose for the hand (relative to the body) that looks down in front.
DEFAULT_HAND_LOOK_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 6))
DEFAULT_HAND_DROP_OBJECT_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=-0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2))
DEFAULT_HAND_LOOK_FLOOR_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 3))
DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2))
DEFAULT_HAND_LOOK_STRAIGHT_DOWN_POSE_HIGH = math_helpers.SE3Pose(
    x=0.65, y=0.0, z=0.32, rot=math_helpers.Quat.from_pitch(np.pi / 2.5))
DEFAULT_HAND_PRE_DUMP_LIFT_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.3, rot=math_helpers.Quat.from_pitch(2 * np.pi / 3))
DEFAULT_HAND_PRE_DUMP_POSE = math_helpers.SE3Pose(
    x=0.80,
    y=0.0,
    z=0.25,
    rot=math_helpers.Quat.from_pitch(np.pi / 2) *
    math_helpers.Quat.from_yaw(np.pi / 1.1))
DEFAULT_HAND_POST_DUMP_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 2))
DEFAULT_SIM_ROBOT_Z_OFFSET = 0.6


# Spot-specific types.
class _Spot3DShape(Enum):
    """Stored as an object 'shape' feature."""
    CUBOID = 1
    CYLINDER = 2


_robot_type = Type(
    "robot",
    ["gripper_open_percentage", "x", "y", "z", "qw", "qx", "qy", "qz"])
# NOTE: include a unique object identifier in the object state to allow for
# object-specific sampler learning (e.g., pick hammer vs pick brush).
_base_object_type = Type("base-object", [
    "x",
    "y",
    "z",
    "qw",
    "qx",
    "qy",
    "qz",
    "shape",
    "height",
    "width",
    "length",
    "object_id",
])
_movable_object_type = Type(
    "movable",
    list(_base_object_type.feature_names) +
    ["placeable", "held", "lost", "in_hand_view", "in_view", "is_sweeper"],
    parent=_base_object_type)
_immovable_object_type = Type("immovable",
                              list(_base_object_type.feature_names) +
                              ["flat_top_surface"],
                              parent=_base_object_type)
_container_type = Type("container",
                       list(_movable_object_type.feature_names),
                       parent=_movable_object_type)
_dustpan_type = Type("dustpan",
                     list(_movable_object_type.feature_names),
                     parent=_movable_object_type)
_broom_type = Type("broom",
                   list(_movable_object_type.feature_names),
                   parent=_movable_object_type)
_wrappers_type = Type("wrappers",
                      list(_movable_object_type.feature_names),
                      parent=_movable_object_type)


def get_collision_geoms_for_nav(state: State) -> List[_Geom2D]:
    """Get all relevant collision geometries for navigating."""
    # We want to consider collisions with all objects that:
    # (1) aren't the robot
    # (2) aren't in an excluded object list defined below
    # (3) aren't being currently held.
    excluded_objects = ["robot", "floor", "brush", "train_toy", "football"]
    collision_geoms = []
    for obj in set(state):
        if obj.name not in excluded_objects:
            if obj.type == _movable_object_type:
                if state.get(obj, "held") > 0.5:
                    continue
            collision_geoms.append(object_to_top_down_geom(obj, state))
    return collision_geoms


def object_to_top_down_geom(
        obj: Object,
        state: State,
        size_buffer: float = 0.0,
        put_on_robot_if_held: bool = True) -> utils._Geom2D:
    """Convert object to top-down view geometry."""
    assert obj.is_instance(_base_object_type)
    shape_type = int(np.round(state.get(obj, "shape")))
    if put_on_robot_if_held and \
        obj.is_instance(_movable_object_type) and state.get(obj, "held") > 0.5:
        robot, = state.get_objects(_robot_type)
        se3_pose = utils.get_se3_pose_from_state(state, robot)
    else:
        se3_pose = utils.get_se3_pose_from_state(state, obj)
    angle = se3_pose.rot.to_yaw()
    center_x = se3_pose.x
    center_y = se3_pose.y
    width = state.get(obj, "width") + size_buffer
    length = state.get(obj, "length") + size_buffer
    if shape_type == _Spot3DShape.CUBOID.value:
        return utils.Rectangle.from_center(center_x, center_y, width, length,
                                           angle)
    assert shape_type == _Spot3DShape.CYLINDER.value
    assert np.isclose(width, length)
    radius = width / 2
    return utils.Circle(center_x, center_y, radius)


def object_to_side_view_geom(
        obj: Object,
        state: State,
        size_buffer: float = 0.0,
        put_on_robot_if_held: bool = True) -> utils._Geom2D:
    """Convert object to side view geometry."""
    assert obj.is_instance(_base_object_type)
    # The shape doesn't matter because all shapes are rectangles from the side.
    # If the object is held, use the robot's pose.
    if put_on_robot_if_held and \
        obj.is_instance(_movable_object_type) and state.get(obj, "held") > 0.5:
        robot, = state.get_objects(_robot_type)
        se3_pose = utils.get_se3_pose_from_state(state, robot)
    else:
        se3_pose = utils.get_se3_pose_from_state(state, obj)
    center_y = se3_pose.y
    center_z = se3_pose.z
    length = state.get(obj, "length") + size_buffer
    height = state.get(obj, "height") + size_buffer
    return utils.Rectangle.from_center(center_y, center_z, length, height, 0.0)


@functools.lru_cache(maxsize=None)
def get_allowed_map_regions() -> Collection[Delaunay]:
    """Gets Delaunay regions from metadata that correspond to free space."""
    metadata = load_spot_metadata()
    allowed_regions = metadata.get("allowed-regions", {})
    convex_hulls = []
    for region_pts in allowed_regions.values():
        dealunay_hull = Delaunay(np.array(region_pts))
        convex_hulls.append(dealunay_hull)
    return convex_hulls


def get_graph_nav_dir() -> Path:
    """Get the path to the graph nav directory."""
    upload_dir = Path(__file__).parent / "graph_nav_maps"
    return upload_dir / CFG.spot_graph_nav_map


def load_spot_metadata() -> Dict:
    """Load from the YAML config."""
    config_filepath = get_graph_nav_dir() / "metadata.yaml"
    with open(config_filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def get_spot_home_pose() -> math_helpers.SE2Pose:
    """Load the home pose for the robot."""
    metadata = load_spot_metadata()
    home_pose_dict = metadata["spot-home-pose"]
    x = home_pose_dict["x"]
    y = home_pose_dict["y"]
    angle = home_pose_dict["angle"]
    return math_helpers.SE2Pose(x, y, angle)


def get_april_tag_transform(april_tag: int) -> math_helpers.SE3Pose:
    """Load the world frame transform for an april tag.

    Returns identity if no config is found.
    """
    metadata = load_spot_metadata()
    transform_dict = metadata["april-tag-offsets"]
    try:
        april_tag_transform_dict = transform_dict[f"tag-{april_tag}"]
    except KeyError:
        return math_helpers.SE3Pose(0, 0, 0, rot=math_helpers.Quat())
    x = april_tag_transform_dict["x"]
    y = april_tag_transform_dict["y"]
    z = april_tag_transform_dict["z"]
    return math_helpers.SE3Pose(x, y, z, rot=math_helpers.Quat())


def verify_estop(robot: Robot) -> None:
    """Verify the robot is not estopped."""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external" + \
            " E-Stop client, such as the estop SDK example, to" + \
            " configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)


def get_pixel_from_user(rgb: NDArray[np.uint8]) -> Tuple[int, int]:
    """Use open CV GUI to select a pixel on the given image."""

    image_click: Optional[Tuple[int, int]] = None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    def _callback(event: int, x: int, y: int, flags: int, param: None) -> None:
        """Callback for the click-to-grasp functionality with the Spot API's
        grasping interface."""
        del flags, param
        nonlocal image_click
        if event == cv2.EVENT_LBUTTONUP:
            image_click = (x, y)

    image_title = "Click to grasp"
    cv2.namedWindow(image_title)
    cv2.setMouseCallback(image_title, _callback)
    cv2.imshow(image_title, bgr)

    while image_click is None:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            # Quit and terminate the process (if you're panicking.)
            sys.exit()

    cv2.destroyAllWindows()

    return image_click


def get_relative_se2_from_se3(
        robot_pose: math_helpers.SE3Pose,
        target_pose: math_helpers.SE3Pose,
        target_offset_distance: float = 0.0,
        target_offset_angle: float = 0.0) -> math_helpers.SE2Pose:
    """Given a current se3 pose and a target se3 pose on the same plane, return
    a relative se2 pose for moving from the current to the target.

    Also add an angle and distance offset to the target pose. The returned
    se2 pose is facing toward the target.

    Typical use case: we know the current se3 pose for the body of the robot
    and the se3 pose for a table, and we want to move in front of the table.
    """
    dx = np.cos(target_offset_angle) * target_offset_distance
    dy = np.sin(target_offset_angle) * target_offset_distance
    x = target_pose.x + dx
    y = target_pose.y + dy
    # Face towards the target.
    rot = target_offset_angle + np.pi
    target_se2 = math_helpers.SE2Pose(x, y, rot)
    robot_se2 = robot_pose.get_closest_se2_transform()
    return robot_se2.inverse() * target_se2


def valid_navigation_position(
        robot_geom: Rectangle,
        collision_geoms: Collection[_Geom2D],
        allowed_regions: Collection[scipy.spatial.Delaunay],  # pylint: disable=no-member
) -> bool:
    """Checks whether a given robot geom is not in collision and also within
    some allowed region."""
    # Check for out-of-bounds. To do this, we're looking for
    # one allowed region where all four points defining the
    # robot will be within the region in the new pose.
    oob = True
    for region in allowed_regions:
        for cx, cy in robot_geom.vertices:
            if region.find_simplex(np.array([cx, cy])) < 0:
                break
        else:
            oob = False
            break
    if oob:
        return False
    # Check for collisions.
    collision = False
    for collision_geom in collision_geoms:
        if collision_geom.intersects(robot_geom):
            collision = True
            break
    # Success!
    return not collision


def sample_random_nearby_point_to_move(
    robot_geom: Rectangle,
    collision_geoms: Collection[_Geom2D],
    rng: np.random.Generator,
    max_distance_away: float,
    allowed_regions: Collection[scipy.spatial.Delaunay],  # pylint: disable=no-member
    max_samples: int = 1000
) -> Tuple[float, float, Rectangle]:
    """Sampler for navigating to a randomly selected point within some distance
    from the current robot's position. Useful when trying to find lost objects.

    Returns a distance and an angle in radians. Also returns the next
    robot geom for visualization and debugging convenience.
    """
    for _ in range(max_samples):
        distance = rng.uniform(0.1, max_distance_away)
        angle = rng.uniform(-np.pi, np.pi)
        dx = np.cos(angle) * distance
        dy = np.sin(angle) * distance
        x = robot_geom.x + dx
        y = robot_geom.y + dy
        # Face towards the target.
        rot = angle + np.pi if angle < 0 else angle - np.pi
        cand_geom = Rectangle.from_center(x, y, robot_geom.width,
                                          robot_geom.height, rot)
        if valid_navigation_position(cand_geom, collision_geoms,
                                     allowed_regions):
            return (distance, angle, cand_geom)

    raise RuntimeError(f"Sampling failed after {max_samples} attempts")


def sample_move_offset_from_target(
    target_origin: Tuple[float, float],
    robot_geom: Rectangle,
    collision_geoms: Collection[_Geom2D],
    rng: np.random.Generator,
    min_distance: float,
    max_distance: float,
    allowed_regions: Collection[scipy.spatial.Delaunay],  # pylint: disable=no-member
    max_samples: int = 1000,
    min_angle: float = -np.pi,
    max_angle: float = np.pi,
) -> Tuple[float, float, Rectangle]:
    """Sampler for navigating to a target object.

    Returns a distance and an angle in radians. Also returns the next
    robot geom for visualization and debugging convenience.
    """
    for _ in range(max_samples):
        distance = rng.uniform(min_distance, max_distance)
        angle = rng.uniform(min_angle, max_angle)
        dx = np.cos(angle) * distance
        dy = np.sin(angle) * distance
        x = target_origin[0] + dx
        y = target_origin[1] + dy
        # Face towards the target.
        rot = angle + np.pi if angle < 0 else angle - np.pi
        cand_geom = Rectangle.from_center(x, y, robot_geom.width,
                                          robot_geom.height, rot)
        if valid_navigation_position(cand_geom, collision_geoms,
                                     allowed_regions):
            return (distance, angle, cand_geom)

    raise RuntimeError(f"Sampling failed after {max_samples} attempts")


def get_robot_state(robot: Robot,
                    timeout_per_call: float = 20,
                    num_retries: int = 10) -> robot_state_pb2.RobotState:
    """Get the robot state."""
    robot_state_client = robot.ensure_client(
        RobotStateClient.default_service_name)
    for _ in range(num_retries):
        try:
            robot_state = robot_state_client.get_robot_state(
                timeout=timeout_per_call)
            return robot_state
        except (TimedOutError, ProxyConnectionError):
            print("WARNING: get robot state failed once, retrying...")
    raise RuntimeError("get_robot_state() failed permanently.")


def get_robot_gripper_open_percentage(robot: Robot) -> float:
    """Get the current state of how open the gripper is."""
    robot_state = get_robot_state(robot)
    return float(robot_state.manipulator_state.gripper_open_percentage)


def spot_pose_to_geom2d(pose: math_helpers.SE3Pose) -> Rectangle:
    """Use known dimensions for spot robot to create a bounding box for the
    robot (top-down view).

    The origin of the rectangle is the back RIGHT leg of the spot.

    NOTE: the spot's x axis in the body frame points forward and the y axis
    points leftward. See the link below for an illustration of the frame.
    https://dev.bostondynamics.com/docs/concepts/geometry_and_frames
    """
    # We want to create a rectangle whose center is (pose.x, pose.y),
    # whose width (x direction) is front_to_back_length, whose height
    # (y direction) is side_length, and whose rotation is the pose yaw.
    front_to_back_length = 0.85  # meters, approximately
    side_length = 0.25
    yaw = pose.rot.to_yaw()
    return Rectangle.from_center(pose.x, pose.y, front_to_back_length,
                                 side_length, yaw)


def update_pbrspot_robot_conf(robot: Robot,
                              sim_robot: pbrspot.spot.Spot) -> None:
    """Simply updates the simulated spot to mirror the configuration of the
    real robot."""
    curr_robot_state = get_robot_state(robot)
    robot_joint_to_curr_conf: Dict[str, float] = {}
    for joint in curr_robot_state.kinematic_state.joint_states:
        robot_joint_to_curr_conf[joint.name.replace(".", "_").replace(
            "arm0", "arm")] = joint.position.value
    sim_robot.set_joint_positions(sim_robot.body_joint_names, [
        robot_joint_to_curr_conf[j_name]
        for j_name in sim_robot.body_joint_names
    ])
    arm_conf = [
        robot_joint_to_curr_conf[j_name]
        for j_name in sim_robot.arm_joint_names
    ]
    sim_robot.arm.set_configuration(arm_conf)


def update_pbrspot_given_state(sim_robot: pbrspot.spot.Spot,
                               obj_name_to_sim_obj: Dict[str,
                                                         pbrspot.body.Body],
                               state: State) -> None:
    """Update simulated environment to match state."""
    for obj in state:
        # The floor object is loaded during init and thus treated
        # separately.
        if obj.name == "floor":
            continue
        if obj.type.name == "robot":
            sim_robot.set_pose(([
                state.get(obj, "x"),
                state.get(obj, "y"),
                state.get(obj, "z") - DEFAULT_SIM_ROBOT_Z_OFFSET
            ], [
                state.get(obj, "qx"),
                state.get(obj, "qy"),
                state.get(obj, "qz"),
                state.get(obj, "qw")
            ]))
        else:
            sim_obj = obj_name_to_sim_obj[obj.name]
            sim_obj.set_point([
                state.get(obj, "x"),
                state.get(obj, "y"),
                state.get(obj, "z")
            ])


def construct_state_given_pbrspot(sim_robot: pbrspot.spot.Spot,
                                  obj_name_to_sim_obj: Dict[str,
                                                            pbrspot.body.Body],
                                  state: State) -> State:
    """Construct state to match new simulated env state.

    Return an updated copy of the state.
    """
    sim_robot_pose = sim_robot.get_pose()
    next_state = state.copy()
    for obj in state:
        if obj.name == "floor":
            continue
        if obj.type.name == "robot":
            next_state.set(obj, "x", sim_robot_pose[0][0])
            next_state.set(obj, "y", sim_robot_pose[0][1])
            next_state.set(obj, "z",
                           sim_robot_pose[0][2] + DEFAULT_SIM_ROBOT_Z_OFFSET)
            next_state.set(obj, "qx", sim_robot_pose[1][0])
            next_state.set(obj, "qy", sim_robot_pose[1][1])
            next_state.set(obj, "qz", sim_robot_pose[1][2])
            next_state.set(obj, "qw", sim_robot_pose[1][3])
            next_state.set(obj, "gripper_open_percentage",
                           sim_robot.hand.GetJointPositions() * -100.0)
        else:
            sim_obj = obj_name_to_sim_obj[obj.name]
            sim_obj_pose = sim_obj.get_pose()
            next_state.set(obj, "x", sim_obj_pose[0][0])
            next_state.set(obj, "y", sim_obj_pose[0][1])
            next_state.set(obj, "z", sim_obj_pose[0][2])
    return next_state
