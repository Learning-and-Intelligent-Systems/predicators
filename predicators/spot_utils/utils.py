"""Small utility functions for spot."""

import sys
from typing import Collection, Optional, Tuple

import cv2
import numpy as np
from bosdyn.api import estop_pb2, robot_state_pb2
from bosdyn.client import math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.exceptions import ProxyConnectionError, TimedOutError
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot
from numpy.typing import NDArray

from predicators.utils import Rectangle, _Geom2D

# Pose for the hand (relative to the body) that looks down in front.
DEFAULT_HAND_LOOK_DOWN_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 6))
DEFAULT_HAND_LOOK_FLOOR_POSE = math_helpers.SE3Pose(
    x=0.80, y=0.0, z=0.25, rot=math_helpers.Quat.from_pitch(np.pi / 3))

# Center of the fourth floor room.
HOME_POSE = math_helpers.SE2Pose(x=1.25, y=0.0, angle=np.pi / 2)


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


def sample_move_offset_from_target(
        target_origin: Tuple[float, float],
        robot_geom: Rectangle,
        collision_geoms: Collection[_Geom2D],
        rng: np.random.Generator,
        max_distance: float,
        room_bounds: Tuple[float, float, float, float],
        max_samples: int = 100) -> Tuple[float, float, Rectangle]:
    """Sampler for navigating to a target object.

    Returns a distance an an angle in radians. Also returns the next
    robot geom for visualization and debugging convenience.
    """
    min_x, min_y, max_x, max_y = room_bounds
    for _ in range(max_samples):
        distance = rng.uniform(0.0, max_distance)
        angle = rng.uniform(-np.pi, np.pi)
        dx = np.cos(angle) * distance
        dy = np.sin(angle) * distance
        x = target_origin[0] + dx
        y = target_origin[1] + dy
        # Face towards the target.
        rot = angle + np.pi if angle < 0 else angle - np.pi
        cand_geom = Rectangle(x, y, robot_geom.width, robot_geom.height, rot)
        # Check for out-of-bounds.
        oob = False
        for cx, cy in cand_geom.vertices:
            if cx < min_x or cy < min_y or cx > max_x or cy > max_y:
                oob = True
                break
        if oob:
            continue
        # Check for collisions.
        collision = False
        for collision_geom in collision_geoms:
            if collision_geom.intersects(cand_geom):
                collision = True
                break
        # Success!
        if not collision:
            return distance, angle, cand_geom

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
    robot (top-down view)."""
    length = 0.75  # meters
    width = 0.2
    x = pose.x
    y = pose.y
    theta = pose.rot.to_yaw()
    return Rectangle(x, y, length, width, theta)
