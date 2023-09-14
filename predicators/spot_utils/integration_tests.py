"""Integration tests for spot utilities.

Run with --spot_robot_ip and any other flags.
"""

import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.api.geometry_pb2 import SE2Velocity, SE2VelocityLimit, Vec2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, \
    get_se2_a_tform_b
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate

from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.perception.perception_structs import \
    ObjectDetectionID
from predicators.spot_utils.skills.spot_find_objects import find_objects
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import verify_estop


def test_find_move_pick_place(
    manipuland_id: ObjectDetectionID,
    init_surface_id: ObjectDetectionID,
    target_surface_id: ObjectDetectionID,
    surface_nav_distance: float,
    surface_nav_angle: float,
) -> None:
    """Find the given object and surfaces, pick the object from the first
    surface, and place it on the second surface.

    The surface nav parameters determine where the robot should navigate
    with respect to the surfaces when picking and placing. The
    intelligence for choosing these offsets is external to the skills
    (e.g., they might be sampled).
    """
    # Set up the robot and localizer.
    hostname = CFG.spot_robot_ip
    upload_dir = Path(__file__).parent / "graph_nav_maps"
    path = upload_dir / CFG.spot_graph_nav_map
    sdk = create_standard_sdk("TestClient")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(lease_client,
                                     must_acquire=True,
                                     return_at_exit=True)
    assert path.exists()
    localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)

    # Find objects.
    object_ids = [manipuland_id, init_surface_id, target_surface_id]
    detections, artifacts = find_objects(robot, localizer, object_ids)

    # Compute waypoints given the surface offsets.
    def _get_waypoint(
            surface_pose: math_helpers.SE3Pose) -> math_helpers.SE2Pose:
        dx = np.cos(surface_nav_angle) * surface_nav_distance
        dy = np.sin(surface_nav_angle) * surface_nav_distance
        x = surface_pose.x + dx
        y = surface_pose.y + dy
        # Face towards the center.
        rot = 2 * np.pi - surface_nav_angle
        return math_helpers.SE2Pose(x, y, rot)

    start_waypoint = _get_waypoint(detections[init_surface_id])
    end_waypoint = _get_waypoint(detections[target_surface_id])

    # Navigate to the first waypoint.
    robot_pose = localizer.get_last_robot_pose().get_closest_se2_transform()
    rel_pose = robot_pose.inverse() * start_waypoint
    navigate_to_relative_pose(robot, rel_pose)
    localizer.localize()

    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    from predicators.spot_utils.perception.object_detection import \
        AprilTagObjectDetectionID

    # Parse flags.
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.update_config(args)

    # Run tests.
    init_surface = AprilTagObjectDetectionID(
        408, math_helpers.SE3Pose(0.0, 0.5, 0.0, math_helpers.Quat()))
    target_surface = AprilTagObjectDetectionID(
        409, math_helpers.SE3Pose(0.0, 0.5, 0.0, math_helpers.Quat()))
    cube = AprilTagObjectDetectionID(
        410, math_helpers.SE3Pose(0.0, 0.0, 0.0, math_helpers.Quat()))
    # Assume that the tables are at the "front" of the room (with the hall
    # on the left in room 408).
    test_find_move_pick_place(cube,
                              init_surface,
                              target_surface,
                              surface_nav_distance=1.0,
                              surface_nav_angle=-np.pi / 2)
