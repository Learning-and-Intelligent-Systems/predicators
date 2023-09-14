"""Integration tests for spot utilities.

Run with --spot_robot_ip and any other flags.
"""

import logging
import time
from pathlib import Path
from typing import Tuple

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
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import verify_estop


def test_find_move_pick_place(
    manipuland_id: ObjectDetectionID,
    init_surface_id: ObjectDetectionID,
    target_surface_id: ObjectDetectionID,
) -> None:
    """Find the given object and surfaces, pick the object from the first
    surface, and place it on the second surface."""
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


if __name__ == "__main__":
    # Parse flags.
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.update_config(args)

    # Run tests.
    test_find_move_pick_place()
