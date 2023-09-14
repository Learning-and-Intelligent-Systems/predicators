"""Integration tests for spot utilities.

Run with --spot_robot_ip and any other flags.
"""

from pathlib import Path

import numpy as np
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.util import authenticate

from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    detect_objects, get_object_center_pixel_from_artifacts
from predicators.spot_utils.perception.perception_structs import \
    ObjectDetectionID
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_find_objects import find_objects
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import go_home, \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, \
    get_relative_se2_from_se3, verify_estop


def test_find_move_pick_place(
    manipuland_id: ObjectDetectionID,
    init_surface_id: ObjectDetectionID,
    target_surface_id: ObjectDetectionID,
    pre_pick_nav_distance: float = 1.25,
    pre_place_nav_distance: float = 1.0,
    pre_pick_nav_angle: float = -np.pi / 2,
    pre_place_nav_angle: float = -np.pi / 2,
    place_offset_z: float = 0.25,
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

    go_home(robot, localizer)
    localizer.localize()

    # Find objects.
    object_ids = [manipuland_id, init_surface_id, target_surface_id]
    detections, _ = find_objects(robot, localizer, object_ids)

    # Get current robot pose.
    robot_pose = localizer.get_last_robot_pose()

    # Navigate to the first surface.
    rel_pose = get_relative_se2_from_se3(robot_pose,
                                         detections[init_surface_id],
                                         pre_pick_nav_distance,
                                         pre_pick_nav_angle)
    navigate_to_relative_pose(robot, rel_pose)
    localizer.localize()

    # Look down at the surface.
    move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_DOWN_POSE)
    open_gripper(robot)

    # Capture an image from the hand camera.
    hand_camera = "hand_color_image"
    rgbds = capture_images(robot, localizer, [hand_camera])

    # Run detection to get a pixel for grasping.
    _, artifacts = detect_objects([manipuland_id], rgbds)
    pixel = get_object_center_pixel_from_artifacts(artifacts, manipuland_id,
                                                   hand_camera)

    # Pick at the pixel with a top-down grasp.
    top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
    grasp_at_pixel(robot, rgbds[hand_camera], pixel, grasp_rot=top_down_rot)
    localizer.localize()

    # Stow the arm.
    stow_arm(robot)

    # Navigate to the other surface.
    robot_pose = localizer.get_last_robot_pose()
    rel_pose = get_relative_se2_from_se3(robot_pose,
                                         detections[target_surface_id],
                                         pre_place_nav_distance,
                                         pre_place_nav_angle)
    navigate_to_relative_pose(robot, rel_pose)
    localizer.localize()

    # Place on the surface.
    robot_pose = localizer.get_last_robot_pose()
    surface_rel_pose = robot_pose.inverse() * detections[target_surface_id]
    place_rel_pos = math_helpers.Vec3(x=surface_rel_pose.x,
                                      y=surface_rel_pose.y,
                                      z=surface_rel_pose.z + place_offset_z)
    place_at_relative_position(robot, place_rel_pos)

    # Finish by stowing arm again.
    stow_arm(robot)

    # Give up the lease.
    lease_client.return_lease()
    lease_keepalive.shutdown()


if __name__ == "__main__":
    from predicators.spot_utils.perception.object_detection import \
        AprilTagObjectDetectionID, LanguageObjectDetectionID

    # Parse flags.
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.update_config(args)

    # Run test with april tag cube.
    init_surface = AprilTagObjectDetectionID(
        408, math_helpers.SE3Pose(0.0, 0.25, 0.0, math_helpers.Quat()))
    target_surface = AprilTagObjectDetectionID(
        409, math_helpers.SE3Pose(0.0, 0.25, 0.0, math_helpers.Quat()))
    cube = AprilTagObjectDetectionID(
        410, math_helpers.SE3Pose(0.0, 0.0, 0.0, math_helpers.Quat()))

    # Assume that the tables are at the "front" of the room (with the hall
    # on the left when on the fourth floor).
    input("Set up the tables and CUBE on the north wall")
    test_find_move_pick_place(cube, init_surface, target_surface)

    # Run test with brush.
    # Assume that the tables are at the "front" of the room (with the hall
    # on the left when on the fourth floor).
    brush = LanguageObjectDetectionID("brush")
    input("Set up the tables and BRUSH on the north wall")
    test_find_move_pick_place(brush, init_surface, target_surface)

    # Run test with tables moved so that the init table is on the wall adjacent
    # to the hallway and the target table is on the opposite wall.
    # Note that we need to change the offsets because the april tags are
    # now rotated.
    input("Set up the tables and CUBE on opposite walls")
    init_surface = AprilTagObjectDetectionID(
        408, math_helpers.SE3Pose(-0.25, 0.0, 0.0, math_helpers.Quat()))
    target_surface = AprilTagObjectDetectionID(
        409, math_helpers.SE3Pose(0.25, 0.0, 0.0, math_helpers.Quat()))
    test_find_move_pick_place(cube,
                              init_surface,
                              target_surface,
                              pre_pick_nav_angle=0,
                              pre_place_nav_angle=np.pi)

    drill = LanguageObjectDetectionID("drill")
    input("Set up the tables and DRILL on opposite walls")
    test_find_move_pick_place(drill,
                              init_surface,
                              target_surface,
                              pre_pick_nav_angle=0,
                              pre_place_nav_angle=np.pi)
