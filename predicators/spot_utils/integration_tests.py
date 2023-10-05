"""Integration tests for spot utilities.

Run with --spot_robot_ip and any other flags.
"""
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate

from predicators import utils
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    AprilTagObjectDetectionID, LanguageObjectDetectionID, detect_objects, \
    get_object_center_pixel_from_artifacts
from predicators.spot_utils.perception.perception_structs import \
    ObjectDetectionID
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_dump import dump_container
from predicators.spot_utils.skills.spot_find_objects import \
    init_search_for_objects
from predicators.spot_utils.skills.spot_grasp import grasp_at_pixel
from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import go_home, \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_place import place_at_relative_position
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, \
    DEFAULT_HAND_LOOK_FLOOR_POSE, get_relative_se2_from_se3, \
    get_spot_home_pose, sample_move_offset_from_target, spot_pose_to_geom2d, \
    verify_estop


def test_find_move_pick_place(
    robot: Robot,
    localizer: SpotLocalizer,
    manipuland_id: ObjectDetectionID,
    init_surface_id: Optional[ObjectDetectionID],
    target_surface_id: ObjectDetectionID,
    pre_pick_surface_nav_distance: float = 1.25,
    pre_pick_floor_nav_distance: float = 1.75,
    pre_place_nav_distance: float = 1.0,
    place_offset_z: float = 0.2,
) -> None:
    """Find the given object and surfaces, pick the object from the first
    surface, and place it on the second surface.

    The surface nav parameters determine where the robot should navigate
    with respect to the surfaces when picking and placing. The
    intelligence for choosing these offsets is external to the skills
    (e.g., they might be sampled).
    """
    localizer.localize()
    go_home(robot, localizer)
    localizer.localize()

    # Test assumes that objects are in front of the robot's home position.
    home_pose = get_spot_home_pose()
    pre_pick_nav_angle = home_pose.angle - np.pi
    pre_place_nav_angle = pre_pick_nav_angle

    # Find objects.
    object_ids = [manipuland_id]
    if init_surface_id is not None:
        object_ids.append(init_surface_id)
    object_ids.append(target_surface_id)
    detections, _ = init_search_for_objects(robot, localizer, object_ids)

    # Get current robot pose.
    robot_pose = localizer.get_last_robot_pose()
    if init_surface_id is not None:
        # Navigate to the first surface.
        rel_pose = get_relative_se2_from_se3(robot_pose,
                                             detections[init_surface_id],
                                             pre_pick_surface_nav_distance,
                                             pre_pick_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()
    else:
        # In this case, we assume the object is on the floor.
        rel_pose = get_relative_se2_from_se3(robot_pose,
                                             detections[manipuland_id],
                                             pre_pick_floor_nav_distance,
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
    grasp_at_pixel(robot, rgbds[hand_camera], pixel)
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


def test_all_find_move_pick_place() -> None:
    """Multiple tests for find, move, pick, place."""

    # Parse flags.
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.update_config(args)

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

    # Run test with april tag cube.
    init_surface = AprilTagObjectDetectionID(408)
    target_surface = AprilTagObjectDetectionID(409)
    cube = AprilTagObjectDetectionID(410)

    # Assume that the tables are at the "front" of the room (with respect to
    # the robot home pose).
    input("Set up the tables and CUBE in front of the robot")
    test_find_move_pick_place(robot, localizer, cube, init_surface,
                              target_surface)

    # Run test with brush.
    brush = LanguageObjectDetectionID("brush")
    input("Set up the tables and BRUSH in front of the robot")
    test_find_move_pick_place(robot, localizer, brush, init_surface,
                              target_surface)

    # Run test with cube on floor.
    input("Place the cube anywhere on the floor")
    test_find_move_pick_place(robot, localizer, cube, None, target_surface)


def test_move_with_sampling() -> None:
    """Test for moving to a surface with a sampled rotation and distance,
    taking into account potential collisions with walls and other surfaces."""

    # Approximate values for the set up on the fourth floor.
    room_bounds = (0.4, -1.0, 4.0, 2.0)  # min x, min y, max x, max y
    surface_radius = 0.2

    num_samples = 10
    max_distance = 1.5

    # Parse flags.
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.update_config(args)

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

    # Run test with april tag cube.
    surface1 = AprilTagObjectDetectionID(408)
    surface2 = AprilTagObjectDetectionID(409)

    go_home(robot, localizer)
    localizer.localize()

    # Find objects.
    object_ids = [surface1, surface2]
    detections, _ = init_search_for_objects(robot, localizer, object_ids)

    # Create collision geoms using known object sizes.
    collision_geoms = [
        utils.Circle(detections[o].x, detections[o].y, surface_radius)
        for o in object_ids
    ]

    # Repeatedly sample valid places to move and move there.
    target_pose = detections[surface1]
    target_origin = (target_pose.x, target_pose.y)
    rng = np.random.default_rng(123)
    for i in range(num_samples):
        localizer.localize()
        robot_pose = localizer.get_last_robot_pose()
        robot_geom = spot_pose_to_geom2d(robot_pose)
        distance, angle, next_robot_geom = sample_move_offset_from_target(
            target_origin,
            robot_geom,
            collision_geoms,
            rng,
            max_distance=max_distance,
            room_bounds=room_bounds,
        )
        # Visualize everything.
        figsize = (1.1 * (room_bounds[2] - room_bounds[0]),
                   1.1 * (room_bounds[3] - room_bounds[1]))
        _, ax = plt.subplots(1, 1, figsize=figsize)
        robot_geom.plot(ax, facecolor="lightgreen", edgecolor="black")
        # Draw the origin of the robot, which should be the back right leg.
        ax.scatter([robot_geom.x], [robot_geom.y],
                   s=120,
                   marker="*",
                   color="gray",
                   zorder=3)
        next_robot_geom.plot(ax,
                             facecolor="lightblue",
                             edgecolor="black",
                             linestyle="--")
        ax.scatter([next_robot_geom.x], [next_robot_geom.y],
                   s=120,
                   marker="*",
                   color="gray",
                   zorder=3)
        for object_id, geom in zip(object_ids, collision_geoms):
            geom.plot(ax, facecolor="lightgray", edgecolor="black")
            if object_id == surface1:
                ax.scatter([geom.x], [geom.y],
                           s=320,
                           marker="*",
                           color="gold",
                           zorder=3)
        # Draw the walls.
        min_x, min_y, max_x, max_y = room_bounds
        ax.plot((min_x, min_x), (min_y, max_y), linestyle="--", color="gray")
        ax.plot((max_x, max_x), (min_y, max_y), linestyle="--", color="gray")
        ax.plot((min_x, max_x), (min_y, min_y), linestyle="--", color="gray")
        ax.plot((min_x, max_x), (max_y, max_y), linestyle="--", color="gray")
        plt.savefig(f"sampling_integration_test_{i}.png")

        # Execute the move.
        rel_pose = get_relative_se2_from_se3(robot_pose, target_pose, distance,
                                             angle)
        navigate_to_relative_pose(robot, rel_pose)


def test_repeated_brush_bucket_dump_pick_place(
        num_repeats: int = 3,
        pre_pick_floor_nav_distance: float = 1.25,
        pre_place_nav_distance: float = 0.75,
        pre_dump_nav_distance: float = 1.25,
        place_offset_x: float = 0.1,
        place_offset_z: float = 0.5,
        post_dump_place_offset_z: float = 0.0,
        bucket_grasp_dr: int = 50) -> None:
    """Test repeatedly picking, placing, and dumping a brush into a bucket.

    The brush should start outside the bucket.
    """

    # Parse flags.
    args = utils.parse_args(env_required=False,
                            seed_required=False,
                            approach_required=False)
    utils.update_config(args)

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

    # Test assumes that objects are in front of the robot's home position.
    home_pose = get_spot_home_pose()
    pre_pick_nav_angle = home_pose.angle - np.pi
    pre_place_nav_angle = pre_pick_nav_angle

    # Use vision-language model to detect bucket and brush.
    bucket = LanguageObjectDetectionID("large red bucket")
    brush = LanguageObjectDetectionID("brush")

    for _ in range(num_repeats):
        # Find the brush and bucket.
        go_home(robot, localizer)
        localizer.localize()

        # Find objects.
        detections, _ = init_search_for_objects(robot, localizer,
                                                [bucket, brush])
        localizer.localize()

        # Move to the brush.
        robot_pose = localizer.get_last_robot_pose()
        rel_pose = get_relative_se2_from_se3(robot_pose, detections[brush],
                                             pre_pick_floor_nav_distance,
                                             pre_pick_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()

        # Look down at the floor.
        move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_FLOOR_POSE)
        open_gripper(robot)
        localizer.localize()

        # Capture an image from the hand camera.
        hand_camera = "hand_color_image"
        rgbds = capture_images(robot, localizer, [hand_camera])

        # Run detection to get a pixel for grasping.
        _, artifacts = detect_objects([brush], rgbds)
        pixel = get_object_center_pixel_from_artifacts(artifacts, brush,
                                                       hand_camera)

        # Pick at the pixel with a top-down grasp.
        grasp_at_pixel(robot, rgbds[hand_camera], pixel)
        localizer.localize()

        # Stow the arm.
        stow_arm(robot)
        localizer.localize()

        # Move to the bucket.
        robot_pose = localizer.get_last_robot_pose()
        rel_pose = get_relative_se2_from_se3(robot_pose, detections[bucket],
                                             pre_place_nav_distance,
                                             pre_place_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()

        # Place in the bucket.
        robot_pose = localizer.get_last_robot_pose()
        surface_rel_pose = robot_pose.inverse() * detections[bucket]
        place_rel_pos = math_helpers.Vec3(
            x=surface_rel_pose.x + place_offset_x,
            y=surface_rel_pose.y,
            z=surface_rel_pose.z + place_offset_z)
        place_at_relative_position(robot,
                                   place_rel_pos,
                                   downward_angle=np.pi / 2)
        localizer.localize()

        # Stow the arm again.
        stow_arm(robot)
        localizer.localize()

        # Navigate to look at the bucket.
        robot_pose = localizer.get_last_robot_pose()
        rel_pose = get_relative_se2_from_se3(robot_pose, detections[bucket],
                                             pre_dump_nav_distance,
                                             pre_place_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()

        # Look at the bucket.
        move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_FLOOR_POSE)
        localizer.localize()

        # Capture an image.
        rgbds = capture_images(robot, localizer, [hand_camera])
        rgbd = rgbds[hand_camera]

        # Choose a grasp.
        _, artifacts = detect_objects([bucket], rgbds)

        r, c = get_object_center_pixel_from_artifacts(artifacts, bucket,
                                                      hand_camera)
        pixel = (r + bucket_grasp_dr, c)

        # Grasp at the pixel with a top-down grasp.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        grasp_at_pixel(robot, rgbd, pixel, grasp_rot=top_down_rot)

        # Dump!
        dump_container(robot, post_dump_place_offset_z)
        localizer.localize()

        # Stow the arm again.
        stow_arm(robot)
        localizer.localize()


if __name__ == "__main__":
    test_all_find_move_pick_place()
    test_move_with_sampling()
    test_repeated_brush_bucket_dump_pick_place()
