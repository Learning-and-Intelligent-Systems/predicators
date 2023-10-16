"""Interface for finding objects by moving around and running detection."""

from typing import Any, Collection, Dict, List, Optional, Tuple

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.lease import LeaseClient
from bosdyn.client.sdk import Robot
from matplotlib import pyplot as plt

from predicators import utils
from predicators.spot_utils.perception.object_detection import \
    detect_objects, display_camera_detections
from predicators.spot_utils.perception.perception_structs import \
    ObjectDetectionID, RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, \
    DEFAULT_HAND_LOOK_FLOOR_POSE


def _find_objects_with_choreographed_moves(
    robot: Robot,
    localizer: SpotLocalizer,
    object_ids: Collection[ObjectDetectionID],
    relative_base_moves: List[math_helpers.SE2Pose],
    relative_hand_moves: Optional[List[math_helpers.SE3Pose]] = None,
    open_and_close_gripper: bool = True,
    use_gui: bool = False,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Helper for object search with hard-coded relative moves."""

    if relative_hand_moves is not None:
        assert len(relative_hand_moves) == len(relative_base_moves)

    # Naively combine detections and artifacts using the most recent ones.
    all_detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    all_artifacts: Dict[str, Any] = {}
    # Save all RGBDs in case of failure so we can analyze them.
    all_rgbds: List[Dict[str, RGBDImageWithContext]] = []

    # Open the hand to mitigate possible occlusions.
    if open_and_close_gripper:
        open_gripper(robot)

    # Run detection once to start before moving.
    rgbds = capture_images(robot, localizer)
    all_rgbds.append(rgbds)
    detections, artifacts = detect_objects(object_ids, rgbds)
    all_detections.update(detections)
    all_artifacts.update(artifacts)

    # Display the detections on screen so that we can follow along.
    if use_gui:
        num_cameras = len(rgbds)
        num_display_rows = int(np.ceil(np.sqrt(num_cameras)))
        num_display_cols = int(np.ceil(num_cameras / num_display_rows))
        display_fig_scale = 10
        fig, display_axes = plt.subplots(
            num_display_rows,
            num_display_cols,
            squeeze=False,
            figsize=(display_fig_scale * num_display_rows,
                     display_fig_scale * num_display_cols))

        plt.ion()
        plt.show()
        plt.pause(0.1)
        display_camera_detections(artifacts, display_axes)
        fig.canvas.draw()
        plt.pause(0.1)

    for i, relative_pose in enumerate(relative_base_moves):
        remaining_object_ids = set(object_ids) - set(all_detections)
        print(f"Found objects: {set(all_detections)}")
        print(f"Remaining objects: {remaining_object_ids}")

        # Success, finish.
        if not remaining_object_ids:
            break

        # Move and re-capture.
        navigate_to_relative_pose(robot, relative_pose)

        if relative_hand_moves is not None:
            hand_move = relative_hand_moves[i]
            move_hand_to_relative_pose(robot, hand_move)

        localizer.localize()
        rgbds = capture_images(robot, localizer)
        all_rgbds.append(rgbds)
        detections, artifacts = detect_objects(object_ids, rgbds)
        all_detections.update(detections)
        all_artifacts.update(artifacts)

        # Update the GUI.
        if use_gui:
            display_camera_detections(artifacts, display_axes)
            fig.canvas.draw()
            plt.pause(0.1)

    # Stop the display.
    plt.close()

    # Close the gripper.
    if open_and_close_gripper:
        close_gripper(robot)

    # Success, finish.
    remaining_object_ids = set(object_ids) - set(all_detections)
    if not remaining_object_ids:
        return all_detections, all_artifacts

    # Fail. Analyze the RGBDs if you want (by uncommenting here).
    # import imageio.v2 as iio
    # for i, rgbds in enumerate(all_rgbds):
    #     for camera, rgbd in rgbds.items():
    #         path = f"init_search_for_objects_angle{i}_{camera}.png"
    #         iio.imsave(path, rgbd.rgb)
    #         print(f"Wrote out to {path}.")

    remaining_object_ids = set(object_ids) - set(all_detections)
    raise RuntimeError(f"Could not find objects: {remaining_object_ids}")


def init_search_for_objects(
    robot: Robot,
    localizer: SpotLocalizer,
    object_ids: Collection[ObjectDetectionID],
    num_spins: int = 8,
    relative_hand_moves: Optional[List[math_helpers.SE3Pose]] = None,
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Spin around in place looking for objects.

    Raise a RuntimeError if an object can't be found after spinning.
    """
    spin_amount = 2 * np.pi / (num_spins + 1)
    relative_pose = math_helpers.SE2Pose(0, 0, spin_amount)
    base_moves = [relative_pose] * num_spins
    return _find_objects_with_choreographed_moves(
        robot,
        localizer,
        object_ids,
        base_moves,
        relative_hand_moves=relative_hand_moves,
        use_gui=True)


def find_objects(
    robot: Robot,
    localizer: SpotLocalizer,
    lease_client: LeaseClient,
    object_ids: Collection[ObjectDetectionID],
) -> None:
    """Execute a hard-coded sequence of movements and hope that one of them
    puts the lost object in view.

    This is very specifically designed for the case where an object has
    fallen in the immediate vicinity.
    """
    moves = [
        # First move way back and don't move the hand. This is useful when the
        # object has not actually fallen, but wasn't grasped.
        (math_helpers.SE2Pose(-0.75, 0.0, 0.0), DEFAULT_HAND_LOOK_DOWN_POSE),
        # Just look down at the floor.
        (math_helpers.SE2Pose(0.0, 0.0, 0.0), DEFAULT_HAND_LOOK_FLOOR_POSE),
        # Spin to the right and look at the floor.
        (math_helpers.SE2Pose(0.0, 0.0,
                              np.pi / 6), DEFAULT_HAND_LOOK_FLOOR_POSE),
        # Spin to the left and look at the floor.
        (math_helpers.SE2Pose(0.0, 0.0,
                              -np.pi / 6), DEFAULT_HAND_LOOK_FLOOR_POSE),
    ]
    base_moves, hand_moves = zip(*moves)
    try:
        # Don't open and close the gripper because we need the object to be
        # in view when the action has finished, and we can't leave the gripper
        # open because then HandEmpty will misfire.
        _find_objects_with_choreographed_moves(robot,
                                               localizer,
                                               object_ids,
                                               base_moves,
                                               hand_moves,
                                               open_and_close_gripper=False)
    except RuntimeError:
        prompt = ("Please take control of the robot and make the object "
                  "become in its view. Hit the 'Enter' key when you're done!")
        utils.prompt_user(prompt)
        lease_client.take()


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # This test assumes that the 408, 409, and 410 april tags can be found.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators.settings import CFG
    from predicators.spot_utils.perception.object_detection import \
        AprilTagObjectDetectionID
    from predicators.spot_utils.utils import get_graph_nav_dir, verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()
        sdk = create_standard_sdk('FindObjectsTestClient')
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

        object_ids = [
            # Table.
            AprilTagObjectDetectionID(408),
            # Table.
            AprilTagObjectDetectionID(409),
            # Cube.
            AprilTagObjectDetectionID(410),
        ]

        # Test running the initial search for objects.
        input("Set up initial object search test")
        init_search_for_objects(robot, localizer, object_ids)

        # Test finding a lost object.
        input("Set up finding lost object test")
        cube = object_ids[2]
        find_objects(robot, localizer, lease_client, {cube})

    _run_manual_test()
