"""Interface for finding objects by moving around and running detection."""

from typing import Any, Collection, Dict, List, Tuple

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.lease import LeaseClient
from bosdyn.client.sdk import Robot

from predicators.spot_utils.perception.object_detection import detect_objects
from predicators.spot_utils.perception.perception_structs import \
    ObjectDetectionID, RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
    move_hand_to_relative_pose, open_gripper
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE


def init_search_for_objects(
    robot: Robot,
    localizer: SpotLocalizer,
    object_ids: Collection[ObjectDetectionID],
    num_spins: int = 8
) -> Tuple[Dict[ObjectDetectionID, math_helpers.SE3Pose], Dict[str, Any]]:
    """Spin around in place looking for objects.

    Raise a RuntimeError if an object can't be found after spinning.
    """
    # Naively combine detections and artifacts using the most recent ones.
    all_detections: Dict[ObjectDetectionID, math_helpers.SE3Pose] = {}
    all_artifacts: Dict[str, Any] = {}
    # Save all RGBDs in case of failure so we can analyze them.
    all_rgbds: List[Dict[str, RGBDImageWithContext]] = []

    # Open the hand to mitigate possible occlusions.
    open_gripper(robot)

    # Run detection once to start before spinning.
    rgbds = capture_images(robot, localizer)
    all_rgbds.append(rgbds)
    detections, artifacts = detect_objects(object_ids, rgbds)
    all_detections.update(detections)
    all_artifacts.update(artifacts)

    spin_amount = 2 * np.pi / (num_spins + 1)
    relative_pose = math_helpers.SE2Pose(0, 0, spin_amount)

    for _ in range(num_spins):
        remaining_object_ids = set(object_ids) - set(all_detections)
        print(f"Found objects: {set(all_detections)}")
        print(f"Remaining objects: {remaining_object_ids}")

        # Success, finish.
        if not remaining_object_ids:
            break

        # Spin and re-capture.
        navigate_to_relative_pose(robot, relative_pose)
        localizer.localize()
        rgbds = capture_images(robot, localizer)
        all_rgbds.append(rgbds)
        detections, artifacts = detect_objects(object_ids, rgbds)
        all_detections.update(detections)
        all_artifacts.update(artifacts)

    # Close the gripper.
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


def find_object(robot: Robot, find_controller_move_queue_idx: int,
                lease_client: LeaseClient) -> None:
    """Execute look around."""
    # Execute a hard-coded sequence of movements and hope that one of them
    # puts the lost object in view. This is very specifically designed for
    # the case where an object has fallen in the immediate vicinity.
    stow_arm(robot)

    # First move way back and don't move the hand. This is useful when the
    # object has not actually fallen, but wasn't grasped.
    if find_controller_move_queue_idx == 1:
        move_back_pose = math_helpers.SE2Pose(-0.75, 0.0, 0.0)
        navigate_to_relative_pose(robot, move_back_pose)
        return

    # Now just look down.
    if find_controller_move_queue_idx == 2:
        pass
    # Move to the right.
    elif find_controller_move_queue_idx == 3:
        look_right_pose = math_helpers.SE2Pose(0.0, 0.0, np.pi / 6)
        navigate_to_relative_pose(robot, look_right_pose)
    # Move to the left.
    elif find_controller_move_queue_idx == 4:
        look_left_pose = math_helpers.SE2Pose(0.0, 0.0, -np.pi / 6)
        navigate_to_relative_pose(robot, look_left_pose)
    else:
        prompt = """Please take control of the robot and make the
        object become in its view. Hit the 'Enter' key
        when you're done!"""
        utils.prompt_user(prompt)
        lease_client.take()
        return

    # Move the hand to get a view of the floor.
    move_hand_to_relative_pose(robot, DEFAULT_HAND_LOOK_DOWN_POSE)
    return


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # This test assumes that the 408, 409, and 410 april tags can be found.

    # pylint: disable=ungrouped-imports
    from pathlib import Path

    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.object_detection import \
        AprilTagObjectDetectionID
    from predicators.spot_utils.utils import verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        upload_dir = Path(__file__).parent.parent / "graph_nav_maps"
        path = upload_dir / CFG.spot_graph_nav_map

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
            AprilTagObjectDetectionID(
                408, math_helpers.SE3Pose(0.0, 0.5, 0.0, math_helpers.Quat())),
            # Table.
            AprilTagObjectDetectionID(
                409, math_helpers.SE3Pose(0.0, 0.5, 0.0, math_helpers.Quat())),
            # Cube.
            AprilTagObjectDetectionID(
                410, math_helpers.SE3Pose(0.0, 0.0, 0.0, math_helpers.Quat())),
        ]

        init_search_for_objects(robot, localizer, object_ids)

    _run_manual_test()
