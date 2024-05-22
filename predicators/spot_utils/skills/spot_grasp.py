"""Interface for spot grasping skill."""

import logging
import time
from typing import Optional, Tuple

import pbrspot
from bosdyn.api import geometry_pb2, manipulation_api_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import VISION_FRAME_NAME, \
    get_vision_tform_body
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.sdk import Robot

from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.skills.spot_hand_move import close_gripper
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.utils import get_robot_state


def grasp_at_pixel(robot: Robot,
                   rgbd: RGBDImageWithContext,
                   pixel: Tuple[int, int],
                   grasp_rot: Optional[math_helpers.Quat] = None,
                   rot_thresh: float = 0.17,
                   move_while_grasping: bool = True,
                   timeout: float = 20.0,
                   retry_with_no_constraints: bool = False) -> None:
    """Grasp an object at a specified pixel in the RGBD image, which should be
    from the hand camera and should be up to date with the robot's state.

    The `move_while_grasping` param dictates whether we're allowing the
    robot to automatically move its feet while grasping or not.

    The `retry_with_no_constraints` dictates whether after failing to grasp we
    try again but with all constraints on the grasp removed.
    """
    assert rgbd.camera_name == "hand_color_image"

    manipulation_client = robot.ensure_client(
        ManipulationApiClient.default_service_name)

    if move_while_grasping:
        # Stow Arm first (only if robot is allowed to move while grasping)
        stow_arm(robot)

    pick_vec = geometry_pb2.Vec2(x=pixel[0], y=pixel[1])

    # Build the proto. Note that the possible settings for walk_gaze_mode
    # can be found here:
    # https://dev.bostondynamics.com/protos/bosdyn/api/proto_reference.html
    walk_gaze_mode = 1 if move_while_grasping else 2
    grasp = manipulation_api_pb2.PickObjectInImage(
        pixel_xy=pick_vec,
        transforms_snapshot_for_camera=rgbd.transforms_snapshot,
        frame_name_image_sensor=rgbd.frame_name_image_sensor,
        camera_model=rgbd.camera_model,
        walk_gaze_mode=walk_gaze_mode)

    # If a desired rotation for the hand was given, add a grasp constraint.
    if grasp_rot is not None:
        robot_state = get_robot_state(robot)
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME  # pylint: disable=no-member
        vision_tform_body = get_vision_tform_body(
            robot_state.kinematic_state.transforms_snapshot)
        # Rotation from the body to our desired grasp.
        vision_rot = vision_tform_body.rotation * grasp_rot
        # Turn into a proto.
        constraint = grasp.grasp_params.allowable_orientation.add()  # pylint: disable=no-member
        constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
            vision_rot.to_proto())
        constraint.rotation_with_tolerance.threshold_radians = rot_thresh

    # Create the request.
    grasp_request = manipulation_api_pb2.ManipulationApiRequest(
        pick_object_in_image=grasp)

    # Send the request.
    cmd_response = manipulation_client.manipulation_api_command(
        manipulation_api_request=grasp_request)

    # Get feedback from the robot and execute grasping, repeating until a
    # proper response is received.
    start_time = time.perf_counter()
    while (time.perf_counter() - start_time) <= timeout:
        feedback_request = manipulation_api_pb2.ManipulationApiFeedbackRequest(
            manipulation_cmd_id=cmd_response.manipulation_cmd_id)
        response = manipulation_client.manipulation_api_feedback_command(
            manipulation_api_feedback_request=feedback_request)
        # Uncomment this for debugging why the grasp might be failing
        # repeatedly.
        # print(response.current_state)
        if response.current_state in [
                manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION
        ]:
            break
    if (time.perf_counter() - start_time) > timeout:
        logging.warning("Timed out waiting for grasp to execute!")

    # Retry grasping with no constraints if the corresponding arg is true.
    if response.current_state in [
            manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION,
            manipulation_api_pb2.MANIP_STATE_GRASP_FAILED
    ] and retry_with_no_constraints:
        print("WARNING: grasp planning failed, retrying with no constraint")
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=rgbd.transforms_snapshot,
            frame_name_image_sensor=rgbd.frame_name_image_sensor,
            camera_model=rgbd.camera_model,
            walk_gaze_mode=1)
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp)
        cmd_response = manipulation_client.manipulation_api_command(
            manipulation_api_request=grasp_request)
        while (time.perf_counter() - start_time) <= timeout:
            feedback_request = manipulation_api_pb2.\
                ManipulationApiFeedbackRequest(
                manipulation_cmd_id=cmd_response.manipulation_cmd_id)
            response = manipulation_client.manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)
            if response.current_state in [
                    manipulation_api_pb2.MANIP_STATE_GRASP_SUCCEEDED,
                    manipulation_api_pb2.MANIP_STATE_GRASP_FAILED,
                    manipulation_api_pb2.MANIP_STATE_GRASP_PLANNING_NO_SOLUTION
            ]:
                break
        if (time.perf_counter() - start_time) > timeout:
            logging.warning("Timed out waiting for grasp to execute!")

    # Sometimes the grasp doesn't properly close the gripper, so force this
    # to ensure a pick has happened!
    close_gripper(robot)
    time.sleep(0.5)


def simulated_grasp_at_pixel(sim_robot: pbrspot.spot.Spot,
                             obj_to_be_grasped: pbrspot.body.Body) -> None:
    """Simulated grasping in pybullet.

    For now, this is really dumb and just teleports the object into the
    hand. In the near future, a simple thing to add is just making sure
    the hand can IK to the object. In the further future, this function
    will hopefully be made be much more reasonable in general (we
    probably want to do BEHAVIOR-style simulated grasping)!
    """
    # Start by opening the hand.
    sim_robot.hand.Open()
    # Now, teleport the object into the hand.
    obj_to_be_grasped.set_pose(sim_robot.hand.get_pose())


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # pylint: disable=ungrouped-imports
    import numpy as np
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.spot_cameras import capture_images
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import get_graph_nav_dir, \
        get_pixel_from_user, verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()
        sdk = create_standard_sdk('GraspSkillTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        lease_keepalive = LeaseKeepAlive(lease_client,
                                         must_acquire=True,
                                         return_at_exit=True)
        robot.time_sync.wait_for_sync()
        localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)

        # Capture an image.
        camera = "hand_color_image"
        rgbd = capture_images(robot, localizer, [camera])[camera]

        # Select a pixel manually.
        pixel = get_pixel_from_user(rgbd.rgb)

        # Grasp at the pixel with a top-down grasp.
        top_down_rot = math_helpers.Quat.from_pitch(np.pi / 2)
        grasp_at_pixel(robot, rgbd, pixel, grasp_rot=top_down_rot)

    _run_manual_test()
