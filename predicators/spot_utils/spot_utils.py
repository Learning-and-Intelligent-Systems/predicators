"""Utility functions to interface with the Boston Dynamics Spot robot."""

import os
import sys
import time
from typing import Any, Sequence

import bosdyn.client
import bosdyn.client.estop
import bosdyn.client.lease
import bosdyn.client.util
import cv2
import numpy as np
from bosdyn.api import arm_command_pb2, basic_command_pb2, estop_pb2, \
    geometry_pb2, image_pb2, manipulation_api_pb2
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
from bosdyn.client import math_helpers
from bosdyn.client.estop import EstopClient
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, \
    GRAV_ALIGNED_BODY_FRAME_NAME, ODOM_FRAME_NAME, VISION_FRAME_NAME, \
    get_a_tform_b, get_se2_a_tform_b, get_vision_tform_body
from bosdyn.client.image import ImageClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, \
    RobotCommandClient, block_until_arm_arrives, blocking_stand
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot

from predicators.settings import CFG
from predicators.spot_utils.helpers.graph_nav_command_line import \
    GraphNavInterface
from predicators.structs import Object

g_image_click = None
g_image_display = None

graph_nav_loc_to_id = {
    "start": "stone-prawn-2b0kiMuHQBhKfgwhNH3PLA==",
    "6-12_front": "linked-puffin-JpuZjUsLbqMMA7Ym5IwyGQ==",
    "6-12_table": "logy-impala-xc9w.jwUDuckZdITrner6g==",
    "front_tool_room": "dented-marlin-HZHTzO56529oo0oGfAFHdg==",
    "tool_room_table": "lumpen-squid-p9fT8Ui8TYI7JWQJvfQwKw==",
    "tool_room_bag": "seared-hare-0JBmyRiYHfbxn58ymEwPaQ==",
    "tool_room_tool_stand": "roving-gibbon-3eduef4VV0itZzkpHZueNQ==",
    "tool_room_platform": "comfy-auk-W0iygJ1WJyKR1eB3qe2mlg==",
    "tool_room_tool_rack": "alight-coyote-Nvl0i02Mk7Ds8ax0sj0Hsw==",
    "6-08_front": "curled-spawn-m19jn1Alc5XrrIcIoOSnaw==",
    "6-08_table": "maiden-oryx-zLyJUTDIg0ZNB3T4J1A8IQ==",
    "6-07_front": "moldy-cuckoo-RUdyBHBeLWD5pmNUEcW1Ng==",
    "6-07_table": "combed-gaur-dGWSpydRmOFNEAlBaABI4g==",
    "center": "unmown-botfly-8OaXOF1VG5LJWF47h.dRpQ==",
    "6-09_front": "ninth-jackal-g6onxC+AUQyIDznGUH3Fkw==",
    "6-09_bike": "sodden-hare-OxrK.cEZ1ZKHjb8jjwKrWA==",
    "6-09_table": "ranked-oxen-G0kq38CpHN7H7R.0FCm7DA==",
    "outside_table": "causal-fleece-fPaNlwN1dMO5vQ00ZjocmQ==",
    "6-13_corner": "gooey-mamba-nbmRlRr8J0KPsCztt0Wkyw==",
    "trash": "holy-aphid-SuqZLSjvRUjUxCDywLIFhw=="
}


# pylint: disable=no-member
class SpotControllers():
    """Implementation of interface with low-level controllers for the Spot
    robot.

    Controllers:

    navigateToController(objs, [float:dx, float:dy, float:dyaw])
    graspController(objs, [(0:Any,1:Top,-1:Side)])
    placeOntopController(objs, [float:distance])
    """

    def __init__(self) -> None:
        self._hostname = CFG.spot_robot_ip
        self._verbose = False
        self._force_45_angle_grasp = False
        self._force_horizontal_grasp = False
        self._force_squeeze_grasp = False
        self._force_top_down_grasp = False
        self._image_source = "hand_color_image"

        self.hand_x, self.hand_y, self.hand_z = (0.80, 0, 0.45)

        # See hello_spot.py for an explanation of these lines.
        bosdyn.client.util.setup_logging(self._verbose)

        self.sdk = bosdyn.client.create_standard_sdk('SesameClient')
        self.robot: Robot = self.sdk.create_robot(self._hostname)
        if not os.environ.get('BOSDYN_CLIENT_USERNAME') or not os.environ.get(
                'BOSDYN_CLIENT_PASSWORD'):
            raise RuntimeError("Spot environment variables unset.")
        bosdyn.client.util.authenticate(self.robot)
        self.robot.time_sync.wait_for_sync()

        assert self.robot.has_arm(
        ), "Robot requires an arm to run this example."

        # Verify the robot is not estopped and that an external application has
        # registered and holds an estop endpoint.
        self.verify_estop(self.robot)

        self.lease_client = self.robot.ensure_client(
            bosdyn.client.lease.LeaseClient.default_service_name)
        self.robot_state_client: RobotStateClient = self.robot.ensure_client(
            RobotStateClient.default_service_name)
        self.robot_command_client: RobotCommandClient = \
            self.robot.ensure_client(RobotCommandClient.default_service_name)
        self.image_client = self.robot.ensure_client(
            ImageClient.default_service_name)
        self.manipulation_api_client = self.robot.ensure_client(
            ManipulationApiClient.default_service_name)
        self.lease_keepalive = bosdyn.client.lease.LeaseKeepAlive(
            self.lease_client, must_acquire=True, return_at_exit=True)

        # Create Graph Nav Command Line
        self.upload_filepath = "predicators/spot_utils/bike_env/" + \
            "downloaded_graph/"
        self.graph_nav_command_line = GraphNavInterface(
            self.robot, self.upload_filepath, self.lease_client,
            self.lease_keepalive)

        # Initializing Spot
        self.robot.logger.info(
            "Powering on robot... This may take a several seconds.")
        self.robot.power_on(timeout_sec=20)
        assert self.robot.is_powered_on(), "Robot power on failed."

        self.robot.logger.info("Commanding robot to stand...")
        blocking_stand(self.robot_command_client, timeout_sec=10)
        self.robot.logger.info("Robot standing.")

    def navigateToController(self, objs: Sequence[Object],
                             params: Sequence[float]) -> None:
        """Controller that navigates to specific pre-specified locations.

        Params are [dx, dy, d-yaw]
        """
        print("NavigateTo", objs)
        assert len(params) == 3

        waypoint_id = ""
        if 'bag' in objs[1].name:
            waypoint_id = "ranked-oxen-G0kq38CpHN7H7R.0FCm7DA=="
        else:
            waypoint_id = "lumpen-squid-p9fT8Ui8TYI7JWQJvfQwKw=="
        self.navigate_to(waypoint_id, params)

    def graspController(self, objs: Sequence[Object],
                        params: Sequence[float]) -> None:
        """Wrapper method for grasp controller.

        Params are just one-dimensional corresponding to a top-down
        grasp (1), side grasp (-1) or any (0).
        """
        print("Grasp", objs)
        assert len(params) == 1
        assert params[0] in [0, 1, -1]
        if params[0] == 1:
            self._force_horizontal_grasp = False
            self._force_top_down_grasp = True
        elif params[0] == -1:
            self._force_horizontal_grasp = True
            self._force_top_down_grasp = False
        self.arm_object_grasp()

    def placeOntopController(self, objs: Sequence[Object],
                             params: Sequence[float]) -> None:
        """Wrapper method for placeOnTop controller.

        Params is one-dimensional corresponding to the extension of the
        arm from the robot when placing.
        """
        print("PlaceOntop", objs)
        assert len(params) == 1
        self.hand_movement(params)

    def verify_estop(self, robot: Any) -> None:
        """Verify the robot is not estopped."""

        client = robot.ensure_client(EstopClient.default_service_name)
        if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
            error_message = "Robot is estopped. Please use an external" + \
                " E-Stop client, such as the estop SDK example, to" + \
                " configure E-Stop."
            robot.logger.error(error_message)
            raise Exception(error_message)

    # NOTE: We want to deprecate this over the long-term!
    def cv_mouse_callback(self, event, x, y, flags, param):  # type: ignore
        """Callback for the click-to-grasp functionality with the Spot API's
        grasping interface."""
        del flags, param
        # pylint: disable=global-variable-not-assigned
        global g_image_click, g_image_display
        clone = g_image_display.copy()
        if event == cv2.EVENT_LBUTTONUP:
            g_image_click = (x, y)
        else:
            # Draw some lines on the image.
            #print('mouse', x, y)
            color = (30, 30, 30)
            thickness = 2
            image_title = 'Click to grasp'
            height = clone.shape[0]
            width = clone.shape[1]
            cv2.line(clone, (0, y), (width, y), color, thickness)
            cv2.line(clone, (x, 0), (x, height), color, thickness)
            cv2.imshow(image_title, clone)

    def add_grasp_constraint(
        self, grasp: manipulation_api_pb2.PickObjectInImage,
        robot_state_client: RobotStateClient
    ) -> manipulation_api_pb2.PickObjectInImage:
        """Method to constrain desirable grasps."""
        # There are 3 types of constraints:
        #   1. Vector alignment
        #   2. Full rotation
        #   3. Squeeze grasp
        #
        # You can specify more than one if you want and they will be
        # OR'ed together.

        # For these options, we'll use a vector alignment constraint.
        use_vector_constraint = self._force_top_down_grasp or \
            self._force_horizontal_grasp

        # Specify the frame we're using.
        grasp.grasp_params.grasp_params_frame_name = VISION_FRAME_NAME

        if use_vector_constraint:
            if self._force_top_down_grasp:
                # Add a constraint that requests that the x-axis of the
                # gripper is pointing in the negative-z direction in the
                # vision frame.

                # The axis on the gripper is the x-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=1, y=0, z=0)

                # The axis in the vision frame is the negative z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=-1)

            if self._force_horizontal_grasp:
                # Add a constraint that requests that the y-axis of the
                # gripper is pointing in the positive-z direction in the
                # vision frame.  That means that the gripper is
                # constrained to be rolled 90 degrees and pointed at the
                # horizon.

                # The axis on the gripper is the y-axis.
                axis_on_gripper_ewrt_gripper = geometry_pb2.Vec3(x=0, y=1, z=0)

                # The axis in the vision frame is the positive z-axis
                axis_to_align_with_ewrt_vo = geometry_pb2.Vec3(x=0, y=0, z=1)

            # Add the vector constraint to our proto.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.vector_alignment_with_tolerance.\
                axis_on_gripper_ewrt_gripper.\
                    CopyFrom(axis_on_gripper_ewrt_gripper)
            constraint.vector_alignment_with_tolerance.\
                axis_to_align_with_ewrt_frame.\
                    CopyFrom(axis_to_align_with_ewrt_vo)

            # We'll take anything within about 10 degrees for top-down or
            # horizontal grasps.
            constraint.vector_alignment_with_tolerance.\
                threshold_radians = 0.17

        elif self._force_45_angle_grasp:
            # Demonstration of a RotationWithTolerance constraint.
            # This constraint allows you to specify a full orientation you
            # want the hand to be in, along with a threshold.
            # You might want this feature when grasping an object with known
            # geometry and you want to make sure you grasp a specific part
            # of it. Here, since we don't have anything in particular we
            # want to grasp,  we'll specify an orientation that will have the
            # hand aligned with robot and rotated down 45 degrees as an
            # example.

            # First, get the robot's position in the world.
            robot_state = robot_state_client.get_robot_state()
            vision_T_body = get_vision_tform_body(
                robot_state.kinematic_state.transforms_snapshot)

            # Rotation from the body to our desired grasp.
            body_Q_grasp = math_helpers.Quat.from_pitch(0.785398)  # 45 degrees
            vision_Q_grasp = vision_T_body.rotation * body_Q_grasp

            # Turn into a proto
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.rotation_with_tolerance.rotation_ewrt_frame.CopyFrom(
                vision_Q_grasp.to_proto())

            # We'll accept anything within +/- 10 degrees
            constraint.rotation_with_tolerance.threshold_radians = 0.17

        elif self._force_squeeze_grasp:
            # Tell the robot to just squeeze on the ground at the given point.
            constraint = grasp.grasp_params.allowable_orientation.add()
            constraint.squeeze_grasp.SetInParent()

        return grasp

    def arm_object_grasp(self) -> None:
        """A simple example of using the Boston Dynamics API to command Spot's
        arm."""
        assert self.robot.is_powered_on(), "Robot power on failed."
        assert basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING

        # Take a picture with a camera
        self.robot.logger.info(f'Getting an image from: {self._image_source}')
        image_responses = self.image_client.get_image_from_sources(
            [self._image_source])

        if len(image_responses) != 1:
            print(f'Got invalid number of images: {str(len(image_responses))}')
            print(image_responses)
            assert False

        image = image_responses[0]
        if image.shot.image.pixel_format == image_pb2.Image.\
            PIXEL_FORMAT_DEPTH_U16:
            dtype = np.uint16  # type: ignore
        else:
            dtype = np.uint8  # type: ignore
        img = np.fromstring(image.shot.image.data, dtype=dtype)  # type: ignore
        if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
            img = img.reshape(image.shot.image.rows, image.shot.image.cols)
        else:
            img = cv2.imdecode(img, -1)

        # Show the image to the user and wait for them to click on a pixel
        self.robot.logger.info('Click on an object to start grasping...')
        image_title = 'Click to grasp'
        cv2.namedWindow(image_title)
        cv2.setMouseCallback(image_title, self.cv_mouse_callback)

        # pylint: disable=global-variable-not-assigned, global-statement
        global g_image_click, g_image_display
        g_image_display = img
        cv2.imshow(image_title, g_image_display)
        while g_image_click is None:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                # Quit
                print('"q" pressed, exiting.')
                sys.exit()

        # pylint: disable=unsubscriptable-object
        self.robot.\
            logger.info(f"Object at ({g_image_click[0]}, {g_image_click[1]})")
        # pylint: disable=unsubscriptable-object
        pick_vec = geometry_pb2.Vec2(x=g_image_click[0], y=g_image_click[1])

        # Build the proto
        grasp = manipulation_api_pb2.PickObjectInImage(
            pixel_xy=pick_vec,
            transforms_snapshot_for_camera=image.shot.transforms_snapshot,
            frame_name_image_sensor=image.shot.frame_name_image_sensor,
            camera_model=image.source.pinhole)

        # Optionally add a grasp constraint.  This lets you tell the robot you
        # only want top-down grasps or side-on grasps.
        grasp = self.add_grasp_constraint(grasp, self.robot_state_client)

        # Ask the robot to pick up the object
        grasp_request = manipulation_api_pb2.ManipulationApiRequest(
            pick_object_in_image=grasp)

        # Send the request
        cmd_response = self.manipulation_api_client.manipulation_api_command(
            manipulation_api_request=grasp_request)

        # Get feedback from the robot
        while True:
            feedback_request = manipulation_api_pb2.\
                ManipulationApiFeedbackRequest(manipulation_cmd_id=\
                    cmd_response.manipulation_cmd_id)

            # Send the request
            response = self.manipulation_api_client.\
                manipulation_api_feedback_command(
                manipulation_api_feedback_request=feedback_request)

            print(
                'Current state: ',
                manipulation_api_pb2.ManipulationFeedbackState.Name(
                    response.current_state))

            if response.current_state in [manipulation_api_pb2.\
                MANIP_STATE_GRASP_SUCCEEDED, manipulation_api_pb2.\
                MANIP_STATE_GRASP_FAILED]:
                break

        # Unstow the arm
        unstow = RobotCommandBuilder.arm_ready_command()

        # Issue the command via the RobotCommandClient
        unstow_command_id = self.robot_command_client.robot_command(unstow)

        self.robot.logger.info("Unstow command issued.")
        block_until_arm_arrives(self.robot_command_client, unstow_command_id,
                                3.0)

        time.sleep(1.0)

        # Allow Stowing and Stow Arm
        grasp_carry_state_override = manipulation_api_pb2.\
            ApiGraspedCarryStateOverride(override_request=3)
        grasp_override_request = manipulation_api_pb2.\
            ApiGraspOverrideRequest(
            carry_state_override=grasp_carry_state_override)
        cmd_response = self.manipulation_api_client.\
            grasp_override_command(grasp_override_request)

        stow_cmd = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.robot_command_client.robot_command(stow_cmd)
        self.robot.logger.info("Stow command issued.")
        block_until_arm_arrives(self.robot_command_client, stow_command_id,
                                3.0)

        self.robot.logger.info('Finished grasp.')

        g_image_click = None
        g_image_display = None

        time.sleep(2.0)

    def block_until_arm_arrives_with_prints(self, robot: Robot,
                                            command_client: RobotCommandClient,
                                            cmd_id: int) -> None:
        """Block until the arm arrives at the goal and print the distance
        remaining.

        Note: a version of this function is available as a helper in
        robot_command without the prints.
        """
        while True:
            feedback_resp = command_client.robot_command_feedback(cmd_id)

            if feedback_resp.feedback.synchronized_feedback.\
                arm_command_feedback.arm_cartesian_feedback.status == \
                arm_command_pb2.ArmCartesianCommand.Feedback.\
                    STATUS_TRAJECTORY_COMPLETE:
                robot.logger.info('Move complete.')
                break
            time.sleep(0.1)

    def hand_movement(self, params: Sequence[float]) -> None:
        """Move arm to infront of robot an open gripper."""
        # Move the arm to a spot in front of the robot, and open the gripper.
        assert self.robot.is_powered_on(), "Robot power on failed."
        assert basic_command_pb2.StandCommand.Feedback.STATUS_IS_STANDING

        # Rotation as a quaternion
        qw = np.cos((np.pi / 4) / 2)
        qx = 0
        qy = np.sin((np.pi / 4) / 2)
        qz = 0
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        # Make the arm pose RobotCommand
        # Build a position to move the arm to (in meters, relative to and
        # expressed in the gravity aligned body frame).
        assert params[0] >= -0.5 and params[0] <= 0.5
        x = self.hand_x + params[0]  # dx hand
        y = self.hand_y
        z = self.hand_z
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = self.robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(
            robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME,
            GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_obj(
            flat_body_T_hand)

        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w,
            odom_T_hand.rot.x, odom_T_hand.rot.y, odom_T_hand.rot.z,
            ODOM_FRAME_NAME, seconds)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.\
            claw_gripper_open_fraction_command(0.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(
            gripper_command, arm_command)

        # Send the request
        cmd_id: int = self.robot_command_client.robot_command(command)
        self.robot.logger.info('Moving arm to position.')

        # Wait until the arm arrives at the goal.
        self.block_until_arm_arrives_with_prints(self.robot,
                                                 self.robot_command_client,
                                                 cmd_id)

        time.sleep(2)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.\
            claw_gripper_open_fraction_command(1.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(
            gripper_command, arm_command)

        # Send the request
        cmd_id = self.robot_command_client.robot_command(command)
        self.robot.logger.info('Moving arm to position.')

        # Wait until the arm arrives at the goal.
        self.block_until_arm_arrives_with_prints(self.robot,
                                                 self.robot_command_client,
                                                 cmd_id)

        time.sleep(2)

        stow_cmd = RobotCommandBuilder.arm_stow_command()
        stow_command_id = self.robot_command_client.robot_command(stow_cmd)
        self.robot.logger.info("Stow command issued.")
        block_until_arm_arrives(self.robot_command_client, stow_command_id,
                                3.0)

    def navigate_to(self, waypoint_id: str, params: Sequence[float]) -> None:
        """Use GraphNavInterface to localize robot and go to a location."""
        # pylint: disable=broad-except
        try:
            # (1) Initialize location
            self.graph_nav_command_line.set_initial_localization_fiducial()

            # (2) Get localization state
            self.graph_nav_command_line.get_localization_state()

            # (4) Navigate to
            self.graph_nav_command_line.navigate_to([waypoint_id])

            # (5) Offset by params
            if params != [0.0, 0.0, 0.0]:
                self.relative_move(params[0], params[1], params[2])

        except Exception as e:
            print(e)

    def relative_move(self,
                      dx: float,
                      dy: float,
                      dyaw: float,
                      stairs: bool = False) -> bool:
        """Move to relative robot position in body frame."""
        transforms = self.robot_state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot

        # Build the transform for where we want the robot to be
        # relative to where the body currently is.
        body_tform_goal = math_helpers.SE2Pose(x=dx, y=dy, angle=dyaw)
        # We do not want to command this goal in body frame because
        # the body will move, thus shifting our goal. Instead, we
        # transform this offset to get the goal position in the output
        # frame (which will be either odom or vision).
        out_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME,
                                           BODY_FRAME_NAME)
        out_tform_goal = out_tform_body * body_tform_goal

        # Command the robot to go to the goal point in the specified
        # frame. The command will stop at the new position.
        robot_cmd = RobotCommandBuilder.synchro_se2_trajectory_point_command(
            goal_x=out_tform_goal.x,
            goal_y=out_tform_goal.y,
            goal_heading=out_tform_goal.angle,
            frame_name=ODOM_FRAME_NAME,
            params=RobotCommandBuilder.mobility_params(stair_hint=stairs))
        end_time = 10.0
        cmd_id = self.robot_command_client.robot_command(
            lease=None,
            command=robot_cmd,
            end_time_secs=time.time() + end_time)
        # Wait until the robot has reached the goal.
        while True:
            feedback = self.robot_command_client.\
                robot_command_feedback(cmd_id)
            mobility_feedback = feedback.feedback.\
                synchronized_feedback.mobility_command_feedback
            if mobility_feedback.status != \
                RobotCommandFeedbackStatus.STATUS_PROCESSING:
                print("Failed to reach the goal")
                return False
            traj_feedback = mobility_feedback.se2_trajectory_feedback
            if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL
                    and traj_feedback.body_movement_status
                    == traj_feedback.BODY_STATUS_SETTLED):
                print("Arrived at the goal.")
                return True
            time.sleep(1)
