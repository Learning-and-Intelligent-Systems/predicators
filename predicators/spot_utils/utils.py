"""Small utility functions for spot."""

import sys
from typing import Optional, Tuple

import cv2
import numpy as np
from bosdyn.api import estop_pb2, manipulation_api_pb2
from bosdyn.client.estop import EstopClient
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, \
    RobotCommandClient, block_until_arm_arrives
from bosdyn.client.sdk import Robot
from numpy.typing import NDArray


def verify_estop(robot: Robot) -> None:
    """Verify the robot is not estopped."""

    client = robot.ensure_client(EstopClient.default_service_name)
    if client.get_status().stop_level != estop_pb2.ESTOP_LEVEL_NONE:
        error_message = "Robot is estopped. Please use an external" + \
            " E-Stop client, such as the estop SDK example, to" + \
            " configure E-Stop."
        robot.logger.error(error_message)
        raise Exception(error_message)


def stow_arm(robot: Robot, timeout: float = 5) -> None:
    """Execute a stow arm command."""

    manipulation_client = robot.ensure_client(
        ManipulationApiClient.default_service_name)
    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)

    # Enable stowing and stow Arm.
    grasp_carry_state_override = manipulation_api_pb2.ApiGraspedCarryStateOverride(
        override_request=3)
    grasp_override_request = manipulation_api_pb2.ApiGraspOverrideRequest(
        carry_state_override=grasp_carry_state_override)
    manipulation_client.grasp_override_command(grasp_override_request)

    # Build the commands.
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    close_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)

    # Combine the arm and gripper commands into one RobotCommand
    combo_cmd = RobotCommandBuilder.build_synchro_command(close_cmd, stow_cmd)
    cmd_id = robot_command_client.robot_command(combo_cmd)

    # Send the command.
    block_until_arm_arrives(robot_command_client, cmd_id, timeout)


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

    image_title = 'Click to grasp'
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
