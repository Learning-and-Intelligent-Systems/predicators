"""Interface for stowing the spot arm."""
from bosdyn.api import manipulation_api_pb2
from bosdyn.client.manipulation_api_client import ManipulationApiClient
from bosdyn.client.robot_command import RobotCommandBuilder, \
    RobotCommandClient, block_until_arm_arrives
from bosdyn.client.sdk import Robot


def stow_arm(robot: Robot, timeout: float = 5) -> None:
    """Execute a stow arm command."""

    manipulation_client = robot.ensure_client(
        ManipulationApiClient.default_service_name)
    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)

    # Enable stowing.
    override = manipulation_api_pb2.ApiGraspedCarryStateOverride(
        override_request=3)
    grasp_override_request = manipulation_api_pb2.ApiGraspOverrideRequest(
        carry_state_override=override)
    manipulation_client.grasp_override_command(grasp_override_request)

    # Build the commands.
    stow_cmd = RobotCommandBuilder.arm_stow_command()
    close_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)

    # Combine the arm and gripper commands into one RobotCommand
    combo_cmd = RobotCommandBuilder.build_synchro_command(close_cmd, stow_cmd)
    cmd_id = robot_command_client.robot_command(combo_cmd)

    # Send the command.
    block_until_arm_arrives(robot_command_client, cmd_id, timeout)
