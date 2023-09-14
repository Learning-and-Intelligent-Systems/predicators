"""Interface for moving the spot hand."""

from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME
from bosdyn.client.robot_command import RobotCommandBuilder, \
    RobotCommandClient, block_until_arm_arrives
from bosdyn.client.sdk import Robot


def move_hand_to_relative_pose(
    robot: Robot,
    body_tform_goal: math_helpers.SE3Pose,
    open_gripper: bool = False,
    close_gripper: bool = False,
    timeout: float = 2,
) -> None:
    """Move the spot hand while optionally opening or closing the gripper.

    The target pose is relative to the robot's body.
    """

    assert not (open_gripper and close_gripper)

    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)

    # Build the arm command.
    arm_cmd = RobotCommandBuilder.arm_pose_command(
        body_tform_goal.x, body_tform_goal.y, body_tform_goal.z,
        body_tform_goal.rot.w, body_tform_goal.rot.x, body_tform_goal.rot.y,
        body_tform_goal.rot.z, BODY_FRAME_NAME, timeout)

    # Build the gripper command (optional).
    if open_gripper:
        grip_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)
        cmd = RobotCommandBuilder.build_synchro_command(grip_cmd, arm_cmd)
    elif close_gripper:
        grip_cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(0.0)
        cmd = RobotCommandBuilder.build_synchro_command(grip_cmd, arm_cmd)
    else:
        cmd = arm_cmd

    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)

    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(robot_command_client, cmd_id, timeout)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # pylint: disable=ungrouped-imports
    import numpy as np
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.utils import verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip

        sdk = create_standard_sdk('MoveHandSkillTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        robot.time_sync.wait_for_sync()
        resting_pose = math_helpers.SE3Pose(x=0.80,
                                            y=0,
                                            z=0.45,
                                            rot=math_helpers.Quat())
        relative_down_pose = math_helpers.SE3Pose(
            x=0.0, y=0, z=0.0, rot=math_helpers.Quat.from_pitch(np.pi / 4))
        resting_down_pose = resting_pose * relative_down_pose

        print("Moving to a resting pose in front of the robot.")
        move_hand_to_relative_pose(robot, resting_pose)
        input("Press enter when ready to move on")

        print("Moving to the same pose, but now opening the gripper.")
        move_hand_to_relative_pose(robot, resting_pose, open_gripper=True)
        input("Press enter when ready to move on")

        print("Moving to the same pose (should have no change).")
        move_hand_to_relative_pose(robot, resting_pose)
        input("Press enter when ready to move on")

        print("Moving to the same pose, but now closing the gripper.")
        move_hand_to_relative_pose(robot, resting_pose, close_gripper=True)
        input("Press enter when ready to move on")

        print("Looking down and opening the gripper.")
        move_hand_to_relative_pose(robot, resting_down_pose, open_gripper=True)

    _run_manual_test()
