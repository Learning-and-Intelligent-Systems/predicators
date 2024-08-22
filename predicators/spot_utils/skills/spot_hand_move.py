"""Interface for moving the spot hand."""

import time

from bosdyn.api import arm_command_pb2, robot_command_pb2, \
    synchronized_command_pb2, trajectory_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import BODY_FRAME_NAME, ODOM_FRAME_NAME, \
    get_a_tform_b
from bosdyn.client.robot_command import RobotCommandBuilder, \
    RobotCommandClient, block_until_arm_arrives
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.sdk import Robot
from bosdyn.util import seconds_to_duration
from google.protobuf.wrappers_pb2 import \
    DoubleValue  # pylint: disable=no-name-in-module


def move_hand_to_relative_pose(
    robot: Robot,
    body_tform_goal: math_helpers.SE3Pose,
) -> None:
    """Move the spot hand.

    The target pose is relative to the robot's body.
    """
    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    # Build the arm command.
    cmd = RobotCommandBuilder.arm_pose_command(
        body_tform_goal.x, body_tform_goal.y, body_tform_goal.z,
        body_tform_goal.rot.w, body_tform_goal.rot.x, body_tform_goal.rot.y,
        body_tform_goal.rot.z, BODY_FRAME_NAME, 2.0)
    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)
    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(robot_command_client, cmd_id, 2.0)


def move_hand_to_relative_pose_with_velocity(
    robot: Robot,
    curr_hand_pose: math_helpers.SE3Pose,
    body_tform_goal: math_helpers.SE3Pose,
    duration: float = 2.0,
) -> None:
    """Move the spot hand with a certain velocity specified as a duration (so
    velocity will become 1/duration)

    The curr hand pose and target pose are relative to the robot's body.
    """
    sweep_start_pose_traj_point = trajectory_pb2.SE3TrajectoryPoint(
        pose=curr_hand_pose.to_proto(),
        time_since_reference=seconds_to_duration(0.0))
    sweep_end_pose_traj_point = trajectory_pb2.SE3TrajectoryPoint(
        pose=body_tform_goal.to_proto(),
        time_since_reference=seconds_to_duration(duration))

    # Build the trajectory proto by combining the points.
    hand_traj = trajectory_pb2.SE3Trajectory(
        points=[sweep_start_pose_traj_point, sweep_end_pose_traj_point])

    # Build the command by taking the trajectory and specifying the frame it
    # is expressed in. Note that we set the max linear and angular velocity
    # absurdly high to remove the default limits and enable the duration
    # to significantly impact the speed of arm movement.
    # IMPORTANT: don't max-out the acceleration limit on this command;
    # leads to weird jerking motions.
    arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
        pose_trajectory_in_task=hand_traj,
        root_frame_name=BODY_FRAME_NAME,
        max_linear_velocity=DoubleValue(value=100),
        max_angular_velocity=DoubleValue(value=100))

    # Pack everything up in protos.
    arm_command = arm_command_pb2.ArmCommand.Request(
        arm_cartesian_command=arm_cartesian_command)

    synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
        arm_command=arm_command)

    robot_command = robot_command_pb2.RobotCommand(
        synchronized_command=synchronized_command)
    # Send the trajectory to the robot.
    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    cmd_id = robot_command_client.robot_command(robot_command)
    while True:
        feedback_resp = robot_command_client.robot_command_feedback(cmd_id)
        if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.\
            arm_cartesian_feedback.status in [
                arm_command_pb2.ArmCartesianCommand.Feedback. # pylint: disable=no-member
                STATUS_TRAJECTORY_COMPLETE, arm_command_pb2. # pylint: disable=no-member
                ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_STALLED
        ]:
            break
        time.sleep(0.1)


def gaze_at_relative_pose(
    robot: Robot,
    gaze_target: math_helpers.Vec3,
    duration: float = 2.0,
) -> None:
    """Gaze at a point relative to the robot's body frame."""
    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    # Transform the gaze target from the body frame to the odom frame because
    # the gaze command results in shaking in the body frame.
    robot_state_client = robot.ensure_client(
        RobotStateClient.default_service_name)
    robot_state = robot_state_client.get_robot_state()
    odom_tform_body = get_a_tform_b(
        robot_state.kinematic_state.transforms_snapshot, ODOM_FRAME_NAME,
        BODY_FRAME_NAME)
    gaze_target = odom_tform_body.transform_vec3(gaze_target)
    # Build the arm command.
    cmd = RobotCommandBuilder.arm_gaze_command(gaze_target.x, gaze_target.y,
                                               gaze_target.z, ODOM_FRAME_NAME)
    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)
    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(robot_command_client, cmd_id, duration)
    time.sleep(1.0)


def change_gripper(
    robot: Robot,
    fraction: float,
    duration: float = 2.0,
) -> None:
    """Change the spot gripper angle."""
    assert 0.0 <= fraction <= 1.0
    robot_command_client = robot.ensure_client(
        RobotCommandClient.default_service_name)
    # Build the command.
    cmd = RobotCommandBuilder.claw_gripper_open_fraction_command(fraction)
    # Send the request.
    cmd_id = robot_command_client.robot_command(cmd)
    # Wait until the arm arrives at the goal.
    block_until_arm_arrives(robot_command_client, cmd_id, duration)


def open_gripper(
    robot: Robot,
    duration: float = 2.0,
) -> None:
    """Open the spot gripper."""
    return change_gripper(robot, fraction=1.0, duration=duration)


def close_gripper(
    robot: Robot,
    duration: float = 2.0,
) -> None:
    """Close the spot gripper."""
    return change_gripper(robot, fraction=0.0, duration=duration)


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
        looking_down_and_rotated_right_pose = math_helpers.SE3Pose(
            x=0.9,
            y=0,
            z=0.0,
            rot=math_helpers.Quat.from_pitch(np.pi / 2) *
            math_helpers.Quat.from_roll(np.pi / 2))

        print(
            "Moving to a pose that looks down and rotates the gripper to the "
            + "right.")
        move_hand_to_relative_pose(robot, looking_down_and_rotated_right_pose)
        input("Press enter when ready to move on")

        print("Moving to a resting pose in front of the robot.")
        move_hand_to_relative_pose(robot, resting_pose)
        input("Press enter when ready to move on")

        print("Opening the gripper.")
        open_gripper(robot)
        input("Press enter when ready to move on")

        print("Moving to the same pose (should have no change).")
        move_hand_to_relative_pose(robot, resting_pose)
        input("Press enter when ready to move on")

        print("Closing the gripper.")
        move_hand_to_relative_pose(robot, resting_pose)
        close_gripper(robot)
        input("Press enter when ready to move on")

        print("Looking down and opening the gripper.")
        move_hand_to_relative_pose(robot, resting_down_pose)
        open_gripper(robot)

    _run_manual_test()
