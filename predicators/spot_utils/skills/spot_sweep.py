"""Interface for spot sweeping skill."""

import time

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, move_hand_to_relative_pose_with_velocity
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_stow_arm import stow_arm


def sweep(robot: Robot, sweep_start_pose: math_helpers.SE3Pose, move_dx: float,
          move_dy: float, move_dz: float, duration: float) -> None:
    """Sweep in the xy plane, starting at the start pose and then moving."""
    # Move first in the yz direction (perpendicular to robot body) to avoid
    # knocking the target object over.
    sweep_start_y_move = math_helpers.SE3Pose(
        x=0.5,  # sensible default
        y=sweep_start_pose.y,
        z=sweep_start_pose.z,
        rot=math_helpers.Quat(),
    )
    move_hand_to_relative_pose(robot, sweep_start_y_move)
    # Now rotate the hand, still not moving in the x direction.
    sweep_start_rotate_move = math_helpers.SE3Pose(
        x=sweep_start_y_move.x,  # sensible default
        y=sweep_start_y_move.y,
        z=sweep_start_y_move.z,
        rot=sweep_start_pose.rot,
    )
    move_hand_to_relative_pose(robot, sweep_start_rotate_move)
    # Now move the remaining way to the start pose.
    move_hand_to_relative_pose(robot, sweep_start_pose)
    time.sleep(0.1)
    # Conservatively assuming the start pose is somewhere above the surface
    # we're sweeping, move down by a bit to actually contact the surface.
    sweep_start_conformant = math_helpers.SE3Pose(x=sweep_start_pose.x,
                                                  y=sweep_start_pose.y,
                                                  z=sweep_start_pose.z - 0.09,
                                                  rot=sweep_start_pose.rot)
    move_hand_to_relative_pose(robot, sweep_start_conformant)
    time.sleep(0.1)
    # Calculate the end pose.
    relative_hand_move = math_helpers.SE3Pose(x=move_dx,
                                              y=move_dy,
                                              z=move_dz,
                                              rot=math_helpers.Quat())
    sweep_end_pose = relative_hand_move * sweep_start_conformant
    move_hand_to_relative_pose_with_velocity(robot, sweep_start_conformant,
                                             sweep_end_pose, duration)
    # Stow arm.
    stow_arm(robot)
    # Back up a little bit so that we can see the result of sweeping.
    # NOTE: this assumes that sweeping is done to the right side of the robot,
    # which is an assumption we could remove later.
    body_rel_pose = math_helpers.SE2Pose(
        x=-0.1,
        y=-0.5,
        angle=0.0,
    )
    navigate_to_relative_pose(robot, body_rel_pose)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: this test assumes that the robot is standing in front of a table
    # that has a train_toy on it. The test starts by running object detection to
    # get the pose of the train_toy. Then the robot opens its gripper and pauses
    # until a brush is put in the gripper, with the bristles facing down and
    # forward. The robot should then brush the train_toy to the right.

    # pylint: disable=ungrouped-imports
    from bosdyn.client import create_standard_sdk
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.util import authenticate

    from predicators import utils
    from predicators.settings import CFG
    from predicators.spot_utils.perception.perception_structs import \
        LanguageObjectDetectionID
    from predicators.spot_utils.skills.spot_find_objects import \
        init_search_for_objects
    from predicators.spot_utils.skills.spot_hand_move import close_gripper, \
        open_gripper
    from predicators.spot_utils.skills.spot_navigation import go_home
    from predicators.spot_utils.spot_localization import SpotLocalizer
    from predicators.spot_utils.utils import get_graph_nav_dir, \
        get_relative_se2_from_se3, verify_estop

    def _run_manual_test() -> None:
        # Put inside a function to avoid variable scoping issues.
        args = utils.parse_args(env_required=False,
                                seed_required=False,
                                approach_required=False)
        utils.update_config(args)

        # Get constants.
        hostname = CFG.spot_robot_ip
        path = get_graph_nav_dir()

        sdk = create_standard_sdk('SweepSkillTestClient')
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
        localizer.localize()

        # IMPORTANT: Only true for the floor8-sweeping map
        black_table_pose = math_helpers.SE3Pose(2.3, -2.0, -0.32,
                                                math_helpers.Quat())
        black_table_width = 0.37
        black_table_height = 0.37
        # Define the upper left pose we want to put the brush at, and the
        # middle (horizontally) bottom pose. We will always move the hand to
        # some pose between these.
        upper_left_black_table_pose = math_helpers.SE3Pose(
            x=black_table_pose.x + black_table_width / 2.0,
            y=black_table_pose.y - black_table_height / 2.0 + 0.15,
            z=0.25,
            rot=black_table_pose.rot)
        middle_bottom_black_table_pose = math_helpers.SE3Pose(
            x=black_table_pose.x - 0.18,
            y=black_table_pose.y + black_table_height / 2.0,
            z=0.25,
            rot=black_table_pose.rot)

        # Go home.
        stow_arm(robot)
        go_home(robot, localizer)
        localizer.localize()

        # Find the train_toy can.
        train_toy_prompt = "/".join([
            "small white ambulance toy",
            "car_(automobile) toy",
            "egg",
        ])
        train_toy_detection_id = LanguageObjectDetectionID(train_toy_prompt)
        detections, _ = init_search_for_objects(robot, localizer,
                                                {train_toy_detection_id})
        train_toy_pose = detections[train_toy_detection_id]

        # Move the hand to the side so that the brush can face forward.
        hand_side_pose = math_helpers.SE3Pose(x=0.80,
                                              y=0.0,
                                              z=0.25,
                                              rot=math_helpers.Quat.from_yaw(
                                                  -np.pi / 2))
        move_hand_to_relative_pose(robot, hand_side_pose)

        # Ask for the brush.
        open_gripper(robot)
        # Press any key, instead of just enter. Useful for remote control.
        msg = "Put the brush in the robot's gripper, then press any key"
        utils.wait_for_any_button_press(msg)
        close_gripper(robot)

        # Move to in front of the train_toy.
        stow_arm(robot)
        pre_sweep_nav_distance = 0.7
        pre_sweep_nav_angle = np.pi / 2
        localizer.localize()
        robot_pose = localizer.get_last_robot_pose()
        rel_pose = get_relative_se2_from_se3(robot_pose, train_toy_pose,
                                             pre_sweep_nav_distance,
                                             pre_sweep_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()

        # Calculate sweep parameters.
        robot_pose = localizer.get_last_robot_pose()
        train_toy_rel_pose = robot_pose.inverse() * train_toy_pose
        upper_left_black_table_rel_pose = robot_pose.inverse(
        ) * upper_left_black_table_pose
        middle_bottom_black_table_rel_pose = robot_pose.inverse(
        ) * middle_bottom_black_table_pose
        start_dx = 0.175
        start_dy = 0.4
        # clamp the x and y poses to the top left of the table worst case.
        start_x = np.clip(middle_bottom_black_table_rel_pose.x,
                          train_toy_rel_pose.x + start_dx,
                          upper_left_black_table_rel_pose.x)
        start_y = np.clip(middle_bottom_black_table_rel_pose.y,
                          train_toy_rel_pose.y + start_dy,
                          upper_left_black_table_rel_pose.y)
        # use absolute value so that we don't get messed up by noise in the
        # perception estimate
        start_z = 0.19
        pitch = math_helpers.Quat.from_pitch(np.pi / 2)
        yaw = math_helpers.Quat.from_yaw(np.pi / 4)
        rot = pitch * yaw
        sweep_start_pose = math_helpers.SE3Pose(x=start_x,
                                                y=start_y,
                                                z=start_z,
                                                rot=rot)
        # Calculate the yaw and distance for the sweep.
        sweep_move_dx = 0.0
        sweep_move_dy = -0.8
        sweep_move_dz = 0.0
        duration = 0.5

        # Execute the sweep.
        sweep(robot, sweep_start_pose, sweep_move_dx, sweep_move_dy,
              sweep_move_dz, duration)

        # Stow to finish.
        stow_arm(robot)

    _run_manual_test()
