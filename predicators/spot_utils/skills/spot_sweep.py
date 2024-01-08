"""Interface for spot sweeping skill."""

import numpy as np
from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose
from predicators.spot_utils.skills.spot_navigation import \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_stow_arm import stow_arm


def sweep(robot: Robot, sweep_start_pose: math_helpers.SE3Pose, move_dx: float,
          move_dy: float, move_dz: float) -> None:
    """Sweep in the xy plane, starting at the start pose and then moving."""
    # Move first in the yz direction (perpendicular to robot body) to avoid
    # knocking the target object over.
    sweep_start_y_move = math_helpers.SE3Pose(
        x=0.4,  # sensible default
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
    # Calculate the end pose.
    relative_hand_move = math_helpers.SE3Pose(x=move_dx,
                                              y=move_dy,
                                              z=move_dz,
                                              rot=math_helpers.Quat())
    sweep_end_pose = relative_hand_move * sweep_start_pose
    # Move the hand to the end pose. This is the main sweep, go slowly!
    move_hand_to_relative_pose(robot, sweep_end_pose, duration=4.0)
    # Stow arm.
    stow_arm(robot)
    # Back up a little bit so that we can see the result of sweeping.
    body_rel_pose = math_helpers.SE2Pose(
        x=-0.2,
        y=0.0,
        angle=0.0,
    )
    navigate_to_relative_pose(robot, body_rel_pose)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # NOTE: this test assumes that the robot is standing in front of a table
    # that has a soda can on it. The test starts by running object detection to
    # get the pose of the soda can. Then the robot opens its gripper and pauses
    # until a brush is put in the gripper, with the bristles facing down and
    # forward. The robot should then brush the soda can to the right.

    # pylint: disable=ungrouped-imports
    import curses

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
        get_relative_se2_from_se3, get_spot_home_pose, verify_estop

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

        # Go home.
        go_home(robot, localizer)
        localizer.localize()

        # Find the soda can.
        soda_detection_id = LanguageObjectDetectionID("soda can/beer can")
        detections, _ = init_search_for_objects(robot, localizer,
                                                {soda_detection_id})
        soda_pose = detections[soda_detection_id]

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
        stdscr = curses.initscr()
        curses.noecho()
        stdscr.addstr(msg)
        stdscr.getkey()
        curses.endwin()
        close_gripper(robot)

        # Move to in front of the soda can.
        stow_arm(robot)
        pre_sweep_nav_distance = 0.7
        home_pose = get_spot_home_pose()
        pre_sweep_nav_angle = home_pose.angle - np.pi
        localizer.localize()
        robot_pose = localizer.get_last_robot_pose()
        rel_pose = get_relative_se2_from_se3(robot_pose, soda_pose,
                                             pre_sweep_nav_distance,
                                             pre_sweep_nav_angle)
        navigate_to_relative_pose(robot, rel_pose)
        localizer.localize()

        # Calculate sweep parameters.
        robot_pose = localizer.get_last_robot_pose()
        soda_rel_pose = robot_pose.inverse() * soda_pose
        start_dx = 0.0
        start_dy = 0.4
        start_dz = 0.1
        start_x = soda_rel_pose.x + start_dx
        start_y = soda_rel_pose.y + start_dy
        start_z = soda_rel_pose.z + start_dz
        pitch = math_helpers.Quat.from_pitch(np.pi / 2)
        yaw = math_helpers.Quat.from_yaw(np.pi / 4)
        rot = pitch * yaw
        sweep_start_pose = math_helpers.SE3Pose(x=start_x,
                                                y=start_y,
                                                z=start_z,
                                                rot=rot)
        # Calculate the yaw and distance for the sweep.
        sweep_move_dx = -start_dx
        sweep_move_dy = -start_dy
        sweep_move_dz = -3 * start_dz

        # Execute the sweep.
        sweep(robot, sweep_start_pose, sweep_move_dx, sweep_move_dy,
              sweep_move_dz)

        # Stow to finish.
        stow_arm(robot)

    _run_manual_test()
