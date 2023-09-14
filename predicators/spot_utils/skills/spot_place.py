"""Interface for spot placing skill."""

from bosdyn.client import math_helpers
from bosdyn.client.sdk import Robot

from predicators.spot_utils.skills.spot_hand_move import \
    move_hand_to_relative_pose, open_gripper


def place_at_relative_pose(robot: Robot,
                           body_tform_goal: math_helpers.SE3Pose,
                           pause_duration: float = 1.0) -> None:
    """Assuming something is held, place is at the given pose.

    The pose is relative to the robot's body. It is the responsibility
    of the user of this method to specify a pose that makes sense, e.g.,
    one with an angle facing downward to facilitate the place.
    """
    # First move the hand to the target pose.
    move_hand_to_relative_pose(robot, body_tform_goal)
    # Open the hand.
    open_gripper(robot)


if __name__ == "__main__":
    # Run this file alone to test manually.
    # Make sure to pass in --spot_robot_ip.

    # To run this test, the robot should already be holding something and at
    # the location where it wants to place.

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

        sdk = create_standard_sdk('PlaceSkillTestClient')
        robot = sdk.create_robot(hostname)
        authenticate(robot)
        verify_estop(robot)
        lease_client = robot.ensure_client(LeaseClient.default_service_name)
        lease_client.take()
        robot.time_sync.wait_for_sync()
        target_pose = math_helpers.SE3Pose(x=0.80,
                                           y=0,
                                           z=0.25,
                                           rot=math_helpers.Quat.from_pitch(
                                               np.pi / 3))
        place_at_relative_pose(robot, target_pose)

    _run_manual_test()
