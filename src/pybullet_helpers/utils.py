"""Other Pybullet utility functions."""

from typing import List

from predicators.src.pybullet_helpers.joint import get_joint_info


def get_kinematic_chain(robot: int, end_effector: int,
                        physics_client_id: int) -> List[int]:
    """Get all the free joints from end effector to base.

    Parameters
    ----------
    robot: body ID of robot
    end_effector: joint ID of end effector
    physics_client_id: physics client ID

    Returns
    -------
    List of joint IDs from end effector to base.
    """
    kinematic_chain = []
    current_joint = end_effector

    while current_joint > -1:
        joint_info = get_joint_info(robot, current_joint, physics_client_id)
        if joint_info.qIndex > -1:
            kinematic_chain.append(current_joint)
        current_joint = joint_info.parentIndex

    return kinematic_chain
