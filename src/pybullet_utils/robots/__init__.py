"""Interfaces to PyBullet robots."""
import abc

from predicators.src.structs import Array, Pose3D


class _SingleArmPyBulletRobot(abc.ABC):
    """A single-arm fixed-base PyBullet robot with a two-finger gripper.

    The action space for the robot is 4D. The first three dimensions are
    a change in the (x, y, z) of the end effector. The last dimension is
    a change in the finger joint(s), which are constrained to be
    symmetric.
    """

    def __init__(self, ee_home_pose: Pose3D, open_fingers: float,
                 closed_fingers: float, finger_action_tol: float,
                 physics_client_id: int) -> None:
        # Initial position for the end effector.
        self._ee_home_pose = ee_home_pose
        # The value at which the finger joints should be open.
        self._open_fingers = open_fingers
        # The value at which the finger joints should be closed.
        self._closed_fingers = closed_fingers
        # If an f_delta is less than this magnitude, it's considered a noop.
        self._finger_action_tol = finger_action_tol
        self._physics_client_id = physics_client_id
        self._initialize()

    @abc.abstractmethod
    def _initialize(self) -> None:
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def robot_id(self) -> int:
        """The PyBullet ID for the robot."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def end_effector_id(self) -> int:
        """The PyBullet ID for the end effector."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def left_finger_id(self) -> int:
        """The PyBullet ID for the left finger."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def right_finger_id(self) -> int:
        """The PyBullet ID for the right finger."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def reset_state(self, robot_state: Array) -> None:
        """Reset the robot state to match the input state.

        The robot_state corresponds to the State vector for the robot
        object.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_state(self) -> Array:
        """Get the robot state vector based on the current PyBullet state.

        The robot_state corresponds to the State vector for the robot
        object.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def set_motors(self, ee_delta: Pose3D, f_delta: float) -> None:
        """Update the motors to execute the given action in PyBullet given a
        delta on the end effector and finger joint(s)."""
        raise NotImplementedError("Override me!")


def create_single_arm_pybullet_robot(
        robot_name: str, ee_home_pose: Pose3D, open_fingers: float,
        closed_fingers: float, finger_action_tol: float,
        physics_client_id: int) -> _SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    from predicators.src.pybullet_utils.robots.fetch import FetchPyBulletRobot
    from predicators.src.pybullet_utils.robots.panda import PandaPyBulletRobot

    if robot_name == "fetch":
        return FetchPyBulletRobot(ee_home_pose, open_fingers, closed_fingers,
                                  finger_action_tol, physics_client_id)
    elif robot_name == "panda":
        return PandaPyBulletRobot(ee_home_pose, open_fingers, closed_fingers,
                                  finger_action_tol, physics_client_id)
    raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")
