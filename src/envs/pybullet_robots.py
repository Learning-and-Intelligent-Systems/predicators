"""Interfaces to PyBullet robots."""

import abc
import logging
import time
from functools import cached_property
from typing import (
    ClassVar,
    Collection,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np
import pybullet as p
import pybullet_data
from gym.spaces import Box

from predicators.src import utils
from predicators.src.pybullet_utils.ikfast import ikfast_inverse_kinematics
from predicators.src.settings import CFG
from predicators.src.structs import (
    Array,
    JointsState,
    Pose3D,
)


class SingleArmPyBulletRobot(abc.ABC):
    """A single-arm fixed-base PyBullet robot with a two-finger gripper."""

    def __init__(
        self,
        ee_home_pose: Pose3D,
        ee_orientation: Sequence[float],
        physics_client_id: int,
    ) -> None:
        # Initial position for the end effector.
        self._ee_home_pose = ee_home_pose
        # Orientation for the end effector.
        self._ee_orientation = ee_orientation
        self._physics_client_id = physics_client_id
        self._initialize()

    @cached_property
    def initial_joints_state(self) -> JointsState:
        """The joint values for the robot in its home pose."""
        initial_joints_state = self.inverse_kinematics(
            self._ee_home_pose, validate=True
        )
        # The initial joint values for the fingers should be open. IK may
        # return anything for them.
        initial_joints_state[self.left_finger_joint_idx] = self.open_fingers
        initial_joints_state[self.right_finger_joint_idx] = self.open_fingers
        return initial_joints_state

    @property
    def action_space(self) -> Box:
        """
        The action space for the robot.
        Represents position control of the arm and finger joints.
        """
        return Box(
            np.array(self.joint_lower_limits, dtype=np.float32),
            np.array(self.joint_upper_limits, dtype=np.float32),
            dtype=np.float32,
        )

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the name of the robot."""
        raise NotImplementedError("Override me!")

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
    def end_effector_name(self) -> str:
        """The name of the end effector."""
        raise NotImplementedError("Override me!")

    @property
    def end_effector_id(self) -> int:
        """The PyBullet ID for the end effector."""
        return self.joint_names.index(self.end_effector_name)

    @cached_property
    def arm_joints(self) -> List[int]:
        """The Pybullet IDs of the joints of the robot arm."""
        joint_ids = get_kinematic_chain(
            self.robot_id,
            self.end_effector_id,
            physics_client_id=self._physics_client_id,
        )
        # NOTE: pybullet tools assumes sorted arm joints.
        joint_ids = sorted(joint_ids)
        joint_ids.extend([self.left_finger_id, self.right_finger_id])
        return joint_ids

    @cached_property
    def num_joints(self) -> int:
        """The number of joints in the robot."""
        return p.getNumJoints(self.robot_id, physicsClientId=self._physics_client_id)

    @cached_property
    def joint_names(self) -> List[str]:
        """Get the names of the joints in the robot."""
        joint_names = [
            p.getJointInfo(self.robot_id, i, physicsClientId=self._physics_client_id)[
                1
            ].decode("utf-8")
            for i in range(self.num_joints)
        ]
        print("ROBOT:", joint_names)
        return joint_names

    @property
    @abc.abstractmethod
    def left_finger_joint_name(self) -> str:
        """The name of the left finger joint."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def right_finger_joint_name(self) -> str:
        """The name of the right finger joint."""
        raise NotImplementedError("Override me!")

    @cached_property
    def left_finger_id(self) -> int:
        """The PyBullet ID for the left finger."""
        return self.joint_names.index(self.left_finger_joint_name)

    @cached_property
    def right_finger_id(self) -> int:
        """The PyBullet ID for the right finger."""
        return self.joint_names.index(self.right_finger_joint_name)

    @cached_property
    def left_finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the left finger."""
        return self.arm_joints.index(self.left_finger_id)

    @cached_property
    def right_finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the right finger."""
        return self.arm_joints.index(self.right_finger_id)

    @cached_property
    def _joint_limits(self) -> Tuple[JointsState, JointsState]:
        """Return the lower and upper joint limits."""
        joint_lower_limits, joint_upper_limits = [], []

        for i in self.arm_joints:
            info = p.getJointInfo(
                self.robot_id, i, physicsClientId=self._physics_client_id
            )
            lower_limit = info[8]
            upper_limit = info[9]
            # Per PyBullet documentation, values ignored if upper < lower.
            if upper_limit < lower_limit:
                joint_lower_limits.append(-np.inf)
                joint_upper_limits.append(np.inf)
            else:
                joint_lower_limits.append(lower_limit)
                joint_upper_limits.append(upper_limit)

        return joint_lower_limits, joint_upper_limits

    @property
    def joint_lower_limits(self) -> JointsState:
        """Lower bound on the arm joint limits."""
        return self._joint_limits[0]

    @property
    def joint_upper_limits(self) -> JointsState:
        """Upper bound on the arm joint limits."""
        return self._joint_limits[1]

    @property
    @abc.abstractmethod
    def open_fingers(self) -> float:
        """The value at which the finger joints should be open."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def closed_fingers(self) -> float:
        """The value at which the finger joints should be closed."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def reset_state(self, robot_state: Array) -> None:
        """Reset the robot state to match the input state.

        The robot_state corresponds to the State vector for the robot
        object.
        """
        raise NotImplementedError("Override me!")

    def get_state(self) -> Array:
        """Get the robot state vector based on the current PyBullet state.

        This corresponds to the State vector for the robot object.
        """
        ee_link_state = p.getLinkState(
            self.robot_id, self.end_effector_id, physicsClientId=self._physics_client_id
        )
        rx, ry, rz = ee_link_state[4]
        # Note: we assume both left and right gripper have the same joint position
        rf = p.getJointState(
            self.robot_id,
            self.left_finger_id,
            physicsClientId=self._physics_client_id,
        )[0]
        # pose_x, pose_y, pose_z, fingers
        return np.array([rx, ry, rz, rf], dtype=np.float32)

    def get_joints(self) -> JointsState:
        """Get the joints state from the current PyBullet state."""
        joint_state = [
            joint_info[0]  # extract joint position only
            for joint_info in p.getJointStates(
                self.robot_id, self.arm_joints, physicsClientId=self._physics_client_id
            )
        ]
        return joint_state

    def set_joints(self, joints_state: JointsState) -> None:
        """Directly set the joint states.

        Outside of resetting to an initial state, this should not be
        used with the robot that uses stepSimulation(); it should only
        be used for motion planning, collision checks, etc., in a robot
        that does not maintain state.
        """
        assert len(joints_state) == len(self.arm_joints)
        for joint_id, joint_val in zip(self.arm_joints, joints_state):
            p.resetJointState(
                self.robot_id,
                joint_id,
                targetValue=joint_val,
                targetVelocity=0,
                physicsClientId=self._physics_client_id,
            )

    def set_motors(self, joints_state: JointsState) -> None:
        """Update the motors to move toward the given joints state."""
        assert len(joints_state) == len(self.arm_joints)

        # Set arm joint motors.
        if CFG.pybullet_control_mode == "position":
            p.setJointMotorControlArray(
                bodyUniqueId=self.robot_id,
                jointIndices=self.arm_joints,
                controlMode=p.POSITION_CONTROL,
                targetPositions=joints_state,
                physicsClientId=self._physics_client_id,
            )
        elif CFG.pybullet_control_mode == "reset":
            self.set_joints(joints_state)
        else:
            raise NotImplementedError(
                f"Unrecognized pybullet_control_mode: {CFG.pybullet_control_mode}"
            )

    def forward_kinematics(self, joints_state: JointsState) -> Pose3D:
        """Compute the end effector position that would result if the robot arm
        joints state was equal to the input joints_state.

        WARNING: This method will make use of resetJointState(), and so it
        should NOT be used during simulation.
        """
        self.set_joints(joints_state)
        ee_link_state = p.getLinkState(
            self.robot_id,
            self.end_effector_id,
            computeForwardKinematics=True,
            physicsClientId=self._physics_client_id,
        )
        position = ee_link_state[4]
        return position

    @abc.abstractmethod
    def inverse_kinematics(
        self, end_effector_pose: Pose3D, validate: bool
    ) -> JointsState:
        """Compute a joints state from a target end effector position.

        The target orientation is always self._ee_orientation.

        If validate is True, guarantee that the returned joints state
        would result in end_effector_pose if run through
        forward_kinematics.

        WARNING: if validate is True, physics may be overridden, and so it
        should not be used within simulation.
        """
        raise NotImplementedError("Override me!")


class FetchPyBulletRobot(SingleArmPyBulletRobot):
    """A Fetch robot with a fixed base and only one arm in use."""

    # Parameters that aren't important enough to need to clog up settings.py
    _base_pose: ClassVar[Pose3D] = (0.75, 0.7441, 0.0)
    _base_orientation: ClassVar[Sequence[float]] = [0.0, 0.0, 0.0, 1.0]

    @property
    def end_effector_name(self) -> str:
        return "gripper_axis"

    @property
    def left_finger_joint_name(self) -> str:
        return "l_gripper_finger_joint"

    @property
    def right_finger_joint_name(self) -> str:
        return "r_gripper_finger_joint"

    @classmethod
    def get_name(cls) -> str:
        return "fetch"

    def _initialize(self) -> None:
        self._fetch_id = p.loadURDF(
            utils.get_env_asset_path("urdf/robots/fetch.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id,
        )
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id,
        )

    @property
    def robot_id(self) -> int:
        return self._fetch_id

    @property
    def open_fingers(self) -> float:
        return 0.04

    @property
    def closed_fingers(self) -> float:
        return 0.01

    def reset_state(self, robot_state: Array) -> None:
        rx, ry, rz, rf = robot_state
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id,
        )
        # First, reset the joint values to self._initial_joints_state,
        # so that IK is consistent (less sensitive to initialization).
        self.set_joints(self.initial_joints_state)
        # Now run IK to get to the actual starting rx, ry, rz. We use
        # validate=True to ensure that this initialization works.
        joints_state = self.inverse_kinematics((rx, ry, rz), validate=True)
        self.set_joints(joints_state)
        # Handle setting the robot finger joints.
        for finger_id in [self.left_finger_id, self.right_finger_id]:
            p.resetJointState(
                self._fetch_id, finger_id, rf, physicsClientId=self._physics_client_id
            )

    def inverse_kinematics(
        self, end_effector_pose: Pose3D, validate: bool
    ) -> JointsState:
        return pybullet_inverse_kinematics(
            self._fetch_id,
            self.end_effector_id,
            end_effector_pose,
            self._ee_orientation,
            self.arm_joints,
            physics_client_id=self._physics_client_id,
            validate=validate,
        )


class PandaPyBulletRobot(SingleArmPyBulletRobot):
    """Franka Emika Panda which we assume is fixed on some base."""

    @property
    def end_effector_name(self) -> str:
        # TODO explain or change this
        return "tool_joint"

    @property
    def left_finger_joint_name(self) -> str:
        return "panda_finger_joint1"

    @property
    def right_finger_joint_name(self) -> str:
        return "panda_finger_joint2"

    # Parameters that aren't important enough to need to clog up settings.py
    # _base_pose: ClassVar[Pose3D] = (0.75, 0.7441, 0.25)
    _base_pose: ClassVar[Pose3D] = (0, 0, 0)
    _base_orientation: ClassVar[Sequence[float]] = [0.0, 0.0, 0.0, 1.0]

    @classmethod
    def get_name(cls) -> str:
        return "panda"

    def _initialize(self) -> None:

        # self._ikfast_info = IKFastInfo(
        #     module_name="franka_panda.ikfast_panda_arm",
        #     base_link="panda_link0",
        #     ee_link="panda_link8",
        #     free_joints=["panda_joint7"],
        # )

        # TODO!!! fix this
        self._ee_orientation = [-1.0, 0.0, 0.0, 0.0]

        self._panda_id = p.loadURDF(
            utils.get_env_asset_path(
                "urdf/franka_description/robots/panda_arm_hand.urdf"
            ),
            useFixedBase=True,
            physicsClientId=self._physics_client_id,
        )

        p.resetBasePositionAndOrientation(
            self._panda_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id,
        )

        # Extract IDs for individual robot links and joints.

        # TODO: change this, because it's highly confusing that this is not
        # the tool tip, since end_effector_id is the tool tip.
        self._end_effector_link = get_link_from_name(
            self._panda_id, "panda_link8", self._physics_client_id
        )
        self._tool_link = get_link_from_name(
            self._panda_id, "tool_link", self._physics_client_id
        )

        # TODO: factor out common code here and elsewhere.
        joint_names = self.joint_names

        self._initial_joint_values = self.inverse_kinematics(
            self._ee_home_pose, validate=True
        )
        # The initial joint values for the fingers should be open. IK may
        # return anything for them.
        self._initial_joint_values[-2] = self.open_fingers
        self._initial_joint_values[-1] = self.open_fingers

    @property
    def robot_id(self) -> int:
        return self._panda_id

    @property
    def open_fingers(self) -> float:
        return 0.04

    @property
    def closed_fingers(self) -> float:
        return 0.03

    def reset_state(self, robot_state: Array) -> None:
        rx, ry, rz, rf = robot_state
        p.resetBasePositionAndOrientation(
            self._panda_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id,
        )
        # First, reset the joint values to self._initial_joint_values,
        # so that IK is consistent (less sensitive to initialization).
        joint_values = self._initial_joint_values
        for joint_id, joint_val in zip(self._arm_joints, joint_values):
            p.resetJointState(
                self._panda_id,
                joint_id,
                joint_val,
                physicsClientId=self._physics_client_id,
            )
        # Now run IK to get to the actual starting rx, ry, rz. We use
        # validate=True to ensure that this initialization works.
        joint_values = self.inverse_kinematics((rx, ry, rz), validate=True)
        for joint_id, joint_val in zip(self._arm_joints, joint_values):
            p.resetJointState(
                self._panda_id,
                joint_id,
                joint_val,
                physicsClientId=self._physics_client_id,
            )
        # Handle setting the robot finger joints.
        for finger_id in [self._left_finger_id, self._right_finger_id]:
            p.resetJointState(
                self._panda_id, finger_id, rf, physicsClientId=self._physics_client_id
            )

    def inverse_kinematics(
        self, end_effector_pose: Pose3D, validate: bool
    ) -> List[float]:

        # TODO handle validate argument

        # TODO explain
        # TODO check if we can compute some of these just once
        tool_from_ee = get_relative_link_pose(
            self.robot_id,
            self._end_effector_link,
            self._tool_link,
            self._physics_client_id,
        )
        world_from_base = p.getBasePositionAndOrientation(
            self.robot_id, physicsClientId=self._physics_client_id
        )
        base_from_ee = p.multiplyTransforms(
            *p.multiplyTransforms(
                *p.invertTransform(*world_from_base),
                end_effector_pose,
                self._ee_orientation,
            ),
            *tool_from_ee,
        )

        joints_state = ikfast_inverse_kinematics(
            "panda_arm",
            base_from_ee[0],
            base_from_ee[1],
            physics_client_id=self._physics_client_id,
        )

        # Add fingers.
        final_joint_state = list(joints_state)
        first_finger_idx, second_finger_idx = sorted(
            [self.left_finger_joint_idx, self.right_finger_joint_idx]
        )
        final_joint_state.insert(first_finger_idx, self.open_fingers)
        final_joint_state.insert(second_finger_idx, self.open_fingers)
        return final_joint_state


def create_single_arm_pybullet_robot(
    robot_name: str,
    ee_home_pose: Pose3D,
    ee_orientation: Sequence[float],
    physics_client_id: int,
) -> SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    for cls in utils.get_all_subclasses(SingleArmPyBulletRobot):
        if not cls.__abstractmethods__ and cls.get_name() == robot_name:
            robot = cls(ee_home_pose, ee_orientation, physics_client_id)
            break
    else:
        raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")
    return robot


########################### Other utility functions ###########################


def get_link_from_name(body: int, name: str, physics_client_id: int) -> int:
    """Get the link ID from the name of the link."""
    base_info = p.getBodyInfo(body, physicsClientId=physics_client_id)
    base_name = base_info[0].decode(encoding="UTF-8")
    if name == base_name:
        return -1  # base link
    for link in range(p.getNumJoints(body, physicsClientId=physics_client_id)):
        joint_info = p.getJointInfo(body, link, physicsClientId=physics_client_id)
        joint_name = joint_info[12].decode("UTF-8")
        print(joint_name)
        if joint_name == name:
            return link
    raise ValueError(f"Body {body} has no link with name {name}.")


def get_link_pose(
    body: int, link: int, physics_client_id: int
) -> Tuple[Pose3D, Sequence[float]]:
    """Get the position and orientation for a link."""
    link_state = p.getLinkState(body, link, physicsClientId=physics_client_id)
    return link_state[0], link_state[1]


def get_relative_link_pose(
    body: int, link1: int, link2: int, physics_client_id: int
) -> Tuple[Pose3D, Sequence[float]]:
    """Get the pose of one link relative to another link on the same body."""
    world_from_link1 = get_link_pose(body, link1, physics_client_id)
    world_from_link2 = get_link_pose(body, link2, physics_client_id)
    link2_from_link1 = p.multiplyTransforms(
        *p.invertTransform(*world_from_link2), *world_from_link1
    )
    return link2_from_link1


def get_kinematic_chain(
    robot: int, end_effector: int, physics_client_id: int
) -> List[int]:
    """Get all of the free joints from robot base to end effector.

    Includes the end effector.
    """
    kinematic_chain = []
    while end_effector > -1:
        joint_info = p.getJointInfo(
            robot, end_effector, physicsClientId=physics_client_id
        )
        if joint_info[3] > -1:
            kinematic_chain.append(end_effector)
        end_effector = joint_info[-1]
    return kinematic_chain


def pybullet_inverse_kinematics(
    robot: int,
    end_effector: int,
    target_position: Pose3D,
    target_orientation: Sequence[float],
    joints: Sequence[int],
    physics_client_id: int,
    validate: bool = True,
) -> JointsState:
    """Runs IK and returns a joints state for the given (free) joints.

    If validate is True, the PyBullet IK solver is called multiple
    times, resetting the robot state each time, until the target
    position is reached. If the target position is not reached after a
    maximum number of iters, an exception is raised.
    """
    # Figure out which joint each dimension of the return of IK corresponds to.
    free_joints = []
    num_joints = p.getNumJoints(robot, physicsClientId=physics_client_id)
    for idx in range(num_joints):
        joint_info = p.getJointInfo(robot, idx, physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            free_joints.append(idx)
    assert set(joints).issubset(set(free_joints))

    # Record the initial state of the joints so that we can reset them after.
    if validate:
        initial_joints_states = p.getJointStates(
            robot, free_joints, physicsClientId=physics_client_id
        )
        assert len(initial_joints_states) == len(free_joints)

    # Running IK once is often insufficient, so we run it multiple times until
    # convergence. If it does not converge, an error is raised.
    convergence_tol = CFG.pybullet_ik_tol
    for _ in range(CFG.pybullet_max_ik_iters):
        free_joint_vals = p.calculateInverseKinematics(
            robot,
            end_effector,
            target_position,
            targetOrientation=target_orientation,
            physicsClientId=physics_client_id,
        )
        assert len(free_joints) == len(free_joint_vals)
        if not validate:
            break
        # Update the robot state and check if the desired position and
        # orientation are reached.
        for joint, joint_val in zip(free_joints, free_joint_vals):
            p.resetJointState(
                robot, joint, targetValue=joint_val, physicsClientId=physics_client_id
            )
        # TODO can this be replaced with get_link_pose?
        ee_link_state = p.getLinkState(
            robot,
            end_effector,
            computeForwardKinematics=True,
            physicsClientId=physics_client_id,
        )
        position = ee_link_state[4]
        # Note: we are checking positions only for convergence.
        if np.allclose(position, target_position, atol=convergence_tol):
            break
    else:
        raise Exception("Inverse kinematics failed to converge.")

    # Reset the joint states to their initial values to avoid modifying the
    # PyBullet internal state.
    if validate:
        for joint, (pos, vel, _, _) in zip(free_joints, initial_joints_states):
            p.resetJointState(
                robot,
                joint,
                targetValue=pos,
                targetVelocity=vel,
                physicsClientId=physics_client_id,
            )
    # Order the found free_joint_vals based on the requested joints.
    joint_vals = []
    for joint in joints:
        free_joint_idx = free_joints.index(joint)
        joint_val = free_joint_vals[free_joint_idx]
        joint_vals.append(joint_val)

    return joint_vals


def run_motion_planning(
    robot: SingleArmPyBulletRobot,
    initial_state: JointsState,
    target_state: JointsState,
    collision_bodies: Collection[int],
    seed: int,
    physics_client_id: int,
) -> Optional[Sequence[JointsState]]:
    """Run BiRRT to find a collision-free sequence of joint states.

    Note that this function changes the state of the robot.
    """
    rng = np.random.default_rng(seed)
    joint_space = robot.action_space
    joint_space.seed(seed)
    _sample_fn = lambda _: joint_space.sample()
    num_interp = CFG.pybullet_birrt_extend_num_interp

    def _extend_fn(pt1: JointsState, pt2: JointsState) -> Iterator[JointsState]:
        pt1_arr = np.array(pt1)
        pt2_arr = np.array(pt2)
        num = int(np.ceil(max(abs(pt1_arr - pt2_arr)))) * num_interp
        if num == 0:
            yield pt2
        for i in range(1, num + 1):
            yield list(pt1_arr * (1 - i / num) + pt2_arr * i / num)

    def _collision_fn(pt: JointsState) -> bool:
        robot.set_joints(pt)
        p.performCollisionDetection(physicsClientId=physics_client_id)
        for body in collision_bodies:
            if p.getContactPoints(
                robot.robot_id, body, physicsClientId=physics_client_id
            ):
                return True
        return False

    def _distance_fn(from_pt: JointsState, to_pt: JointsState) -> float:
        from_ee = robot.forward_kinematics(from_pt)
        to_ee = robot.forward_kinematics(to_pt)
        return sum(np.subtract(from_ee, to_ee) ** 2)

    birrt = utils.BiRRT(
        _sample_fn,
        _extend_fn,
        _collision_fn,
        _distance_fn,
        rng,
        num_attempts=CFG.pybullet_birrt_num_attempts,
        num_iters=CFG.pybullet_birrt_num_iters,
        smooth_amt=CFG.pybullet_birrt_smooth_amt,
    )

    return birrt.query(initial_state, target_state)


if __name__ == "__main__":
    from pybullet_tools.utils import wait_for_user, get_bodies

    logging.basicConfig(
        level=logging.DEBUG, format="%(message)s", handlers=[logging.StreamHandler()]
    )

    physics_client_id = p.connect(p.GUI)
    p.resetSimulation(physicsClientId=physics_client_id)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Disable the preview windows for faster rendering.
    p.configureDebugVisualizer(
        p.COV_ENABLE_GUI, False, physicsClientId=physics_client_id
    )
    p.configureDebugVisualizer(
        p.COV_ENABLE_RGB_BUFFER_PREVIEW, False, physicsClientId=physics_client_id
    )
    p.configureDebugVisualizer(
        p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, False, physicsClientId=physics_client_id
    )
    p.configureDebugVisualizer(
        p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, False, physicsClientId=physics_client_id
    )
    CFG.seed = 0

    panda = create_single_arm_pybullet_robot(
        "fetch",
        (0, 0.5, 0.2),
        p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi]),
        physics_client_id,
    )
    plane_id = p.loadURDF("plane.urdf")
    p.setGravity(0.0, 0.0, -10.0, physicsClientId=physics_client_id)

    logging.info(panda.initial_joints_state)
    panda.set_motors(panda.initial_joints_state)
    # wait_for_user("test")
    print(panda.joint_names)
    print(panda.joint_lower_limits)
    print(panda.joint_upper_limits)
    print(get_bodies())

    wait_for_user("start?")
    for _ in range(50):
        p.stepSimulation(physicsClientId=physics_client_id)
        print(panda.get_joints())
        print(panda.get_state())
        time.sleep(0.1)

    wait_for_user("terminate?")
