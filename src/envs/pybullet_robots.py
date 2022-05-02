"""Interfaces to PyBullet robots."""

import abc
from typing import Callable, ClassVar, Dict, List, Sequence, Tuple, cast

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, JointsState, Object, \
    ParameterizedOption, Pose3D, State, Type


class _SingleArmPyBulletRobot(abc.ABC):
    """A single-arm fixed-base PyBullet robot with a two-finger gripper.

    The action space for the robot is 4D. The first three dimensions are
    a change in the (x, y, z) of the end effector. The last dimension is
    a change in the finger joint(s), which are constrained to be
    symmetric.
    """

    def __init__(self, ee_home_pose: Pose3D, ee_orientation: Sequence[float],
                 move_to_pose_tol: float, max_vel_norm: float,
                 grasp_tol: float, physics_client_id: int) -> None:
        # Initial position for the end effector.
        self._ee_home_pose = ee_home_pose
        # Orientation for the end effector.
        self._ee_orientation = ee_orientation
        # The tolerance used in create_move_end_effector_to_pose_option().
        self._move_to_pose_tol = move_to_pose_tol
        # Used for the action space.
        self._max_vel_norm = max_vel_norm
        # Used for detecting when an object is considered grasped.
        self._grasp_tol = grasp_tol
        self._physics_client_id = physics_client_id
        # These get overridden in initialize(), but type checking needs to be
        # aware that it exists.
        self._initial_joints_state: JointsState = []
        self._initialize()

    @property
    def initial_joints_state(self) -> JointsState:
        """The joint values for the robot in its home pose."""
        return self._initial_joints_state

    @property
    def action_space(self) -> Box:
        """The action space for the robot.

        Represents position control of the arm and finger joints.
        """
        return Box(np.array(self.joint_lower_limits, dtype=np.float32),
                   np.array(self.joint_upper_limits, dtype=np.float32),
                   dtype=np.float32)

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

    @property
    @abc.abstractmethod
    def left_finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the left finger."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def right_finger_joint_idx(self) -> int:
        """The index into the joints corresponding to the right finger."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def joint_lower_limits(self) -> JointsState:
        """Lower bound on the arm joint limits."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def joint_upper_limits(self) -> JointsState:
        """Upper bound on the arm joint limits."""
        raise NotImplementedError("Override me!")

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

    @abc.abstractmethod
    def get_state(self) -> Array:
        """Get the robot state vector based on the current PyBullet state.

        This corresponds to the State vector for the robot object.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_joints(self) -> JointsState:
        """Get the joint states from the current PyBullet state."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def set_motors(self, joints_state: JointsState) -> None:
        """Update the motors to move toward the given joints state."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def forward_kinematics(self, joints_state: JointsState) -> Pose3D:
        """Compute the end effector pose that if the robot arm joint states
        were equal to the input joints_state.

        WARNING: This method will make use of resetJointState(), and so it
        should NOT be used during simulation.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def inverse_kinematics(self, end_effector_pose: Pose3D,
                           validate: bool) -> JointsState:
        """Compute a joint state from an end effector pose.

        If validate is True, guarantee that the returned joint state
        would result in end_effector_pose if run through
        forward_kinematics.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def create_move_end_effector_to_pose_option(
        self,
        name: str,
        types: Sequence[Type],
        params_space: Box,
        get_current_and_target_pose_and_finger_status: Callable[
            [State, Sequence[Object], Array], Tuple[Pose3D, Pose3D, str]],
    ) -> ParameterizedOption:
        """A generic utility that creates a ParameterizedOption for moving the
        end effector to a target pose, given a function that takes in the
        current state, objects, and parameters, and returns the current pose
        and target pose of the end effector, and the finger status."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def create_change_fingers_option(
        self, name: str, types: Sequence[Type], params_space: Box,
        get_current_and_target_val: Callable[[State, Sequence[Object], Array],
                                             Tuple[float, float]]
    ) -> ParameterizedOption:
        """A generic utility that creates a ParameterizedOption for changing
        the robot fingers, given a function that takes in the current state,
        objects, and parameters, and returns the current and target finger
        joint values."""
        raise NotImplementedError("Override me!")


class FetchPyBulletRobot(_SingleArmPyBulletRobot):
    """A Fetch robot with a fixed base and only one arm in use."""

    # Parameters that aren't important enough to need to clog up settings.py
    _base_pose: ClassVar[Pose3D] = (0.75, 0.7441, 0.0)
    _base_orientation: ClassVar[Sequence[float]] = [0., 0., 0., 1.]
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

    def _initialize(self) -> None:
        self._fetch_id = p.loadURDF(
            utils.get_env_asset_path("urdf/robots/fetch.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id)

        # Extract IDs for individual robot links and joints.
        joint_names = [
            p.getJointInfo(
                self._fetch_id, i,
                physicsClientId=self._physics_client_id)[1].decode("utf-8")
            for i in range(
                p.getNumJoints(self._fetch_id,
                               physicsClientId=self._physics_client_id))
        ]
        self._ee_id = joint_names.index('gripper_axis')
        self._arm_joints = get_kinematic_chain(
            self._fetch_id,
            self._ee_id,
            physics_client_id=self._physics_client_id)
        self._left_finger_id = joint_names.index("l_gripper_finger_joint")
        self._right_finger_id = joint_names.index("r_gripper_finger_joint")
        self._arm_joints.append(self._left_finger_id)
        self._arm_joints.append(self._right_finger_id)

        self._initial_joints_state = self.inverse_kinematics(
            self._ee_home_pose, validate=True)
        # The initial joint values for the fingers should be open. IK may
        # return anything for them.
        self._initial_joints_state[-2] = self.open_fingers
        self._initial_joints_state[-1] = self.open_fingers
        # Establish the lower and upper limits for the arm joints.
        self._joint_lower_limits = []
        self._joint_upper_limits = []
        for i in self._arm_joints:
            info = p.getJointInfo(self._fetch_id,
                                  i,
                                  physicsClientId=self._physics_client_id)
            lower_limit = info[8]
            upper_limit = info[9]
            # Per PyBullet documentation, values ignored if upper < lower.
            if upper_limit < lower_limit:
                self._joint_lower_limits.append(-np.inf)
                self._joint_upper_limits.append(np.inf)
            else:
                self._joint_lower_limits.append(lower_limit)
                self._joint_upper_limits.append(upper_limit)

    @property
    def robot_id(self) -> int:
        return self._fetch_id

    @property
    def end_effector_id(self) -> int:
        return self._ee_id

    @property
    def left_finger_id(self) -> int:
        return self._left_finger_id

    @property
    def right_finger_id(self) -> int:
        return self._right_finger_id

    @property
    def left_finger_joint_idx(self) -> int:
        return len(self._arm_joints) - 2

    @property
    def right_finger_joint_idx(self) -> int:
        return len(self._arm_joints) - 1

    @property
    def joint_lower_limits(self) -> JointsState:
        return self._joint_lower_limits

    @property
    def joint_upper_limits(self) -> JointsState:
        return self._joint_upper_limits

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
            physicsClientId=self._physics_client_id)
        # First, reset the joint values to self._initial_joints_state,
        # so that IK is consistent (less sensitive to initialization).
        joints_state = self._initial_joints_state
        for joint_id, joint_val in zip(self._arm_joints, joints_state):
            p.resetJointState(self._fetch_id,
                              joint_id,
                              joint_val,
                              physicsClientId=self._physics_client_id)
        # Now run IK to get to the actual starting rx, ry, rz. We use
        # validate=True to ensure that this initialization works.
        joints_state = self.inverse_kinematics((rx, ry, rz), validate=True)
        for joint_id, joint_val in zip(self._arm_joints, joints_state):
            p.resetJointState(self._fetch_id,
                              joint_id,
                              joint_val,
                              physicsClientId=self._physics_client_id)
        # Handle setting the robot finger joints.
        for finger_id in [self._left_finger_id, self._right_finger_id]:
            p.resetJointState(self._fetch_id,
                              finger_id,
                              rf,
                              physicsClientId=self._physics_client_id)

    def get_state(self) -> Array:
        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       physicsClientId=self._physics_client_id)
        rx, ry, rz = ee_link_state[4]
        rf = p.getJointState(self._fetch_id,
                             self._left_finger_id,
                             physicsClientId=self._physics_client_id)[0]
        # pose_x, pose_y, pose_z, fingers
        return np.array([rx, ry, rz, rf], dtype=np.float32)

    def get_joints(self) -> JointsState:
        joints_state = []
        for joint_idx in self._arm_joints:
            joint_val = p.getJointState(
                self._fetch_id,
                joint_idx,
                physicsClientId=self._physics_client_id)[0]
            joints_state.append(joint_val)
        return joints_state

    def set_motors(self, joints_state: JointsState) -> None:
        assert len(joints_state) == len(self._arm_joints)

        # Set arm joint motors.
        p.setJointMotorControlArray(bodyUniqueId=self._fetch_id,
                                    jointIndices=self._arm_joints,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=joints_state,
                                    physicsClientId=self._physics_client_id)

    def forward_kinematics(self, joints_state: JointsState) -> Pose3D:
        assert len(joints_state) == len(self._arm_joints)
        for joint_id, joint_val in zip(self._arm_joints, joints_state):
            p.resetJointState(self._fetch_id,
                              joint_id,
                              joint_val,
                              physicsClientId=self._physics_client_id)
        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       computeForwardKinematics=True,
                                       physicsClientId=self._physics_client_id)
        position = ee_link_state[4]
        return position

    def inverse_kinematics(self, end_effector_pose: Pose3D,
                           validate: bool) -> JointsState:
        return inverse_kinematics(self._fetch_id,
                                  self._ee_id,
                                  end_effector_pose,
                                  self._ee_orientation,
                                  self._arm_joints,
                                  physics_client_id=self._physics_client_id,
                                  validate=validate)

    def create_move_end_effector_to_pose_option(
        self,
        name: str,
        types: Sequence[Type],
        params_space: Box,
        get_current_and_target_pose_and_finger_status: Callable[
            [State, Sequence[Object], Array], Tuple[Pose3D, Pose3D, str]],
    ) -> ParameterizedOption:

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory  # unused
            # First handle the main arm joints.
            current, target, finger_status = \
                get_current_and_target_pose_and_finger_status(
                    state, objects, params)
            # Run IK to determine the target joint positions.
            ee_delta = np.subtract(target, current)
            # Reduce the target to conform to the max velocity constraint.
            ee_norm = np.linalg.norm(ee_delta)  # type: ignore
            if ee_norm > self._max_vel_norm:
                ee_delta = ee_delta * self._max_vel_norm / ee_norm
            ee_action = np.add(current, ee_delta)
            # Keep validate as False because validate=True would update the
            # state of the robot during simulation, which overrides physics.
            joints_state = self.inverse_kinematics(
                (ee_action[0], ee_action[1], ee_action[2]), validate=False)
            # Handle the fingers. Fingers drift if left alone.
            # When the fingers are not explicitly being opened or closed, we
            # nudge the fingers toward being open or closed according to the
            # finger status.
            if finger_status == "open":
                finger_delta = self._finger_action_nudge_magnitude
            else:
                assert finger_status == "closed"
                finger_delta = -self._finger_action_nudge_magnitude
            # Extract the current finger state.
            state = cast(utils.PyBulletState, state)
            finger_state = state.joints_state[self.left_finger_joint_idx]
            # The finger action is an absolute joint position for the fingers.
            f_action = finger_state + finger_delta
            # Override the meaningless finger values in joint_action.
            joints_state[self.left_finger_joint_idx] = f_action
            joints_state[self.right_finger_joint_idx] = f_action
            action_arr = np.array(joints_state, dtype=np.float32)
            # This clipping is needed sometimes for the joint limits.
            action_arr = np.clip(action_arr, self.action_space.low,
                                 self.action_space.high)
            assert self.action_space.contains(action_arr)
            return Action(action_arr)

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory  # unused
            current, target, _ = \
                get_current_and_target_pose_and_finger_status(
                    state, objects, params)
            squared_dist = np.sum(np.square(np.subtract(current, target)))
            return squared_dist < self._move_to_pose_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=lambda _1, _2, _3, _4: True,
                                   terminal=_terminal)

    def create_change_fingers_option(
        self, name: str, types: Sequence[Type], params_space: Box,
        get_current_and_target_val: Callable[[State, Sequence[Object], Array],
                                             Tuple[float, float]]
    ) -> ParameterizedOption:

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory  # unused
            current_val, target_val = get_current_and_target_val(
                state, objects, params)
            f_delta = target_val - current_val
            f_delta = np.clip(f_delta, -self._max_vel_norm, self._max_vel_norm)
            f_action = current_val + f_delta
            # Don't change the rest of the joints.
            state = cast(utils.PyBulletState, state)
            target = np.array(state.joints_state, dtype=np.float32)
            target[self.left_finger_joint_idx] = f_action
            target[self.right_finger_joint_idx] = f_action
            # This clipping is needed sometimes for the joint limits.
            target = np.clip(target, self.action_space.low,
                             self.action_space.high)
            assert self.action_space.contains(target)
            return Action(target)

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory  # unused
            current_val, target_val = get_current_and_target_val(
                state, objects, params)
            squared_dist = (target_val - current_val)**2
            return squared_dist < self._grasp_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=lambda _1, _2, _3, _4: True,
                                   terminal=_terminal)


def create_single_arm_pybullet_robot(
        robot_name: str, ee_home_pose: Pose3D, ee_orientation: Sequence[float],
        move_to_pose_tol: float, max_vel_norm: float, grasp_tol: float,
        physics_client_id: int) -> _SingleArmPyBulletRobot:
    """Create a single-arm PyBullet robot."""
    if robot_name == "fetch":
        return FetchPyBulletRobot(ee_home_pose, ee_orientation,
                                  move_to_pose_tol, max_vel_norm, grasp_tol,
                                  physics_client_id)
    raise NotImplementedError(f"Unrecognized robot name: {robot_name}.")


def get_kinematic_chain(robot: int, end_effector: int,
                        physics_client_id: int) -> List[int]:
    """Get all of the free joints from robot base to end effector.

    Includes the end effector.
    """
    kinematic_chain = []
    while end_effector > -1:
        joint_info = p.getJointInfo(robot,
                                    end_effector,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            kinematic_chain.append(end_effector)
        end_effector = joint_info[-1]
    return kinematic_chain


def inverse_kinematics(
    robot: int,
    end_effector: int,
    target_position: Sequence[float],
    target_orientation: Sequence[float],
    joints: Sequence[int],
    physics_client_id: int,
    validate: bool = True,
) -> JointsState:
    """Runs IK and returns joint state for the given (free) joints.

    If validate is True, the PyBullet IK solver is called multiple
    times, resetting the robot state each time, until the target
    position is reached. If the target position is not reached after a
    maximum number of iters, an exception is raised.
    """
    # Figure out which joint each dimension of the return of IK corresponds to.
    free_joints = []
    num_joints = p.getNumJoints(robot, physicsClientId=physics_client_id)
    for idx in range(num_joints):
        joint_info = p.getJointInfo(robot,
                                    idx,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            free_joints.append(idx)
    assert set(joints).issubset(set(free_joints))

    # Record the initial state of the joints so that we can reset them after.
    if validate:
        initial_joints_states = p.getJointStates(
            robot, free_joints, physicsClientId=physics_client_id)
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
            physicsClientId=physics_client_id)
        assert len(free_joints) == len(free_joint_vals)
        if not validate:
            break
        # Update the robot state and check if the desired position and
        # orientation are reached.
        for joint, joint_val in zip(free_joints, free_joint_vals):
            p.resetJointState(robot,
                              joint,
                              targetValue=joint_val,
                              physicsClientId=physics_client_id)
        ee_link_state = p.getLinkState(robot,
                                       end_effector,
                                       computeForwardKinematics=True,
                                       physicsClientId=physics_client_id)
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
            p.resetJointState(robot,
                              joint,
                              targetValue=pos,
                              targetVelocity=vel,
                              physicsClientId=physics_client_id)

    # Order the found free_joint_vals based on the requested joints.
    joint_vals = []
    for joint in joints:
        free_joint_idx = free_joints.index(joint)
        joint_val = free_joint_vals[free_joint_idx]
        joint_vals.append(joint_val)

    return joint_vals
