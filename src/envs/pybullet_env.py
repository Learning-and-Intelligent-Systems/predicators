"""Base class for a PyBullet environment.

Contains useful common code.
"""

import abc
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Tuple, \
    cast

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.envs.pybullet_robots import _SingleArmPyBulletRobot
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, Image, Object, \
    ParameterizedOption, Pose3D, State, Task, Type


class PyBulletEnv(BaseEnv):
    """Base class for a PyBullet environment."""
    # Parameters that aren't important enough to need to clog up settings.py

    # General robot parameters.
    _max_vel_norm: ClassVar[float] = 0.05
    _grasp_tol: ClassVar[float] = 0.05
    _move_to_pose_tol: ClassVar[float] = 0.0001
    _finger_action_tol: ClassVar[float] = 0.0001
    _finger_action_nudge_magnitude: ClassVar[float] = 0.001

    # Object parameters.
    _obj_mass: ClassVar[float] = 0.5
    _obj_friction: ClassVar[float] = 1.2
    _obj_colors: ClassVar[Sequence[Tuple[float, float, float, float]]] = [
        (0.95, 0.05, 0.1, 1.),
        (0.05, 0.95, 0.1, 1.),
        (0.1, 0.05, 0.95, 1.),
        (0.4, 0.05, 0.6, 1.),
        (0.6, 0.4, 0.05, 1.),
        (0.05, 0.04, 0.6, 1.),
        (0.95, 0.95, 0.1, 1.),
        (0.95, 0.05, 0.95, 1.),
        (0.05, 0.95, 0.95, 1.),
    ]
    _out_of_view_xy: ClassVar[Sequence[float]] = [10.0, 10.0]

    # Camera parameters.
    _camera_distance: ClassVar[float] = 0.8
    _camera_yaw: ClassVar[float] = 90.0
    _camera_pitch: ClassVar[float] = -24
    _camera_target: ClassVar[Pose3D] = (1.65, 0.75, 0.42)
    _debug_text_position: ClassVar[Pose3D] = (1.65, 0.25, 0.75)

    def __init__(self) -> None:
        super().__init__()

        # When an object is held, a constraint is created to prevent slippage.
        self._held_constraint_id: Optional[int] = None
        self._held_obj_id: Optional[int] = None

        # Set up all the static PyBullet content.
        self._initialize_pybullet()

    def _initialize_pybullet(self) -> None:
        """One-time initialization of PyBullet assets."""
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_use_gui:  # pragma: no cover
            self._physics_client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                self._camera_distance,
                self._camera_yaw,
                self._camera_pitch,
                self._camera_target,
                physicsClientId=self._physics_client_id)
        else:
            self._physics_client_id = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._physics_client_id)

        # Load plane.
        p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"), [0, 0, -1],
                   useFixedBase=True,
                   physicsClientId=self._physics_client_id)

        # Load robot.
        self._pybullet_robot = self._create_pybullet_robot()

        # Set gravity.
        p.setGravity(0., 0., -10., physicsClientId=self._physics_client_id)

    @abc.abstractmethod
    def _create_pybullet_robot(self) -> _SingleArmPyBulletRobot:
        """Make and return a PyBullet robot object, which will be saved as
        self._pybullet_robot."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _extract_robot_state(self, state: State) -> Array:
        """Given a State, extract the robot state, to be passed into
        self._pybullet_robot.reset_state().

        This should be the same type as the return value of
        self._pybullet_robot.get_state().
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return a list of pybullet IDs corresponding to objects in the
        simulator that should be checked when determining whether one is
        held."""
        raise NotImplementedError("Override me!")

    @property
    def action_space(self) -> Box:
        return self._pybullet_robot.action_space

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("A PyBullet environment cannot simulate.")

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        raise NotImplementedError("A PyBullet environment cannot render "
                                  "arbitrary states.")

    def reset(self, train_or_test: str, task_idx: int) -> State:
        state = super().reset(train_or_test, task_idx)
        self._reset_state(state)
        # We could call self._get_state() here, but there could be small
        # inconsistencies between that and the state expected as the initial
        # train task state. Giving the expected initial state in this way
        # leads to a tiny improvement in performance.
        joint_state = list(self._pybullet_robot.initial_joint_values)
        return _PyBulletState(state.data, simulator_state=joint_state)

    def _reset_state(self, state: State) -> None:
        """Helper for reset and testing."""
        # Tear down the old PyBullet scene.
        if self._held_constraint_id is not None:
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None
        self._held_obj_id = None

        # Reset robot.
        self._pybullet_robot.reset_state(self._extract_robot_state(state))

    def render(
            self,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> List[Image]:  # pragma: no cover
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        del caption  # unused

        if not CFG.pybullet_use_gui:
            raise Exception(
                "Rendering only works with GUI on. See "
                "https://github.com/bulletphysics/bullet3/issues/1157")

        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=self._camera_target,
            distance=self._camera_distance,
            yaw=self._camera_yaw,
            pitch=self._camera_pitch,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._physics_client_id)

        width = CFG.pybullet_camera_width
        height = CFG.pybullet_camera_height

        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=float(width / height),
            nearVal=0.1,
            farVal=100.0,
            physicsClientId=self._physics_client_id)

        (_, _, px, _,
         _) = p.getCameraImage(width=width,
                               height=height,
                               viewMatrix=view_matrix,
                               projectionMatrix=proj_matrix,
                               renderer=p.ER_BULLET_HARDWARE_OPENGL,
                               physicsClientId=self._physics_client_id)

        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return [rgb_array]

    def step(self, action: Action) -> State:
        # Send the action to the robot.
        ee_delta = (action.arr[0], action.arr[1], action.arr[2])
        f_delta = action.arr[3]
        self._pybullet_robot.set_motors(ee_delta, f_delta)

        # Step the simulation here before adding or removing constraints
        # because detect_held_object() should use the updated state.
        for _ in range(CFG.pybullet_sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # If not currently holding something, and fingers are closing, check
        # for a new grasp.
        if self._held_constraint_id is None and \
            f_delta < -self._finger_action_tol:
            # Detect whether an object is held.
            self._held_obj_id = self._detect_held_object()
            if self._held_obj_id is not None:
                # Create a grasp constraint.
                base_link_to_world = np.r_[p.invertTransform(*p.getLinkState(
                    self._pybullet_robot.robot_id,
                    self._pybullet_robot.end_effector_id,
                    physicsClientId=self._physics_client_id)[:2])]
                world_to_obj = np.r_[p.getBasePositionAndOrientation(
                    self._held_obj_id,
                    physicsClientId=self._physics_client_id)]
                base_link_to_obj = p.invertTransform(*p.multiplyTransforms(
                    base_link_to_world[:3], base_link_to_world[3:],
                    world_to_obj[:3], world_to_obj[3:]))
                self._held_constraint_id = p.createConstraint(
                    parentBodyUniqueId=self._pybullet_robot.robot_id,
                    parentLinkIndex=self._pybullet_robot.end_effector_id,
                    childBodyUniqueId=self._held_obj_id,
                    childLinkIndex=-1,  # -1 for the base
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=base_link_to_obj[0],
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFrameOrientation=base_link_to_obj[1],
                    physicsClientId=self._physics_client_id)

        # If placing, remove the grasp constraint.
        if self._held_constraint_id is not None and \
            f_delta > self._finger_action_tol:
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None
            self._held_obj_id = None

        self._current_state = self._get_state()
        return self._current_state.copy()

    def _detect_held_object(self) -> Optional[int]:
        """Return the PyBullet object ID of the held object if one exists.

        If multiple objects are within the grasp tolerance, return the
        one that is closest.
        """
        expected_finger_normals = {
            self._pybullet_robot.left_finger_id: np.array([0., 1., 0.]),
            self._pybullet_robot.right_finger_id: np.array([0., -1., 0.]),
        }
        closest_held_obj = None
        closest_held_obj_dist = float("inf")
        for obj_id in self._get_object_ids_for_held_check():
            for finger_id, expected_normal in expected_finger_normals.items():
                # Find points on the object that are within grasp_tol distance
                # of the finger. Note that we use getClosestPoints instead of
                # getContactPoints because we still want to consider the object
                # held even if there is a tiny distance between the fingers and
                # the object.
                closest_points = p.getClosestPoints(
                    bodyA=self._pybullet_robot.robot_id,
                    bodyB=obj_id,
                    distance=self._grasp_tol,
                    linkIndexA=finger_id,
                    physicsClientId=self._physics_client_id)
                for point in closest_points:
                    # If the contact normal is substantially different from
                    # the expected contact normal, this is probably an object
                    # on the outside of the fingers, rather than the inside.
                    # A perfect score here is 1.0 (normals are unit vectors).
                    contact_normal = point[7]
                    if expected_normal.dot(contact_normal) < 0.5:
                        continue
                    # Handle the case where multiple objects pass this check
                    # by taking the closest one. This should be rare, but it
                    # can happen when two objects are stacked and the robot is
                    # unstacking the top one.
                    contact_distance = point[8]
                    if contact_distance < closest_held_obj_dist:
                        closest_held_obj = obj_id
                        closest_held_obj_dist = contact_distance
        return closest_held_obj

    def _create_move_end_effector_to_pose_option(
        self,
        name: str,
        types: Sequence[Type],
        params_space: Box,
        get_current_and_target_pose: Callable[[State, Sequence[Object], Array],
                                              Tuple[Pose3D, Pose3D]],
        finger_status: str,
    ) -> ParameterizedOption:
        """A generic utility that creates a ParameterizedOption for moving the
        end effector to a target pose, given a function that takes in the
        current state, objects, and parameters, and returns the current pose
        and target pose of the end effector.

        Fingers drift if left alone. When the fingers are not explicitly
        being opened or closed, we nudge the fingers toward being open
        or closed according to the finger_status argument.
        """

        if finger_status == "open":
            finger_action = self._finger_action_nudge_magnitude
        else:
            assert finger_status == "closed"
            finger_action = -self._finger_action_nudge_magnitude

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory  # unused
            current, target = get_current_and_target_pose(
                state, objects, params)
            action = np.subtract(target, current)
            action_norm = np.linalg.norm(action)  # type: ignore
            if action_norm > self._max_vel_norm:
                action = action * self._max_vel_norm / action_norm
            action = np.r_[action, finger_action].astype(np.float32)
            assert self.action_space.contains(action)
            return Action(action)

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory  # unused
            current, target = get_current_and_target_pose(
                state, objects, params)
            squared_dist = np.sum(np.square(np.subtract(current, target)))
            return squared_dist < self._move_to_pose_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=lambda _1, _2, _3, _4: True,
                                   terminal=_terminal)

    def _create_change_fingers_option(self, name: str, target_val: float,
                                      types: Sequence[Type], params_space: Box,
                                      robot: Object) -> ParameterizedOption:
        """A generic utility that creates a ParameterizedOption for changing
        the robot fingers."""

        assert types[0] == robot.type

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory, objects, params  # unused
            current_val = state.get(robot, "fingers")
            f_delta = target_val - current_val
            f_delta = np.clip(f_delta, self.action_space.low[3],
                              self.action_space.high[3])
            return Action(np.array([0., 0., 0., f_delta], dtype=np.float32))

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, objects, params  # unused
            current_val = state.get(robot, "fingers")
            squared_dist = (target_val - current_val)**2
            return squared_dist < self._grasp_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=lambda _1, _2, _3, _4: True,
                                   terminal=_terminal)


class _PyBulletState(State):
    """A PyBullet state that stores the robot joint states in addition to the
    features that are exposed in the object-centric state."""

    @property
    def joint_state(self) -> Sequence[float]:
        """Expose the current joint state in the simulator_state."""
        return cast(Sequence[float], self.simulator_state)

    def allclose(self, other: State) -> bool:
        # Ignores the simulator state.
        return State(self.data).allclose(State(other.data))

    def copy(self) -> State:
        state_dict_copy = super().copy().data
        simulator_state_copy = list(self.joint_state)
        return _PyBulletState(state_dict_copy, simulator_state_copy)


def create_pybullet_block(color: Tuple[float, float, float,
                                       float], size: float, mass: float,
                          friction: float, orientation: Sequence[float],
                          physics_client_id: int) -> int:
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    pose = (0, 0, 0)
    half_extents = [size / 2.] * 3

    # Create the collision shape.
    collision_id = p.createCollisionShape(p.GEOM_BOX,
                                          halfExtents=half_extents,
                                          physicsClientId=physics_client_id)

    # Create the visual_shape.
    visual_id = p.createVisualShape(p.GEOM_BOX,
                                    halfExtents=half_extents,
                                    rgbaColor=color,
                                    physicsClientId=physics_client_id)

    # Create the body.
    block_id = p.createMultiBody(baseMass=mass,
                                 baseCollisionShapeIndex=collision_id,
                                 baseVisualShapeIndex=visual_id,
                                 basePosition=pose,
                                 baseOrientation=orientation,
                                 physicsClientId=physics_client_id)
    p.changeDynamics(
        block_id,
        linkIndex=-1,  # -1 for the base
        lateralFriction=friction,
        physicsClientId=physics_client_id)

    return block_id
