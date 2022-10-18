"""Base class for a PyBullet environment.

Contains useful common code.
"""

import abc
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, cast

import matplotlib
import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, State, Task, Video


class PyBulletEnv(BaseEnv):
    """Base class for a PyBullet environment."""
    # Parameters that aren't important enough to need to clog up settings.py

    # General robot parameters.
    _max_vel_norm: ClassVar[float] = 0.05
    _grasp_tol: ClassVar[float] = 0.05
    _finger_action_tol: ClassVar[float] = 1e-4
    _finger_action_nudge_magnitude: ClassVar[float] = 1e-3

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
    _default_orn: ClassVar[Sequence[float]] = [0.0, 0.0, 0.0, 1.0]

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
        self._held_obj_to_base_link: Optional[Any] = None
        self._held_obj_id: Optional[int] = None

        # Set up all the static PyBullet content.
        self._initialize_pybullet()

    def _initialize_pybullet(self) -> None:
        """One-time initialization of PyBullet assets."""
        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_use_gui:  # pragma: no cover
            self._physics_client_id = p.connect(p.GUI)
            # Disable the preview windows for faster rendering.
            p.configureDebugVisualizer(p.COV_ENABLE_GUI,
                                       False,
                                       physicsClientId=self._physics_client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,
                                       False,
                                       physicsClientId=self._physics_client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,
                                       False,
                                       physicsClientId=self._physics_client_id)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,
                                       False,
                                       physicsClientId=self._physics_client_id)
            p.resetDebugVisualizerCamera(
                self._camera_distance,
                self._camera_yaw,
                self._camera_pitch,
                self._camera_target,
                physicsClientId=self._physics_client_id)
        else:
            self._physics_client_id = p.connect(p.DIRECT)
        # This second connection can be useful for stateless operations.
        self._physics_client_id2 = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.resetSimulation(physicsClientId=self._physics_client_id2)

        # Load plane.
        p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"), [0, 0, -1],
                   useFixedBase=True,
                   physicsClientId=self._physics_client_id)
        p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"), [0, 0, -1],
                   useFixedBase=True,
                   physicsClientId=self._physics_client_id2)

        # Load robot.
        self._pybullet_robot = self._create_pybullet_robot(
            self._physics_client_id)
        self._pybullet_robot2 = self._create_pybullet_robot(
            self._physics_client_id2)

        # Set gravity.
        p.setGravity(0., 0., -10., physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -10., physicsClientId=self._physics_client_id2)

    @abc.abstractmethod
    def _create_pybullet_robot(
            self, physics_client_id: int) -> SingleArmPyBulletRobot:
        """Make and return a PyBullet robot object in the given
        physics_client_id.

        It will be saved as either self._pybullet_robot or
        self._pybullet_robot2.
        """
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

    @abc.abstractmethod
    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        """Get the expected finger normals, used in detect_held_object(), as a
        mapping from finger link index to a unit-length normal vector.

        This is environment-specific because it depends on the end
        effector's orientation when grasping.
        """
        raise NotImplementedError("Override me!")

    @property
    def action_space(self) -> Box:
        return self._pybullet_robot.action_space

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("A PyBullet environment cannot simulate.")

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("A PyBullet environment cannot render "
                                  "arbitrary states.")

    def reset(self, train_or_test: str, task_idx: int) -> State:
        state = super().reset(train_or_test, task_idx)
        self._reset_state(state)
        # Converts the State into a PyBulletState.
        self._current_state = self._get_state()
        return self._current_state.copy()

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

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:  # pragma: no cover
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

        rgb_array = np.array(px).reshape((height, width, 4))
        rgb_array = rgb_array[:, :, :3]
        return [rgb_array]

    def step(self, action: Action) -> State:
        # Send the action to the robot.
        target_joint_positions = action.arr.tolist()
        self._pybullet_robot.set_motors(target_joint_positions)

        # If we are setting the robot joints directly, and if there is a held
        # object, we need to reset the pose of the held object directly. This
        # is because the PyBullet constraints don't seem to play nicely with
        # resetJointState (the robot will sometimes drop the object).
        if CFG.pybullet_control_mode == "reset" and \
            self._held_obj_id is not None:
            world_to_base_link = get_link_state(
                self._pybullet_robot.robot_id,
                self._pybullet_robot.end_effector_id,
                physics_client_id=self._physics_client_id).com_pose
            base_link_to_held_obj = p.invertTransform(
                *self._held_obj_to_base_link)
            world_to_held_obj = p.multiplyTransforms(world_to_base_link[0],
                                                     world_to_base_link[1],
                                                     base_link_to_held_obj[0],
                                                     base_link_to_held_obj[1])
            p.resetBasePositionAndOrientation(
                self._held_obj_id,
                world_to_held_obj[0],
                world_to_held_obj[1],
                physicsClientId=self._physics_client_id)

        # Step the simulation here before adding or removing constraints
        # because detect_held_object() should use the updated state.
        if CFG.pybullet_control_mode != "reset":
            for _ in range(CFG.pybullet_sim_steps_per_action):
                p.stepSimulation(physicsClientId=self._physics_client_id)

        # If not currently holding something, and fingers are closing, check
        # for a new grasp.
        if self._held_constraint_id is None and self._fingers_closing(action):
            # Detect if an object is held. If so, create a grasp constraint.
            self._held_obj_id = self._detect_held_object()
            if self._held_obj_id is not None:
                self._create_grasp_constraint()

        # If placing, remove the grasp constraint.
        if self._held_constraint_id is not None and \
            self._fingers_opening(action):
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
        expected_finger_normals = self._get_expected_finger_normals()
        closest_held_obj = None
        closest_held_obj_dist = float("inf")
        for obj_id in self._get_object_ids_for_held_check():
            for finger_id, expected_normal in expected_finger_normals.items():
                assert abs(np.linalg.norm(expected_normal) - 1.0) < 1e-5
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
                    score = expected_normal.dot(contact_normal)
                    assert -1.0 <= score <= 1.0

                    # Take absolute as object/gripper could be rotated 180
                    # degrees in the given axis.
                    if np.abs(score) < 0.9:
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

    def _create_grasp_constraint(self) -> None:
        assert self._held_obj_id is not None
        base_link_to_world = np.r_[p.invertTransform(
            *p.getLinkState(self._pybullet_robot.robot_id,
                            self._pybullet_robot.end_effector_id,
                            physicsClientId=self._physics_client_id)[:2])]
        world_to_obj = np.r_[p.getBasePositionAndOrientation(
            self._held_obj_id, physicsClientId=self._physics_client_id)]
        self._held_obj_to_base_link = p.invertTransform(*p.multiplyTransforms(
            base_link_to_world[:3], base_link_to_world[3:], world_to_obj[:3],
            world_to_obj[3:]))
        self._held_constraint_id = p.createConstraint(
            parentBodyUniqueId=self._pybullet_robot.robot_id,
            parentLinkIndex=self._pybullet_robot.end_effector_id,
            childBodyUniqueId=self._held_obj_id,
            childLinkIndex=-1,  # -1 for the base
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=self._held_obj_to_base_link[0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=self._held_obj_to_base_link[1],
            physicsClientId=self._physics_client_id)

    def _fingers_closing(self, action: Action) -> bool:
        """Check whether this action is working toward closing the fingers."""
        f_delta = self._action_to_finger_delta(action)
        return f_delta < -self._finger_action_tol

    def _fingers_opening(self, action: Action) -> bool:
        """Check whether this action is working toward opening the fingers."""
        f_delta = self._action_to_finger_delta(action)
        return f_delta > self._finger_action_tol

    def _get_finger_position(self, state: State) -> float:
        # Arbitrarily use the left finger as reference.
        state = cast(utils.PyBulletState, state)
        finger_joint_idx = self._pybullet_robot.left_finger_joint_idx
        return state.joint_positions[finger_joint_idx]

    def _action_to_finger_delta(self, action: Action) -> float:
        finger_position = self._get_finger_position(self._current_state)
        target = action.arr[-1]
        return target - finger_position

    def _add_pybullet_state_to_tasks(self, tasks: List[Task]) -> List[Task]:
        """Converts the task initial states into PyBulletStates."""
        pybullet_tasks = []
        for task in tasks:
            # Reset the robot.
            init = task.init
            self._pybullet_robot.reset_state(self._extract_robot_state(init))
            # Extract the joints.
            joint_positions = self._pybullet_robot.get_joints()
            pybullet_init = utils.PyBulletState(
                init.data.copy(), simulator_state=joint_positions)
            pybullet_task = Task(pybullet_init, task.goal)
            pybullet_tasks.append(pybullet_task)
        return pybullet_tasks


def create_pybullet_block(color: Tuple[float, float, float, float],
                          half_extents: Tuple[float, float,
                                              float], mass: float,
                          friction: float, orientation: Sequence[float],
                          physics_client_id: int) -> int:
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    pose = (0, 0, 0)

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
