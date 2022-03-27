"""A PyBullet version of Blocks."""

import abc
import logging
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.envs.pybullet_utils import get_kinematic_chain, \
    inverse_kinematics
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, Image, Object, \
    ParameterizedOption, Pose3D, State, Task, Type

# TODO: probably move the robots into a separate file


class _SingleArmPyBulletRobot:
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
        # The value at which the finger joints should be open.
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
    def step(self, ee_delta: Pose3D, f_delta: float) -> None:
        """Execute the given action in PyBullet given a delta on the end
        effector and finger joint(s)."""
        raise NotImplementedError("Override me!")


class _FetchPyBulletRobot(_SingleArmPyBulletRobot):
    """A Fetch robot with a fixed base and only one arm in use."""

    # Parameters that aren't important enough to need to clog up settings.py
    _base_pose: Pose3D = (0.75, 0.7441, 0.0)
    _base_orientation: Sequence[float] = [0., 0., 0., 1.]
    _ee_orientation: Sequence[float] = [1., 0., -1., 0.]
    _finger_action_nudge_magnitude: float = 0.001

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

        self._initial_joint_values = inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            self._ee_home_pose,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id)

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

    def reset_state(self, robot_state: Array) -> None:
        rx, ry, rz, rf = robot_state
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id)
        assert np.allclose((rx, ry, rz), self._ee_home_pose)
        joint_values = self._initial_joint_values
        for joint_id, joint_val in zip(self._arm_joints, joint_values):
            p.resetJointState(self._fetch_id,
                              joint_id,
                              joint_val,
                              physicsClientId=self._physics_client_id)
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

    def step(self, ee_delta: Pose3D, f_delta: float) -> None:
        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       physicsClientId=self._physics_client_id)
        current_position = ee_link_state[4]
        target_position = np.add(current_position, ee_delta).tolist()

        # We assume that the robot is already close enough to the target
        # position that IK will succeed with one call, so validate is False.
        # Furthermore, updating the state of the robot during simulation, which
        # validate=True would do, is risky and discouraged by PyBullet.
        joint_values = inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            target_position,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id,
            validate=False)

        # Set arm joint motors.
        for joint_idx, joint_val in zip(self._arm_joints, joint_values):
            p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                    jointIndex=joint_idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_val,
                                    physicsClientId=self._physics_client_id)

        # Set finger joint motors.
        for finger_id in [self._left_finger_id, self._right_finger_id]:
            current_val = p.getJointState(
                self._fetch_id,
                finger_id,
                physicsClientId=self._physics_client_id)[0]
            # Fingers drift if left alone. If the finger action is near zero,
            # nudge the fingers toward being open or closed, based on which end
            # of the spectrum they are currently closer to.
            if abs(f_delta) < self._finger_action_tol:
                assert self._open_fingers > self._closed_fingers
                if abs(current_val -
                       self._open_fingers) < abs(current_val -
                                                 self._closed_fingers):
                    nudge = self._finger_action_nudge_magnitude
                else:
                    nudge = -self._finger_action_nudge_magnitude
                target_val = current_val + nudge
            else:
                target_val = current_val + f_delta
            p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                    jointIndex=finger_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_val,
                                    physicsClientId=self._physics_client_id)


class PyBulletBlocksEnv(BlocksEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # General robot parameters.
    _move_gain: float = 1.0
    _max_vel_norm: float = 0.05
    _grasp_offset_z: float = 0.01
    _place_offset_z: float = 0.01
    _grasp_tol: float = 0.05
    _move_to_pose_tol: float = 0.0001
    _finger_action_tol = 0.0001

    # Table parameters.
    _table_pose: Pose3D = (1.35, 0.75, 0.0)
    _table_orientation: Sequence[float] = [0., 0., 0., 1.]

    # Block parameters.
    _block_orientation: Sequence[float] = [0., 0., 0., 1.]
    _block_mass = 0.5
    _block_friction = 1.2
    _block_colors: Sequence[Tuple[float, float, float, float]] = [
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
    _out_of_view_xy: Sequence[float] = [10.0, 10.0]

    # Camera parameters.
    _camera_distance: float = 0.8
    _camera_yaw: float = 90.0
    _camera_pitch: float = -24
    _camera_target: Pose3D = (1.65, 0.75, 0.42)
    _debug_text_position: Pose3D = (1.65, 0.25, 0.75)

    def __init__(self) -> None:
        super().__init__()

        # Override options, keeping the types and parameter spaces the same.
        types = self._Pick.types
        params_space = self._Pick.params_space
        self._Pick: ParameterizedOption = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Creates a ParameterizedOption for moving to the position that
                # has z equal to self.pick_z, and x and y equal to that of the
                # block object parameter. In other words, move the end effector
                # to high above the block in preparation for picking.
                self._create_move_end_effector_relative_to_block_option(
                    "MoveEndEffectorToAboveBlock", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
                # Open fingers.
                self._create_change_fingers_option(
                    "OpenFingers", self.open_fingers, types, params_space),
                # Move down to grasp.
                self._create_move_end_effector_relative_to_block_option(
                    "MoveEndEffectorToGrasp", (0., 0., self._grasp_offset_z),
                    ("rel", "rel", "rel"), types, params_space),
                # Grasp.
                self._create_change_fingers_option(
                    "Grasp", self.closed_fingers, types, params_space),
                # Move up.
                self._create_move_end_effector_relative_to_block_option(
                    "MoveEndEffectorToAboveBlock", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
            ])

        types = self._Stack.types
        params_space = self._Stack.params_space
        self._Stack: ParameterizedOption = \
            utils.LinearChainParameterizedOption("Stack",
            [
                # Move to above the block on which we will stack.
                self._create_move_end_effector_relative_to_block_option(
                    "MoveEndEffectorToAboveBlock", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
                # Move down to place.
                self._create_move_end_effector_relative_to_block_option(
                    "MoveEndEffectorToPlace",
                    (0., 0., self.block_size + self._place_offset_z),
                    ("rel", "rel", "rel"), types, params_space),
                # Open fingers.
                self._create_change_fingers_option("OpenFingers",
                    self.open_fingers, types, params_space),
                # Move up.
                self._create_move_end_effector_relative_to_block_option(
                    "MoveEndEffectorToAboveBlock", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
            ])

        types = self._PutOnTable.types
        params_space = self._PutOnTable.params_space
        self._PutOnTable: ParameterizedOption = \
            utils.LinearChainParameterizedOption("PutOnTable",
            [
                # Move to above the table at the (x, y) where we will place.
                self._create_move_end_effector_relative_to_table_option(
                    "MoveEndEffectorToAboveTable", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
                # Move down to place.
                self._create_move_end_effector_relative_to_table_option(
                    "MoveEndEffectorToPlaceOnTable",
                    (0., 0., self.block_size + self._place_offset_z),
                    ("rel", "rel", "rel"), types, params_space),
                # Open fingers.
                self._create_change_fingers_option("OpenFingers",
                    self.open_fingers, types, params_space),
                # Move up.
                self._create_move_end_effector_relative_to_table_option(
                    "MoveEndEffectorToAboveTable", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
            ])

        # We track the correspondence between PyBullet object IDs and Object
        # instances for blocks. This correspondence changes with the task.
        self._block_id_to_block: Dict[int, Object] = {}

        # When a block is held, a constraint is created to prevent slippage.
        self._held_constraint_id: Optional[int] = None
        self._held_block_id: Optional[int] = None

        # Set up all the static PyBullet content.
        self._initialize_pybullet()

    def _initialize_pybullet(self) -> None:
        """One-time initialization of pybullet assets."""
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

        # Load and reset robot.
        self._pybullet_robot = self._create_robot()

        # Load table.
        self._table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id)

        # Skip test coverage because GUI is too expensive to use in unit tests
        # and cannot be used in headless mode.
        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert CFG.pybullet_use_gui, \
                "pybullet_use_gui must be True to use pybullet_draw_debug."
            # Draw the workspace on the table for clarity.
            p.addUserDebugLine([self.x_lb, self.y_lb, self.table_height],
                               [self.x_ub, self.y_lb, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0)
            p.addUserDebugLine([self.x_lb, self.y_ub, self.table_height],
                               [self.x_ub, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0)
            p.addUserDebugLine([self.x_lb, self.y_lb, self.table_height],
                               [self.x_lb, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0)
            p.addUserDebugLine([self.x_ub, self.y_lb, self.table_height],
                               [self.x_ub, self.y_ub, self.table_height],
                               [1.0, 0.0, 0.0],
                               lineWidth=5.0)
            # Draw coordinate frame labels for reference.
            p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0])
            p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0])
            p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0])
            # Draw the pick z location at the x/y midpoint.
            mid_x = (self.x_ub + self.x_lb) / 2
            mid_y = (self.y_ub + self.y_lb) / 2
            p.addUserDebugText("*", [mid_x, mid_y, self.pick_z],
                               [1.0, 0.0, 0.0])

        # Set gravity.
        p.setGravity(0., 0., -10., physicsClientId=self._physics_client_id)

        # Create blocks. Note that we create the maximum number once, and then
        # remove blocks from the workspace (teleporting them far away) based on
        # the number involved in the state.
        num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
        self._block_ids = [self._create_block(i) for i in range(num_blocks)]

    @property
    def action_space(self) -> Box:
        # dimensions: [dx, dy, dz, dfingers]
        return Box(low=-self._max_vel_norm,
                   high=self._max_vel_norm,
                   shape=(4, ),
                   dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_blocks"

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("PyBulletBlocksEnv cannot simulate.")

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        raise NotImplementedError("PyBulletBlocksEnv cannot render arbitrary "
                                  "states.")

    def reset(self, train_or_test: str, task_idx: int) -> State:
        state = super().reset(train_or_test, task_idx)
        self._reset_state(state)
        return state

    def _reset_state(self, state: State) -> None:
        """Helper for reset and testing."""
        # Tear down the old PyBullet scene.
        if self._held_constraint_id is not None:
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None
        self._held_block_id = None

        # Reset robot.
        self._pybullet_robot.reset_state(state[self._robot])

        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block = {}
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            self._block_id_to_block[block_id] = block_obj
            bx, by, bz, _ = state[block_obj]
            # Assume not holding in the initial state
            assert self._get_held_block(state) is None
            p.resetBasePositionAndOrientation(
                block_id, [bx, by, bz],
                self._block_orientation,
                physicsClientId=self._physics_client_id)

        # For any blocks not involved, put them out of view.
        h = self.block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            assert block_id not in self._block_id_to_block
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * h],
                self._block_orientation,
                physicsClientId=self._physics_client_id)

        # Assert that the state was properly reconstructed.
        reconstructed_state = self._get_state()
        if not reconstructed_state.allclose(state):
            logging.debug("Desired state:")
            logging.debug(state.pretty_str())
            logging.debug("Reconstructed state:")
            logging.debug(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _create_robot(self) -> _SingleArmPyBulletRobot:
        ee_home = (self.robot_init_x, self.robot_init_y, self.robot_init_z)
        if CFG.pybullet_robot == "fetch":
            return _FetchPyBulletRobot(ee_home, self.open_fingers,
                                       self.closed_fingers,
                                       self._finger_action_tol,
                                       self._physics_client_id)
        raise NotImplementedError("Unrecognized pybullet_robot: "
                                  f"{CFG.pybullet_robot}")

    def _create_block(self, block_num: int) -> int:
        """Returns the body ID."""
        color = self._block_colors[block_num % len(self._block_colors)]

        # The poses here are not important because they are overwritten by
        # the state values when a task is reset. By default, we just stack all
        # the blocks into one pile at the center of the table.
        h = self.block_size
        x = (self.x_lb + self.x_ub) / 2
        y = (self.y_lb + self.y_ub) / 2
        z = self.table_height + (0.5 * h) + (h * block_num)
        pose = (x, y, z)
        half_extents = [self.block_size / 2.] * 3

        # Create the collision shape.
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._physics_client_id)

        # Create the visual_shape.
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._physics_client_id)

        # Create the body.
        block_id = p.createMultiBody(baseMass=self._block_mass,
                                     baseCollisionShapeIndex=collision_id,
                                     baseVisualShapeIndex=visual_id,
                                     basePosition=pose,
                                     baseOrientation=self._block_orientation,
                                     physicsClientId=self._physics_client_id)
        p.changeDynamics(
            block_id,
            linkIndex=-1,  # -1 for the base
            lateralFriction=self._block_friction,
            physicsClientId=self._physics_client_id)

        return block_id

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
        ee_delta, f_delta = action.arr[:3], action.arr[3]
        self._pybullet_robot.step(ee_delta, f_delta)

        # Step the simulation here before adding or removing constraints
        # because detect_held_block() should use the updated state.
        for _ in range(CFG.pybullet_sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # If not currently holding something, and fingers are closing, check
        # for a new grasp.
        if self._held_constraint_id is None and \
            f_delta < -self._finger_action_tol:
            # Detect whether an object is held.
            self._held_block_id = self._detect_held_block()
            if self._held_block_id is not None:
                # Create a grasp constraint.
                base_link_to_world = np.r_[p.invertTransform(*p.getLinkState(
                    self._pybullet_robot.robot_id,
                    self._pybullet_robot.end_effector_id,
                    physicsClientId=self._physics_client_id)[:2])]
                world_to_obj = np.r_[p.getBasePositionAndOrientation(
                    self._held_block_id,
                    physicsClientId=self._physics_client_id)]
                base_link_to_obj = p.invertTransform(*p.multiplyTransforms(
                    base_link_to_world[:3], base_link_to_world[3:],
                    world_to_obj[:3], world_to_obj[3:]))
                self._held_constraint_id = p.createConstraint(
                    parentBodyUniqueId=self._pybullet_robot.robot_id,
                    parentLinkIndex=self._pybullet_robot.end_effector_id,
                    childBodyUniqueId=self._held_block_id,
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
            self._held_block_id = None

        self._current_state = self._get_state()
        return self._current_state.copy()

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_id_to_block and self._held_block_id. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        state_dict = {}

        # Get robot state.
        state_dict[self._robot] = self._pybullet_robot.get_state()

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_block_id)
            # pose_x, pose_y, pose_z, held
            state_dict[block] = np.array([bx, by, bz, held], dtype=np.float32)

        state = State(state_dict)
        assert set(state) == set(self._current_state), \
            (f"Reconstructed state has objects {set(state)}, but "
             f"self._current_state has objects {set(self._current_state)}.")

        return state

    def _detect_held_block(self) -> Optional[int]:
        """Return the PyBullet object ID of the held object if one exists.

        If multiple blocks are within the grasp tolerance, return the
        one that is closest.
        """
        expected_finger_normals = {
            self._pybullet_robot.left_finger_id: np.array([0., 1., 0.]),
            self._pybullet_robot.right_finger_id: np.array([0., -1., 0.]),
        }
        closest_held_block = None
        closest_held_block_dist = float("inf")
        for block_id in self._block_id_to_block:
            for finger_id, expected_normal in expected_finger_normals.items():
                # Find points on the block that are within grasp_tol distance
                # of the finger. Note that we use getClosestPoints instead of
                # getContactPoints because we still want to consider the block
                # held even if there is a tiny distance between the fingers and
                # the block.
                closest_points = p.getClosestPoints(
                    bodyA=self._pybullet_robot.robot_id,
                    bodyB=block_id,
                    distance=self._grasp_tol,
                    linkIndexA=finger_id,
                    physicsClientId=self._physics_client_id)
                for point in closest_points:
                    # If the contact normal is substantially different from
                    # the expected contact normal, this is probably a block
                    # on the outside of the fingers, rather than the inside.
                    # A perfect score here is 1.0 (normals are unit vectors).
                    contact_normal = point[7]
                    if expected_normal.dot(contact_normal) < 0.5:
                        continue
                    # Handle the case where multiple blocks pass this check
                    # by taking the closest one. This should be rare, but it
                    # can happen when two blocks are stacked and the robot is
                    # unstacking the top one.
                    contact_distance = point[8]
                    if contact_distance < closest_held_block_dist:
                        closest_held_block = block_id
                        closest_held_block_dist = contact_distance
        return closest_held_block

    ########################## Parameterized Options ##########################

    def _create_move_end_effector_to_pose_option(
        self, name: str, types: Sequence[Type], params_space: Box,
        get_current_and_target_pose: Callable[[State, Sequence[Object], Array],
                                              Tuple[Pose3D, Pose3D]]
    ) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving the end effector to a
        target pose, given a function that takes in the current state and
        objects and returns the current pose and target pose of the robot."""

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory  # unused
            current, target = get_current_and_target_pose(
                state, objects, params)
            return self._get_end_effector_move_toward_waypoint_action(
                current, target)

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

    def _create_move_end_effector_relative_to_block_option(
            self, name: str, rel_or_abs_target_pose: Pose3D,
            rel_or_abs: Tuple[str, str, str], types: Sequence[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving the end effector to a
        target pose relative to a block.

        Each of the three dimensions of the target pose can be specified
        relatively or absolutely. The rel_or_abs argument indicates
        whether each dimension is relative ("rel") or absolute ("abs").
        """
        assert types == [self._robot_type, self._block_type]

        def _get_current_and_target_pose(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D]:
            del params  # not used
            robot, block = objects
            current_pose = (state[robot][0], state[robot][1], state[robot][2])
            block_pose = (state[block][0], state[block][1], state[block][2])
            target_pose = self._convert_rel_abs_to_abs(block_pose,
                                                       rel_or_abs_target_pose,
                                                       rel_or_abs)
            return current_pose, target_pose

        return self._create_move_end_effector_to_pose_option(
            name, types, params_space, _get_current_and_target_pose)

    def _create_move_end_effector_relative_to_table_option(
            self, name: str, rel_or_abs_target_pose: Pose3D,
            rel_or_abs: Tuple[str, str, str], types: Sequence[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving the end effector to a
        target pose relative to the table.

        Each of the three dimensions of the target pose can be specified
        relatively or absolutely. The rel_or_abs argument indicates
        whether each dimension is relative ("rel") or absolute ("abs").
        """

        def _get_current_and_target_pose(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D]:
            robot, = objects
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            x = self.x_lb + (self.x_ub - self.x_lb) * x_norm
            y = self.y_lb + (self.y_ub - self.y_lb) * y_norm
            current_pose = (state[robot][0], state[robot][1], state[robot][2])
            table_pose = (x, y, self.table_height)
            target_pose = self._convert_rel_abs_to_abs(table_pose,
                                                       rel_or_abs_target_pose,
                                                       rel_or_abs)
            return current_pose, target_pose

        return self._create_move_end_effector_to_pose_option(
            name, types, params_space, _get_current_and_target_pose)

    def _create_change_fingers_option(
            self, name: str, target_val: float, types: Sequence[Type],
            params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for changing the robot fingers."""

        assert types[0] == self._robot_type

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory, objects, params  # unused
            current_val = state.get(self._robot, "fingers")
            f_delta = target_val - current_val
            f_delta = np.clip(f_delta, self.action_space.low[3],
                              self.action_space.high[3])
            return Action(np.array([0., 0., 0., f_delta], dtype=np.float32))

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, objects, params  # unused
            current_val = state.get(self._robot, "fingers")
            squared_dist = (target_val - current_val)**2
            return squared_dist < self._grasp_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=lambda _1, _2, _3, _4: True,
                                   terminal=_terminal)

    def _get_end_effector_move_toward_waypoint_action(
            self, ee_pose: Pose3D, target_pose: Pose3D) -> Action:
        action = self._move_gain * np.subtract(target_pose, ee_pose)
        action_norm = np.linalg.norm(action)  # type: ignore
        if action_norm > self._max_vel_norm:
            action = action * self._max_vel_norm / action_norm
        action = np.r_[action, 0.0].astype(np.float32)
        assert self.action_space.contains(action)
        return Action(action)

    @staticmethod
    def _convert_rel_abs_to_abs(current_pose: Pose3D, rel_or_abs_pose: Pose3D,
                                rel_or_abs: Tuple[str, str, str]) -> Pose3D:
        abs_pos = []
        assert len(current_pose) == len(rel_or_abs_pose) == len(rel_or_abs)
        for i in range(len(current_pose)):
            if rel_or_abs[i] == "rel":
                pose_i = current_pose[i] + rel_or_abs_pose[i]
            else:
                assert rel_or_abs[i] == "abs"
                pose_i = rel_or_abs_pose[i]
            abs_pos.append(pose_i)
        return (abs_pos[0], abs_pos[1], abs_pos[2])
