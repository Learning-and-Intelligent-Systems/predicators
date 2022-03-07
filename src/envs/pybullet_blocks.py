"""A PyBullet version of Blocks."""

from typing import Sequence, Tuple, Dict, Optional, Callable, List
from gym.spaces import Box
import numpy as np
import pybullet as p
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.structs import State, Action, Object, Array, \
    ParameterizedOption, Type, Image
from predicators.src import utils
from predicators.src.pybullet_utils import get_kinematic_chain, \
    inverse_kinematics, Pose3D
from predicators.src.settings import CFG


class PyBulletBlocksEnv(BlocksEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Fetch robot parameters.
    _base_pose: Pose3D = (0.75, 0.7441, 0.0)
    _base_orientation: Sequence[float] = [0., 0., 0., 1.]
    _ee_orientation: Sequence[float] = [1., 0., -1., 0.]
    _move_gain: float = 1.0
    _max_vel_norm: float = 0.5
    _grasp_tol: float = 0.05
    _grasp_offset_z: float = 0.01
    _place_offset_z: float = 0.005

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
    _camera_width: int = 1674
    _camera_height: int = 900
    _camera_target: Pose3D = (1.65, 0.75, 0.42)
    _debug_text_position: Pose3D = (1.65, 0.25, 0.75)

    def __init__(self) -> None:
        super().__init__()

        # Override options.
        types = [self._robot_type, self._block_type]
        params_space = Box(0, 1, (0, ))
        self._Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Creates a ParameterizedOption for moving to the position that
                # has z equal to self.pick_z, and x and y equal to that of the
                # block object paramereter. In other words, move the end
                # effector to high above the block in preparation for picking.
                self._move_relative_to_block(
                    "MoveToAboveBlock", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
                # Open grippers.
                self._change_grippers("OpenGrippers", self.open_fingers, types,
                                      params_space),
                # Move down to grasp.
                self._move_relative_to_block(
                    "MoveToGrasp", (0., 0., self._grasp_offset_z),
                    ("rel", "rel", "rel"), types, params_space),
                # Grasp.
                self._change_grippers("Grasp", self.closed_fingers, types,
                                      params_space),
                # Move up.
                self._move_relative_to_block(
                    "MoveToAboveBlock", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
            ])

        types = [self._robot_type, self._block_type]
        params_space = Box(0, 1, (0, ))
        self._Stack = utils.LinearChainParameterizedOption(
            "Stack",
            [
                # Move to above the block on which we will stack.
                self._move_relative_to_block(
                    "MoveToAboveBlock", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
                # Move down to place.
                self._move_relative_to_block(
                    "MoveToPlace",
                    (0., 0., self.block_size + self._place_offset_z),
                    ("rel", "rel", "rel"), types, params_space),
                # Open grippers.
                self._change_grippers("OpenGrippers", self.open_fingers, types,
                                      params_space),
                # Move up.
                self._move_relative_to_block(
                    "MoveToAboveBlock", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
            ])

        types = [self._robot_type]
        params_space = Box(0, 1, (2, ))  # normalized w.r.t. workspace
        self._PutOnTable = utils.LinearChainParameterizedOption(
            "PutOnTable",
            [
                # Move to above the block on which we will stack.
                self._move_relative_to_table(
                    "MoveToAboveTable", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
                # Move down to place.
                self._move_relative_to_table(
                    "MoveToPlaceOnTable",
                    (0., 0., self.block_size + self._place_offset_z),
                    ("rel", "rel", "rel"), types, params_space),
                # Open grippers.
                self._change_grippers("OpenGrippers", self.open_fingers, types,
                                      params_space),
                # Move up.
                self._move_relative_to_table(
                    "MoveToAboveTable", (0., 0., self.pick_z),
                    ("rel", "rel", "abs"), types, params_space),
            ])

        # One-time initialization of pybullet assets. Note that this happens
        # in __init__ because many class attributes are created.
        if CFG.pybullet_use_gui:
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

        # Load Fetch robot.
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

        if CFG.pybullet_draw_debug:
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
                               [1.0, 0.0, 0.0],
                               lifeTime=CFG.pybullet_draw_debug_lifetime)

        # Set gravity.
        p.setGravity(0., 0., -10., physicsClientId=self._physics_client_id)

        # Determine the initial joint values.
        self._robot_home_pose = (self.robot_init_x, self.robot_init_y,
                                 self.robot_init_z)
        self._initial_joint_values = inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            self._robot_home_pose,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id)

        # Create blocks. Note that we create the maximum number once, and then
        # remove blocks from view based on the number involved in the state.
        num_blocks = max(max(CFG.blocks_num_blocks_train),
                         max(CFG.blocks_num_blocks_test))
        self._block_ids = [self._create_block(i) for i in range(num_blocks)]
        self._block_id_to_block: Dict[int, Object] = {}

        # When a block is held, a constraint is created to prevent slippage.
        self._held_constraint_id: Optional[int] = None
        self._held_block_id: Optional[int] = None

    @property
    def action_space(self) -> Box:
        # dimensions: [dx, dy, dz, dfingers]
        return Box(low=-0.05, high=0.05, shape=(4, ), dtype=np.float32)

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
        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_pose,
            self._base_orientation,
            physicsClientId=self._physics_client_id)
        rx, ry, rz, rf = state[self._robot]
        # Avoid running IK in the common case where the robot is at home.
        if np.allclose((rx, ry, rz), self._robot_home_pose):
            joint_values = self._initial_joint_values
        else:
            joint_values = inverse_kinematics(
                self._fetch_id,
                self._ee_id, (rx, ry, rz),
                self._ee_orientation,
                self._arm_joints,
                physics_client_id=self._physics_client_id)
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

        # Reset blocks based on the state.
        block_objs = state.get_objects(self._block_type)
        self._block_id_to_block = {}
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            self._block_id_to_block[block_id] = block_obj
            bx, by, bz, held = state[block_obj]
            assert held < 1.0  # not holding in the initial state
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
            print("Desired state:")
            print(state.pretty_str())
            print("Reconstructed state:")
            print(reconstructed_state.pretty_str())
            raise ValueError("Could not reconstruct state.")

    def _create_block(self, block_num: int) -> int:
        """Returns the body ID."""
        color = self._block_colors[block_num % len(self._block_colors)]

        # The poses here are not important because they are overwritten by
        # the state values when a task is reset. By default, we just stack all
        # the blocks into one pile at the center of the table so we can see.
        h = self.block_size
        x = (self.x_lb + self.x_ub) / 2
        y = (self.y_lb + self.y_ub) / 2
        z = self.table_height + (0.5 * h) + (h * block_num)
        pose = (x, y, z)

        # Create the collision shape.
        half_extents = [self.block_size / 2.] * 3
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
        p.changeDynamics(block_id,
                         -1,
                         lateralFriction=self._block_friction,
                         physicsClientId=self._physics_client_id)

        return block_id

    def render(self, action: Optional[Action] = None) -> List[Image]:
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

        width, height = self._camera_width, self._camera_height

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
        ee_delta, finger_action = action.arr[:3], action.arr[3]

        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       physicsClientId=self._physics_client_id)
        current_position = ee_link_state[4]
        target_position = np.add(current_position, ee_delta)

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
            target_val = current_val + finger_action
            p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                    jointIndex=finger_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_val,
                                    physicsClientId=self._physics_client_id)

        # Step the simulation here before adding or removing constraints
        # because detect_held_block() should use the updated state.
        for _ in range(CFG.pybullet_sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        # If not currently holding something, and gripper is closing,
        # check for a new grasp.
        if self._held_constraint_id is None and finger_action < 0:
            # Detect whether an object is held.
            self._held_block_id = self._detect_held_block()
            if self._held_block_id is not None:
                # Create a grasp constraint.
                base_link_to_world = np.r_[p.invertTransform(*p.getLinkState(
                    self._fetch_id,
                    self._ee_id,
                    physicsClientId=self._physics_client_id)[:2])]
                world_to_obj = np.r_[p.getBasePositionAndOrientation(
                    self._held_block_id,
                    physicsClientId=self._physics_client_id)]
                base_link_to_obj = p.invertTransform(*p.multiplyTransforms(
                    base_link_to_world[:3], base_link_to_world[3:],
                    world_to_obj[:3], world_to_obj[3:]))
                self._held_constraint_id = p.createConstraint(
                    parentBodyUniqueId=self._fetch_id,
                    parentLinkIndex=self._ee_id,
                    childBodyUniqueId=self._held_block_id,
                    childLinkIndex=-1,
                    jointType=p.JOINT_FIXED,
                    jointAxis=[0, 0, 0],
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=base_link_to_obj[0],
                    parentFrameOrientation=[0, 0, 0, 1],
                    childFrameOrientation=base_link_to_obj[1],
                    physicsClientId=self._physics_client_id)

        # If placing, remove the grasp constraint.
        if self._held_constraint_id is not None and finger_action > 0:
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None
            self._held_block_id = None

        self._current_state = self._get_state()
        return self._current_state.copy()

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state."""
        state = self._current_state.copy()

        # Get robot state.
        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       physicsClientId=self._physics_client_id)
        rx, ry, rz = ee_link_state[4]
        rf = p.getJointState(self._fetch_id,
                             self._left_finger_id,
                             physicsClientId=self._physics_client_id)[0]
        state.set(self._robot, "pose_x", rx)
        state.set(self._robot, "pose_y", ry)
        state.set(self._robot, "pose_z", rz)
        state.set(self._robot, "fingers", rf)

        # Get block states.
        for block_id, block in self._block_id_to_block.items():
            (bx, by, bz), _ = p.getBasePositionAndOrientation(
                block_id, physicsClientId=self._physics_client_id)
            held = (block_id == self._held_block_id)
            state.set(block, "pose_x", bx)
            state.set(block, "pose_y", by)
            state.set(block, "pose_z", bz)
            state.set(block, "held", held)

        return state

    def _detect_held_block(self) -> Optional[int]:
        """Return the PyBullet object ID of the held object if one exists.

        If multiple blocks are within the grasp tolerance, return the
        one that is closest.
        """
        expected_finger_normals = {
            self._left_finger_id: np.array([0., 1., 0.]),
            self._right_finger_id: np.array([0., -1., 0.]),
        }
        closest_held_block = None
        closest_held_block_dist = float("inf")
        for block_id in self._block_id_to_block:
            for finger_id, expected_normal in expected_finger_normals.items():
                # Find points on the block that are within grasp_tol distance
                # of the finger.
                closest_points = p.getClosestPoints(
                    bodyA=self._fetch_id,
                    bodyB=block_id,
                    distance=self._grasp_tol,
                    linkIndexA=finger_id,
                    physicsClientId=self._physics_client_id)
                for point in closest_points:
                    # If the contact normal is substantially different from
                    # the expected contact normal, this is probably a block
                    # on the outside of the gripper, rather than the inside.
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

    def _move_to_pose(
        self, name: str, types: Sequence[Type], params_space: Box,
        get_current_and_target_pose: Callable[[State, Sequence[Object], Array],
                                              Tuple[Pose3D, Pose3D]]
    ) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a target pose, given a
        function that takes in the current state and objects and returns the
        current pose and target pose of the robot."""

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory  # unused
            current, target = get_current_and_target_pose(
                state, objects, params)
            return self._get_move_toward_waypoint_action(current, target)

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory  # unused
            current, target = get_current_and_target_pose(
                state, objects, params)
            dist = np.sum(np.subtract(current, target)**2)
            return dist < self.pick_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=lambda _1, _2, _3, _4: True,
                                   terminal=_terminal)

    def _move_relative_to_block(self, name: str,
                                rel_or_abs_target_pose: Pose3D,
                                rel_or_abs: Tuple[str, str,
                                                  str], types: Sequence[Type],
                                params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a target pose relative
        to a block.

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

        return self._move_to_pose(name, types, params_space,
                                  _get_current_and_target_pose)

    def _change_grippers(self, name: str, target_val: float,
                         types: Sequence[Type],
                         params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for changing the robot grippers."""

        assert types[0] == self._robot_type

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory, objects, params  # unused
            current_val = state.get(self._robot, "fingers")
            gripper_action = target_val - current_val
            gripper_action = np.clip(gripper_action, self.action_space.low[3],
                                     self.action_space.high[3])
            return Action(
                np.array([0., 0., 0., gripper_action], dtype=np.float32))

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, objects, params  # unused
            current_val = state.get(self._robot, "fingers")
            dist = (target_val - current_val)**2
            return dist < self._grasp_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=lambda _1, _2, _3, _4: True,
                                   terminal=_terminal)

    def _move_relative_to_table(self, name: str,
                                rel_or_abs_target_pose: Pose3D,
                                rel_or_abs: Tuple[str, str,
                                                  str], types: Sequence[Type],
                                params_space: Box) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a target pose relative
        to the table.

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
            x = self.x_lb + self.block_size / 2 + (self.x_ub - self.x_lb -
                                                   self.block_size) * x_norm
            y = self.y_lb + self.block_size / 2 + (self.y_ub - self.y_lb -
                                                   self.block_size) * y_norm
            current_pose = (state[robot][0], state[robot][1], state[robot][2])
            table_pose = (x, y, self.table_height)
            target_pose = self._convert_rel_abs_to_abs(table_pose,
                                                       rel_or_abs_target_pose,
                                                       rel_or_abs)
            return current_pose, target_pose

        return self._move_to_pose(name, types, params_space,
                                  _get_current_and_target_pose)

    def _get_move_toward_waypoint_action(self, gripper_pose: Pose3D,
                                         target_pose: Pose3D) -> Action:
        action = self._move_gain * np.subtract(target_pose, gripper_pose)
        action_norm = np.linalg.norm(action)  # type: ignore
        if action_norm > self._max_vel_norm:
            action = action * self._max_vel_norm / action_norm
        action = np.r_[action, 0.0]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return Action(action.astype(np.float32))

    @staticmethod
    def _convert_rel_abs_to_abs(current_pose: Pose3D, rel_or_abs_pose: Pose3D,
                                rel_or_abs: Tuple[str, str, str]) -> Pose3D:
        abs_pos = []
        for i in range(3):
            if rel_or_abs[i] == "rel":
                pose_i = current_pose[i] + rel_or_abs_pose[i]
            else:
                assert rel_or_abs[i] == "abs"
                pose_i = rel_or_abs_pose[i]
            abs_pos.append(pose_i)
        return (abs_pos[0], abs_pos[1], abs_pos[2])
