"""A PyBullet version of Blocks."""

from typing import Sequence, Tuple, Dict, Optional
from gym.spaces import Box
import numpy as np
import pybullet as p
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.structs import State, Action, Object, Array, \
    ParameterizedOption
from predicators.src import utils
from predicators.src.pybullet_utils import get_kinematic_chain, \
    inverse_kinematics
from predicators.src.settings import CFG


class PyBulletBlocksEnv(BlocksEnv):
    """PyBullet Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py

    # Fetch robot parameters.
    _base_position: Sequence[float] = [0.75, 0.7441, 0.0]
    _base_orientation: Sequence[float] = [0., 0., 0., 1.]
    _ee_orientation: Sequence[float] = [1., 0., -1., 0.]
    _move_gain: float = 1.0
    _max_vel_norm: float = 0.5
    _grasp_tol: float = 0.0001

    # Table parameters.
    _table_position: Sequence[float] = [1.35, 0.75, 0.0]
    _table_orientation: Sequence[float] = [0., 0., 0., 1.]

    # Block parameters.
    _block_orientation: Sequence[float] = [0., 0., 0., 1.]
    _block_mass = 0.04
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
    _camera_distance: float = 1.5
    _yaw: float = 90.0
    _pitch: float = -24
    _camera_target: Sequence[float] = [1.65, 0.75, 0.42]
    _debug_text_position = [1.65, 0.25, 0.75]

    def __init__(self) -> None:
        super().__init__()

        # Override options.
        self._Pick = utils.LinearChainParameterizedOption(
            "Pick",
            [
                # Creates a ParameterizedOption for moving to the position that
                # has z equal to self.pick_z, and x and y equal to that of the
                # block object paramereter. In other words, move the end
                # effector to high above the block in preparation for picking.
                # Note that the params space here is trivial (size 0) and
                # the types are [robot, block].
                self._move_relative_to_block("MoveToAboveBlock",
                                             (0., 0., self.pick_z),
                                             ("rel", "rel", "abs")),
                # Open grippers.
                self._change_grippers("OpenGrippers", 0.95),
                # Move down to grasp.
                self._move_relative_to_block("MoveToGrasp",
                                             (0., 0., 0.),
                                             ("rel", "rel", "rel")),
                # Grasp.        
                self._change_grippers("Grasp", 0.98),
                # Move up.
                self._move_relative_to_block("MoveToAboveBlock",
                                             (0., 0., self.pick_z),
                                             ("rel", "rel", "abs")),                
                # TODO more.
            ])
        # TODO: override Stack and Place.

        # One-time initialization of pybullet assets. Note that this happens
        # in __init__ because many class attributes are created.
        if CFG.pybullet_use_gui:
            self._physics_client_id = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                self._camera_distance,
                self._yaw,
                self._pitch,
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
            self._base_position,
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
            self._table_position,
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
        self._initial_joint_values = inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            [self.robot_init_x, self.robot_init_y, self.robot_init_z],
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id)

        # Create blocks. Note that we create the maximum number once, and then
        # remove blocks from view based on the number involved in the state.
        num_blocks = max(max(self.num_blocks_train), max(self.num_blocks_test))
        self._block_ids = [self._create_block(i) for i in range(num_blocks)]

        # When a block is held, a constraint is created to prevent slippage.
        self._held_constraint_id: Optional[int] = None

        # while True:
        #     p.stepSimulation(physicsClientId=self._physics_client_id)

    @property
    def action_space(self) -> Box:
        # dimensions: [dx, dy, dz, fingers]
        return Box(low=-0.05, high=0.05, shape=(4, ), dtype=np.float32)

    def reset(self, train_or_test: str, task_idx: int) -> State:
        # Resets current_state and current_task.
        state = super().reset(train_or_test, task_idx)

        # Tear down the old PyBullet scene.
        if self._held_constraint_id is not None:
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None

        p.resetBasePositionAndOrientation(
            self._fetch_id,
            self._base_position,
            self._base_orientation,
            physicsClientId=self._physics_client_id)

        for joint_idx, joint_val in zip(self._arm_joints,
                                        self._initial_joint_values):
            p.resetJointState(self._fetch_id,
                              joint_idx,
                              joint_val,
                              physicsClientId=self._physics_client_id)

        # Prevent collisions between robot and blocks during scene init.
        # up_action = Action(np.array([-0.5, -0.5, 0.5, 0.0], dtype=np.float32))
        # for _ in range(10):
        #     self.step(up_action)

        # Reset block positions based on the state.
        block_objs = list(o for o in state if o.type == self._block_type)
        for i, block_obj in enumerate(block_objs):
            block_id = self._block_ids[i]
            x, y, z, held = state[block_obj]
            assert held < 1.0  # not holding in the initial state
            p.resetBasePositionAndOrientation(
                block_id, [x, y, z],
                self._block_orientation,
                physicsClientId=self._physics_client_id)

        # For any blocks not involved, put them out of view.
        h = self.block_size
        oov_x, oov_y = self._out_of_view_xy
        for i in range(len(block_objs), len(self._block_ids)):
            block_id = self._block_ids[i]
            p.resetBasePositionAndOrientation(
                block_id, [oov_x, oov_y, i * h],
                self._block_orientation,
                physicsClientId=self._physics_client_id)

        # while True:
        #     p.stepSimulation(physicsClientId=self._physics_client_id)

        # TODO: assert that the initial state is properly reconstructed

        return state

    def _create_block(self, block_num: int) -> int:
        """Returns the body ID."""
        color = self._block_colors[block_num % len(self._block_colors)]

        # The positions here are not important because they are overwritten by
        # the state values when a task is reset. By default, we just stack all
        # the blocks into one pile at the center of the table so we can see.
        h = self.block_size
        x = (self.x_lb + self.x_ub) / 2
        y = (self.y_lb + self.y_ub) / 2
        z = self.table_height + (0.5 * h) + (h * block_num)
        position = [x, y, z]

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
                                     basePosition=position,
                                     baseOrientation=self._block_orientation,
                                     physicsClientId=self._physics_client_id)
        p.changeDynamics(block_id,
                         -1,
                         lateralFriction=self._block_friction,
                         physicsClientId=self._physics_client_id)

        return block_id

    def step(self, action: Action) -> State:
        ee_delta, finger_action = action.arr[:3], action.arr[3]

        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       physicsClientId=self._physics_client_id)
        current_position = ee_link_state[4]
        target_position = np.add(current_position, ee_delta)

        joint_values = inverse_kinematics(
            self._fetch_id,
            self._ee_id,
            target_position,
            self._ee_orientation,
            self._arm_joints,
            physics_client_id=self._physics_client_id)

        # Set arm joint motors.
        for joint_idx, joint_val in zip(self._arm_joints, joint_values):
            p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                    jointIndex=joint_idx,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=joint_val,
                                    physicsClientId=self._physics_client_id)

        # Set finger joint motors
        for finger_id in [self._left_finger_id, self._right_finger_id]:
            current_val = p.getJointState(
                self._fetch_id,
                finger_id,
                physicsClientId=self._physics_client_id)[0]
            target_val = current_val - finger_action
            p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                    jointIndex=finger_id,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPosition=target_val,
                                    physicsClientId=self._physics_client_id)

        # If picking, create a grasp constraint.
        held_block_id = self._detect_held_block()
        if self._held_constraint_id is None and held_block_id:
            base_link_to_world = np.r_[p.invertTransform(
                *p.getLinkState(self._fetch_id, self._ee_id,
                                physicsClientId=self._physics_client_id)[:2])]
            world_to_obj = np.r_[p.getBasePositionAndOrientation(
                held_block_id, physicsClientId=self._physics_client_id)]
            base_link_to_obj = p.invertTransform(*p.multiplyTransforms(
                base_link_to_world[:3], base_link_to_world[3:],
                world_to_obj[:3], world_to_obj[3:]))
            self._held_constraint_id = p.createConstraint(
                parentBodyUniqueId=self._fetch_id,
                parentLinkIndex=self._ee_id,
                childBodyUniqueId=held_block_id,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=base_link_to_obj[0],
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=base_link_to_obj[1],
                physicsClientId=self._physics_client_id)

        # TODO other things...

        for _ in range(CFG.pybullet_sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._physics_client_id)

        return self._get_state()

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state."""
        state = self._current_state.copy()

        ee_link_state = p.getLinkState(self._fetch_id,
                                       self._ee_id,
                                       physicsClientId=self._physics_client_id)
        x, y, z = ee_link_state[4]
        fingers = 1.0 - p.getJointState(
            self._fetch_id,
            self._left_finger_id,
            physicsClientId=self._physics_client_id)[0]

        # TODO A BUNCH OF OTHER STUFF
        state.set(self._robot, "pose_x", x)
        state.set(self._robot, "pose_y", y)
        state.set(self._robot, "pose_z", z)
        state.set(self._robot, "fingers", fingers)

        return state

    def _detect_held_block(self) -> Optional[int]:
        """Return the PyBullet object ID of the held object if one exists."""
        for block_id in self._block_ids:
            contact_points = p.getContactPoints(self._fetch_id, block_id, self._left_finger_id)
            if contact_points:
                return block_id
        return None

    ########################## Parameterized Options ##########################

    def _move_relative_to_block(
            self, name: str, rel_or_abs_target_pose: Tuple[float, float,
                                                           float],
            rel_or_abs: Tuple[str, str, str]) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a target pose.

        Each of the three dimensions of the target pose can be specified
        relatively or absolutely. The rel_or_abs argument indicates
        whether each dimension is relative ("rel") or absolute ("abs").
        """

        def _get_current_and_target_pose(
            state: State, objects: Sequence[Object]
        ) -> Tuple[Sequence[float], Sequence[float]]:
            robot, block = objects
            current_pose = state[robot][:3]
            block_pose = state[block][:3]
            target_pose = []
            for i in range(3):
                if rel_or_abs[i] == "rel":
                    pose_i = block_pose[i] + rel_or_abs_target_pose[i]
                else:
                    assert rel_or_abs[i] == "abs"
                    pose_i = rel_or_abs_target_pose[i]
                target_pose.append(pose_i)
            return current_pose, target_pose

        types = [self._robot_type, self._block_type]
        params_space = Box(0, 1, (0, ))

        def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                       params: Array) -> bool:
            del memory, params  # unused
            _, target = _get_current_and_target_pose(state, objects)
            if CFG.pybullet_draw_debug:
                p.addUserDebugText("*",
                                   target, [1.0, 0.0, 0.0],
                                   lifeTime=CFG.pybullet_draw_debug_lifetime)
            return True

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory, params  # unused
            current, target = _get_current_and_target_pose(state, objects)
            return self._get_move_toward_waypoint_action(current, target)

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, params  # unused
            current, target = _get_current_and_target_pose(state, objects)
            dist = np.sum(np.subtract(current, target)**2)
            return dist < self.pick_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=_initiable,
                                   terminal=_terminal)

    def _change_grippers(self, name: str,
                         target_val: float) -> ParameterizedOption:
        """Creates a ParameterizedOption for changing the robot grippers."""
        # TODO probably factor this out.
        types = [self._robot_type, self._block_type]
        params_space = Box(0, 1, (0, ))

        def _policy(state: State, memory: Dict, objects: Sequence[Object],
                    params: Array) -> Action:
            del memory, params  # unused
            current_val = state.get(self._robot, "fingers")
            gripper_action = target_val - current_val
            gripper_action = np.clip(gripper_action, self.action_space.low[3],
                                     self.action_space.high[3])
            return Action(
                np.array([0., 0., 0., gripper_action], dtype=np.float32))

        def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                      params: Array) -> bool:
            del memory, params  # unused
            current_val = state.get(self._robot, "fingers")
            dist = (target_val - current_val)**2
            return dist < self._grasp_tol

        return ParameterizedOption(name,
                                   types=types,
                                   params_space=params_space,
                                   policy=_policy,
                                   initiable=lambda _1, _2, _3, _4: True,
                                   terminal=_terminal)

    def _get_move_toward_waypoint_action(
            self, gripper_position: Sequence[float],
            target_position: Sequence[float]) -> Action:
        delta_position = np.subtract(target_position, gripper_position)
        action = self._move_gain * delta_position
        action_norm = np.linalg.norm(action)  # type: ignore
        if action_norm > self._max_vel_norm:
            action = action * self._max_vel_norm / action_norm
        action = np.r_[action, 0.0]
        action = np.clip(action, self.action_space.low, self.action_space.high)
        return Action(action.astype(np.float32))
