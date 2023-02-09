"""A simple environment where a robot must pick a block from the table with a
top grasp and put it into a high-up shelf with a side grasp.

The main purpose of this environment is to develop PyBullet options that
involve changing the end-effector orientation.
"""

import logging
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Sequence, Set, Tuple, Type

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.controllers import \
    create_change_fingers_option, create_move_end_effector_to_pose_option
from predicators.pybullet_helpers.geometry import Pose3D, Quaternion
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot, \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import RGBA, Array, Object, ParameterizedOption, \
    Predicate, State, Task


class PyBulletShelfEnv(PyBulletEnv):
    """PyBullet shelf domain."""
    # Parameters that aren't important enough to need to clog up settings.py
    # The table x bounds are (1.1, 1.6), but the workspace is smaller.
    x_lb: ClassVar[float] = 1.2
    x_ub: ClassVar[float] = 1.5
    # The table y bounds are (0.3, 1.2), but the workspace is smaller.
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1

    # Option parameters.
    _offset_z: ClassVar[float] = 0.01
    _pick_z: ClassVar[float] = 0.5

    # Table parameters.
    _table_pose: ClassVar[Pose3D] = (1.35, 0.75, 0.0)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)
    _table_height: ClassVar[float] = 0.2

    # Robot parameters.
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = _pick_z
    _move_to_pose_tol: ClassVar[float] = 1e-4

    # Block parameters.
    _block_color: ClassVar[RGBA] = (1.0, 0.0, 0.0, 1.0)
    _block_size: ClassVar[float] = 0.04

    # Shelf parameters.
    shelf_width: ClassVar[float] = (x_ub - x_lb) * 0.4
    shelf_length: ClassVar[float] = (y_ub - y_lb) * 0.2
    shelf_base_height: ClassVar[float] = _pick_z * 0.8
    shelf_ceiling_height: ClassVar[float] = _pick_z * 0.2
    shelf_ceiling_thickness: ClassVar[float] = 0.01
    shelf_pole_girth: ClassVar[float] = 0.01
    shelf_color: ClassVar[RGBA] = (0.5, 0.3, 0.05, 1.0)
    shelf_x: ClassVar[float] = x_ub - shelf_width / 2
    shelf_y: ClassVar[float] = y_ub - shelf_length

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return set()

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return set()

    @property
    def types(self) -> Set[Type]:
        return set()

    @property
    def options(self) -> Set[ParameterizedOption]:
        return set()

    @property
    def action_space(self) -> Box:
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)

    def _initialize_pybullet(self) -> None:
        """Run super(), then handle blocks-specific initialization."""
        super()._initialize_pybullet()

        # Load table in both the main client and the copy.
        self._table_id = p.loadURDF(
            utils.get_env_asset_path("urdf/table.urdf"),
            useFixedBase=True,
            physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id)
        p.loadURDF(utils.get_env_asset_path("urdf/table.urdf"),
                   useFixedBase=True,
                   physicsClientId=self._physics_client_id2)
        p.resetBasePositionAndOrientation(
            self._table_id,
            self._table_pose,
            self._table_orientation,
            physicsClientId=self._physics_client_id2)

        # Create shelf.
        color = self.shelf_color
        orientation = self._default_orn
        base_pose = (self.shelf_x, self.shelf_y, self.shelf_base_height / 2)
        # Holder base.
        # Create the collision shape.
        base_half_extents = [
            self.shelf_width / 2, self.shelf_length / 2,
            self.shelf_base_height / 2
        ]
        base_collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=base_half_extents,
            physicsClientId=self._physics_client_id)
        # Create the visual shape.
        base_visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=base_half_extents,
            rgbaColor=color,
            physicsClientId=self._physics_client_id)
        # Create the ceiling.
        link_positions = []
        link_collision_shape_indices = []
        link_visual_shape_indices = []
        pose = (
            0, 0,
            self.shelf_base_height / 2 + self.shelf_ceiling_height - \
                self.shelf_ceiling_thickness / 2
        )
        link_positions.append(pose)
        half_extents = [
            self.shelf_width / 2, self.shelf_length / 2,
            self.shelf_ceiling_thickness / 2
        ]
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self._physics_client_id)
        link_collision_shape_indices.append(collision_id)
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self._physics_client_id)
        link_visual_shape_indices.append(visual_id)
        # Create poles connecting the base to the ceiling.
        for x_sign in [-1, 1]:
            for y_sign in [-1, 1]:
                pose = (x_sign * (self.shelf_width - self.shelf_pole_girth) /
                        2, y_sign *
                        (self.shelf_length - self.shelf_pole_girth) / 2,
                        self.shelf_base_height / 2 +
                        self.shelf_ceiling_height / 2)
                link_positions.append(pose)
                half_extents = [
                    self.shelf_pole_girth / 2, self.shelf_pole_girth / 2,
                    self.shelf_ceiling_height / 2
                ]
                collision_id = p.createCollisionShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    physicsClientId=self._physics_client_id)
                link_collision_shape_indices.append(collision_id)
                visual_id = p.createVisualShape(
                    p.GEOM_BOX,
                    halfExtents=half_extents,
                    rgbaColor=color,
                    physicsClientId=self._physics_client_id)
                link_visual_shape_indices.append(visual_id)

        # Create the whole body.
        num_links = len(link_positions)
        assert len(link_collision_shape_indices) == num_links
        assert len(link_visual_shape_indices) == num_links
        link_masses = [0.1 for _ in range(num_links)]
        link_orientations = [orientation for _ in range(num_links)]
        link_intertial_frame_positions = [[0, 0, 0] for _ in range(num_links)]
        link_intertial_frame_orns = [[0, 0, 0, 1] for _ in range(num_links)]
        link_parent_indices = [0 for _ in range(num_links)]
        link_joint_types = [p.JOINT_FIXED for _ in range(num_links)]
        link_joint_axis = [[0, 0, 0] for _ in range(num_links)]
        self._holder_id = p.createMultiBody(
            baseCollisionShapeIndex=base_collision_id,
            baseVisualShapeIndex=base_visual_id,
            basePosition=base_pose,
            baseOrientation=orientation,
            linkMasses=link_masses,
            linkCollisionShapeIndices=link_collision_shape_indices,
            linkVisualShapeIndices=link_visual_shape_indices,
            linkPositions=link_positions,
            linkOrientations=link_orientations,
            linkInertialFramePositions=link_intertial_frame_positions,
            linkInertialFrameOrientations=link_intertial_frame_orns,
            linkParentIndices=link_parent_indices,
            linkJointTypes=link_joint_types,
            linkJointAxis=link_joint_axis,
            physicsClientId=self._physics_client_id)

        # Create block.
        color = self._block_color
        half_extents = (self._block_size / 2.0, self._block_size / 2.0,
                        self._block_size / 2.0)
        self._block_id = create_pybullet_block(color, half_extents,
                                               self._obj_mass,
                                               self._obj_friction,
                                               self._default_orn,
                                               self._physics_client_id)

        import time
        while True:
            p.stepSimulation(physicsClientId=self._physics_client_id)
            time.sleep(0.001)

    def _create_pybullet_robot(
            self, physics_client_id: int) -> SingleArmPyBulletRobot:
        ee_home = (self.robot_init_x, self.robot_init_y, self.robot_init_z)
        return create_single_arm_pybullet_robot(CFG.pybullet_robot,
                                                physics_client_id, ee_home)

    def _extract_robot_state(self, state: State) -> Array:
        return np.array([
            state.get(self._robot, "pose_x"),
            state.get(self._robot, "pose_y"),
            state.get(self._robot, "pose_z"),
            self._fingers_state_to_joint(state.get(self._robot, "fingers")),
        ],
                        dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_shelf"

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle blocks-specific resetting."""
        super()._reset_state(state)

        import ipdb
        ipdb.set_trace()

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_id_to_block and self._held_obj_id. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        import ipdb
        ipdb.set_trace()

    def _get_tasks(self, num_tasks: int,
                   rng: np.random.Generator) -> List[Task]:
        import ipdb
        ipdb.set_trace()

    def _load_task_from_json(self, json_file: Path) -> Task:
        task = super()._load_task_from_json(json_file)
        return self._add_pybullet_state_to_tasks([task])[0]

    def _get_object_ids_for_held_check(self) -> List[int]:
        return sorted(self._block_id_to_block)

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        if CFG.pybullet_robot == "panda":
            # gripper rotated 90deg so parallel to x-axis
            normal = np.array([1., 0., 0.], dtype=np.float32)
        elif CFG.pybullet_robot == "fetch":
            # gripper parallel to y-axis
            normal = np.array([0., 1., 0.], dtype=np.float32)
        else:  # pragma: no cover
            # Shouldn't happen unless we introduce a new robot.
            raise ValueError(f"Unknown robot {CFG.pybullet_robot}")

        return {
            self._pybullet_robot.left_finger_id: normal,
            self._pybullet_robot.right_finger_id: -1 * normal,
        }

    def _force_grasp_object(self, block: Object) -> None:
        block_to_block_id = {b: i for i, b in self._block_id_to_block.items()}
        block_id = block_to_block_id[block]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        assert block_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = block_id
        self._create_grasp_constraint()

    def _create_blocks_move_to_above_block_option(
            self, name: str, z_func: Callable[[float], float],
            finger_status: str) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        block argument.

        The parameter z_func maps the block's z position to the target z
        position.
        """
        types = [self._robot_type, self._block_type]
        params_space = Box(0, 1, (0, ))

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D, str]:
            assert not params
            robot, block = objects
            current_pose = (state.get(robot,
                                      "pose_x"), state.get(robot, "pose_y"),
                            state.get(robot, "pose_z"))
            target_pose = (state.get(block,
                                     "pose_x"), state.get(block, "pose_y"),
                           z_func(state.get(block, "pose_z")))
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            self._pybullet_robot_sim, name, types, params_space,
            _get_current_and_target_pose_and_finger_status,
            self._move_to_pose_tol, self._max_vel_norm,
            self._finger_action_nudge_magnitude)

    def _create_blocks_move_to_above_table_option(
            self, name: str, z: float,
            finger_status: str) -> ParameterizedOption:
        """Creates a ParameterizedOption for moving to a pose above that of the
        table.

        The z position of the target pose must be provided.
        """
        types = [self._robot_type]
        params_space = Box(0, 1, (2, ))

        def _get_current_and_target_pose_and_finger_status(
                state: State, objects: Sequence[Object],
                params: Array) -> Tuple[Pose3D, Pose3D, str]:
            robot, = objects
            current_pose = (state.get(robot,
                                      "pose_x"), state.get(robot, "pose_y"),
                            state.get(robot, "pose_z"))
            # De-normalize parameters to actual table coordinates.
            x_norm, y_norm = params
            target_pose = (self.x_lb + (self.x_ub - self.x_lb) * x_norm,
                           self.y_lb + (self.y_ub - self.y_lb) * y_norm, z)
            return current_pose, target_pose, finger_status

        return create_move_end_effector_to_pose_option(
            self._pybullet_robot_sim, name, types, params_space,
            _get_current_and_target_pose_and_finger_status,
            self._move_to_pose_tol, self._max_vel_norm,
            self._finger_action_nudge_magnitude)

    def _fingers_state_to_joint(self, fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either self._pybullet_robot.closed_fingers or
        self._pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        return closed_f if fingers_state == 0.0 else open_f

    def _fingers_joint_to_state(self, fingers_joint: float) -> float:
        """Convert the finger joint values in PyBullet to values for the State.

        The joint values given as input are the ones coming out of
        self._pybullet_robot.get_state().
        """
        open_f = self._pybullet_robot.open_fingers
        closed_f = self._pybullet_robot.closed_fingers
        # Fingers in the State should be either 0 or 1.
        return int(fingers_joint > (open_f + closed_f) / 2)
