from dataclasses import dataclass, field
import functools
import itertools
import logging
import pickle
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Sequence, Set, Tuple, cast

from experiments.envs.utils import DottedDict
from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion, matrix_from_quat, multiply_poses
from predicators.pybullet_helpers.inverse_kinematics import InverseKinematicsError
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.motion_planning import run_motion_planning
from predicators.pybullet_helpers.robots import create_single_arm_pybullet_robot
from predicators.pybullet_helpers.robots.panda import PandaPyBulletRobot
from predicators.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, Object, Predicate, State, Type
from pybullet_utils.transformations import quaternion_from_euler

import numpy as np
import numpy.typing as npt
import pybullet as p
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate, rotate
from shapely import wkt
import os
import time

from predicators.utils import PyBulletState

def get_asset_path(filename: str) -> str:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', filename)
    assert os.path.exists(path), f"Asset {path} does not exist"
    return path

def get_tagged_block_sizes() -> List[Tuple[float, float, float]]:
    tags_path = get_asset_path('tags')
    return [
        tuple(pickle.load(open(os.path.join(tags_path, block_info_fname), 'rb'))['dimensions'])
        for block_info_fname in os.listdir(tags_path)
    ]

BlockId = int
BoxId = int

@dataclass
class PyBulletPackingState(PyBulletState):
    _block_id_to_block: Dict[BlockId, Object] = field(default_factory=dict)
    _box_id: BoxId = -10000
    _placement_offsets: List[Tuple[float, float]] = field(default_factory=list)

    def copy(self) -> State:
        state_dict_copy = super().copy().data
        simulator_state_copy = list(self.joint_positions)
        return PyBulletPackingState(
            state_dict_copy, simulator_state_copy,
            self._block_id_to_block,
            self._box_id, self._placement_offsets
        )

class PyBulletPackingEnv(PyBulletEnv):
    # Assets
    block_poses: ClassVar[Polygon] = wkt.load(open(get_asset_path("block-poses.wkt")))
    box_poses: ClassVar[Polygon] = wkt.load(open(get_asset_path("box-poses.wkt")))

    # Settings
    ## Task Generation
    range_train_box_cols: ClassVar[Tuple[int, int]] = (2, 4)
    num_box_rows: ClassVar[int] = 2
    box_front_margin = 0.20
    block_side_margin = 0.1
    block_vert_offset = 0.001

    num_tries: ClassVar[int] = 100000

    ## Object Descriptors
    block_sizes: ClassVar[Dict[BlockId, Tuple[float, float, float]]] = dict(enumerate(itertools.product(
        [0.07, 0.06, 0.05],
        [0.07, 0.06, 0.05],
        [0.13, 0.14, 0.15]
    ), 1000))
    box_col_counts: ClassVar[Dict[BoxId, int]] = dict(enumerate(range(2, 4+1), 1000))

    ## World Shape Parameters
    hiding_pose: ClassVar[Pose] = Pose((0.0, 0.0, -10.0), PyBulletEnv._default_orn)

    box_color: ClassVar[Tuple[float, float, float, float]] = (0.1, 0.1, 0.1, 0.7)
    box_height: ClassVar[float] = 0.4
    box_col_margins: ClassVar[float] = 0.05
    box_row_margins: ClassVar[float] = 0.03

    gripper_width: ClassVar[float] = 0.21
    max_finger_width: ClassVar[float] = 0.14

    robot_base_pos: ClassVar[Pose3D] = (-0.0716, .0, .0)
    robot_ee_init_pose: ClassVar[Pose] = Pose((0.3, .0, 0.3), (.0, 1.0, .0, .0))
    robot_finger_normals: ClassVar[Tuple[Pose3D, Pose3D]] = ((0, 1, 0), (0, -1, 0))

    debug_space_height: ClassVar[float] = 1.21
    debug_space_width: ClassVar[float] = 1.4
    debug_space_size: ClassVar[float] = 0.5

    ## Predicate thresholds
    held_thresh: ClassVar[float] = 0.5

    # Types
    _robot_type: ClassVar[Type] = Type("robot", ["x", "y", "z", "qx", "qy", "qz", "qw", "fingers"])
    _block_type: ClassVar[Type] = Type("block", ["x", "y", "z", "d", "w", "h", "qx", "qy", "qz", "qw", "held"])
    _box_type: ClassVar[Type] = Type("box", ["x", "y", "z", "d", "w", "h"])

    # Predicates
    ## InBox
    def _InBox_holds(state: State, objects: Sequence[Object]) -> bool: # type: ignore
        block, box = objects
        box_corners = PyBulletPackingEnv._get_box_corners(state, box)
        block_corners = PyBulletPackingEnv._get_block_corners(state, block)
        return (box_corners.min(axis=0) <= block_corners.min(axis=0)).all() and (box_corners.max(axis=0) >= block_corners.max(axis=0)).all()

    _InBox: ClassVar[Predicate] = Predicate("InBox", [_block_type, _box_type], _InBox_holds)

    ## OnTable
    def _OnTable_holds(state: State, objects: Sequence[Object]) -> bool: # type: ignore
        block, = objects
        return not PyBulletPackingEnv._InBox_holds(state, [block, PyBulletPackingEnv._box]) # type: ignore

    _OnTable: ClassVar[Predicate] = Predicate("OnTable", [_block_type], _OnTable_holds)

    # Common Objects
    _robot: ClassVar[Object] = Object("robot", _robot_type)
    _box: ClassVar[Object] = Object("box", _box_type)

    def __init__(self, use_gui: bool=True):
        super().__init__(use_gui)

        # For detecting multiprocessing and when we potentially need to reinitialize pybullet
        self._pid = os.getpid()

        # For keeping track of things in the active task and faster lookup
        self._block_id_to_block: Dict[BlockId, Object] = {}
        self._placement_offsets: List[Tuple[float, float]] = []
        self._box_id: BoxId = -10000

        if self._real_block_poses:
            self._test_tasks = [self._generate_real_task()]

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_packing"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._OnTable, self._InBox}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InBox}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._block_type, self._box_type}

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        if not self._train_tasks:
            self._train_tasks = self._generate_tasks(
                rng=self._train_rng,
                num_tasks=CFG.num_train_tasks,
                range_box_cols=self.range_train_box_cols,
            )
        return self._train_tasks

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if not self._test_tasks:
            self._test_tasks = self._generate_tasks(
                rng=self._test_rng,
                num_tasks=CFG.num_test_tasks,
                range_box_cols=(CFG.pybullet_packing_test_num_box_cols, CFG.pybullet_packing_test_num_box_cols),
            )
        return self._test_tasks

    def _generate_tasks(
        self,
        rng: np.random.Generator,
        num_tasks: int,
        range_box_cols: Tuple[int, int]
    ) -> List[EnvironmentTask]:
        return [
            self._generate_task(
                rng,
                range_box_cols,
            ) for _ in range(num_tasks)
        ]

    def _generate_task(
        self,
        rng: np.random.Generator,
        range_box_cols: Tuple[int, int]
    ) -> EnvironmentTask:
        num_box_cols = rng.integers(*range_box_cols, endpoint=True)
        collision_poly = MultiPolygon()

        # Creating objects
        block_id_to_block = {
            block_id: Object(f"block{idx}", self._block_type)
            for idx, block_id in enumerate(rng.choice(list(self.block_sizes.keys()), num_box_cols * self.num_box_rows, replace=False))
        }
        box_id = self._get_box_id(num_box_cols)

        # Setting up placement offsets
        placement_indices = np.arange(num_box_cols * self.num_box_rows).reshape(self.num_box_rows, num_box_cols)
        for row in range(self.num_box_rows):
            rng.shuffle(placement_indices[row])
        placement_indices = placement_indices.flatten()

        placement_coords = -np.vstack([
            placement_indices // num_box_cols - (self.num_box_rows - 1) / 2,
            placement_indices % num_box_cols - (num_box_cols - 1) / 2
        ]).T

        y_sep, _, x_sep, _ = self._get_box_creation_params()
        placement_offsets = list(map(tuple, placement_coords * [x_sep, y_sep]))

        # Creating the state and goal
        data: Dict[Object, npt.NDArray] = {}
        state = PyBulletPackingState(data, self._robot_initial_joints, block_id_to_block, box_id, placement_offsets)
        goal = {self._InBox([block, self._box]) for block in block_id_to_block.values()}

        # Setting up the robot and joint positions
        data[self._robot] = np.array(list(itertools.chain(
            self.robot_ee_init_pose.position,
            self.robot_ee_init_pose.orientation,
            [1.0]
        )))

        # Setting up the box
        box_d, box_w, box_h = self._get_box_dims(box_id)
        box_poly = box(-box_d/2, -box_w/2, box_d/2, box_w/2)
        _, box_y_min, box_x_max, box_y_max = self.box_poses.bounds

        box_x = box_x_max - box_d/2 - 0.0001
        box_y = 0

        data[self._box] = np.array(list(itertools.chain(
            (box_x, box_y, box_h / 2), [box_d, box_w, box_h],
        )))
        collision_poly = collision_poly.union(
            translate(box(-box_d/2 - self.box_front_margin, -box_w/2, box_d/2, box_w/2), box_x, box_y)
        )
        collision_poly = collision_poly.union(
            translate(box(-box_d/2, box_y_min, box_d/2, box_y_max), box_x, box_y)
        )

        # Setting up the blocks
        block_x_min, block_y_min, block_x_max, block_y_max = self.block_poses.bounds
        for block_id, block in block_id_to_block.items():
            block_d, block_w, block_h = self.block_sizes[block_id]
            for _ in range(self.num_tries):
                block_x_vertical = rng.choice([True, False])
                block_axis_up = rng.choice([True, False])
                block_x, block_y, block_rot = rng.uniform(
                    [block_x_min, block_y_min, -np.pi],
                    [block_x_max, block_y_max, np.pi]
                )

                block_w_2d = block_w if block_x_vertical else block_d
                block_poly_margins = translate(rotate(
                    box(-block_h/2, -block_w_2d/2 - self.block_side_margin, block_h/2, block_w_2d/2 + self.block_side_margin),
                block_rot, use_radians=True), block_x, block_y)
                if self.block_poses.contains(block_poly_margins) and not collision_poly.intersects(block_poly_margins) >= self.block_side_margin and block_x > -0.0:
                    break
            else:
                print([self.block_sizes[block_id] for block_id in block_id_to_block])
                raise ValueError('Could not generate a task with given settings')

            block_quat = multiply_poses(
                Pose((0, 0, 0), quaternion_from_euler(0, 0, block_rot)), # type: ignore
                Pose((0, 0, 0), quaternion_from_euler(0 if block_axis_up else np.pi, 0, 0)), # type: ignore
                Pose((0, 0, 0), quaternion_from_euler(0 if block_x_vertical else np.pi/2, 0, 0)), # type: ignore
                Pose((0, 0, 0), quaternion_from_euler(0, np.pi/2, 0)), # type: ignore
            ).orientation
            block_up_size = block_d if block_x_vertical else block_w
            data[block] = np.array(list(itertools.chain(
                # PyBullet sometimes bugs out if the blocks are on the same height
                [block_x, block_y, block_up_size/2 + self.block_vert_offset + rng.uniform(-1e-6, 1e-6)],
                [block_d, block_w, block_h],
                block_quat,
                [0],
            )))
            collision_poly = collision_poly.union(translate(rotate(
                box(-block_h/2, -block_w_2d/2, block_h/2, block_w_2d/2),
            block_rot, use_radians=True), block_x, block_y))
        return EnvironmentTask(state, goal)

    def _generate_real_task(
        self,
    ) -> EnvironmentTask:
        num_box_cols = CFG.pybullet_packing_test_num_box_cols
        assert len(self._real_block_poses) == num_box_cols * self.num_box_rows

        # Creating objects
        block_id_to_block = {
            block_id: Object(f"block{block_id}", self._block_type)
            for block_id in self._real_block_poses
        }
        box_id = self._get_box_id(CFG.pybullet_packing_test_num_box_cols)

        # Setting up placement offsets
        placement_indices = np.arange(num_box_cols * self.num_box_rows).reshape(self.num_box_rows, num_box_cols)
        for row in range(self.num_box_rows):
            self._test_rng.shuffle(placement_indices[row])
        placement_indices = placement_indices.flatten()

        placement_coords = -np.vstack([
            placement_indices // num_box_cols - (self.num_box_rows - 1) / 2,
            placement_indices % num_box_cols - (num_box_cols - 1) / 2
        ]).T

        y_sep, _, x_sep, _ = self._get_box_creation_params()
        placement_offsets = list(map(tuple, placement_coords * [x_sep, y_sep]))

        # Creating the state and goal
        data: Dict[Object, npt.NDArray] = {}
        state = PyBulletPackingState(data, self._robot_initial_joints, block_id_to_block, box_id, placement_offsets)
        goal = {self._InBox([block, self._box]) for block in block_id_to_block.values()}

        # Setting up the robot and joint positions
        data[self._robot] = np.array(list(itertools.chain(
            self.robot_ee_init_pose.position,
            self.robot_ee_init_pose.orientation,
            [1.0]
        )))

        # Setting up the box
        box_d, box_w, box_h = self._get_box_dims(box_id)
        _, _, box_x_max, _ = self.box_poses.bounds
        data[self._box] = np.array(list(itertools.chain(
            (box_x_max - box_d/2 - 0.0001, 0, box_h / 2), [box_d, box_w, box_h],
        )))

        # Setting up the blocks
        for block_id, block in block_id_to_block.items():
            pose = Pose((0, 0, self.block_vert_offset)).multiply(self._real_block_poses[block_id])
            data[block] = np.array(list(itertools.chain(
                pose.position,
                self.block_sizes[block_id],
                pose.orientation,
                [0],
            )))
        return EnvironmentTask(state, goal)

    @classmethod
    @functools.cache
    def initialize_pybullet( # type: ignore
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, DottedDict]:
        """Run super(), then handle packing-specific initialization."""
        physics_client_id, pybullet_robot, bodies =  super(
        ).initialize_pybullet(using_gui)
        bodies = DottedDict(bodies)

        # Getting the initial robot joints
        bodies.robot_initial_joints = pybullet_robot.get_joints()

        # Creating the bounds object
        bodies.bounds_obj_id = p.loadURDF(get_asset_path("bounds.urdf"), useFixedBase=True, physicsClientId=physics_client_id)
        p.setCollisionFilterPair(pybullet_robot.robot_id, bodies.bounds_obj_id, -1, -1, False, physicsClientId=physics_client_id)

        # Creating the box objects
        assert max(cls.box_col_counts.values()) >= max(*cls.range_train_box_cols, CFG.pybullet_packing_test_num_box_cols)
        bodies.box_id_to_obj_id = {box_id: cls._create_box(box_id, physics_client_id) for box_id in cls.box_col_counts}

        # Creating the block objects
        assert max(cls.box_col_counts.values()) * cls.num_box_rows <= len(cls.block_sizes)
        bodies.block_id_to_obj_id = {
            block_id: cls._create_block(dimensions, color, physics_client_id)
            for color, (block_id, dimensions) in zip(np.linspace([1, 0, 0, 1], [0, 0, 1, 1], len(cls.block_sizes)), cls.block_sizes.items())
        }

        # Parsing the real world task info
        bodies.real_block_poses = {}
        if CFG.pybullet_packing_task_info:
            real_block_info = pickle.load(open(CFG.pybullet_packing_task_info, 'rb'))
            for real_block_pose, real_block_dimensions in real_block_info:
                for block_id, block_dimensions in cls.block_sizes.items():
                    if np.allclose(block_dimensions, real_block_dimensions):
                        bodies.real_block_poses[block_id] = real_block_pose
                        break
                else:
                    raise ValueError("Real world block not matching any known block sizes")
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, bodies: Dict[str, Any]) -> None: # type: ignore
        assert isinstance(bodies, DottedDict)
        self._robot_initial_joints: JointPositions = bodies.robot_initial_joints
        self._bounds_obj_id: int = bodies.bounds_obj_id
        self._box_id_to_obj_id: Dict[BoxId, int] = bodies.box_id_to_obj_id
        self._block_id_to_obj_id: Dict[BlockId, int] = bodies.block_id_to_obj_id
        self._real_block_poses: Dict[BlockId, Pose] = bodies.real_block_poses

    @classmethod
    def _create_pybullet_robot(cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        return PandaPyBulletRobot(cls.robot_ee_init_pose, physics_client_id, Pose(cls.robot_base_pos, cls._default_orn))

    def _extract_robot_state(self, state: State) -> Array:
        raise NotImplementedError("Extracting the robot state not needed here")

    def simulate(self, state: State, action: Action) -> State:
        """Additionally check for collisions"""
        logging.info("SIMULATE")
        self._restart_pybullet() # Check if the env is running in a new process and needs a new pybullet instance

        self._current_observation = state
        self._reset_state(state)
        next_state = self.step(action)

        self._remove_grasp_constraint() # Clean up afterwards for functions like run_motion_planning

        if CFG.pybullet_control_mode == "reset" and self.check_collisions(itertools.chain(
            (self._block_id_to_obj_id[block_id] for block_id in self._block_id_to_block),
            [self._bounds_obj_id, self._box_id_to_obj_id[self._box_id]]
        )):
            return state
        return next_state

    @classmethod
    def run_motion_planning(cls, state: State, target_joint_positions: JointPositions, use_gui: bool=False) -> Optional[Sequence[JointPositions]]:
        assert isinstance(state, PyBulletPackingState)
        physics_client_id, robot, bodies = cls.initialize_pybullet(use_gui)

        target_joint_positions[robot.left_finger_joint_idx] = state.joint_positions[robot.left_finger_joint_idx]
        target_joint_positions[robot.right_finger_joint_idx] = state.joint_positions[robot.right_finger_joint_idx]

        cls._reset_pybullet(state, robot, bodies.block_id_to_obj_id, bodies.box_id_to_obj_id, physics_client_id)

        held_blocks = [(block_id, block) for block_id, block in state._block_id_to_block.items() if state.get(block, "held") >= cls.held_thresh]
        held_block_id, held_block = held_blocks[0]

        if held_blocks:
            (held_block_id, held_block), = held_blocks
            bx, by, bz = state[held_block][:3]
            bqx, bqy, bqz, bqw = state[held_block][6:10]
            rx, ry, rz, rqx, rqy, rqz, rqw = state[cls._robot][:7]
            base_link_to_held_obj = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw)).invert().multiply(Pose((bx, by, bz), (bqx, bqy, bqz, bqw)))
        else:
            base_link_to_held_obj = None

        return run_motion_planning(
            robot = robot,
            initial_positions = state.joint_positions,
            target_positions = target_joint_positions,
            collision_bodies = list(itertools.chain(
                (bodies.block_id_to_obj_id[block_id] for block_id in state._block_id_to_block),
                [bodies.bounds_obj_id, bodies.box_id_to_obj_id[state._box_id]]
            )),
            seed = CFG.seed,
            physics_client_id = physics_client_id,
            held_object = bodies.block_id_to_obj_id[held_block_id] if held_blocks else None,
            base_link_to_held_obj = base_link_to_held_obj, # type: ignore
        )

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle packing-specific resetting."""
        state = cast(PyBulletPackingState, state)

        # Set the state information
        self._block_id_to_block = state._block_id_to_block
        self._box_id = state._box_id
        self._placement_offsets = state._placement_offsets

        # Remove the old grasp constraint
        self._remove_grasp_constraint()

        # Reset pybullet
        self._reset_pybullet(
            state,
            self._pybullet_robot,
            self._block_id_to_obj_id,
            self._box_id_to_obj_id,
            self._physics_client_id
        )

        # Set the held block constraint
        self._held_obj_id = None
        for block_id, block in state._block_id_to_block.items():
            if state.get(block, 'held') >= self.held_thresh:
                assert self._held_obj_id is None
                self._force_grasp_object(block_id)

        assert State({k: np.abs(v) for k, v in self._get_state().data.items()}).allclose(State({k: np.abs(v) for k, v in state.data.items()}))

    @classmethod
    def _reset_pybullet(
        cls,
        state: PyBulletPackingState,
        pybullet_robot: SingleArmPyBulletRobot,
        block_id_to_obj_id: Dict[BlockId, int],
        box_id_to_obj_id: Dict[BoxId, int],
        physics_client_id: int
    ) -> None:
        # Set robot position
        pybullet_robot.set_joints(state.joint_positions)

        # Set block positions
        for block_id, obj_id in block_id_to_obj_id.items():
            block = state._block_id_to_block.get(block_id, None)
            if block is None:
                cls._hide_object(obj_id, physics_client_id)
            else:
                p.resetBasePositionAndOrientation(
                    obj_id,
                    state[block][:3],
                    state[block][6:10],
                    physicsClientId=physics_client_id,
                )

        # Set box positions
        for box_id, obj_id in box_id_to_obj_id.items():
            if box_id != state._box_id:
                cls._hide_object(obj_id, physics_client_id)
            else:
                p.resetBasePositionAndOrientation(
                    obj_id,
                    state[cls._box][:3],
                    cls._default_orn,
                    physicsClientId=physics_client_id,
                )

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_id_to_block, self._box_id and self._placement_offsets. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        data = {}
        state = PyBulletPackingState(
            data, self._pybullet_robot.get_joints(),
            self._block_id_to_block, self._box_id,
            self._placement_offsets,
        )

        # Getting the robot state
        rx, ry, rz, rqx, rqy, rqz, rqw, rf = self._pybullet_robot.get_state()
        data[self._robot] = np.array([rx, ry, rz, rqx, rqy, rqz, rqw, self._fingers_joint_to_state(rf)])

        # Getting the box state
        box_position = p.getBasePositionAndOrientation(
            self._box_id_to_obj_id[self._box_id],
            physicsClientId=self._physics_client_id
        )[0]
        data[self._box] = np.array(list(itertools.chain(
            box_position, self._get_box_dims(self._box_id)
        )))

        # Getting the blocks states
        for block_id, block in self._block_id_to_block.items():
            block_pose = p.getBasePositionAndOrientation(
                self._block_id_to_obj_id[block_id],
                physicsClientId=self._physics_client_id
            )
            data[block] = np.array(list(itertools.chain(
                block_pose[0], self.block_sizes[block_id],
                block_pose[1], [0.0]
            )))
        if self._held_obj_id is not None:
            for block_id, block in self._block_id_to_block.items():
                if self._held_obj_id == self._block_id_to_obj_id[block_id]:
                    state.set(block, "held", 1.0)
                    break
        return state

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return a list of pybullet IDs corresponding to objects in the
        simulator that should be checked when determining whether one is
        held."""
        return [self._block_id_to_obj_id[block_id] for block_id in self._block_id_to_block]

    def _get_expected_finger_normals(self) -> Dict[int, Array]:
        """Get the expected finger normals, used in detect_held_object(), as a
        mapping from finger link index to a unit-length normal vector.

        This is environment-specific because it depends on the end
        effector's orientation when grasping.
        """
        left_normal = self._get_finger_rot_matrix(self._pybullet_robot.left_finger_id) @ np.array(self.robot_finger_normals[0])
        right_normal = self._get_finger_rot_matrix(self._pybullet_robot.right_finger_id) @ np.array(self.robot_finger_normals[1])
        return {
            self._pybullet_robot.left_finger_id: left_normal,
            self._pybullet_robot.right_finger_id: right_normal,
        }

    def _restart_pybullet(self):
        if self._pid == os.getpid():
            return
        logging.info(f"RESTARTING PYBULLET FOR PID {os.getpid()}")
        self._physics_client_id, self._pybullet_robot, pybullet_bodies = \
            self.initialize_pybullet(False)
        self._store_pybullet_bodies(pybullet_bodies)
        self._pid = os.getpid()

    def _force_grasp_object(self, block_id: BlockId) -> None:
        block_obj_id = self._block_id_to_obj_id[block_id]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        assert block_obj_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = block_obj_id
        self._create_grasp_constraint()

    def _remove_grasp_constraint(self) -> None:
        if self._held_constraint_id is not None:
            p.removeConstraint(self._held_constraint_id,
                               physicsClientId=self._physics_client_id)
            self._held_constraint_id = None

    def _get_finger_rot_matrix(self, finger_id: int) -> npt.NDArray[np.float32]:
        finger_state = get_link_state(self._pybullet_robot.robot_id, finger_id, self._physics_client_id)
        return matrix_from_quat(finger_state.linkWorldOrientation)

    def check_collisions(self, obj_ids_iter: Iterator[int]) -> bool:
        block_obj_ids = set(self._block_id_to_obj_id.values())
        obj_ids = list(obj_ids_iter)

        p.performCollisionDetection(physicsClientId=self._physics_client_id)
        for idx, obj_id in enumerate(obj_ids):
            if obj_id == self._held_obj_id:
                continue
            if p.getContactPoints(self._pybullet_robot.robot_id, obj_id, physicsClientId=self._physics_client_id):
                p1, p2 = p.getContactPoints(self._pybullet_robot.robot_id, obj_id, physicsClientId=self._physics_client_id)[0][5:7]
                logging.info(f"ROBOT COLLISION {p.getBodyInfo(obj_id, physicsClientId=self._physics_client_id)} {idx}/{len(obj_ids)}")
                if obj_id in block_obj_ids:
                    logging.info("BLOCK")
                elif obj_id == self._box_id_to_obj_id[self._box_id]:
                    logging.info("BOX")
                else:
                    logging.info("BOUNDS")
                return True
            if self._held_obj_id is not None and p.getContactPoints(self._held_obj_id, obj_id, physicsClientId=self._physics_client_id):
                p1, p2 = p.getContactPoints(self._held_obj_id, obj_id, physicsClientId=self._physics_client_id)[0][5:7]
                logging.info(f"BLOCK COLLISION {p.getBodyInfo(obj_id, physicsClientId=self._physics_client_id)} {idx}/{len(obj_ids)}")
                if obj_id in block_obj_ids:
                    logging.info("BLOCK")
                elif obj_id == self._box_id_to_obj_id[self._box_id]:
                    logging.info("BOX")
                else:
                    logging.info("BOUNDS")
                return True
        return False

    @classmethod
    def fingers_state_to_joint(cls, pybullet_robot: SingleArmPyBulletRobot,
                               fingers_state: float) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either pybullet_robot.closed_fingers or
        pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
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

    @classmethod
    def _hide_object(cls, obj_id: int, physics_client_id: int) -> None:
        p.resetBasePositionAndOrientation(
            obj_id,
            cls.hiding_pose.position,
            cls.hiding_pose.orientation,
            physicsClientId=physics_client_id
        )

    @classmethod
    def _create_block(
        cls,
        dimensions: Tuple[float, float, float],
        color: Tuple[float, float, float, float],
        physics_client_id: int
    ) -> int:
        d, w, h = dimensions
        half_extents = (d/2, w/2, h/2)
        return create_pybullet_block(
            color = color,
            half_extents = half_extents,
            mass = cls._obj_mass,
            friction = cls._obj_friction,
            orientation = cls._default_orn,
            physics_client_id = physics_client_id
        )

    @classmethod
    def _get_box_creation_params(cls) -> Tuple[float, float, float, float]:
        max_block_width = max(size for sizes in cls.block_sizes.values() for size in sizes[:2])

        block_width_sep = max(max_block_width/2, cls.max_finger_width/2) + cls.box_col_margins + max_block_width/2
        block_width_margin = cls.gripper_width / 2 + cls.box_col_margins

        block_depth_sep = max_block_width + cls.box_row_margins
        block_depth_margin = max_block_width / 2 + cls.box_row_margins

        return block_width_sep, block_width_margin, block_depth_sep, block_depth_margin

    @classmethod
    def _get_box_dims(cls, box_id: BoxId) -> Tuple[float, float, float]:
        block_width_sep, block_width_margin, block_depth_sep, block_depth_margin = cls._get_box_creation_params()
        box_depth = block_depth_sep * (cls.num_box_rows - 1) + block_depth_margin * 2
        box_width = block_width_sep * (cls.box_col_counts[box_id] - 1) + block_width_margin * 2
        return (box_depth, box_width, cls.box_height)

    @classmethod
    def _create_box(cls, box_id: int, physics_client_id: int) -> int:
        return create_pybullet_box(
            cls.box_color,
            cls._get_box_dims(box_id),
            0, 1, cls._default_orn,
            physics_client_id,
        )

    @classmethod
    def _get_box_id(cls, col_count: int) -> BoxId:
        for box_id, box_col_count in cls.box_col_counts.items():
            if box_col_count == col_count:
                return box_id
        raise ValueError(f"Box with column count {col_count} does not exist")

    @classmethod
    def _get_box_corners(cls, state: State, block: Object) -> npt.NDArray[np.float32]:
        pos = np.array([state.get(block, "x"), state.get(block, "y"), state.get(block, "z")])
        dims = np.array([state.get(block, "d"), state.get(block, "w"), state.get(block, "h")])
        return pos + np.vstack([
            dims / 2 * mult for mult in itertools.product(*([[-1, 1]]*3))
        ])

    @classmethod
    def _get_block_corners(cls, state: State, block: Object) -> npt.NDArray[np.float32]:
        assert block.is_instance(cls._block_type)
        pos = np.array([state.get(block, "x"), state.get(block, "y"), state.get(block, "z")])
        dims = np.array([state.get(block, "d"), state.get(block, "w"), state.get(block, "h")])
        quaternion = state.get(block, "qx"), state.get(block, "qy"), state.get(block, "qz"), state.get(block, "qw")
        relative_corners = matrix_from_quat(quaternion) @ np.vstack([
            dims / 2 * mult for mult in itertools.product(*([[-1, 1]]*3))
        ]).T
        return pos + relative_corners.T

def create_pybullet_box(
    color: Tuple[float, float, float, float],
    scale: Tuple[float, float, float],
    mass: float, friction: float, orientation: Quaternion,
    physics_client_id: int
) -> int:
    """A generic utility for creating a new block.

    Returns the PyBullet ID of the newly created block.
    """
    # The poses here are not important because they are overwritten by
    # the state values when a task is reset.
    pose = (0, 0, 0)

    # Create the collision shape.
    collision_id = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                          fileName=get_asset_path("box.obj"),
                                          meshScale=scale,
                                          flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                          physicsClientId=physics_client_id)

    # Create the visual_shape.
    visual_id = p.createVisualShape(shapeType=p.GEOM_MESH,
                                    fileName=get_asset_path("box.obj"),
                                    meshScale=scale,
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