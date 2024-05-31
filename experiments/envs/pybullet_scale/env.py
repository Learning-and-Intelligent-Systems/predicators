import copy
from dataclasses import dataclass, field
import functools
import itertools
import logging
import pickle
import time
from typing import Any, ClassVar, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, cast

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
from predicators.utils import BiRRT, PyBulletState
from pybullet_utils.transformations import quaternion_from_euler

import numpy as np
import numpy.typing as npt
import pybullet as p
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate, rotate
from shapely import wkt
import os
from gym.spaces import Box


BlockId = int
BoxId = int

def get_asset_path(filename: str) -> str:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', filename)
    assert os.path.exists(path), f"Asset {path} does not exist"
    return path

def get_tagged_block_sizes() -> List[Tuple[float, float, float]]:
    tags_path = get_asset_path('tags')
    return [
        dimensions
        for block_info_fname in os.listdir(tags_path)
        for d, w, h in [sorted(pickle.load(open(os.path.join(tags_path, block_info_fname), 'rb'))['dimensions'])]
        for dimensions in [(d, w, h), (w, d, h)]
    ]

def get_box_creation_params(max_block_width: float, max_finger_width: float, gripper_width: float, col_margins: float, row_margins: float):
    box_width_sep = max(max_block_width/2, max_finger_width/2) + col_margins + max_block_width/2
    box_width_margin = gripper_width / 2 + col_margins

    box_depth_sep = max_block_width + row_margins
    box_depth_margin = max_block_width / 2 + row_margins

    return box_width_sep, box_width_margin, box_depth_sep, box_depth_margin

def get_box_dimensions(
    num_rows: int,
    num_cols: int,
    height: float,
    width_sep: float,
    width_margin: float,
    depth_sep: float,
    depth_margin: float,
) -> Tuple[float, float, float]:
    return (depth_sep * (num_rows - 1) + depth_margin * 2, width_sep * (num_cols - 1) + width_margin * 2, height)

@dataclass
class PyBulletScaleState(PyBulletState):
    _block_id_to_block: Dict[BlockId, Object] = field(default_factory=dict)
    _placement_offsets: List[Tuple[float, float, bool]] = field(default_factory=list)

    def copy(self) -> State:
        super_copy = super().copy()
        assert isinstance(super_copy, PyBulletState)
        return PyBulletScaleState(
            super_copy.data, super_copy.joint_positions,
            self._block_id_to_block,
            self._placement_offsets
        )

ScaleMotionPlanningState = Tuple[npt.NDArray, bool, bool] # (joints, is starting state, ok for block to be too close)

class PyBulletScaleEnv(PyBulletEnv):
    # Assets
    block_poses: ClassVar[Polygon] = wkt.load(open(get_asset_path("block-poses.wkt")))
    block_poses_margins: ClassVar[Polygon] = wkt.load(open(get_asset_path("block-poses-margins.wkt")))

    # Settings
    ## Task Generation
    range_train_per_side_blocks: ClassVar[Tuple[int, int]] = (2, 5)
    box_front_margin: ClassVar[float] = 0.25
    block_margin: ClassVar[float] = 0.1
    block_vert_offset: ClassVar[float] = 1e-3

    num_tries: ClassVar[int] = 100000

    ## Object Descriptors
    ### Blocks
    blocks_ids_to_dimensions: ClassVar[Dict[BlockId, Tuple[float, float, float]]] = dict(enumerate(itertools.product(
        np.linspace(0.035, 0.055, 3),
        np.linspace(0.035, 0.055, 3),
        np.linspace(0.11, 0.13, 3),
    ), 1000))

    ## World Shape Parameters
    hiding_pose: ClassVar[Pose] = Pose((0.0, 0.0, -10.0), PyBulletEnv._default_orn)

    robot_base_pos: ClassVar[Pose3D] = (-0.0716, .0, .0)
    robot_ee_init_pose: ClassVar[Pose] = Pose((0.4, .0, 0.5), (.0, 1.0, .0, .0))
    robot_finger_normals: ClassVar[Tuple[Pose3D, Pose3D]] = ((0, 1, 0), (0, -1, 0))

    scale_poses: ClassVar[Tuple[Pose, Pose]] = (
        Pose((0.4, 0.2, 0.0)),
        Pose.from_rpy((0.4, -0.2, 0.0), (0, 0, np.pi)),
    )
    scale_valid_block_placements: ClassVar[Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]] = (
        list(itertools.product(np.linspace(0.15, 0.65, 6), np.linspace(0.1, 0.35, 2)))[1:-1],
        list(itertools.product(np.linspace(0.15, 0.65, 6), np.linspace(-0.1, -0.35, 2)))[1:-1],
    )

    scale_size: ClassVar[Tuple[float, float, float]] = (0.7, 0.4, 0.15)

    ## Collision Thresholds
    bounds_min_distance: ClassVar[float] = 0.01
    scale_min_distance: ClassVar[float] = 0.005
    block_min_distance: ClassVar[float] = 0.003

    ## Predicate thresholds
    held_thresh: ClassVar[float] = 0.5
    scale_checked_thresh: ClassVar[float] = 0.5

    # Types
    _robot_type: ClassVar[Type] = Type("robot", ["x", "y", "z", "qx", "qy", "qz", "qw", "fingers"])
    _block_type: ClassVar[Type] = Type("block", ["x", "y", "z", "d", "w", "h", "qx", "qy", "qz", "qw", "held"])
    _scale_type: ClassVar[Type] = Type("scale", ["checked"])

    # Predicates
    ## ScaleChecked
    def _ScaleChecked_holds(state: State, objects: Sequence[Object]) -> bool: # type: ignore
        scale, = objects
        blocks = state.get_objects(PyBulletScaleEnv._block_type)
        assert len(blocks) % 2 == 0
        num_left_scale_blocks = len([
            block
            for block in blocks
            if PyBulletScaleEnv._check_block_on_scale(state, block, True)
        ])
        num_right_scale_blocks = len([
            block
            for block in blocks
            if PyBulletScaleEnv._check_block_on_scale(state, block, False)
        ])

        return state.get(scale, "checked") >= PyBulletScaleEnv.scale_checked_thresh and \
            len(blocks) / 2 == num_left_scale_blocks == num_right_scale_blocks

    _ScaleChecked: ClassVar[Predicate] = Predicate("ScaleChecked", [_scale_type], _ScaleChecked_holds)

    ## ScaleNotChecked
    def _ScaleNotChecked_holds(state: State, objects: Sequence[Object]) -> bool: # type: ignore
        scale, = objects
        return state.get(scale, "checked") < PyBulletScaleEnv.scale_checked_thresh # type: ignore

    _ScaleNotChecked: ClassVar[Predicate] = Predicate("ScaleNotChecked", [_scale_type], _ScaleNotChecked_holds)

    ## OnScale
    def _OnScale_holds(state: State, objects: Sequence[Object]) -> bool: # type: ignore
        block, _ = objects
        return PyBulletScaleEnv._check_block_on_scale(state, block, True) or \
            PyBulletScaleEnv._check_block_on_scale(state, block, False)

    _OnScale: ClassVar[Predicate] = Predicate("OnScale", [_block_type, _scale_type], _OnScale_holds)

    ## OnTable
    def _OnTable_holds(state: State, objects: Sequence[Object]) -> bool: # type: ignore
        block, = objects
        return not PyBulletScaleEnv._OnScale_holds(state, [block, PyBulletScaleEnv._scale]) # type: ignore

    _OnTable: ClassVar[Predicate] = Predicate("OnTable", [_block_type], _OnTable_holds)

    # Common Objects
    _robot: ClassVar[Object] = Object("robot", _robot_type)
    _scale: ClassVar[Object] = Object("scale", _scale_type)

    def __init__(self, use_gui: bool=True):
        super().__init__(use_gui)

        # For detecting multiprocessing and when we potentially need to reinitialize pybullet
        self._pid = os.getpid()

        # For keeping track of things in the active task and faster lookup
        self._block_id_to_block: Dict[BlockId, Object] = {}
        self._placement_offsets: List[Tuple[float, float, bool]] = []
        self._scale_checked = False

    @classmethod
    def get_name(cls) -> str:
        return "pybullet_scale"

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._OnTable, self._OnScale, self._ScaleChecked, self._ScaleNotChecked}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._OnScale, self._ScaleChecked}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._block_type, self._scale_type}

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        if not self._train_tasks:
            self._train_tasks = self._generate_tasks(
                rng=self._train_rng,
                num_tasks=CFG.num_train_tasks,
                range_blocks_per_side=self.range_train_per_side_blocks,
            )
        return self._train_tasks

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if not self._test_tasks:
            self._test_tasks = self._generate_tasks(
                rng=self._test_rng,
                num_tasks=CFG.num_test_tasks,
                range_blocks_per_side=(CFG.pybullet_scale_test_num_blocks_per_side, CFG.pybullet_scale_test_num_blocks_per_side),
            )
        return self._test_tasks

    def _generate_tasks(
        self,
        rng: np.random.Generator,
        num_tasks: int,
        range_blocks_per_side: Tuple[int, int]
    ) -> List[EnvironmentTask]:
        return [
            self._generate_task(
                rng,
                range_blocks_per_side,
            ) for _ in range(num_tasks)
        ]

    def _generate_task(
        self,
        rng: np.random.Generator,
        range_blocks_per_side: Tuple[int, int]
    ) -> EnvironmentTask:
        num_blocks_per_side = rng.integers(*range_blocks_per_side, endpoint=True)

        # Setting up the state
        ## Creating objects
        block_id_to_block = {
            block_id: Object(f"block{idx}", self._block_type)
            for idx, block_id in enumerate(rng.choice(list(self.blocks_ids_to_dimensions.keys()), num_blocks_per_side * 2, replace=False))
        }

        ## Setting up placement offsets
        valid_block_placements = (np.asarray(self.scale_valid_block_placements, dtype=np.float32).transpose((1, 0, 2)) - [
            self.scale_poses[0].position[:2], self.scale_poses[1].position[:2]
        ])
        rng.shuffle(valid_block_placements, axis=0)

        sides = np.hstack([np.zeros(num_blocks_per_side, dtype=np.int64), np.ones(num_blocks_per_side, dtype=np.int64)])
        rng.shuffle(sides)

        side_indices = np.zeros((2, num_blocks_per_side * 2), dtype=np.int64)
        side_indices[sides, np.arange(num_blocks_per_side * 2)] = 1
        side_indices = np.cumsum(side_indices, axis=1)

        placement_offsets = np.hstack([valid_block_placements[side_indices[sides, np.arange(num_blocks_per_side * 2)], sides], sides[:, None].astype(np.bool_)])

        ## Creating the state and goal
        data: Dict[Object, npt.NDArray] = {}
        state = PyBulletScaleState(data, self._robot_initial_joints, block_id_to_block, placement_offsets)
        goal = {self._OnScale([block, self._scale]) for block in block_id_to_block.values()} | {self._ScaleChecked([self._scale])}

        # Setting up object data
        ## Setting up the scale
        data[self._scale] = np.array([0.0])

        ## Setting up the robot and joint positions
        data[self._robot] = np.array(list(itertools.chain(
            self.robot_ee_init_pose.position,
            self.robot_ee_init_pose.orientation,
            [1.0]
        )))

        ## Setting up the blocks
        block_x_min, block_y_min, block_x_max, block_y_max = self.block_poses.bounds
        blocks_poly = MultiPolygon()
        blocks_margins_poly = MultiPolygon()
        for block_id, block in block_id_to_block.items():
            block_d, block_w, block_h = self.blocks_ids_to_dimensions[block_id]
            for _ in range(self.num_tries):
                block_x, block_y, block_rot = rng.uniform(
                    [block_x_min, block_y_min, -np.pi],
                    [block_x_max, block_y_max, np.pi]
                )

                block_transform = lambda geom: translate(rotate(geom, block_rot, use_radians=True), block_x, block_y)
                block_poly = block_transform(box(-block_d/2, -block_w/2, block_h/2, block_w/2))
                block_poly_margins = block_poly.buffer(max(self.block_min_distance, self.block_margin))
                if self.block_poses.contains(block_poly) and not self.block_poses_margins.intersects(block_poly_margins) and \
                        not blocks_poly.intersects(block_poly_margins) and not blocks_margins_poly.intersects(block_poly):
                    break
            else:
                raise ValueError('Could not generate a task with given settings')

            block_quat = quaternion_from_euler(0, 0, block_rot)
            data[block] = np.array(list(itertools.chain(
                # PyBullet sometimes bugs out if the blocks are on the same height
                [block_x, block_y, block_h/2 + self.block_vert_offset + rng.uniform(-1e-6, 1e-6)],
                [block_d, block_w, block_h],
                block_quat,
                [0],
            )))
            blocks_poly = blocks_poly.union(block_poly)
            blocks_margins_poly = blocks_margins_poly.union(block_poly_margins)
        return EnvironmentTask(state, goal)

    @classmethod
    @functools.cache
    def initialize_pybullet( # type: ignore
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, DottedDict]:
        """Run super(), then handle packing-specific initialization."""
        # if not using_gui:
        #     return cls.initialize_pybullet(True)
        physics_client_id, pybullet_robot, bodies =  super(
        ).initialize_pybullet(using_gui)
        bodies = DottedDict(bodies)

        # Getting the initial robot joints
        bodies.robot_initial_joints = pybullet_robot.get_joints()

        # Creating the bounds object
        bodies.bounds_obj_id = p.loadURDF(get_asset_path("bounds.urdf"), useFixedBase=True, physicsClientId=physics_client_id)
        p.setCollisionFilterPair(pybullet_robot.robot_id, bodies.bounds_obj_id, -1, -1, False, physicsClientId=physics_client_id)

        # Creating the scale objects
        bodies.scale_obj_ids = (
            p.loadURDF(get_asset_path("scale.urdf"), *cls.scale_poses[0]),
            p.loadURDF(get_asset_path("scale.urdf"), *cls.scale_poses[1])
        )

        # Creating the block objects
        assert max(
            CFG.pybullet_scale_test_num_blocks_per_side, *cls.range_train_per_side_blocks
        ) * 2 <= len(cls.blocks_ids_to_dimensions)
        bodies.block_id_to_obj_id = {
            block_id: cls._create_block(dimensions, color, physics_client_id)
            for color, (block_id, dimensions) in zip(np.linspace([1, 0, 0, 1], [0, 0, 1, 1], len(cls.blocks_ids_to_dimensions)), cls.blocks_ids_to_dimensions.items())
        }
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, bodies: Dict[str, Any]) -> None: # type: ignore
        assert isinstance(bodies, DottedDict)
        self._robot_initial_joints: JointPositions = bodies.robot_initial_joints
        self._bounds_obj_id: int = bodies.bounds_obj_id
        self._scale_obj_ids: Tuple[int, int] = bodies.scale_obj_ids
        self._block_id_to_obj_id: Dict[BlockId, int] = bodies.block_id_to_obj_id

    @classmethod
    def _create_pybullet_robot(cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        return PandaPyBulletRobot(cls.robot_ee_init_pose, physics_client_id, Pose(cls.robot_base_pos, cls._default_orn))

    def _extract_robot_state(self, state: State) -> Array:
        raise NotImplementedError("Extracting the robot state not needed here")

    def step(self, action: Action) -> State:
        state: State = self._current_observation
        check_scale = action.extra_info if type(action.extra_info) == bool else False
        if check_scale:
            next_state = state.copy()
            next_state.set(self._scale, "checked", 1.0)
            return next_state

        was_block_grasped = self._held_obj_id is not None
        self._restart_pybullet()
        self._reset_state(state)

        # Running the actual transition
        self._scale_checked |= check_scale
        if self._scale_checked:
            self._remove_grasp_constraint()
            return state.copy()
        next_state = super().step(action)

        # Check collisions
        assert isinstance(self._pybullet_robot, PandaPyBulletRobot)
        collision, block_bounds_too_close = self.check_collisions(
            robot = self._pybullet_robot,
            bounds_obj_id = self._bounds_obj_id,
            scale_obj_ids = self._scale_obj_ids,
            block_obj_ids = [self._block_id_to_obj_id[block_id] for block_id in self._block_id_to_block],
            held_obj_id = self._held_obj_id,
            physics_client_id = self._physics_client_id
        )
        self._remove_grasp_constraint()

        if CFG.pybullet_control_mode == "reset" and collision and not (block_bounds_too_close and not was_block_grasped):
            logging.info("COLLISION")
            return state
        return next_state

    def simulate(self, state: State, action: Action) -> State:
        """Additionally check for collisions"""
        logging.info("SIMULATE")
        self._current_observation = state
        next_state = self.step(action)
        return next_state

    @classmethod
    def run_motion_planning(cls, state: State, target_joint_positions: JointPositions, use_gui: bool=False) -> Optional[Sequence[JointPositions]]:
        assert isinstance(state, PyBulletScaleState)
        physics_client_id, robot, bodies = cls.initialize_pybullet(use_gui)

        target_joint_positions[robot.left_finger_joint_idx] = state.joint_positions[robot.left_finger_joint_idx]
        target_joint_positions[robot.right_finger_joint_idx] = state.joint_positions[robot.right_finger_joint_idx]

        cls._reset_pybullet(state, robot, bodies.block_id_to_obj_id, physics_client_id)

        held_blocks = [(block_id, block) for block_id, block in state._block_id_to_block.items() if state.get(block, "held") >= cls.held_thresh]

        held_obj_id = None
        if held_blocks:
            (held_block_id, held_block), = held_blocks
            held_obj_id = bodies.block_id_to_obj_id[held_block_id]
            bx, by, bz = state[held_block][:3]
            bqx, bqy, bqz, bqw = state[held_block][6:10]
            rx, ry, rz, rqx, rqy, rqz, rqw = state[cls._robot][:7]
            base_link_to_held_block = Pose((rx, ry, rz), (rqx, rqy, rqz, rqw)).invert().multiply(Pose((bx, by, bz), (bqx, bqy, bqz, bqw)))
        else:
            base_link_to_held_block = None

        joint_space = robot.action_space
        joint_space.seed(CFG.seed)
        num_interp = CFG.pybullet_birrt_extend_num_interp

        def _sample_fn(pt: ScaleMotionPlanningState) -> ScaleMotionPlanningState:
            joints, _, _ = pt
            new_joints = joint_space.sample()
            # Don't change the fingers.
            new_joints[robot.left_finger_joint_idx] = joints[robot.left_finger_joint_idx]
            new_joints[robot.right_finger_joint_idx] = joints[robot.right_finger_joint_idx]
            return new_joints, False, False # Not a starting state and the block cannot be too close

        def _extend_fn(pt1: ScaleMotionPlanningState,
                    pt2: ScaleMotionPlanningState) -> Iterator[ScaleMotionPlanningState]:
            joints_start, start_is_starting_state, _ = pt1
            joints_end, end_is_starting_state, _ = pt2
            is_starting_state = start_is_starting_state | end_is_starting_state
            if np.allclose(joints_start, joints_end):
                yield joints_end, False, is_starting_state
            for joints in np.linspace(joints_start, joints_end, num_interp + 1)[1:]:
                yield joints, False, is_starting_state

        def _collision_fn(pt: ScaleMotionPlanningState) -> bool:
            joints, _, block_too_close_to_bounds_ok = pt
            robot.set_joints(list(joints))
            if base_link_to_held_block is not None:
                assert held_obj_id is not None
                world_to_base_link = get_link_state(
                    robot.robot_id,
                    robot.end_effector_id,
                    physics_client_id=physics_client_id).com_pose
                world_to_held_block = world_to_base_link.multiply(base_link_to_held_block)
                p.resetBasePositionAndOrientation(
                    held_obj_id,
                    world_to_held_block.position,
                    world_to_held_block.orientation,
                    physicsClientId=physics_client_id)

            assert isinstance(robot, PandaPyBulletRobot)
            collision, block_too_close_to_bounds = cls.check_collisions(
                robot = robot,
                bounds_obj_id = bodies.bounds_obj_id,
                scale_obj_ids = bodies.scale_obj_ids,
                block_obj_ids = [bodies.block_id_to_obj_id[block_id] for block_id in state._block_id_to_block],
                held_obj_id = held_obj_id,
                physics_client_id = physics_client_id,
            )
            return collision and not (block_too_close_to_bounds and block_too_close_to_bounds_ok)

        def _distance_fn(from_pt: ScaleMotionPlanningState, to_pt: ScaleMotionPlanningState) -> float:
            from_joints, _, _ = from_pt
            to_joints, _, _ = to_pt
            return ((from_joints - to_joints) ** 2).sum()

        joint_positions_list, _, _ = zip(*BiRRT(
            _sample_fn,
            _extend_fn,
            _collision_fn,
            _distance_fn,
            np.random.default_rng(CFG.seed),
            num_attempts=CFG.pybullet_birrt_num_attempts,
            num_iters=CFG.pybullet_birrt_num_iters,
            smooth_amt=CFG.pybullet_birrt_smooth_amt
        ).query((np.array(state.joint_positions), True, True), (np.array(target_joint_positions), False, False)))

        keep_joints = [True] + [not np.allclose(np.subtract(next, curr), np.subtract(curr, prev)) for prev, curr, next in zip(joint_positions_list[:-2], joint_positions_list[1:-1], joint_positions_list[2:])] + [True]
        filtered_joint_positions, _ = zip(*filter(lambda x: x[1], zip(joint_positions_list, keep_joints)))
        return filtered_joint_positions

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle packing-specific resetting."""
        state = cast(PyBulletScaleState, state)

        # Set the state information
        self._block_id_to_block = state._block_id_to_block
        self._placement_offsets = state._placement_offsets
        self._scale_checked = state.get(self._scale, "checked") > self.scale_checked_thresh

        # Remove the old grasp constraint
        self._remove_grasp_constraint()

        # Reset pybullet
        self._reset_pybullet(
            state,
            self._pybullet_robot,
            self._block_id_to_obj_id,
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
        state: PyBulletScaleState,
        pybullet_robot: SingleArmPyBulletRobot,
        block_id_to_obj_id: Dict[BlockId, int],
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

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_id_to_block, self._box_id and self._placement_offsets. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        data = {}
        state = PyBulletScaleState(
            data, self._pybullet_robot.get_joints(),
            self._block_id_to_block,
            self._placement_offsets,
        )

        # Getting the scale state
        state.set(self._scale, "checked", self._scale_checked)

        # Getting the robot state
        rx, ry, rz, rqx, rqy, rqz, rqw, rf = self._pybullet_robot.get_state()
        data[self._robot] = np.array([rx, ry, rz, rqx, rqy, rqz, rqw, self.fingers_joint_to_state(rf, self._pybullet_robot)])

        # Getting the blocks states
        for block_id, block in self._block_id_to_block.items():
            block_pose = p.getBasePositionAndOrientation(
                self._block_id_to_obj_id[block_id],
                physicsClientId=self._physics_client_id
            )
            data[block] = np.array(list(itertools.chain(
                block_pose[0], self.blocks_ids_to_dimensions[block_id],
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

    @classmethod
    def check_collisions(
        cls,
        robot: PandaPyBulletRobot,
        bounds_obj_id: int,
        scale_obj_ids: Tuple[int, int],
        block_obj_ids: Iterable[int],
        held_obj_id: Optional[int],
        physics_client_id: int,
    ) -> Tuple[bool, bool]:
        block_obj_ids = set(block_obj_ids)
        if robot.check_self_collision(robot.get_joints()):
            logging.info("ROBOT SELF COLLISION")
            return True, False

        if held_obj_id is not None:
            block_obj_ids.discard(held_obj_id)

        collisions = cls._check_collisions_single_mesh(
            robot.robot_id, bounds_obj_id, scale_obj_ids, block_obj_ids, physics_client_id, exclude_bounds_base_collision_check=True
        )
        for obj_id, distance in collisions:
            if obj_id == bounds_obj_id:
                logging.info(f"ROBOT COLLISION WITH BOUNDS AT DISTANCE {distance}")
            elif obj_id in scale_obj_ids:
                logging.info(f"ROBOT COLLISION WITH {'LEFT' if obj_id == scale_obj_ids[0] else 'RIGHT'} SCALE AT DISTANCE {distance}")
            else:
                logging.info(f"ROBOT COLLISION WITH BLOCK WITH ID {obj_id} AT DISTANCE {distance}")
        if collisions:
            return True, False

        if held_obj_id is not None:
            collisions = cls._check_collisions_single_mesh(held_obj_id, bounds_obj_id, scale_obj_ids, block_obj_ids, physics_client_id)
            held_obj_too_close_to_bounds = True
            for obj_id, distance in collisions:
                if obj_id == bounds_obj_id:
                    logging.info(f"BLOCK COLLISION WITH BOUNDS AT DISTANCE {distance}")
                elif obj_id in scale_obj_ids:
                    logging.info(f"BLOCK COLLISION WITH {'LEFT' if obj_id == scale_obj_ids[0] else 'RIGHT'} SCALE AT DISTANCE {distance}")
                else:
                    logging.info(f"BLOCK COLLISION WITH BLOCK WITH ID {obj_id} AT DISTANCE {distance}")
                held_obj_too_close_to_bounds &= obj_id == bounds_obj_id and distance > 0.0
            if collisions:
                return True, held_obj_too_close_to_bounds
        return False, False

    @classmethod
    def _check_collisions_single_mesh(
        cls,
        obj_id: int,
        bounds_obj_id: int,
        scale_obj_ids: Tuple[int, int],
        block_obj_ids: Iterable[int],
        physics_client_id: int,
        exclude_bounds_base_collision_check: bool = False
    ) -> List[Tuple[int, float]]:
        collisions = []
        collisions += cls._check_collisions_two_meshes(
            obj_id, bounds_obj_id, cls.bounds_min_distance, physics_client_id, base_collision = not exclude_bounds_base_collision_check
        )
        for scale_obj_id in scale_obj_ids:
            collisions += cls._check_collisions_two_meshes(obj_id, scale_obj_id, cls.scale_min_distance, physics_client_id)
        for block_obj_id in block_obj_ids:
            collisions += cls._check_collisions_two_meshes(obj_id, block_obj_id, cls.block_min_distance, physics_client_id)
        return collisions

    @classmethod
    def _check_collisions_two_meshes(
        cls,
        obj_id: int,
        other_obj_id: int,
        min_distance: float,
        physics_client_id: int,
        base_collision: bool = True,
    ) -> List[Tuple[int, float]]:
        if obj_id == other_obj_id:
            return []
        closest_points = p.getClosestPoints(obj_id, other_obj_id, min_distance, physicsClientId=physics_client_id)
        if not base_collision:
            closest_points = [p for p in closest_points if p[3] != -1 or p[4] != -1]
        if closest_points:
            return [(other_obj_id, min(closest_point[8] for closest_point in closest_points))]
        return []

    @classmethod
    def fingers_state_to_joint(
        cls,
        pybullet_robot: SingleArmPyBulletRobot,
        fingers_state: float,
    ) -> float:
        """Convert the fingers in the given State to joint values for PyBullet.

        The fingers in the State are either 0 or 1. Transform them to be
        either pybullet_robot.closed_fingers or
        pybullet_robot.open_fingers.
        """
        assert fingers_state in (0.0, 1.0)
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
        return closed_f if fingers_state == 0.0 else open_f

    @classmethod
    def fingers_joint_to_state(
        cls,
        fingers_joint: float,
        pybullet_robot: SingleArmPyBulletRobot,
    ) -> float:
        """Convert the finger joint values in PyBullet to values for the State.
        """
        open_f = pybullet_robot.open_fingers
        closed_f = pybullet_robot.closed_fingers
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

    @classmethod
    def _check_block_on_scale(cls, state: State, block: Object, left: bool) -> bool:
        assert block.is_instance(cls._block_type)
        block_position = state[block][0:3]
        block_height = state.get(block, 'h')
        scale_position = (cls.scale_poses[0] if left else cls.scale_poses[1]).position
        return (np.abs(np.subtract(block_position[:2], scale_position[:2])) * 2 <= cls.scale_size[:2]).all() and \
            scale_position[2] + cls.scale_size[2] <= block_position[2] - block_height/2 # type: ignore

