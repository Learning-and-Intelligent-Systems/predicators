from dataclasses import dataclass, field
import functools
import itertools
import logging
from typing import Any, ClassVar, Dict, Iterator, List, Optional, Sequence, Set, Tuple, cast

from predicators.envs.pybullet_env import PyBulletEnv, create_pybullet_block
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion, matrix_from_quat, multiply_poses
from predicators.pybullet_helpers.joint import JointPositions
from predicators.pybullet_helpers.link import get_link_state
from predicators.pybullet_helpers.robots import create_single_arm_pybullet_robot
from predicators.pybullet_helpers.robots.panda import PandaPyBulletRobot
from predicators.pybullet_helpers.robots.single_arm import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, Object, Predicate, State, Type

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

@dataclass
class PyBulletPackingState(PyBulletState):
    _block_to_block_id: Dict[Object, int] = field(default_factory=dict)
    _block_id_to_block: Dict[int, Object] = field(default_factory=dict)
    _box_id: int = -10000
    _placement_offsets: List[Tuple[float, float]] = field(default_factory=list)

    def copy(self) -> State:
        state_dict_copy = super().copy().data
        simulator_state_copy = list(self.joint_positions)
        return PyBulletPackingState(
            state_dict_copy, simulator_state_copy,
            self._block_to_block_id, self._block_id_to_block,
            self._box_id, self._placement_offsets
        )

class PyBulletPackingEnv(PyBulletEnv):
    # Multiprocessing stuff


    # Settings
    ## Task Generation
    range_train_box_cols: ClassVar[Tuple[int, int]] = (2, 4)
    num_box_rows: ClassVar[int] = 2
    box_front_margin = 0.20
    block_side_margin = 0.08
    block_vert_offset = 0.0001

    num_tries: ClassVar[int] = 100000

    ## World Shape Parameters
    values_range: ClassVar[Tuple[float, float]] = (-10, 10)

    hiding_pose: ClassVar[Pose] = Pose((0.0, 0.0, -10.0), PyBulletEnv._default_orn)

    block_sizes: ClassVar[List[Tuple[float, float, float]]] = list(itertools.product(
        [0.07, 0.06, 0.05],
        [0.07, 0.06, 0.05],
        [0.13, 0.14, 0.15]
    ))
    box_color: ClassVar[Tuple[float, float, float, float]] = (0.1, 0.1, 0.1, 0.7)
    box_height: ClassVar[float] = 0.4
    box_col_margins: ClassVar[float] = 0.05
    box_row_margins: ClassVar[float] = 0.03
    gripper_width: ClassVar[float] = 0.19
    max_finger_width: ClassVar[float] = 0.14

    robot_base_pos: ClassVar[Pose3D] = (-0.0716, .0, .0)
    robot_ee_init_pose: ClassVar[Pose] = Pose((0.3, .0, 0.3), (.0, 1.0, .0, .0))
    robot_finger_normals: ClassVar[Tuple[Pose3D, Pose3D]] = ((0, -1, 0), (0, 1, 0))

    debug_space_height: ClassVar[float] = 1.21
    debug_space_width: ClassVar[float] = 1.4
    debug_space_size: ClassVar[float] = 0.5

    block_poses: ClassVar[Polygon] = wkt.load(open(get_asset_path("block-poses.wkt")))
    box_poses: ClassVar[Polygon] = wkt.load(open(get_asset_path("box-poses.wkt")))

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
        return not PyBulletPackingEnv._InBox_holds(state, [block, PyBulletPackingEnv._box])

    _OnTable: ClassVar[Predicate] = Predicate("OnTable", [_block_type], _OnTable_holds)

    # Common Objects
    _robot: ClassVar[Object] = Object("robot", _robot_type)
    _box: ClassVar[Object] = Object("box", _box_type)

    def __init__(self, use_gui: bool=True):
        super().__init__(use_gui)

        self._block_to_block_id: Dict[Object, int] = {}
        self._block_id_to_block: Dict[int, Object] = {}
        self._placement_offsets = []
        self._box_id: int = -10000

        self._pid = os.getpid() # For detecting multiprocessing and when we potentially need to reinitialize pybullet

        self._robot_initial_joints = self._pybullet_robot.inverse_kinematics(
            end_effector_pose=self.robot_ee_init_pose,
            validate=True, set_joints=False
        )

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
        block_to_block_id = {
            Object(f"block{idx}", self._block_type): block_id
            for idx, block_id in enumerate(rng.choice(list(self._block_id_to_dims.keys()), num_box_cols * self.num_box_rows, replace=False))
        }
        block_id_to_block = {block_id: block for block, block_id in block_to_block_id.items()}
        box_id = self._box_ids[num_box_cols]

        # Setting up placement offsets
        box_d, box_w, box_h = self._box_id_to_dims[box_id]

        placement_indices = np.arange(num_box_cols * self.num_box_rows).reshape(self.num_box_rows, num_box_cols)
        for row in range(self.num_box_rows):
            rng.shuffle(placement_indices[row])
        placement_indices = placement_indices.flatten()

        placement_coords = -np.vstack([
            placement_indices // num_box_cols - (self.num_box_rows - 1) / 2,
            placement_indices % num_box_cols - (num_box_cols - 1) / 2
        ]).T

        y_sep, _, x_sep, _ = self._box_creation_params()
        placement_offsets = list(map(tuple, placement_coords * [x_sep, y_sep]))

        # Creating the state and goal
        data: Dict[Object, npt.NDArray] = {}
        state = PyBulletPackingState(data, None, block_to_block_id, block_id_to_block, box_id, placement_offsets)
        goal = {self._InBox([block, self._box]) for block in block_to_block_id.keys()}

        # Setting up the robot and joint positions
        data[self._robot] = np.array(list(itertools.chain(
            self.robot_ee_init_pose.position,
            self.robot_ee_init_pose.orientation,
            [1.0]
        )))
        state.simulator_state = self._robot_initial_joints
        # Setting up the box
        box_poly = box(-box_d/2, -box_w/2, box_d/2, box_w/2)
        box_x_min, box_y_min, box_x_max, box_y_max = self.box_poses.bounds
        for _ in range(self.num_tries):
            box_x, box_y = rng.uniform([box_x_min, box_y_min], [box_x_max, box_y_max])
            box_x = box_x_max - box_d/2 - 0.0001
            box_y = 0
            if self.box_poses.contains(translate(box_poly, box_x, box_y)):
                break
        else:
            raise ValueError('Could not generate a task with given settings')

        data[self._box] = np.array(list(itertools.chain(
            (box_x, box_y, self.box_height / 2), [box_d, box_w, box_h],
        )))
        collision_poly = collision_poly.union(
            translate(box(-box_d/2 - self.box_front_margin, -box_w/2, box_d/2, box_w/2), box_x, box_y)
        )
        collision_poly = collision_poly.union(
            translate(box(-box_d/2, box_y_min, box_d/2, box_y_max), box_x, box_y)
        )

        # Setting up the blocks
        collision_poly = collision_poly.buffer(self.block_side_margin/2, join_style='mitre')
        block_poses = self.block_poses.buffer(-self.block_side_margin/2, join_style='mitre')

        block_x_min, block_y_min, block_x_max, block_y_max = self.block_poses.bounds
        for block, block_id in block_to_block_id.items():
            block_d, block_w, block_h = self._block_id_to_dims[block_id]
            for _ in range(self.num_tries):
                block_x_vertical = rng.choice([True, False])
                block_axis_up = rng.choice([True, False])
                block_x, block_y, block_rot = rng.uniform(
                    [block_x_min, block_y_min, -np.pi],
                    [block_x_max, block_y_max, np.pi]
                )

                block_w_2d = block_w if block_x_vertical else block_d
                block_poly = translate(rotate(
                    box(-block_h/2, -block_w_2d/2 - self.block_side_margin/2, block_h/2, block_w_2d/2 + self.block_side_margin/2),
                block_rot, use_radians=True), block_x, block_y)
                if block_poses.contains(block_poly) and not collision_poly.intersects(block_poly):
                    break
            else:
                raise ValueError('Could not generate a task with given settings')

            block_quat = multiply_poses(
                Pose((0, 0, 0), p.getQuaternionFromEuler([0, 0, block_rot])),
                Pose((0, 0, 0), p.getQuaternionFromEuler([0 if block_axis_up else np.pi, 0, 0])),
                Pose((0, 0, 0), p.getQuaternionFromEuler([0 if block_x_vertical else np.pi/2, 0, 0])),
                Pose((0, 0, 0), p.getQuaternionFromEuler([0, np.pi/2, 0]))
            ).orientation
            block_up_size = block_d if block_x_vertical else block_w
            data[block] = np.array(list(itertools.chain(
                    # PyBullet sometimes bugs out if the blocks are on the same height
                [block_x, block_y, block_up_size/2 + self.block_vert_offset + rng.uniform(-1e-6, 1e-6)],
                [block_d, block_w, block_h],
                block_quat,
                [0],
            )))
            collision_poly = collision_poly.union(block_poly)
        return EnvironmentTask(state, goal)

    @classmethod
    @functools.cache
    def initialize_pybullet( # type: ignore
            cls, using_gui: bool
    ) -> Tuple[int, SingleArmPyBulletRobot, Dict[str, Any]]:
        """Run super(), then handle packing-specific initialization."""
        logging.info(f"INITIALIZE PYBULLET FOR PID {os.getpid()}")
        physics_client_id, pybullet_robot, bodies = super(
        ).initialize_pybullet(using_gui)

        if CFG.pybullet_draw_debug:  # pragma: no cover
            assert using_gui, \
                "using_gui must be True to use pybullet_draw_debug."
            cls._draw_debug_pybullet(physics_client_id)

        # Creating the bounds of the world
        bodies["bounds"] = p.loadURDF(get_asset_path("bounds.urdf"), useFixedBase=True, physicsClientId=physics_client_id)
        p.setCollisionFilterPair(pybullet_robot.robot_id, bodies["bounds"], -1, -1, False, physicsClientId=physics_client_id)

        # Creating the box to put the things into
        max_box_cols = max(*cls.range_train_box_cols, CFG.pybullet_packing_test_num_box_cols)
        bodies["boxes"] = [(-10000, (0, 0, 0))] + [cls._create_box(width, physics_client_id) for width in reversed(range(max_box_cols, 0, -1))]

        assert max_box_cols * cls.num_box_rows <= len(cls.block_sizes)
        bodies["blocks"] = dict(cls._create_block(idx, physics_client_id) for idx in range(len(cls.block_sizes)))
        return physics_client_id, pybullet_robot, bodies

    def _store_pybullet_bodies(self, pybullet_bodies: Dict[str, Any]) -> None:
        self._bounds_id: int = pybullet_bodies["bounds"]
        self._box_ids: List[int] = list(map(lambda kv: kv[0], pybullet_bodies["boxes"]))
        self._box_id_to_dims: Dict[int, Tuple[float, float, float]] = dict(pybullet_bodies["boxes"])
        self._block_id_to_dims: Dict[int, Tuple[float, float, float]] = pybullet_bodies["blocks"]

    @classmethod
    def _create_pybullet_robot(cls, physics_client_id: int) -> SingleArmPyBulletRobot:
        return PandaPyBulletRobot(cls.robot_ee_init_pose, physics_client_id, Pose(cls.robot_base_pos, cls._default_orn))

    def _extract_robot_state(self, state: State) -> Array:
        f = self.fingers_state_to_joint(self._pybullet_robot,
                                        state.get(self._robot, "fingers"))
        return np.array([
            state.get(self._robot, "x"),
            state.get(self._robot, "y"),
            state.get(self._robot, "z"),
            state.get(self._robot, "qx"),
            state.get(self._robot, "qy"),
            state.get(self._robot, "qz"),
            state.get(self._robot, "qw"),
            f,
        ], dtype=np.float32)

    def simulate(self, state: State, action: Action) -> State:
        """Additionally check for collisions"""
        logging.info("SIMULATE")
        if self._pid != os.getpid():
            self._restart_pybullet()
            self._pid = os.getpid()

        self._current_observation = state
        self._reset_state(state)
        next_state = self.step(action)
        if CFG.pybullet_control_mode == "reset" and self.check_collisions(itertools.chain(
            (obj_id for obj in state.get_objects(self._block_type) for obj_id in [self._block_to_block_id[obj]]),
            [self._bounds_id, self._box_id]
        )):
            return state
        return next_state

    def _reset_state(self, state: State) -> None:
        """Run super(), then handle packing-specific resetting."""
        state = cast(PyBulletPackingState, state)
        super()._reset_state(state)
        self._block_id_to_block = state._block_id_to_block
        self._block_to_block_id = state._block_to_block_id
        self._box_id = state._box_id
        self._placement_offsets = state._placement_offsets

        held_block = False
        for block_id in self._block_id_to_dims.keys():
            block = state._block_id_to_block.get(block_id, None)
            if block is None:
                self._hide_object(block_id)
                continue
            p.resetBasePositionAndOrientation(
                block_id,
                state[block][:3],
                state[block][6:10],
                physicsClientId=self._physics_client_id,
            )

        for block_id, block in state._block_id_to_block.items():
            if state.get(block, 'held') >= self.held_thresh:
                assert not held_block
                self._force_grasp_object(block)
                held_block = True

        for box_id in self._box_ids:
            if box_id != state._box_id:
                self._hide_object(box_id)
                continue
            p.resetBasePositionAndOrientation(
                box_id,
                state[self._box][:3],
                self._default_orn,
                physicsClientId=self._physics_client_id,
            )
        new_state = self._get_state()
        assert State({k: np.abs(v) for k, v in new_state.data.items()}).allclose(State({k: np.abs(v) for k, v in state.data.items()}))

    def _get_state(self) -> State:
        """Create a State based on the current PyBullet state.

        Note that in addition to the state inside PyBullet itself, this
        uses self._block_to_block_id, self._block_id_to_block and self._box_id. As long as
        the PyBullet internal state is only modified through reset() and
        step(), these all should remain in sync.
        """
        data = {}
        state = PyBulletPackingState(
            data, self._pybullet_robot.get_joints(),
            self._block_to_block_id, self._block_id_to_block,
            self._box_id, self._placement_offsets,
        )

        # Getting the robot state
        rx, ry, rz, rqx, rqy, rqz, rqw, rf = self._pybullet_robot.get_state()
        data[self._robot] = np.array([rx, ry, rz, rqx, rqy, rqz, rqw, self._fingers_joint_to_state(rf)])

        # Getting the box state
        box_position = p.getBasePositionAndOrientation(self._box_id, physicsClientId=self._physics_client_id)[0]
        data[self._box] = np.array(list(itertools.chain(
            box_position, self._box_id_to_dims[self._box_id]
        )))

        # Getting the blocks states
        for block, block_id in self._block_to_block_id.items():
            block_pose = p.getBasePositionAndOrientation(block_id, physicsClientId=self._physics_client_id)
            data[block] = np.array(list(itertools.chain(
                block_pose[0], self._block_id_to_dims[block_id],
                block_pose[1], [0.0]
            )))
        if self._held_obj_id is not None:
            state.set(self._block_id_to_block[self._held_obj_id], "held", 1.0)
        return state

    def _get_object_ids_for_held_check(self) -> List[int]:
        """Return a list of pybullet IDs corresponding to objects in the
        simulator that should be checked when determining whether one is
        held."""
        return list(self._block_id_to_block.keys())

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
        logging.info(f"RESTARTING PYBULLET FOR PID {os.getpid()}")
        self._physics_client_id, self._pybullet_robot, pybullet_bodies = \
            self.initialize_pybullet(False)
        self._store_pybullet_bodies(pybullet_bodies)

    def _force_grasp_object(self, block: Object) -> None:
        block_id = self._block_to_block_id[block]
        # The block should already be held. Otherwise, the position of the
        # block was wrong in the state.
        held_obj_id = self._detect_held_object()
        logging.info((block_id, held_obj_id))
        if block_id != held_obj_id:
            time.sleep(3000)
        assert block_id == held_obj_id
        # Create the grasp constraint.
        self._held_obj_id = block_id
        self._create_grasp_constraint()

    def _get_finger_rot_matrix(self, finger_id: int) -> npt.NDArray[np.float32]:
        finger_state = get_link_state(self._pybullet_robot.robot_id, finger_id, self._physics_client_id)
        return np.array(p.getMatrixFromQuaternion(finger_state.linkWorldOrientation)).reshape(3, 3)

    def check_collisions(self, obj_ids_iter: Iterator[int]) -> bool:
        p.performCollisionDetection(physicsClientId=self._physics_client_id)
        obj_ids = list(obj_ids_iter)
        for idx, obj_id in enumerate(obj_ids):
            if obj_id == self._held_obj_id:
                continue
            if p.getContactPoints(self._pybullet_robot.robot_id, obj_id, physicsClientId=self._physics_client_id):
                p1, p2 = p.getContactPoints(self._pybullet_robot.robot_id, obj_id, physicsClientId=self._physics_client_id)[0][5:7]
                # p.addUserDebugLine(p1, p2,
                #     [1.0, 0.0, 0.0],
                #     lineWidth=5.0,
                #     physicsClientId=self._physics_client_id)
                logging.info(f"ROBOT COLLISION {p.getBodyInfo(obj_id, physicsClientId=self._physics_client_id)} {idx}/{len(obj_ids)}")
                if obj_id in self._block_id_to_block:
                    logging.info(f"BLOCK {self._block_id_to_block[obj_id]}")
                elif obj_id == self._box_id:
                    logging.info("BOX")
                else:
                    logging.info("BOUNDS")
                return True
            if self._held_obj_id is not None and p.getContactPoints(self._held_obj_id, obj_id, physicsClientId=self._physics_client_id):
                p1, p2 = p.getContactPoints(self._held_obj_id, obj_id, physicsClientId=self._physics_client_id)[0][5:7]
                # p.addUserDebugLine(p1, p2,
                #     [1.0, 0.0, 0.0],
                #     lineWidth=5.0,
                #     physicsClientId=self._physics_client_id)
                logging.info(f"BLOCK COLLISION {p.getBodyInfo(obj_id, physicsClientId=self._physics_client_id)} {idx}/{len(obj_ids)}")
                if obj_id in self._block_id_to_block:
                    logging.info(f"BLOCK {self._block_id_to_block[obj_id]}")
                elif obj_id == self._box_id:
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
    def _draw_debug_pybullet(cls, physics_client_id: int):
        # Draw the workspace on the table for clarity.
        p.addUserDebugLine([-cls.debug_space_width/2, 0, cls.debug_space_height],
                            [cls.debug_space_width/2, 0, cls.debug_space_height],
                            [1.0, 0.0, 0.0],
                            lineWidth=5.0,
                            physicsClientId=physics_client_id)
        p.addUserDebugLine([-cls.debug_space_width/2, cls.debug_space_size, cls.debug_space_height],
                            [cls.debug_space_width/2, cls.debug_space_size, cls.debug_space_height],
                            [1.0, 0.0, 0.0],
                            lineWidth=5.0,
                            physicsClientId=physics_client_id)
        p.addUserDebugLine([-cls.debug_space_width/2, cls.debug_space_size * 2, cls.debug_space_height],
                            [cls.debug_space_width/2, cls.debug_space_size * 2, cls.debug_space_height],
                            [1.0, 0.0, 0.0],
                            lineWidth=5.0,
                            physicsClientId=physics_client_id)
        p.addUserDebugLine([-cls.debug_space_width/2, cls.debug_space_size * 3, cls.debug_space_height],
                            [cls.debug_space_width/2, cls.debug_space_size * 3, cls.debug_space_height],
                            [1.0, 0.0, 0.0],
                            lineWidth=5.0,
                            physicsClientId=physics_client_id)
        # Draw coordinate frame labels for reference.
        p.addUserDebugText("x", [0.25, 0, 0], [0.0, 0.0, 0.0],
                            physicsClientId=physics_client_id)
        p.addUserDebugText("y", [0, 0.25, 0], [0.0, 0.0, 0.0],
                            physicsClientId=physics_client_id)
        p.addUserDebugText("z", [0, 0, 0.25], [0.0, 0.0, 0.0],
                            physicsClientId=physics_client_id)

    def _hide_object(self, object_id: int) -> None:
        p.resetBasePositionAndOrientation(object_id, self.hiding_pose.position, self.hiding_pose.orientation, physicsClientId=self._physics_client_id)

    @classmethod
    def _create_block(cls, idx: int, physics_client_id: int) -> Tuple[int, Tuple[float, float, float]]:
        color = cls._obj_colors[idx % len(cls._obj_colors)]
        half_extents = (
            cls.block_sizes[idx][0] / 2,
            cls.block_sizes[idx][1] / 2,
            cls.block_sizes[idx][2] / 2,
        )
        return create_pybullet_block(
            color = color,
            half_extents = half_extents,
            mass = np.prod(cls.block_sizes[idx], dtype=float) * cls._obj_mass,
            friction = cls._obj_friction,
            orientation = cls._default_orn,
            physics_client_id = physics_client_id
        ), cls.block_sizes[idx]

    @classmethod
    def _box_creation_params(cls) -> Tuple[float, float, float, float]:
        max_block_width = max(size for sizes in cls.block_sizes for size in sizes[:2])

        block_width_sep = max(max_block_width/2, cls.max_finger_width/2) + cls.box_col_margins + max_block_width/2
        block_width_margin = cls.gripper_width / 2 + cls.box_col_margins

        block_depth_sep = max_block_width + cls.box_row_margins
        block_depth_margin = max_block_width / 2 + cls.box_row_margins

        return block_width_sep, block_width_margin, block_depth_sep, block_depth_margin

    @classmethod
    def _create_box(cls, num_box_cols: int, physics_client_id: int) -> Tuple[int, Tuple[float, float, float]]:
        block_width_sep, block_width_margin, block_depth_sep, block_depth_margin = cls._box_creation_params()

        box_width = block_width_sep * (num_box_cols - 1) + block_width_margin * 2
        box_depth = block_depth_sep * (cls.num_box_rows - 1) + block_depth_margin * 2
        return create_pybullet_box(
            cls.box_color,
            (box_depth, box_width, cls.box_height),
            0, 1, cls._default_orn,
            physics_client_id,
        ), (box_depth, box_width, cls.box_height)

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