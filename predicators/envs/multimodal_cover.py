"""Blocks domain.

This environment IS downward refinable and DOESN'T require any
backtracking (as long as all the blocks can fit comfortably on the
table, which is true here because the block size and number of blocks
are much less than the table dimensions). The simplicity of this
environment makes it a good testbed for predicate invention.
"""
import itertools
import json
import logging
import os
import numpy.typing as npt
from pathlib import Path
from typing import ClassVar, Collection, Dict, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtGui import qPixelFormatYuv
from gym.spaces import Box
from matplotlib import patches
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.affinity import translate, rotate

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion, matrix_from_quat, multiply_poses
from pybullet_utils.transformations import quaternion_from_euler, euler_from_quaternion
from scipy.spatial.transform import Rotation as R
import pickle as pkl


def get_asset_path(filename: str) -> str:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', filename)
    assert os.path.exists(path), f"Asset {path} does not exist"
    return path


def get_tagged_block_sizes() -> List[Tuple[float, float, float]]:
    tags_path = get_asset_path('block_tags')
    return [
        dimensions
        for block_info_fname in os.listdir(tags_path)
        for d, w, h in [sorted(pkl.load(open(os.path.join(tags_path, block_info_fname), 'rb'))['dimensions'])]
        for dimensions in [(d, w, h), (w, d, h)]
    ]


def get_zone_data() -> any:
    f_name = get_asset_path('zone_data.pkl')
    with (f_name, 'rb') as f:
        return pkl.load(f)


def compute_aabb(corners):
    """
    Compute the Axis-Aligned Bounding Box (AABB) for a set of corners.
    """
    corners = np.array(corners)
    min_coords = corners.min(axis=0)
    max_coords = corners.max(axis=0)
    return min_coords, max_coords  # (min_x, min_y, min_z), (max_x, max_y, max_z)


def is_collision(corners1, corners2):
    """
    Check for collision between two blocks using AABB.
    """
    min1, max1 = compute_aabb(corners1)
    min2, max2 = compute_aabb(corners2)

    # Check for overlap along all axes
    overlap_x = min1[0] <= max2[0] and max1[0] >= min2[0]
    overlap_y = min1[1] <= max2[1] and max1[1] >= min2[1]
    overlap_z = min1[2] <= max2[2] and max1[2] >= min2[2]

    return overlap_x and overlap_y and overlap_z


class MultiModalCoverEnv(BaseEnv):
    """Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py
    table_height: ClassVar[float] = 0.2
    # The table x bounds are (1.1, 1.6), but the workspace is smaller.
    # Make it narrow enough that blocks can be only horizontally arranged.
    # Note that these boundaries are for the block positions, and that a
    # block's origin is its center, so the block itself may extend beyond
    # the boundaries while the origin remains in bounds.

    # The table x bounds are (1.1, 1.6),
    x_lb: ClassVar[float] = 1.17
    x_ub: ClassVar[float] = 1.505
    # The table y bounds are (0.3, 1.2)
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1

    num_tries: ClassVar[int] = 100000
    pick_z: ClassVar[float] = 0.5
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = pick_z


    held_tol: ClassVar[float] = 0.5
    pick_tol: ClassVar[float] = 0.0001
    on_tol: ClassVar[float] = 0.01
    collision_padding: ClassVar[float] = 2.0
    gripper_max_depth = 0.03

    box_front_margin = 0.25
    block_side_margin = 0.1
    block_vert_offset = 0.001
    block_min_distance: ClassVar[float] = 0.003

    zone_extents: ClassVar[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = [((1.175, 0.44), (1.33, 0.70)),
                                                                                     ((1.335, 0.44), (1.490, 0.70)),
                                                                                     ((1.175, 0.80), (1.33, 1.06)),
                                                                                     ((1.335, 0.80), (1.490, 1.06))]

    _target_height: ClassVar[float] = 0.0001
    goal_zone = 0
    zone_order = []

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types

        if CFG.multi_modal_cover_real_robot:
            self._block_type = Type("block", [
                "pose_x", "pose_y", "pose_z", "depth", "width", "height", "qx", "qy", "qz", "qw", "held"
            ])
        else:
            self._block_type = Type("block", [
                "pose_x", "pose_y", "pose_z", "depth", "width", "height", "held"
            ])

        self._zone_type = Type("zone", [
            "lower_extent_x", "lower_extent_y", "upper_extent_x", "upper_extent_y"
        ])

        self._dummy_zone_goal_type = Type("dummy_zone_goal", ["zone_idx"])

        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "orn_x", "orn_y", "orn_z", "orn_w", "fingers"])
        # Predicates
        self._On = Predicate("On", [self._block_type, self._block_type],
                             self._On_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)

        self._OnZone = Predicate("OnZone", [self._block_type, self._dummy_zone_goal_type],
                                 self._OnZone_holds)

        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        # Hyperparameters from CFG.
        self._block_size = CFG.blocks_block_size
        self._num_blocks_train = CFG.mcover_num_blocks_train
        self._num_blocks_test = CFG.mcover_num_blocks_test

    @classmethod
    def get_name(cls) -> str:
        return "multimodal_cover"

    @classmethod
    def in_zone(cls, points: List[Tuple[float, float]], extents: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                specific_extents: List[int] = None):




        if specific_extents is None:
            for extent in extents:
                points_inside = True
                for point in points:
                    if not (extent[0][0] < point[0] < extent[1][0] and extent[0][1] < point[1] < extent[1][1]):
                        points_inside = False
                        break

                if points_inside:
                    return True

        else:
            for extent_idx in specific_extents:
                points_inside = True
                for point in points:
                    if not(extents[extent_idx][0][0] < point[0] < extents[extent_idx][1][0] and
                           extents[extent_idx][0][1] < point[1] < extents[extent_idx][1][1]):
                        points_inside = False
                        break

                if points_inside:
                    return True

        return False

    @classmethod
    def _get_block_corners(cls, state: State, block: Object, x_padding=0.0, y_padding=0.0) -> npt.NDArray[np.float32]:

        pos = np.array([state.get(block, "pose_x"), state.get(block, "pose_y"), state.get(block, "pose_z")])
        dims = np.array([state.get(block, "depth")+x_padding,
                         state.get(block, "width")+y_padding,
                         state.get(block, "height")])

        if CFG.multi_modal_cover_real_robot:
            quaternion = state.get(block, "qx"), state.get(block, "qy"), state.get(block, "qz"), state.get(block, "qw")
        else:
            quaternion = (0,0,0,1)
        relative_corners = matrix_from_quat(quaternion) @ np.vstack([
            dims / 2 * mult for mult in itertools.product(*([[-1, 1]] * 3))
        ]).T

        return pos + relative_corners.T

    @classmethod
    def get_rotated_height(cls,
                               original_dimensions: Tuple[float, float, float],
                               quaternion: Tuple[float, float, float, float]):

        d, w, h = original_dimensions
        qx, qy, qz, qw = quaternion

        quaternion = np.array([qx, qy, qz, qw])

        logging.info(f"block rotation: {euler_from_quaternion(quaternion)}")

        rotation = R.from_quat(quaternion)

        depth_vector = np.array([d, 0, 0])
        width_vector = np.array([0, w, 0])
        height_vector = np.array([0, 0, h])

        rotated_depth = rotation.apply(depth_vector)
        rotated_width = rotation.apply(width_vector)
        rotated_height = rotation.apply(height_vector)

        effective_height = abs(rotated_depth[2]) + abs(rotated_width[2]) + abs(rotated_height[2])

        return effective_height

    def _is_block_collision(self, state, block, blocks):
        block_corners = self._get_block_corners(state, block,  x_padding=0.052, y_padding=0000.02)

        for b in blocks:
            b_corners = self._get_block_corners(state, b)

            if is_collision(block_corners, b_corners):
                return True

        return False

    def simulate(self, state: State, action: Action) -> State:

        if len(action.arr) == 5:
            x, y, z, theta, fingers = action.arr
            assert self.action_space.contains([x,y,z,fingers])
        else:
            assert self.action_space.contains(action.arr)
            x, y, z, fingers = action.arr
            theta = 0
        logging.info("transition")
        # Infer which transition function to follow
        if fingers < 0.5:
            return self._transition_pick(state, x, y, z)
        return self._transition_putontable(state, x, y, z, theta)

    def _transition_pick(self, state: State, x: float, y: float,
                         z: float) -> State:
        logging.info("Transition Pick")
        next_state = state.copy()
        # Can only pick if fingers are open
        if not self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_block_at_xyz(state, x, y, z)
        if block is None:  # no block at this pose
            return next_state
        # Can only pick if object is clear
        if not self._block_is_clear(block, state):
            return next_state

        bx = state.get(block, "pose_x")
        by = state.get(block, "pose_y")
        bz = state.get(block, "pose_z")

        h = state.get(block, "height")

        if CFG.multi_modal_cover_real_robot:
            qx = state.get(block, "qx")
            qy = state.get(block, "qy")
            qz = state.get(block, "qz")
            qw = state.get(block, "qw")

            h =  self.get_rotated_height((bx, by, bz), (qx, qy, qz, qw))


        top_z = self.table_height + h

        grab_distance = self.pick_z - (top_z - self.gripper_max_depth)

        logging.info(f"transition pick state: {state}")

        # Execute pick
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", bz + grab_distance - 0.01)
        next_state.set(block, "held", 1.0)
        next_state.set(self._robot, "fingers", 0.0)  # close fingers
        logging.info(f"transition next- pick state: {next_state}")

        return next_state

    def _transition_putontable(self, state: State, x: float, y: float,
                               z: float, theta: float) -> State:
        next_state = state.copy()
        trial_state = state.copy()
        logging.info("Transition PutOnTable")
        # Can only putontable if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state

        block = self._get_held_block(state)


        assert block is not None

        if CFG.multi_modal_cover_real_robot:
            block_quat = np.array([
                float(state.get(block, 'qx')),
                float(state.get(block, 'qy')),
                float(state.get(block, 'qz')),
                float(state.get(block, 'qw'))
            ])
        else:
            block_quat = np.array([0,0,0,1])

        gripper_quat = np.array([
            float(state.get(self._robot, 'orn_x')),
            float(state.get(self._robot, 'orn_y')),
            float(state.get(self._robot, 'orn_z')),
            float(state.get(self._robot, 'orn_w'))
        ])

        # Step 1: Define the incremental rotation (local z-axis of the gripper)
        incremental_rotation_gripper = R.from_rotvec((theta) * np.array([0, 0, 1]))  # z-axis in gripper's frame

        # Step 2: Convert the incremental rotation to the world frame
        gripper_rotation = R.from_quat(gripper_quat)
        incremental_rotation_world = incremental_rotation_gripper

        # Step 3: Combine the incremental rotation with the block's current orientation
        block_rotation = R.from_quat(block_quat)
        updated_block_rotation = incremental_rotation_world * block_rotation

        # Step 4: Extract the updated quaternion
        qx_new, qy_new, qz_new, qw_new = updated_block_rotation.as_quat()

        trial_state.set(block, "pose_x", x)
        trial_state.set(block, "pose_y", y)
        trial_state.set(block, "pose_z", z)

        if CFG.multi_modal_cover_real_robot:
            trial_state.set(block, "qx", qx_new)
            trial_state.set(block, "qy", qy_new)
            trial_state.set(block, "qz", qz_new)
            trial_state.set(block, "qw", qw_new)

        # Check that table surface is clear at this pose
        block_states = [b for b in trial_state if b.is_instance(self._block_type) and b and b != block]
        logging.info(f"transition putontable state: {state}")

        if self._is_block_collision(trial_state, block, block_states):
            logging.info(f"putontable in collision")
            return next_state


        # Execute putontable
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z)
        if CFG.multi_modal_cover_real_robot:
            next_state.set(block, "qx", qx_new)
            next_state.set(block, "qy", qy_new)
            next_state.set(block, "qz", qz_new)
            next_state.set(block, "qw", qw_new)
        next_state.set(block, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers

        logging.info(f"transition putontable next-state: {next_state}")

        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               possible_num_blocks=self._num_blocks_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               possible_num_blocks=self._num_blocks_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnTable, self._GripperOpen, self._Holding, self._OnZone
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        if CFG.blocks_holding_goals:
            return {self._Holding}
        return {self._OnZone}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._robot_type, self._zone_type, self._dummy_zone_goal_type}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers]
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        r = self._block_size * 0.5  # block radius
        #TODO: update for variable block size
        width_ratio = max(
            1. / 5,
            min(
                5.,  # prevent from being too extreme
                (self.y_ub - self.y_lb) / (self.x_ub - self.x_lb)))
        fig, (xz_ax, yz_ax) = plt.subplots(
            1,
            2,
            figsize=(20, 8),
            gridspec_kw={'width_ratios': [1, width_ratio]})
        xz_ax.set_xlabel("x", fontsize=24)
        xz_ax.set_ylabel("z", fontsize=24)
        xz_ax.set_xlim((self.x_lb - 2 * r, self.x_ub + 2 * r))
        xz_ax.set_ylim((self.table_height, r * 16 + 0.1))
        yz_ax.set_xlabel("y", fontsize=24)
        yz_ax.set_ylabel("z", fontsize=24)
        yz_ax.set_xlim((self.y_lb - 2 * r, self.y_ub + 2 * r))
        yz_ax.set_ylim((self.table_height, r * 16 + 0.1))

        blocks = [o for o in state if o.is_instance(self._block_type)]
        held = "None"
        for block in sorted(blocks):
            x = state.get(block, "pose_x")
            y = state.get(block, "pose_y")
            z = state.get(block, "pose_z")
            # RGB values are between 0 and 1.
            if state.get(block, "held") > self.held_tol:
                assert held == "None"
                held = f"{block.name}"

            # xz axis
            xz_rect = patches.Rectangle((x - r, z - r),
                                        2 * r,
                                        2 * r,
                                        zorder=-y,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor='red')
            xz_ax.add_patch(xz_rect)

            # yz axis
            yz_rect = patches.Rectangle((y - r, z - r),
                                        2 * r,
                                        2 * r,
                                        zorder=-x,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor='red')
            yz_ax.add_patch(yz_rect)

        title = f"Held: {held}"
        if caption is not None:
            title += f"; {caption}"
        plt.suptitle(title, fontsize=24, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for _ in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks)

            while True:
                init_state, blocks, goal_zone = self._sample_state(num_blocks, rng)
                goal = self._sample_goal([blocks, goal_zone], num_blocks, rng)
                if not all(goal_atom.holds(init_state) for goal_atom in goal):
                    break

            logging.info(f'GOAL: {goal}')
            tasks.append(EnvironmentTask(init_state, goal))
        logging.info("TASKS GENERATED")
        return tasks

    def _sample_state(self, num_blocks: int,
                      rng: np.random.Generator) -> tuple[State, List[Object], Object]:
        data: Dict[Object, Array] = {}
        # Create objects

        blocks = []
        # Create block states
        init_blocks = set()
        sampled_tags = set()
        block_x_min, block_y_min, block_x_max, block_y_max = self.x_lb, self.y_lb, self.x_ub, self.y_ub
        blocks_poly = MultiPolygon()
        blocks_margins_poly = MultiPolygon()

        block_poses = set()
        block_poses_margins = set()

        for block_idx in range(num_blocks):
            if CFG.multi_modal_cover_real_robot:
                tagged_block_sizes = get_tagged_block_sizes()

                block_tag_id = int(rng.uniform() * len(tagged_block_sizes))
                while block_tag_id in sampled_tags:
                    block_tag_id = int(rng.uniform() * len(tagged_block_sizes))

                sampled_tags.add(block_tag_id)


                logging.info(f"sample block_tag_id: {block_tag_id}")

                block = Object(f"block{block_idx}", self._block_type)
                d = tagged_block_sizes[block_tag_id][0]
                w = tagged_block_sizes[block_tag_id][1]
                h = tagged_block_sizes[block_tag_id][2]

                logging.info(f"block_tag_id: {(d,w,h)}")


                for _ in range(self.num_tries):
                    block_upright = rng.choice([True, False])
                    block_x_vertical = rng.choice([True, False])
                    block_axis_up = rng.choice([True, False])
                    block_x, block_y, block_rot = rng.uniform(
                        [block_x_min, block_y_min, -np.pi],
                        [block_x_max, block_y_max, np.pi]
                    )

                    block_w_2d = w if block_x_vertical else d
                    block_transform = lambda geom: translate(rotate(geom, block_rot, use_radians=True), block_x,
                                                             block_y)

                    block_poly = block_transform(box(-d/2 if block_upright else -h / 2,
                                                     -w/2 if block_upright else -block_w_2d / 2,
                                                     d/2 if block_upright else h / 2,
                                                     w/2 if block_upright else block_w_2d / 2))
                    block_poly_margins = block_transform(box(
                        -d / 2 - self.block_min_distance if block_upright else -h / 2 - self.block_min_distance,
                        -w / 2 - max(self.block_min_distance, self.block_side_margin) if block_upright else -block_w_2d / 2 - max(self.block_min_distance, self.block_side_margin),
                        d / 2 + self.block_min_distance if block_upright else h / 2 + self.block_min_distance,
                        w / 2 + max(self.block_side_margin, self.block_min_distance) if block_upright else block_w_2d / 2 + max(self.block_side_margin, self.block_min_distance)
                    ))

                    if not blocks_poly.intersects(block_poly_margins) and not blocks_margins_poly.intersects(block_poly):
                        break
                else:
                    raise ValueError('Could not generate a task with given settings')
                if block_upright:
                    block_quat = multiply_poses(
                        Pose((0, 0, 0), quaternion_from_euler(0, 0, block_rot)),  # type: ignore
                    ).orientation
                else:
                    block_quat = multiply_poses(
                        Pose((0, 0, 0), quaternion_from_euler(0, 0, block_rot)),  # type: ignore
                        Pose((0, 0, 0), quaternion_from_euler(0 if block_axis_up else np.pi, 0, 0)),  # type: ignore
                        Pose((0, 0, 0), quaternion_from_euler(0 if block_x_vertical else np.pi / 2, 0, 0)),
                        # type: ignore
                        Pose((0, 0, 0), quaternion_from_euler(0, np.pi / 2, 0)),  # type: ignore
                    ).orientation

                block_up_size = h if block_upright else d if block_x_vertical else w
                data[block] = np.array(list(itertools.chain(
                    # PyBullet sometimes bugs out if the blocks are on the same height
                    [block_x, block_y, block_up_size / 2 + self.table_height],
                    [d, w, h],
                    block_quat,
                    [0],
                )))
                blocks_poly = blocks_poly.union(block_poly)
                blocks_margins_poly = blocks_margins_poly.union(block_poly_margins)

            else:
                d = CFG.blocks_block_size
                w = CFG.blocks_block_size
                h = CFG.blocks_block_size

                x, y = self._sample_initial_xy(rng, init_blocks, (d,w,h))
                init_blocks.add((x,y,d,w,h))

                z = self.table_height + h * 0.5
                block = Object(f"block{block_idx}", self._block_type)

                # [pose_x, pose_y, pose_z, held]
                data[block] = np.array([x, y, z, d, w, h, 0.0])

            blocks.append(block)

        for i, zone in enumerate(self.zone_extents):
            lower_extent_x = zone[0][0]
            lower_extent_y = zone[0][1]
            upper_extent_x = zone[1][0]
            upper_extent_y = zone[1][1]

            zone = Object(f"zone{i}", self._zone_type)

            data[zone] = np.array([lower_extent_x, lower_extent_y, upper_extent_x, upper_extent_y])

        dummy_zone = Object(f"zone_goal", self._dummy_zone_goal_type)
        dummy_zone_idx = int(rng.uniform() * len(self.zone_extents))
        data[dummy_zone] = np.array([dummy_zone_idx])

        goal_zone = Object(f"zonegoal", self._zone_type)
        data[goal_zone] = np.array([self.zone_extents[dummy_zone_idx][0][0],
                                    self.zone_extents[dummy_zone_idx][0][1],
                                    self.zone_extents[dummy_zone_idx][1][0],
                                    self.zone_extents[dummy_zone_idx][1][1]])

        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z - rng.uniform(0.002, 0.004)

        rf = 1.0  # fingers start out open
        data[self._robot] = np.array([rx, ry, rz, 0.7071, 0.7071, 0, 0, rf], dtype=np.float32)
        return State(data), blocks, dummy_zone

    def _sample_goal(self, objects: Sequence[Object], num_blocks: int,
                     rng: np.random.Generator) -> Set[GroundAtom]:

        blocks, zone_goal = objects
        # Create goal from blocks
        goal_atoms = set()
        for block in blocks:
            goal_atoms.add(GroundAtom(self._OnZone, [block, zone_goal]))
            goal_atoms.add(GroundAtom(self._OnTable, [block]))

        return goal_atoms

    def _sample_initial_xy(
            self, rng: np.random.Generator,
            existing_xys: Set[Tuple[float, float, float, float, float]],
            block_dims: Tuple[float, float, float]) -> Tuple[float, float]:

        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            y = rng.uniform(self.y_lb, self.y_ub)
            if self._table_xy_is_clear(x, y, existing_xys, block_dims):
                return (x, y)

    def _table_xy_is_clear(self, x: float, y: float,
                           existing_xys: Set[Tuple[float, float, float, float, float]],
                           block_dims: Tuple[float, float, float]) -> bool:

        # if all(
        #         abs(x - other_x) > self.collision_padding * self._block_size
        #         for other_x, _ in existing_xys):
        #     return True
        # if all(
        #         abs(y - other_y) > self.collision_padding * self._block_size
        #         for _, other_y in existing_xys):
        #     return True
        # return False

        finger_padding: float = 0.03
        padding: float = 0.04
        # Only clutter comes from blocks at the moment
        logging.info(f"Checking x:{x}, y:{y} is clear")

        for block in existing_xys:
            upper_x = (block_dims[0] + padding) / 2.0 + x + finger_padding
            upper_y = (block_dims[1] + padding) / 2.0 + y
            lower_x = -(block_dims[0] + padding) / 2.0 + x - finger_padding
            lower_y = -(block_dims[1] + padding) / 2.0 + y

            upper_block_x = (block[2] + padding) / 2.0 + block[0]
            upper_block_y = (block[3] + padding) / 2.0 + block[1]
            lower_block_x = -(block[2] + padding) / 2.0 + block[0]
            lower_block_y = -(block[3] + padding) / 2.0 + block[1]

            if not (lower_x >= upper_block_x or upper_x <= lower_block_x or
                lower_y >= upper_block_y or upper_y <= lower_block_y):
                return False

        return True

    def _block_is_clear(self, block: Object, state: State) -> bool:
        return self._Clear_holds(state, [block])

    def _OnZone_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, zone_goal = objects
        tol = 0.001
        zone_goal_idx = state.get(zone_goal, "zone_idx")

        x = state.get(block, "pose_x")
        y = state.get(block, "pose_y")
        z = state.get(block, "pose_z")

        d = state.get(block, "depth")
        w = state.get(block, "width")
        h = state.get(block, "height")

        if CFG.multi_modal_cover_real_robot:
            qx = state.get(block, "qx")
            qy = state.get(block, "qy")
            qz = state.get(block, "qz")
            qw = state.get(block, "qw")
            h = self.get_rotated_height((x,y,z),(qx,qy,qz,qw))


        block_corners = self._get_block_corners(state, block)


        in_zone = self.in_zone(points=block_corners[:,:2],
                                extents=[self.zone_extents[zone_goal_idx]])


        on_holds = in_zone and self.table_height < z < self.table_height + h / 2 + tol


        logging.info(f"IN ZONE HOLDS: {in_zone}")
        logging.info(f"ON HOLDS: {self.table_height < z < self.table_height + h / 2 + tol}")

        print_block_corners = []
        for corner in block_corners:
            print_block_corners.append((corner[0],corner[1],corner[2]))

        logging.info(f"Block corners: {print_block_corners}")
        logging.info(f"xyz: {x, y, z}")

        if not on_holds:
            logging.info(f"z:{z}, self.table_height + h: {self.table_height + h}, self.table_height: {self.table_height}")

        return on_holds

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        if state.get(block1, "held") >= self.held_tol or \
                state.get(block2, "held") >= self.held_tol:
            return False
        x1 = state.get(block1, "pose_x")
        y1 = state.get(block1, "pose_y")
        z1 = state.get(block1, "pose_z")
        h = state.get(block1, "height")

        if CFG.multi_modal_cover_real_robot:
            qx = state.get(block1, "qx")
            qy = state.get(block1, "qy")
            qz = state.get(block1, "qz")
            qw = state.get(block1, "qw")

            h = self.get_rotated_height((x1,y1,z1),(qx,qy,qz,qw))

        x2 = state.get(block2, "pose_x")
        y2 = state.get(block2, "pose_y")
        z2 = state.get(block2, "pose_z")

        return np.allclose([x1, y1, z1], [x2, y2, z2 + h],
                           atol=self.on_tol)

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        x = state.get(block, "pose_x")
        y = state.get(block, "pose_y")
        z = state.get(block, "pose_z")
        d = state.get(block, "depth")
        w = state.get(block, "width")
        h = state.get(block, "height")

        if CFG.multi_modal_cover_real_robot:
            qx = state.get(block, "qx")
            qy = state.get(block, "qy")
            qz = state.get(block, "qz")
            qw = state.get(block, "qw")
            h = self.get_rotated_height((d, w, h), (qx, qy, qz, qw))

        # Desired Z position calculation
        desired_z = self.table_height + h * 0.5

        # Debugging outputs

        logging.info(f"ON TABLE HOLDS: z:{z}, h:{h}, desired_z:{desired_z}")

        return (state.get(block, "held") < self.held_tol) and \
            (desired_z - self.on_tol < z < desired_z + self.on_tol)


    @staticmethod
    def _GripperOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        rf = state.get(robot, "fingers")
        assert rf in (0.0, 1.0)
        return rf == 1.0

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return self._get_held_block(state) == block

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, objects):
            return False
        block, = objects
        for other_block in state:
            if other_block.type != self._block_type:
                continue
            if self._On_holds(state, [other_block, block]):
                return False
        return True

    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            if state.get(block, "held") >= self.held_tol:
                return block
        return None

    def _get_block_at_xyz(self, state: State, x: float, y: float,
                          z: float) -> Optional[Object]:
        close_blocks = []
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            block_pose = np.array([
                state.get(block, "pose_x"),
                state.get(block, "pose_y"),
                state.get(block, "pose_z")
            ])
            if np.allclose([x, y, z], block_pose, atol=self.pick_tol):
                dist = np.linalg.norm(np.array([x, y, z]) - block_pose)
                close_blocks.append((block, float(dist)))
        if not close_blocks:
            return None
        return min(close_blocks, key=lambda x: x[1])[0]  # min distance

    def _get_highest_block_below(self, state: State, x: float, y: float,
                                 z: float) -> Optional[Object]:
        blocks_here = []
        for block in state:
            if not block.is_instance(self._block_type):
                continue
            block_pose = np.array(
                [state.get(block, "pose_x"),
                 state.get(block, "pose_y")])
            block_z = state.get(block, "pose_z")
            if np.allclose([x, y], block_pose, atol=self.pick_tol) and \
                    block_z < z - self.pick_tol:
                blocks_here.append((block, block_z))
        if not blocks_here:
            return None
        return max(blocks_here, key=lambda x: x[1])[0]  # highest z

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        raise NotImplementedError
        # with open(json_file, "r", encoding="utf-8") as f:
        #     task_spec = json.load(f)
        # # Create the initial state from the task spec.
        # # One day, we can make the block size a feature of the blocks, but
        # # for now, we'll just make sure that the block size in the real env
        # # matches what we expect in sim.
        # assert np.isclose(task_spec["block_size"], self._block_size)
        # state_dict: Dict[Object, Dict[str, float]] = {}
        # id_to_obj: Dict[str, Object] = {}  # used in the goal construction
        # for block_id, block_spec in task_spec["blocks"].items():
        #     block = Object(block_id, self._block_type)
        #     id_to_obj[block_id] = block
        #     x, y, z = block_spec["position"]
        #     # Make sure that the block is in bounds.
        #     if not (self.x_lb <= x <= self.x_ub and \
        #             self.y_lb <= y <= self.y_ub and \
        #             self.table_height <= z):
        #         logging.warning("Block out of bounds in initial state!")
        #     state_dict[block] = {
        #         "pose_x": x,
        #         "pose_y": y,
        #         "pose_z": z,
        #         "held": 0,
        #     }
        # # Add the robot at a constant initial position.
        # rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        #
        # rf = 1.0  # fingers start out open
        # state_dict[self._robot] = {
        #     "pose_x": rx,
        #     "pose_y": ry,
        #     "pose_z": rz,
        #     "fingers": rf,
        # }
        # init_state = utils.create_state_from_dict(state_dict)
        # # Create the goal from the task spec.
        # if "goal" in task_spec:
        #     goal = self._parse_goal_from_json(task_spec["goal"], id_to_obj)
        # elif "language_goal" in task_spec:
        #     goal = self._parse_language_goal_from_json(
        #         task_spec["language_goal"], id_to_obj)
        # else:
        #     raise ValueError("JSON task spec must include 'goal'.")
        # env_task = EnvironmentTask(init_state, goal)
        # assert not env_task.task.goal_holds(init_state)
        # return env_task

    def _get_language_goal_prompt_prefix(self,
                                         object_names: Collection[str]) -> str:
        # pylint:disable=line-too-long
        raise NotImplementedError
