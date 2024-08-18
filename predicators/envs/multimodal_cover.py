"""Blocks domain.

This environment IS downward refinable and DOESN'T require any
backtracking (as long as all the blocks can fit comfortably on the
table, which is true here because the block size and number of blocks
are much less than the table dimensions). The simplicity of this
environment makes it a good testbed for predicate invention.
"""

import json
import logging
import os
from pathlib import Path
from typing import ClassVar, Collection, Dict, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type
import pickle as pkl


def get_asset_path(filename: str) -> str:
    path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'assets', filename)
    assert os.path.exists(path), f"Asset {path} does not exist"
    return path


def get_tagged_block_sizes() -> List[Tuple[float, float, float]]:
    tags_path = get_asset_path('tags')
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

    pick_z: ClassVar[float] = 0.5
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = pick_z
    held_tol: ClassVar[float] = 0.5
    pick_tol: ClassVar[float] = 0.0001
    on_tol: ClassVar[float] = 0.01
    collision_padding: ClassVar[float] = 2.0

    zone_extents: ClassVar[List[Tuple[Tuple[float, float], Tuple[float, float]]]] = [((1.24, 0.54), (1.34, 0.7)),
                                                                                     ((1.355, 0.54), (1.455, 0.7)),
                                                                                     ((1.24, 0.8), (1.34, .96)),
                                                                                     ((1.355, 0.8), (1.455, .96))]
    _target_height: ClassVar[float] = 0.0001
    goal_zone = 0
    zone_order = []

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._block_type = Type("block", [
            "pose_x", "pose_y", "pose_z", "held"
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
    def in_zone(cls, point: Tuple[float, float], extents: List[Tuple[Tuple[float, float], Tuple[float, float]]],
                specific_extents: List[int] = None):
        if specific_extents is None:
            for extent in extents:
                if extent[0][0] < point[0] < extent[1][0] and extent[0][1] < point[1] < extent[1][1]:
                    return True

        else:
            for extent_idx in specific_extents:
                if extents[extent_idx][0][0] < point[0] < extents[extent_idx][1][0] and \
                        extents[extent_idx][0][1] < point[1] < extents[extent_idx][1][1]:
                    return True

        return False

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        x, y, z, fingers = action.arr
        logging.info("transition")
        # Infer which transition function to follow
        if fingers < 0.5:
            return self._transition_pick(state, x, y, z)
        return self._transition_putontable(state, x, y, z)

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
        # Execute pick
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", self.pick_z)
        next_state.set(block, "held", 1.0)
        next_state.set(self._robot, "fingers", 0.0)  # close fingers
        if "clear" in self._block_type.feature_names:
            # See BlocksEnvClear
            next_state.set(block, "clear", 0)
            # If block was on top of other block, set other block to clear
            other_block = self._get_highest_block_below(state, x, y, z)
            if other_block is not None and other_block != block:
                next_state.set(other_block, "clear", 1)
        return next_state

    def _transition_putontable(self, state: State, x: float, y: float,
                               z: float) -> State:
        next_state = state.copy()
        logging.info("Transition PutOnTable")
        # Can only putontable if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_held_block(state)
        assert block is not None
        # Check that table surface is clear at this pose
        poses = [[
            state.get(b, "pose_x"),
            state.get(b, "pose_y"),
            state.get(b, "pose_z")
        ] for b in state if b.is_instance(self._block_type)]
        existing_xys = {(float(p[0]), float(p[1])) for p in poses}
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute putontable
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z)
        next_state.set(block, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers

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
            init_state, blocks, goal_zone = self._sample_state(num_blocks, rng)
            goal = self._sample_goal([blocks, goal_zone], num_blocks, rng)
            logging.info(f'GOAL: {goal}')
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    def _sample_state(self, num_blocks: int,
                      rng: np.random.Generator) -> tuple[State, List[Object], Object]:
        data: Dict[Object, Array] = {}
        # Create objects

        blocks = []
        zones = []
        # Create block states
        for block_idx in range(num_blocks):
            x, y = self._sample_initial_xy(rng, set())

            z = self.table_height + self._block_size * 0.5
            block = Object(f"block{block_idx}", self._block_type)
            # [pose_x, pose_y, pose_z, held]
            data[block] = np.array([x, y, z, 0.0])
            blocks.append(block)

        for i, zone in enumerate(self.zone_extents):
            lower_extent_x = zone[0][0]
            lower_extent_y = zone[0][1]
            upper_extent_x = zone[1][0]
            upper_extent_y = zone[1][1]

            zone = Object(f"zone{i}", self._zone_type)

            data[zone] = np.array([lower_extent_x, lower_extent_y, upper_extent_x, upper_extent_y])
            zones.append(zone)

        dummy_zone = Object(f"zone_goal", self._dummy_zone_goal_type)
        data[dummy_zone] = np.array([int(rng.uniform() * 4)])

        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z

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

        return goal_atoms

    def _sample_initial_xy(
            self, rng: np.random.Generator,
            existing_xys: Set[Tuple[float, float]]) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            y = rng.uniform(self.y_lb, self.y_ub)
            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    def _table_xy_is_clear(self, x: float, y: float,
                           existing_xys: Set[Tuple[float, float]]) -> bool:
        if all(
                abs(x - other_x) > self.collision_padding * self._block_size
                for other_x, _ in existing_xys):
            return True
        if all(
                abs(y - other_y) > self.collision_padding * self._block_size
                for _, other_y in existing_xys):
            return True
        return False

    def _block_is_clear(self, block: Object, state: State) -> bool:
        return self._Clear_holds(state, [block])

    def _OnZone_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, zone_goal = objects

        zone_goal_idx = state.get(zone_goal, "zone_idx")

        x = state.get(block, "pose_x")
        y = state.get(block, "pose_y")
        z = state.get(block, "pose_z")

        on_holds = self.in_zone(point=(x, y),
                                extents=[self.zone_extents[zone_goal_idx]]) and \
                   self.table_height < z < 0.3
        logging.info(f"ON HOLDS: {on_holds}")

        return on_holds

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        if state.get(block1, "held") >= self.held_tol or \
                state.get(block2, "held") >= self.held_tol:
            return False
        x1 = state.get(block1, "pose_x")
        y1 = state.get(block1, "pose_y")
        z1 = state.get(block1, "pose_z")
        x2 = state.get(block2, "pose_x")
        y2 = state.get(block2, "pose_y")
        z2 = state.get(block2, "pose_z")
        return np.allclose([x1, y1, z1], [x2, y2, z2 + self._block_size],
                           atol=self.on_tol)

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        x = state.get(block, "pose_x")
        y = state.get(block, "pose_y")
        z = state.get(block, "pose_z")

        desired_z = self.table_height + self._block_size * 0.5

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
