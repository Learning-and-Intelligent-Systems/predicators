"""Blocks domain.

This environment IS downward refinable and DOESN'T require any
backtracking (as long as all the blocks can fit comfortably on the
table, which is true here because the block size and number of blocks
are much less than the table dimensions). The simplicity of this
environment makes it a good testbed for predicate invention.

Example command with standard planner (without Balanced Precondition):
python predicators/main.py --env pybullet_balance --approach oracle --seed 0\
    --num_train_tasks 0 --num_test_tasks 1\
    --make_test_videos --render_init_state True --pybullet_camera_width 900\
    --pybullet_camera_height 900 --label_objs_with_id_name True\
    --debug --excluded_predicates "Balanced" --sesame_max_skeletons_optimized 4\
    --sesame_max_samples_per_step 1 --make_failure_videos\
    --pybullet_control_mode reset --video_fps 20

Example command with modified planner (with Balanced Precondition):
python predicators/main.py --env pybullet_balance --approach oracle --seed 0\
    --num_train_tasks 0 --num_test_tasks 1\
    --render_init_state True --pybullet_camera_width 900\
    --pybullet_camera_height 900 --label_objs_with_id_name True\
    --debug --sesame_check_dr_reachable False\
    --sesame_filter_unreachable_nsrt False  --sesame_max_skeletons_optimized 4\
    --sesame_max_samples_per_step 1 --make_failure_videos\
    --pybullet_control_mode reset --video_fps 20 --make_test_videos\
    --sesame_check_expected_atoms False
"""

import json
import logging
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
from predicators.pybullet_helpers.geometry import Pose, Pose3D, Quaternion
from predicators.settings import CFG
from predicators.structs import Action, Array, ConceptPredicate, \
    EnvironmentTask, GroundAtom, Object, Predicate, State, Type


class BalanceEnv(BaseEnv):
    """Blocks domain."""
    # Parameters that aren't important enough to need to clog up settings.py
    plate_height: ClassVar[float] = 0.01
    table_height: ClassVar[float] = 0.2
    _table_mid_w = 0.1
    _table_side_w = 0.3
    _table_gap = 0.05
    _table_x, _table2_y, _table_z = 1.35, 0.75, 0.0
    _plate_z = table_height - plate_height
    _table2_pose: ClassVar[Pose3D] = (_table_x, _table2_y, _table_z)
    _plate3_pose: ClassVar[Pose3D] = (_table_x, _table2_y + _table_mid_w / 2 +
                                      _table_side_w / 2 + _table_gap, _plate_z)
    _plate1_pose: ClassVar[Pose3D] = (_table_x, _table2_y - _table_mid_w / 2 -
                                      _table_side_w / 2 - _table_gap, _plate_z)
    _beam_pose: ClassVar[Pose3D] = (_table_x, _table2_y, _plate_z -
                                    2 * plate_height - 2 * plate_height)
    _table_orientation: ClassVar[Quaternion] = (0., 0., 0., 1.)
    _table_mid_half_extents = [0.1, _table_mid_w / 2,
                               table_height]  # depth, w, h
    _table_side_half_extents = [0.25, _table_side_w / 2, table_height]
    _plate_half_extents = [0.25, _table_side_w / 2, plate_height]
    _beam_half_extents = [0.01, 0.3, plate_height]
    _button_radius = 0.04
    _button_color_off = [1, 0, 0, 1]
    _button_color_on = [0, 1, 0, 1]
    button_x, button_y, button_z = _table_x, _table2_y, _table_z + table_height
    button_press_threshold = 1e-3
    # The table x bounds are (1.1, 1.6), but the workspace is smaller.
    # Make it narrow enough that blocks can be only horizontally arranged.
    # Note that these boundaries are for the block positions, and that a
    # block's origin is its center, so the block itself may extend beyond
    # the boundaries while the origin remains in bounds.
    x_lb: ClassVar[float] = 1.325
    x_ub: ClassVar[float] = 1.375
    # The table y bounds are (0.3, 1.2), but the workspace is smaller.
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    # Add a 0.1 padding
    y_plate1_ub: ClassVar[float] = _plate1_pose[1] + _table_side_w / 2 - 0.1
    y_plate3_lb: ClassVar[float] = _plate3_pose[1] - _table_side_w / 2 + 0.1
    # y_plate1_lb: ClassVar[float] = _plate1_pose[1] - _table_side_w / 2
    # y_plate3_ub: ClassVar[float] = _plate3_pose[1] + _table_side_w / 2
    pick_z: ClassVar[float] = 0.7
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = pick_z
    held_tol: ClassVar[float] = 0.5
    pick_tol: ClassVar[float] = 0.0001
    on_tol: ClassVar[float] = 0.01
    collision_padding: ClassVar[float] = 2.0
    max_position_vel: ClassVar[float] = 2.5
    max_angular_vel: ClassVar[float] = np.pi / 4
    max_finger_vel: ClassVar[float] = 1.0

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # Types
        bbox_features = ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"]
        self._block_type = Type("block", [
            "pose_x", "pose_y", "pose_z", "held", "color_r", "color_g",
            "color_b"
        ] + (bbox_features if CFG.env_include_bbox_features else []))
        self._robot_type = Type(
            "robot", ["pose_x", "pose_y", "pose_z", "fingers"] +
            (bbox_features if CFG.env_include_bbox_features else []))
        self._plate_type = Type(
            "plate", (bbox_features if CFG.env_include_bbox_features else []))
        self._machine_type = Type(
            "machine", ["is_on"] +
            (bbox_features if CFG.env_include_bbox_features else []))

        # Predicates
        self._DirectlyOn = Predicate(
            "DirectlyOn", [self._block_type, self._block_type],
            self._DirectlyOn_holds, lambda objs:
            f"{objs[0]} is directly on top of {objs[1]} with no blocks in between."
        )
        self._DirectlyOnPlate = Predicate(
            "DirectlyOnPlate", [self._block_type, self._plate_type],
            self._DirectlyOnPlate_holds, lambda objs:
            f"{objs[0]} is directly resting on the {objs[1]}'s surface.")
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)
        self._MachineOn = Predicate("MachineOn",
                                    [self._machine_type, self._robot_type],
                                    self._MachineOn_holds)
        self._Balanced = Predicate("Balanced",
                                   [self._plate_type, self._plate_type],
                                   self._Balanced_holds)
        self._ClearPlate = Predicate("ClearPlate", [self._plate_type],
                                     self._ClearPlate_holds)

        self._OnPlate_abs = ConceptPredicate(
            "OnPlate",
            [self._block_type, self._plate_type],
            self._OnPlate_CP_holds,
        )
        self._Balanced_abs = ConceptPredicate(
            "Balanced",
            [self._plate_type, self._plate_type],
            # self._EqualBlocksOnPlates_CP_holds,
            self._Balanced_CP_holds,
            untransformed_predicate=self._Balanced)

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._plate1 = Object("plate1", self._plate_type)
        # self._table2 = Object("table2", self._plate_type)
        self._plate3 = Object("plate3", self._plate_type)
        self._machine = Object("mac", self._machine_type)
        # Hyperparameters from CFG.
        self._block_size = CFG.balance_block_size
        self._num_blocks_train = CFG.balance_num_blocks_train
        self._num_blocks_test = CFG.balance_num_blocks_test

    def _OnPlate_CP_holds(self, atoms: Set[GroundAtom],
                          objects: Sequence[Object]) -> bool:
        x, y = objects
        for atom in atoms:
            if atom.predicate == self._DirectlyOnPlate and\
               atom.objects == [x, y]:
                return True
        other_blocks = {
            a.objects[0]
            for a in atoms if a.predicate == self._DirectlyOn
            or a.predicate == self._OnPlate_abs
        }

        for other_block in other_blocks:
            holds1 = False
            for atom in atoms:
                if atom.predicate == self._DirectlyOn and\
                   atom.objects == [x, other_block]:
                    holds1 = True
                    break
            if holds1 and self._OnPlate_CP_holds(atoms, [other_block, y]):
                return True
        return False

    def _ClearPlate_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        plate, = objects
        for block in state.get_objects(self._block_type):
            if self._DirectlyOnPlate_holds(state, [block, plate]):
                return False
        return True

    def _MachineOn_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        machine, _ = objects
        return state.get(machine, "is_on") > 0.5

    def _PressingButton_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        robot, _ = objects
        button_pos = (self.button_x, self.button_y, self.button_z)
        x = state.get(robot, "pose_x")
        y = state.get(robot, "pose_y")
        z = state.get(robot, "pose_z")
        sq_dist_to_button = np.sum(np.subtract(button_pos, (x, y, z))**2)
        return bool(sq_dist_to_button < self.button_press_threshold)

    def _Balanced_holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if the blocks are balanced on the table."""
        plate1, table2 = objects
        if plate1 == table2:
            return False

        # Function to count the number of blocks in the tower
        def count_num_blocks(table):

            def count_recursive(base_obj, count):
                for block in state.get_objects(self._block_type):
                    if base_obj.type == self._block_type and\
                            self._DirectlyOn_holds(state, [block, base_obj]):
                        count = count_recursive(block, count + 1)
                    elif base_obj.type == self._plate_type and\
                            self._DirectlyOnPlate_holds(state, [block,
                                                                base_obj]):
                        count = count_recursive(block, count + 1)
                return count

            return count_recursive(table, 0)

        # Get the height of the blocks using recursion
        height1 = count_num_blocks(plate1)
        height2 = count_num_blocks(table2)

        return height1 == height2

    def _EqualBlocksOnPlates_CP_holds(self, atoms: Set[GroundAtom],
                                      objects: Sequence[Object]) -> bool:
        left_plate, right_plate = objects
        if left_plate == right_plate:
            return False
        left_count = 0
        right_count = 0
        for atom in atoms:
            if atom.predicate == self._OnPlate_abs and\
               atom.objects[1] == left_plate:
                left_count += 1
            if atom.predicate == self._OnPlate_abs and\
               atom.objects[1] == right_plate:
                right_count += 1
        # logging.debug(f"left: {left_count}, right: {right_count}")
        return left_count == right_count

    def _Balanced_CP_holds(self, atoms: Set[GroundAtom],
                           objects: Sequence[Object]) -> bool:
        """Check if the blocks are balanced on the table."""
        plate1, table2 = objects
        if plate1 == table2:
            return False
        # Function to count the number of blocks in the tower
        def count_num_blocks(table):

            def count_recursive(base_obj, count):
                for atom in atoms:
                    if atom.predicate == self._DirectlyOn and\
                            atom.objects[1] == base_obj:
                        count = count_recursive(atom.objects[0], count + 1)
                    elif atom.predicate == self._DirectlyOnPlate and\
                            atom.objects[1] == base_obj:
                        count = count_recursive(atom.objects[0], count + 1)
                return count

            return count_recursive(table, 0)

        # Get the height of the blocks using recursion
        height1 = count_num_blocks(plate1)
        height2 = count_num_blocks(table2)

        return height1 == height2

    @classmethod
    def get_name(cls) -> str:
        return "balance"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        x, y, z, fingers = action.arr
        # Infer which transition function to follow
        next_state = state.copy()
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        next_state.set(self._robot, "pose_z", z)
        next_state.set(self._robot, "fingers", fingers)
        pressing_button = self._PressingButton_holds(
            next_state, [self._robot, self._machine])
        # logging.debug(f"[simulate] pressing_button {pressing_button}")
        if pressing_button:
            return self._transition_pressbutton(state, x, y, z)

        if fingers < 0.5:
            return self._transition_pick(state, x, y, z)
        putOnPlate = z < self.table_height + self._block_size
        # logging.debug(f"[simulate] put_on_plate {putOnPlate}")
        if putOnPlate:
            return self._transition_putOnPlate(state, x, y, z)
        return self._transition_stack(state, x, y, z)

    def _transition_pressbutton(self, state: State, x: float, y: float,
                                z: float) -> State:
        next_state = state.copy()
        machine_was_on = self._MachineOn_holds(state,
                                               [self._machine, self._robot])
        balanced = self._Balanced_holds(state, [self._plate1, self._plate3])
        if not machine_was_on and balanced:
            next_state.set(self._machine, "is_on", 1.0)
            next_state.set(self._robot, "pose_x", x)
            next_state.set(self._robot, "pose_y", y)
            next_state.set(self._robot, "pose_z", z)
        return next_state

    def _transition_pick(self, state: State, x: float, y: float,
                         z: float) -> State:
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

    def _transition_putOnPlate(self, state: State, x: float, y: float,
                               z: float) -> State:
        next_state = state.copy()
        # Can only putOnPlate if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        block = self._get_held_block(state)
        assert block is not None
        # Check that table surface is clear at this pose
        poses = [[
            state.get(b, "pose_x"),
            state.get(b, "pose_y"),
            state.get(b, "pose_z")
        ] for b in state if b.is_instance(self._block_type) and b != block]
        existing_xys = {(float(p[0]), float(p[1])) for p in poses}
        # logging.debug(f"[simulator] table is clear {self._table_xy_is_clear(x, y, existing_xys)}")
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute putOnPlate
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z)
        next_state.set(block, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers
        if "clear" in self._block_type.feature_names:
            # See BlocksEnvClear
            next_state.set(block, "clear", 1)
        # logging.debug(f"[simulator] put_on_plate state:\n{next_state.pretty_str()}")
        return next_state

    def _transition_stack(self, state: State, x: float, y: float,
                          z: float) -> State:
        next_state = state.copy()
        # Can only stack if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        # Check that both blocks exist
        block = self._get_held_block(state)
        assert block is not None
        other_block = self._get_highest_block_below(state, x, y, z)
        if other_block is None:  # no block to stack onto
            return next_state
        # Can't stack onto yourself!
        if block == other_block:
            return next_state
        # Need block we're stacking onto to be clear
        if not self._block_is_clear(other_block, state):
            return next_state
        # Execute stack by snapping into place
        cur_x = state.get(other_block, "pose_x")
        cur_y = state.get(other_block, "pose_y")
        cur_z = state.get(other_block, "pose_z")
        next_state.set(block, "pose_x", cur_x)
        next_state.set(block, "pose_y", cur_y)
        next_state.set(block, "pose_z", cur_z + self._block_size)
        next_state.set(block, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers
        if "clear" in self._block_type.feature_names:
            # See BlocksEnvClear
            next_state.set(block, "clear", 1)
            next_state.set(other_block, "clear", 0)
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
            self._DirectlyOn, self._DirectlyOnPlate, self._GripperOpen,
            self._Holding, self._Clear, self._MachineOn, self._ClearPlate,
            self._Balanced_abs, self._OnPlate_abs
        }

    @property
    def concept_predicates(self) -> Set[ConceptPredicate]:
        return {self._Balanced_abs}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        if CFG.balance_holding_goals:
            return {self._Holding}
        return {self._DirectlyOn, self._DirectlyOnPlate}

    @property
    def types(self) -> Set[Type]:
        return {
            self._block_type, self._robot_type, self._plate_type,
            self._machine_type
        }

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
            color_r = state.get(block, "color_r")
            color_g = state.get(block, "color_g")
            color_b = state.get(block, "color_b")
            color = (color_r, color_g, color_b)
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
                                        facecolor=color)
            xz_ax.add_patch(xz_rect)

            # yz axis
            yz_rect = patches.Rectangle((y - r, z - r),
                                        2 * r,
                                        2 * r,
                                        zorder=-x,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor=color)
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
        for idx in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks, p=[0.3, 0.7])
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            goal = set(
                [GroundAtom(self._MachineOn, [self._machine, self._robot])])
            # while True:  # repeat until goal is not satisfied
            #     goal = self._sample_goal_from_piles(num_blocks, piles, rng)
            #     if not all(goal_atom.holds(init_state) for goal_atom in goal):
            #         break
            # if idx == 0:
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    def _sample_initial_piles(self, num_blocks: int,
                              rng: np.random.Generator) -> List[List[Object]]:
        n_piles = 0
        piles: List[List[Object]] = []
        for block_num in range(num_blocks):
            block = Object(f"block{block_num}", self._block_type)
            # If coin flip, start new pile
            # if (block_num == 0 or rng.uniform() < 0.2) and n_piles < 2:
            # increase the chance of starting a new pile
            if (block_num == 0 or rng.uniform() < 0.4) and n_piles < 2:
                n_piles += 1
                piles.append([])
            # Add block to pile
            piles[-1].append(block)
        return piles

    def _sample_state_from_piles(self, piles: List[List[Object]],
                                 rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        # Create objects
        block_to_pile_idx = {}
        for i, pile in enumerate(piles):
            for j, block in enumerate(pile):
                assert block not in block_to_pile_idx
                block_to_pile_idx[block] = (i, j)
        # Sample pile (x, y)s
        pile_to_xy: Dict[int, Tuple[float, float]] = {}
        for i in range(len(piles)):
            pile_to_xy[i] = self._sample_initial_pile_xy(
                rng, set(pile_to_xy.values()))
        # Create block states
        for block, pile_idx in block_to_pile_idx.items():
            pile_i, pile_j = pile_idx
            x, y = pile_to_xy[pile_i]
            # Example: 0.2 + 0.045 * 0.5
            z = self.table_height + self._block_size * (0.5 + pile_j)
            r, g, b = rng.uniform(size=3)
            if "clear" in self._block_type.feature_names:
                # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b,
                # clear]
                # Block is clear iff it is at the top of a pile
                clear = pile_j == len(piles[pile_i]) - 1
                data[block] = np.array([x, y, z, 0.0, r, g, b, clear])
            else:
                # [pose_x, pose_y, pose_z, held, color_r, color_g, color_b]
                data[block] = np.array([x, y, z, 0.0, r, g, b])
        # [pose_x, pose_y, pose_z, fingers]
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        data[self._robot] = np.array([rx, ry, rz, rf], dtype=np.float32)
        data[self._plate1] = np.array([], dtype=np.float32)
        # data[self._table2] = np.array([], dtype=np.float32)
        data[self._plate3] = np.array([], dtype=np.float32)
        data[self._machine] = np.array([0.0], dtype=np.float32)
        return State(data)

    def _sample_goal_from_piles(self, num_blocks: int,
                                piles: List[List[Object]],
                                rng: np.random.Generator) -> Set[GroundAtom]:
        # Sample a goal that involves holding a block that's on the top of
        # the pile. This is useful for isolating the learning of picking and
        # unstacking. (For just picking, use num_blocks 1).
        if CFG.balance_holding_goals:
            pile_idx = rng.choice(len(piles))
            top_block = piles[pile_idx][-1]
            return {GroundAtom(self._Holding, [top_block])}
        # Sample goal pile that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_blocks, rng)
            if goal_piles != piles:
                break
        # Create goal from piles
        goal_atoms = set()
        for pile in goal_piles:
            goal_atoms.add(GroundAtom(self._DirectlyOnPlate, [pile[0]]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_atoms.add(GroundAtom(self._DirectlyOn, [block1, block2]))
        return goal_atoms

    def _sample_initial_pile_xy(
            self, rng: np.random.Generator,
            existing_xys: Set[Tuple[float, float]]) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            if rng.uniform(0, 1) < 0.5:
                # Table 1
                y = rng.uniform(self.y_lb, self.y_plate1_ub)
            else:
                # Table 3
                y = rng.uniform(self.y_plate3_lb, self.y_ub)

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

    def _DirectlyOn_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
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

    def _DirectlyOnPlate_holds(self, state: State,
                               objects: Sequence[Object]) -> bool:
        block, table = objects
        y = state.get(block, "pose_y")
        z = state.get(block, "pose_z")
        desired_z = self.table_height + self._block_size * 0.5

        if (state.get(block, "held") < self.held_tol) and \
                (desired_z-self.on_tol < z < desired_z+self.on_tol):
            if table.name == "plate1":
                return y < self._table2_y
            elif table.name == "plate3":
                return y > self._table2_y
            else:
                raise ValueError("Invalid table name")
        else:
            return False

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
            if self._DirectlyOn_holds(state, [other_block, block]):
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
        with open(json_file, "r", encoding="utf-8") as f:
            task_spec = json.load(f)
        # Create the initial state from the task spec.
        # DirectlyOne day, we can make the block size a feature of the blocks, but
        # for now, we'll just make sure that the block size in the real env
        # matches what we expect in sim.
        assert np.isclose(task_spec["block_size"], self._block_size)
        state_dict: Dict[Object, Dict[str, float]] = {}
        id_to_obj: Dict[str, Object] = {}  # used in the goal construction
        for block_id, block_spec in task_spec["blocks"].items():
            block = Object(block_id, self._block_type)
            id_to_obj[block_id] = block
            x, y, z = block_spec["position"]
            # Make sure that the block is in bounds.
            if not (self.x_lb <= x <= self.x_ub and \
                    self.y_lb <= y <= self.y_ub and \
                    self.table_height <= z):
                logging.warning("Block out of bounds in initial state!")
            r, g, b = block_spec["color"]
            state_dict[block] = {
                "pose_x": x,
                "pose_y": y,
                "pose_z": z,
                "held": 0,
                "color_r": r,
                "color_b": b,
                "color_g": g,
            }
        # Add the robot at a constant initial position.
        rx, ry, rz = self.robot_init_x, self.robot_init_y, self.robot_init_z
        rf = 1.0  # fingers start out open
        state_dict[self._robot] = {
            "pose_x": rx,
            "pose_y": ry,
            "pose_z": rz,
            "fingers": rf,
        }
        init_state = utils.create_state_from_dict(state_dict)
        # Create the goal from the task spec.
        if "goal" in task_spec:
            goal = self._parse_goal_from_json(task_spec["goal"], id_to_obj)
        elif "language_goal" in task_spec:
            goal = self._parse_language_goal_from_json(
                task_spec["language_goal"], id_to_obj)
        else:
            raise ValueError("JSON task spec must include 'goal'.")
        env_task = EnvironmentTask(init_state, goal)
        assert not env_task.task.goal_holds(init_state)
        return env_task

    def _get_language_goal_prompt_prefix(self,
                                         object_names: Collection[str]) -> str:
        # pylint:disable=line-too-long
        return """# Build a tower of block 1, block 2, and block 3, with block 1 on top
{"DirectlyOn": [["block1", "block2"], ["block2", "block3"]]}

# Put block 4 on block 3 and block 2 on block 1 and block 1 on table
{"DirectlyOn": [["block4", "block3"], ["block2", "block1"]], "DirectlyOnPlate": [["block1"]]}
"""
