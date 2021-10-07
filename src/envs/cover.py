"""Toy cover domain. This environment IS downward refinability (low-level search
won't ever fail), but it still requires backtracking.
"""

from typing import List, Set, Sequence, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


class CoverEnv(BaseEnv):
    """Toy cover domain.
    """
    def __init__(self) -> None:
        super().__init__()
        # Types
        self._block_type = Type(
            "block", ["is_block", "is_target", "width", "pose", "grasp"])
        self._target_type = Type(
            "target", ["is_block", "is_target", "width", "pose"])
        self._robot_type = Type("robot", ["hand"])
        # Predicates
        self._IsBlock = Predicate(
            "IsBlock", [self._block_type], self._IsBlock_holds)
        self._IsTarget = Predicate(
            "IsTarget", [self._target_type], self._IsTarget_holds)
        self._Covers = Predicate(
            "Covers", [self._block_type, self._target_type], self._Covers_holds)
        self._HandEmpty = Predicate(
            "HandEmpty", [], self._HandEmpty_holds)
        self._Holding = Predicate(
            "Holding", [self._block_type], self._Holding_holds)
        # Options
        params_space = Box(0, 1, (1,))
        self._PickPlace = ParameterizedOption(
            "PickPlace", params_space,
            _policy=lambda s, p: Action(p),  # action is simply the parameter
            _initiable=lambda s, p: True,  # can be run from anywhere
            _terminal=lambda s, p: True)  # always 1 timestep
        # Objects
        self._blocks = []
        self._targets = []
        for i in range(CFG.cover_num_blocks):
            self._blocks.append(Object(f"block{i}", self._block_type))
        for i in range(CFG.cover_num_targets):
            self._targets.append(Object(f"target{i}", self._target_type))
        self._robot = Object("robby", self._robot_type)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        pose = action.arr.item()
        next_state = state.copy()
        hand_regions = self._get_hand_regions(state)
        # If we're not in any hand region, no-op.
        if not any(hand_lb <= pose <= hand_rb
                   for hand_lb, hand_rb in hand_regions):
            return next_state
        # Identify which block we're holding and which block we're above.
        held_block = None
        above_block = None
        for block in self._blocks:
            if state[block][4] != -1:
                assert held_block is None
                held_block = block
            block_lb = state[block][3]-state[block][2]/2
            block_ub = state[block][3]+state[block][2]/2
            if state[block][4] == -1 and block_lb <= pose <= block_ub:
                assert above_block is None
                above_block = block
        # If we're not holding anything and we're above a block, grasp it.
        if held_block is None and above_block is not None:
            grasp = pose-state[above_block][3]
            next_state[self._robot][0] = pose
            next_state[above_block][3] = -1000  # out of the way
            next_state[above_block][4] = grasp
        # If we are holding something, place it.
        # Disallow placing on another block or in free space.
        if held_block is not None and above_block is None:
            new_pose = pose-state[held_block][4]
            if not self._any_intersection(
                    new_pose, state[held_block][2], state.data,
                    block_only=True) and \
                any(state[targ][3]-state[targ][2]/2
                    <= pose <=
                    state[targ][3]+state[targ][2]/2
                    for targ in self._targets):
                next_state[self._robot][0] = pose
                next_state[held_block][3] = new_pose
                next_state[held_block][4] = -1
        return next_state

    def get_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks)

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._IsBlock, self._IsTarget, self._Covers,
                self._HandEmpty, self._Holding}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._target_type, self._robot_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._PickPlace}

    @property
    def action_space(self) -> Box:
        # For this env, the action space is the same as the option param space.
        return self._PickPlace.params_space

    def render(self, state: State) -> Image:
        fig, ax = plt.subplots(1, 1)
        # Draw main line
        plt.plot([-0.2, 1.2], [-0.055, -0.055], color="black")
        # Draw hand regions
        hand_regions = self._get_hand_regions(state)
        for i, (hand_lb, hand_rb) in enumerate(hand_regions):
            if i == 0:
                label = "Allowed hand region"
            else:
                label = None
            plt.plot([hand_lb, hand_rb], [-0.08, -0.08], color="red",
                     alpha=0.5, lw=8., label=label)
        # Draw hand
        plt.scatter(state[self._robot][0], 0.05, color="r",
                    s=100, alpha=1., zorder=10, label="Hand")
        lw = 3
        height = 0.1
        cs = ["blue", "purple", "green", "yellow"]
        block_alpha = 0.75
        targ_alpha = 0.25
        # Draw blocks
        for i, block in enumerate(self._blocks):
            c = cs[i]
            if state.get(block, "grasp") != -1:
                lcolor = "red"
                pose = state[self._robot][0]-state.get(block, "grasp")
                suffix = " (grasped)"
            else:
                lcolor = "gray"
                pose =  state.get(block, "pose")
                suffix = ""
            rect = plt.Rectangle(
                (pose-state.get(block, "width")/2., -height/2.),
                state.get(block, "width"), height, linewidth=lw,
                edgecolor=lcolor, facecolor=c, alpha=block_alpha,
                label=f"block{i}"+suffix)
            ax.add_patch(rect)
        # Draw targets
        for i, targ in enumerate(self._targets):
            c = cs[i]
            rect = plt.Rectangle(
                (state.get(targ, "pose")-state.get(targ, "width")/2.,
                 -height/2.),
                state.get(targ, "width"), height, linewidth=lw,
                edgecolor=lcolor,
                facecolor=c, alpha=targ_alpha, label=f"target{i}")
            ax.add_patch(rect)
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.25, 0.5)
        plt.yticks([])
        plt.legend()
        plt.tight_layout()
        img = utils.fig2data(fig)
        plt.close()
        return img

    def _get_hand_regions(self, state: State) -> List[Tuple[float, float]]:
        hand_regions = []
        for block in self._blocks:
            hand_regions.append(
                (state[block][3]-state[block][2]/2,
                 state[block][3]+state[block][2]/2))
        for targ in self._targets:
            hand_regions.append(
                (state[targ][3]-state[targ][2]/10,
                 state[targ][3]+state[targ][2]/10))
        return hand_regions

    def _get_tasks(self, num: int) -> List[Task]:
        tasks = []
        goal1 = {GroundAtom(self._Covers, [self._blocks[0], self._targets[0]])}
        goal2 = {GroundAtom(self._Covers, [self._blocks[1], self._targets[1]])}
        goal3 = {GroundAtom(self._Covers, [self._blocks[0], self._targets[0]]),
                 GroundAtom(self._Covers, [self._blocks[1], self._targets[1]])}
        goals = [goal1, goal2, goal3]
        for i in range(num):
            tasks.append(Task(self._create_initial_state(),
                              goals[i%len(goals)]))
        return tasks

    def _create_initial_state(self) -> State:
        data: Dict[Object, Array] = {}
        assert len(CFG.cover_block_widths) == len(self._blocks)
        for block, width in zip(self._blocks, CFG.cover_block_widths):
            while True:
                pose = self._rng.uniform(width/2, 1.0-width/2)
                if not self._any_intersection(pose, width, data):
                    break
            # [is_block, is_target, width, pose, grasp]
            data[block] = np.array([1.0, 0.0, width, pose, -1.0])
        assert len(CFG.cover_target_widths) == len(self._targets)
        for target, width in zip(self._targets, CFG.cover_target_widths):
            while True:
                pose = self._rng.uniform(width/2, 1.0-width/2)
                if not self._any_intersection(
                        pose, width, data, larger_gap=True):
                    break
            # [is_block, is_target, width, pose]
            data[target] = np.array([0.0, 1.0, width, pose])
        # [hand]
        data[self._robot] = np.array([0.0])
        return State(data)

    @staticmethod
    def _IsBlock_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return block in state

    @staticmethod
    def _IsTarget_holds(state: State, objects: Sequence[Object]) -> bool:
        target, = objects
        return target in state

    @staticmethod
    def _Covers_holds(state: State, objects: Sequence[Object]) -> bool:
        block, target = objects
        block_pose = state[block][3]
        block_width = state[block][2]
        target_pose = state[target][3]
        target_width = state[target][2]
        return (target_pose-target_width/2 <= block_pose-block_width/2) and \
               (target_pose+target_width/2 >= block_pose+block_width/2)

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        assert not objects
        for obj in state:
            if obj.type.name == "block" and state[obj][4] != -1:
                return False
        return True

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state[block][4] != -1

    def _any_intersection(self, pose: float, width: float,
                          data: Dict[Object, Array],
                          block_only: bool=False,
                          larger_gap: bool=False) -> bool:
        mult = 1.5 if larger_gap else 0.5
        for other in data:
            if block_only and other not in self._blocks:
                continue
            other_feats = data[other]
            distance = abs(other_feats[3]-pose)
            if distance <= (width+other_feats[2])*mult:
                return True
        return False
