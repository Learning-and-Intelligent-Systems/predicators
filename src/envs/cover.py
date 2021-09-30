"""Toy cover domain.
"""

from typing import List, Set, Sequence, Dict
import numpy as np
from numpy.typing import NDArray
from gym.spaces import Box  # type: ignore
from predicators.configs.envs import cover_config
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object

Array = NDArray[np.float32]
CONFIG = cover_config.get_config()


class CoverEnv(BaseEnv):
    """Toy cover domain.
    """
    def __init__(self):
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
            _policy=lambda s, p: p,  # action is simply the parameter
            _initiable=lambda s, p: True,  # can be run from anywhere
            _terminal=lambda s, p: True)  # always 1 timestep
        # Objects
        self._blocks = []
        self._targets = []
        for i in range(CONFIG["num_blocks"]):
            self._blocks.append(self._block_type(f"block{i}"))
        for i in range(CONFIG["num_targets"]):
            self._targets.append(self._target_type(f"target{i}"))
        self._robot = self._robot_type("robby")

    def simulate(self, state: State, action: Array) -> State:
        assert self.action_space.contains(action)
        pose = action.item()
        next_state = state.copy()
        # Compute hand_regions (allowed pick/place regions).
        hand_regions = []
        for block in self._blocks:
            hand_regions.append(
                (state[block][3]-state[block][2]/2,
                 state[block][3]+state[block][2]/2))
        for targ in self._targets:
            hand_regions.append(
                (state[targ][3]-state[targ][2]/10,
                 state[targ][3]+state[targ][2]/10))
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
        return self._get_tasks(num=CONFIG["num_train_tasks"])

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CONFIG["num_test_tasks"])

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

    def _get_tasks(self, num: int) -> List[Task]:
        tasks = []
        goal1 = {self._Covers([self._blocks[0], self._targets[0]])}
        goal2 = {self._Covers([self._blocks[1], self._targets[1]])}
        goal3 = {self._Covers([self._blocks[0], self._targets[0]]),
                 self._Covers([self._blocks[1], self._targets[1]])}
        goals = [goal1, goal2, goal3]
        for i in range(num):
            tasks.append(Task(self._create_initial_state(),
                              goals[i%len(goals)]))
        return tasks

    def _create_initial_state(self) -> State:
        data: Dict[Object, Array] = {}
        assert len(CONFIG["block_widths"]) == len(self._blocks)
        for block, width in zip(self._blocks, CONFIG["block_widths"]):
            while True:
                pose = self._rng.uniform(width/2, 1.0-width/2)
                if not self._any_intersection(pose, width, data):
                    break
            # [is_block, is_target, width, pose, grasp]
            data[block] = np.array([1.0, 0.0, width, pose, -1.0])
        assert len(CONFIG["target_widths"]) == len(self._targets)
        for target, width in zip(self._targets, CONFIG["target_widths"]):
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
        return (block_pose-block_width/2 <= target_pose-target_width/2) and \
               (block_pose+block_width/2 >= target_pose+target_width/2)

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
