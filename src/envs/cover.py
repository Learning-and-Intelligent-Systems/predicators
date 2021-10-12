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
        self._PickPlace = ParameterizedOption(
            "PickPlace", types=[], params_space=Box(0, 1, (1,)),
            _policy=lambda s, o, p: Action(p),  # action is simply the parameter
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=lambda s, o, p: True)  # always 1 timestep
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
            if state.get(block, "grasp") != -1:
                assert held_block is None
                held_block = block
            block_lb = state.get(block, "pose")-state.get(block, "width")/2
            block_ub = state.get(block, "pose")+state.get(block, "width")/2
            if state.get(block, "grasp") == -1 and block_lb <= pose <= block_ub:
                assert above_block is None
                above_block = block
        # If we're not holding anything and we're above a block, grasp it.
        if held_block is None and above_block is not None:
            grasp = pose-state.get(above_block, "pose")
            next_state.set(self._robot, "hand", pose)
            next_state.set(above_block, "pose", -1000)  # out of the way
            next_state.set(above_block, "grasp", grasp)
        # If we are holding something, place it.
        # Disallow placing on another block or in free space.
        if held_block is not None and above_block is None:
            new_pose = pose-state.get(held_block, "grasp")
            if not self._any_intersection(
                    new_pose, state.get(held_block, "width"), state.data,
                    block_only=True) and \
                any(state.get(targ, "pose")-state.get(targ, "width")/2
                    <= pose <=
                    state.get(targ, "pose")+state.get(targ, "width")/2
                    for targ in self._targets):
                next_state.set(self._robot, "hand", pose)
                next_state.set(held_block, "pose", new_pose)
                next_state.set(held_block, "grasp", -1)
        return next_state

    def get_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               rng=self._train_rng)

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               rng=self._test_rng)

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
        return Box(0, 1, (1,))  # same as option param space

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
        plt.scatter(state.get(self._robot, "hand"), 0.05, color="r",
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
                pose = state.get(self._robot, "hand")-state.get(block, "grasp")
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
                (state.get(block, "pose")-state.get(block, "width")/2,
                 state.get(block, "pose")+state.get(block, "width")/2))
        for targ in self._targets:
            hand_regions.append(
                (state.get(targ, "pose")-state.get(targ, "width")/10,
                 state.get(targ, "pose")+state.get(targ, "width")/10))
        return hand_regions

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        tasks = []
        goal1 = {GroundAtom(self._Covers, [self._blocks[0], self._targets[0]])}
        goal2 = {GroundAtom(self._Covers, [self._blocks[1], self._targets[1]])}
        goal3 = {GroundAtom(self._Covers, [self._blocks[0], self._targets[0]]),
                 GroundAtom(self._Covers, [self._blocks[1], self._targets[1]])}
        goals = [goal1, goal2, goal3]
        for i in range(num):
            tasks.append(Task(self._create_initial_state(rng),
                              goals[i%len(goals)]))
        return tasks

    def _create_initial_state(self, rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        assert len(CFG.cover_block_widths) == len(self._blocks)
        for block, width in zip(self._blocks, CFG.cover_block_widths):
            while True:
                pose = rng.uniform(width/2, 1.0-width/2)
                if not self._any_intersection(pose, width, data):
                    break
            # [is_block, is_target, width, pose, grasp]
            data[block] = np.array([1.0, 0.0, width, pose, -1.0])
        assert len(CFG.cover_target_widths) == len(self._targets)
        for target, width in zip(self._targets, CFG.cover_target_widths):
            while True:
                pose = rng.uniform(width/2, 1.0-width/2)
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
        block_pose = state.get(block, "pose")
        block_width = state.get(block, "width")
        target_pose = state.get(target, "pose")
        target_width = state.get(target, "width")
        return (block_pose-block_width/2 <= target_pose-target_width/2) and \
               (block_pose+block_width/2 >= target_pose+target_width/2)

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        assert not objects
        for obj in state:
            if obj.type.name == "block" and state.get(obj, "grasp") != -1:
                return False
        return True

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "grasp") != -1

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


class CoverEnvTypedOptions(CoverEnv):
    """Toy cover domain with options that have object arguments. This means
    we need two options (one for block, one for target).
    """
    def __init__(self) -> None:
        super().__init__()
        del self._PickPlace
        self._Pick = ParameterizedOption(
            "Pick", types=[self._block_type], params_space=Box(-0.1, 0.1, (1,)),
            _policy=self._pick_policy,
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=lambda s, o, p: True)  # always 1 timestep
        self._Place = ParameterizedOption(
            "Place", types=[self._target_type], params_space=Box(0, 1, (1,)),
            _policy=lambda s, o, p: Action(p),  # action is simply the parameter
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=lambda s, o, p: True)  # always 1 timestep

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Pick, self._Place}

    @staticmethod
    def _pick_policy(s: State, o: Sequence[Object], p: Array) -> Action:
        # The pick parameter is a RELATIVE position, so we need to
        # add the pose of the object.
        pick_pose = s.get(o[0], "pose") + p[0]
        pick_pose = min(max(pick_pose, 0.0), 1.0)
        return Action(np.array([pick_pose], dtype=np.float32))
