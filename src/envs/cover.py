"""Toy cover domain. This environment IS downward refinable (low-level search
won't ever fail), but it still requires backtracking.
"""

from typing import List, Set, Sequence, Dict, Tuple, Optional
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
            _policy=self._PickPlace_policy,
            _initiable=self._PickPlace_initiable,
            _terminal=self._PickPlace_terminal)
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
        # The grasped block's pose stays the same.
        if held_block is None and above_block is not None:
            grasp = pose-state.get(above_block, "pose")
            next_state.set(self._robot, "hand", pose)
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
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Covers}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._target_type, self._robot_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._PickPlace}

    @property
    def action_space(self) -> Box:
        return Box(0, 1, (1,))  # same as option param space

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> List[Image]:
        del task  # not used by this render function
        del action  # not used by this render function
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
        return [img]

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

    def _HandEmpty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert not objects
        for obj in state:
            if obj.is_instance(self._block_type) and \
               state.get(obj, "grasp") != -1:
                return False
        return True

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "grasp") != -1

    @staticmethod
    def _PickPlace_policy(state: State, objects: Sequence[Object],
                          params: Array) -> Action:
        del state, objects  # unused
        return Action(params)  # action is simply the parameter

    @staticmethod
    def _PickPlace_initiable(state: State, objects: Sequence[Object],
                             params: Array) -> bool:
        del state, objects, params  # unused
        return True  # can be run from anywhere

    @staticmethod
    def _PickPlace_terminal(state: State, objects: Sequence[Object],
                            params: Array) -> bool:
        del state, objects, params  # unused
        return True  # always 1 timestep

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
            _policy=self._Pick_policy,
            _initiable=self._PickPlace_initiable,
            _terminal=self._PickPlace_terminal)
        self._Place = ParameterizedOption(
            "Place", types=[self._target_type], params_space=Box(0, 1, (1,)),
            _policy=self._PickPlace_policy,  # use the parent class's policy
            _initiable=self._PickPlace_initiable,
            _terminal=self._PickPlace_terminal)

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Pick, self._Place}

    @staticmethod
    def _Pick_policy(s: State, o: Sequence[Object], p: Array) -> Action:
        # The pick parameter is a RELATIVE position, so we need to
        # add the pose of the object.
        pick_pose = s.get(o[0], "pose") + p[0]
        pick_pose = min(max(pick_pose, 0.0), 1.0)
        return Action(np.array([pick_pose], dtype=np.float32))

class CoverEnvAugmentedActions(CoverEnv):
    """Toy cover domain with with options that have object arguments and an
    augmented action space that specifies ∆x, ∆y, and toggling of the gripper.
    """
    def __init__(self) -> None:
        super().__init__()
        # Change types to include some more attributes.
        self._block_type = Type("block", ["is_block", "is_target", "width", "pose", "y", "grasped"])
        self._target_type = Type("target", ["is_block", "is_target", "width", "pose"])
        self._robot_type = Type("robot", ["gripper_x", "gripper_y", "gripper_active"])
        # Two new predicates to ensure high-level state changes after executing
        # the move_to operator.
        self._IsBlockBelow = Predicate(
            "IsBlockBelow", [self._block_type, self._robot_type], self._IsObjBelow_holds)
        self._IsTargetBelow = Predicate(
            "IsTargetBelow", [self._target_type, self._robot_type], self._IsObjBelow_holds)
        # Options
        del self._PickPlace
        self._MoveToBlock = ParameterizedOption(
            "MoveToBlock",
            types=[self._robot_type, self._block_type],
            params_space=Box(low=np.array([-1.0, 1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64),
            _policy=self._move_to_obj_policy,
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=self._move_to_obj_terminate_cond
        )
        self._MoveToTarget = ParameterizedOption(
            "MoveToTarget",
            types=[self._robot_type, self._target_type],
            params_space=Box(low=np.array([-1.0, 1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64),
            _policy=self._move_to_obj_policy,
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=self._move_to_obj_terminate_cond
        )
        self._Pick = ParameterizedOption(
            "Pick",
            types=[],
            params_space=Box(low=np.array([-1.0, 1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64),
            _policy=self._pick_policy,
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=lambda s, o, p: True  # always 1 timestep
        )
        self._Place = ParameterizedOption(
            "Place",
            types=[],
            params_space=Box(low=np.array([-1.0, 1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64),
            _policy=self._place_policy,
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=lambda s, o, p: True  # always 1 timestep
        )
        # Change objects to be of the new updated type
        self._blocks = []
        self._targets = []
        for i in range(CFG.cover_num_blocks):
            self._blocks.append(Object(f"block{i}", self._block_type))
        for i in range(CFG.cover_num_targets):
            self._targets.append(Object(f"target{i}", self._target_type))
        self._robot = Object("robby", self._robot_type)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        delta_x, delta_y, next_gripper_active = action.arr
        next_gripper_active = next_gripper_active >= 0.5
        gripper_x = state.get(self._robot, "gripper_x")
        gripper_y = state.get(self._robot, "gripper_y")
        next_gripper_x = gripper_x + delta_x
        next_gripper_y = gripper_y + delta_y
        next_state = state.copy()
        hand_regions = self._get_hand_regions(state)
        gripper_active = state.get(self._robot, "gripper_active")
        if 0 <= (next_gripper_x) <= 1 and 0 <= (next_gripper_y):
            next_state.set(self._robot, "gripper_x", next_gripper_x)
            next_state.set(self._robot, "gripper_y", next_gripper_y)
        else:
            return next_state

        # Identify which block we're holding and which block we're above based
        # on the current state.
        held_block = None
        above_block = None
        for block in self._blocks:
            if state.get(block, "grasped") != 0:
                assert held_block is None
                assert state.get(self._robot, "gripper_active") == 1
                held_block = block

            block_lb = state.get(block, "pose")-state.get(block, "width")/2
            block_ub = state.get(block, "pose")+state.get(block, "width")/2
            if (state.get(block, "grasped") == 0) and (block_lb <= next_gripper_x <= block_ub):
                assert above_block is None
                above_block = block

        if held_block is not None and above_block is not None:
            assert gripper_active
            # We moved somewhere while holding a block.
            # There is a block under us.
            # Not allowed to place a block on top of another block.
            return next_state

        elif held_block is None and above_block is None:
            # We moved somewhere while not holding a block.
            # There is no block under us.
            # Allowed to set gripper configuration arbitrarily.
            next_state.set(self._robot, "gripper_active", next_gripper_active)

        elif held_block is not None and above_block is None:
            # We moved somewhere while holding a block.
            # There is no block directly under our gripper.
            # Allowed to place this block down if the block fits.
            assert gripper_active
            block_x, block_y = state.get(held_block, "pose"), state.get(block, "y")
            next_x = block_x + delta_x
            next_y = block_y + delta_y
            # We can only move if the block stays above the ground.
            if next_y < 0:
                # Revert change.
                next_state.set(self._robot, "gripper_x", gripper_x)
                next_state.set(self._robot, "gripper_y", gripper_y)
            else:
                next_state.set(held_block, "pose", next_x)
                next_state.set(held_block, "y", next_y)
                # Disallow placing on another block, but allow placing in free space.
                if not self._any_intersection(next_x, state.get(held_block, "width"), state.data, block_only=True):
                    if not next_gripper_active:
                        next_state.set(held_block, "y", 0)
                        next_state.set(held_block, "grasped", 0)
                        next_state.set(self._robot, "gripper_active", 0)

        elif held_block is None and above_block is not None:
            # We moved somewhere while not holding a block.
            # There is a block under us.
            # Allowed to grasp this block if we are close enough and if we are
            # in the allowed hand region.
            if next_gripper_y <= 0.5:
                if next_gripper_active and not gripper_active and any(hand_lb <= gripper_x <= hand_rb for hand_lb, hand_rb in hand_regions):
                    next_state.set(above_block, "grasped", 1)
                    next_state.set(self._robot, "gripper_active", 1)

        return next_state

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._IsBlock, self._IsTarget, self._Covers, self._HandEmpty, self._Holding, self._IsBlockBelow, self._IsTargetBelow}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._MoveToBlock, self._MoveToTarget, self._Pick, self._Place}

    @property
    def action_space(self) -> Box:
        return Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float64)

    def render(self, state: State, task: Task,
           action: Optional[Action] = None) -> Image:
        # print("state: ", state)
        del task  # not used by this render function
        del action  # not used by this render function
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
        plt.scatter(x=[state.get(self._robot, "gripper_x")], y=[state.get(self._robot, "gripper_y")], color="r", s=[100], alpha=1.0, zorder=10, label="Gripper")
        lw = 3
        height = 0.1
        cs = ["blue", "purple", "green", "yellow"]
        block_alpha = 0.75
        targ_alpha = 0.25
        # Draw blocks
        for i, block in enumerate(self._blocks):
            c = cs[i]
            if state.get(block, "grasped") != 0:
                lcolor = "red"
                pose = state.get(block, "pose")
                y = state.get(block, "y")
                suffix = " (grasped)"
            else:
                lcolor = "gray"
                pose =  state.get(block, "pose")
                y = state.get(block, "y")
                suffix = ""
            rect = plt.Rectangle(
                (pose-state.get(block, "width")/2., y-height/2.),
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
        plt.xlim(0.0, 1.2)
        plt.ylim(-0.25, 0.5)
        plt.yticks(np.linspace(-0.1, 1, 12))
        plt.legend()
        plt.tight_layout()
        import random
        import string
        import time
        name = str(int(time.time()*1000))
        # name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
        # plt.savefig("images/" + name)
        img = utils.fig2data(fig)
        plt.close()
        return [img]

    def _create_initial_state(self, rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        assert len(CFG.cover_block_widths) == len(self._blocks)
        for block, width in zip(self._blocks, CFG.cover_block_widths):
            while True:
                pose = rng.uniform(width/2, 1.0-width/2)
                if not self._any_intersection(pose, width, data):
                    break
            # [is_block, is_target, width, pose, y, grasped]
            data[block] = np.array([1, 0, width, pose, 0.0, 0])
        assert len(CFG.cover_target_widths) == len(self._targets)
        for target, width in zip(self._targets, CFG.cover_target_widths):
            while True:
                pose = rng.uniform(width/2, 1.0-width/2)
                if not self._any_intersection(
                        pose, width, data, larger_gap=True):
                    break
            # [is_block, is_target, width, pose]
            data[target] = np.array([0.0, 1.0, width, pose])
        # [gripper_x, gripper_y, grasp_active]
        data[self._robot] = np.array([0.27, 0.0, 0.0])
        return State(data)

    def _HandEmpty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert not objects
        for obj in state:
            if obj.is_instance(self._block_type) and \
               state.get(obj, "grasped") != 0:
                return False
        return True

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "grasped") != 0

    @staticmethod
    def _IsObjBelow_holds(state: State, objects: Sequence[Object]) -> bool:
        obj, robot = objects
        gripper_x = state.get(robot, "gripper_x")
        gripper_y = state.get(robot, "gripper_y")
        obj_x = state.get(obj, "pose")
        obj_width = state.get(obj, "width")

        # gripper_y can never be below an object, so only have to check "x".
        if (obj_x - obj_width/2 <= gripper_x <= obj_x + obj_width/2):
            return True
        else:
            return False

    @staticmethod
    def _move_to_obj_policy(s: State, o: Sequence[Object], p: Array) -> Action:
        # if gripper is open keep it open, if it is closed keep it closed
        robot, obj = o
        gripper_status = s.get(robot, "gripper_active")
        gripper_x = s.get(robot, "gripper_x")
        gripper_y = s.get(robot, "gripper_y")
        obj_x = s.get(obj, "pose")
        obj_width = s.get(obj, "width")
        x_offset = p[0] * obj_width

        # Only move laterally when above this height. That is, there are no
        # blocks with height greater than 10.
        CONSTANT = 10.5
        if gripper_y < CONSTANT:
            # We first move up to avoid hitting any other blocks.
            delta_y_commanded = CONSTANT - gripper_y
            delta_x_commanded = 0
        elif gripper_y >= CONSTANT:
            # We are allowed to move laterally.
            # If we are not above the block, move to the block.
            # Otherwise, move down to the block.
            if gripper_x == (obj_x + x_offset):
                delta_y_commanded = -gripper_y + 0.5
                delta_x_commanded = 0
            else:
                delta_y_commanded = 0
                delta_x_commanded = ((obj_x + x_offset) - gripper_x)

        return Action(np.array([delta_x_commanded, delta_y_commanded, gripper_status], dtype=np.float32))

    @staticmethod
    def _move_to_obj_terminate_cond(s: State, o: Sequence[Object], p: Array) -> bool:
        robot, obj = o
        gripper_x = s.get(robot, "gripper_x")
        gripper_y = s.get(robot, "gripper_y")
        obj_x = s.get(obj, "pose")
        obj_width = s.get(obj, "width")
        # Stop if our gripper is near the object.
        if (gripper_y <= 0.5) and (obj_x - obj_width/2 <= gripper_x <= obj_x + obj_width/2):
            return True
        else:
            return False

    @staticmethod
    def _pick_policy(s: State, o: Sequence[Object], p: Array) -> Action:
        # close the gripper
        return Action(np.array([0, 0, 1], dtype=np.float32))

    @staticmethod
    def _place_policy(s: State, o: Sequence[Object], p: Array) -> Action:
        # open the gripper
        return Action(np.array([0, 0, 0], dtype=np.float32))

class CoverEnvHierarchicalTypes(CoverEnv):
    """Toy cover domain with hierarchical types, just for testing.
    """
    def __init__(self) -> None:
        super().__init__()
        # Change blocks to be of a derived type
        self._block_type_derived = Type(
            "block_derived",
            ["is_block", "is_target", "width", "pose", "grasp"],
            parent=self._block_type)
        # Objects
        self._blocks = []
        for i in range(CFG.cover_num_blocks):
            self._blocks.append(Object(f"block{i}", self._block_type_derived))

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._block_type_derived,
                self._target_type, self._robot_type}
