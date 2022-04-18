"""Toy cover domain.

This environment IS downward refinable (low-level search won't ever
fail), but it still requires backtracking.
"""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class CoverEnv(BaseEnv):
    """Toy cover domain."""

    _allow_free_space_placing: ClassVar[bool] = False
    _initial_pick_offsets: ClassVar[List[float]] = []  # see CoverEnvRegrasp

    _workspace_x: ClassVar[float] = 1.35
    _workspace_z: ClassVar[float] = 0.75

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._block_type = Type(
            "block", ["is_block", "is_target", "width", "pose", "grasp"])
        self._target_type = Type("target",
                                 ["is_block", "is_target", "width", "pose"])
        self._robot_type = Type("robot", ["hand", "pose_x", "pose_z"])
        # Predicates
        self._IsBlock = Predicate("IsBlock", [self._block_type],
                                  self._IsBlock_holds)
        self._IsTarget = Predicate("IsTarget", [self._target_type],
                                   self._IsTarget_holds)
        self._Covers = Predicate("Covers",
                                 [self._block_type, self._target_type],
                                 self._Covers_holds)
        self._HandEmpty = Predicate("HandEmpty", [], self._HandEmpty_holds)
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)
        # Options
        self._PickPlace = utils.SingletonParameterizedOption(
            "PickPlace", self._PickPlace_policy, params_space=Box(0, 1, (1, )))
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "cover"

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
        for block in state.get_objects(self._block_type):
            if state.get(block, "grasp") != -1:
                assert held_block is None
                held_block = block
            block_lb = state.get(block, "pose") - state.get(block, "width") / 2
            block_ub = state.get(block, "pose") + state.get(block, "width") / 2
            if state.get(block,
                         "grasp") == -1 and block_lb <= pose <= block_ub:
                assert above_block is None
                above_block = block
        # If we're not holding anything and we're above a block, grasp it.
        # The grasped block's pose stays the same.
        if held_block is None and above_block is not None:
            grasp = pose - state.get(above_block, "pose")
            next_state.set(self._robot, "hand", pose)
            next_state.set(above_block, "grasp", grasp)
        # If we are holding something, place it.
        # Disallow placing on another block.
        if held_block is not None and above_block is None:
            new_pose = pose - state.get(held_block, "grasp")
            # Prevent collisions with other blocks.
            if self._any_intersection(new_pose,
                                      state.get(held_block, "width"),
                                      state.data,
                                      block_only=True,
                                      excluded_object=held_block):
                return next_state
            # Only place if free space placing is allowed, or if we're
            # placing onto some target.
            targets = state.get_objects(self._target_type)
            if self._allow_free_space_placing or \
                any(state.get(targ, "pose")-state.get(targ, "width")/2
                    <= pose <=
                    state.get(targ, "pose")+state.get(targ, "width")/2
                    for targ in targets):
                next_state.set(self._robot, "hand", pose)
                next_state.set(held_block, "pose", new_pose)
                next_state.set(held_block, "grasp", -1)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._IsBlock, self._IsTarget, self._Covers, self._HandEmpty,
            self._Holding
        }

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
        return Box(0, 1, (1, ))  # same as option param space

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
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
            plt.plot([hand_lb, hand_rb], [-0.08, -0.08],
                     color="red",
                     alpha=0.5,
                     lw=8.,
                     label=label)
        # Draw hand
        plt.scatter(state.get(self._robot, "hand"),
                    0.05,
                    color="r",
                    s=100,
                    alpha=1.,
                    zorder=10,
                    label="Hand")
        lw = 3
        height = 0.1
        cs = ["blue", "purple", "green", "yellow"]
        block_alpha = 0.75
        targ_alpha = 0.25
        # Draw blocks
        for i, block in enumerate(state.get_objects(self._block_type)):
            c = cs[i]
            if state.get(block, "grasp") != -1:
                lcolor = "red"
                pose = state.get(self._robot, "hand") - state.get(
                    block, "grasp")
                suffix = " (grasped)"
            else:
                lcolor = "gray"
                pose = state.get(block, "pose")
                suffix = ""
            rect = plt.Rectangle(
                (pose - state.get(block, "width") / 2., -height / 2.),
                state.get(block, "width"),
                height,
                linewidth=lw,
                edgecolor=lcolor,
                facecolor=c,
                alpha=block_alpha,
                label=f"block{i}" + suffix)
            ax.add_patch(rect)
        # Draw targets
        for i, targ in enumerate(state.get_objects(self._target_type)):
            c = cs[i]
            rect = plt.Rectangle(
                (state.get(targ, "pose") - state.get(targ, "width") / 2.,
                 -height / 2.),
                state.get(targ, "width"),
                height,
                linewidth=lw,
                edgecolor=lcolor,
                facecolor=c,
                alpha=targ_alpha,
                label=f"target{i}")
            ax.add_patch(rect)
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.25, 0.5)
        plt.yticks([])
        plt.legend()
        if caption is not None:
            plt.suptitle(caption, wrap=True)
        plt.tight_layout()
        img = utils.fig2data(fig)
        plt.close()
        return [img]

    def _get_hand_regions(self, state: State) -> List[Tuple[float, float]]:
        hand_regions = []
        for block in state.get_objects(self._block_type):
            hand_regions.append(
                (state.get(block, "pose") - state.get(block, "width") / 2,
                 state.get(block, "pose") + state.get(block, "width") / 2))
        for targ in state.get_objects(self._target_type):
            hand_regions.append(
                (state.get(targ, "pose") - state.get(targ, "width") / 10,
                 state.get(targ, "pose") + state.get(targ, "width") / 10))
        return hand_regions

    def _create_blocks_and_targets(self) -> Tuple[List[Object], List[Object]]:
        blocks = []
        targets = []
        for i in range(CFG.cover_num_blocks):
            blocks.append(Object(f"block{i}", self._block_type))
        for i in range(CFG.cover_num_targets):
            targets.append(Object(f"target{i}", self._target_type))
        return blocks, targets

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        tasks = []
        # Create blocks and targets.
        blocks, targets = self._create_blocks_and_targets()
        # Create goals.
        goal1 = {GroundAtom(self._Covers, [blocks[0], targets[0]])}
        goals = [goal1]
        if len(blocks) > 1 and len(targets) > 1:
            goal2 = {GroundAtom(self._Covers, [blocks[1], targets[1]])}
            goals.append(goal2)
            goal3 = {
                GroundAtom(self._Covers, [blocks[0], targets[0]]),
                GroundAtom(self._Covers, [blocks[1], targets[1]])
            }
            goals.append(goal3)
        for i in range(num):
            init = self._create_initial_state(blocks, targets, rng)
            assert init.get_objects(self._block_type) == blocks
            assert init.get_objects(self._target_type) == targets
            tasks.append(Task(init, goals[i % len(goals)]))
        return tasks

    def _create_initial_state(self, blocks: List[Object],
                              targets: List[Object],
                              rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        assert len(CFG.cover_block_widths) == len(blocks)
        for block, width in zip(blocks, CFG.cover_block_widths):
            while True:
                pose = rng.uniform(width / 2, 1.0 - width / 2)
                if not self._any_intersection(pose, width, data):
                    break
            # [is_block, is_target, width, pose, grasp]
            data[block] = np.array([1.0, 0.0, width, pose, -1.0])
        assert len(CFG.cover_target_widths) == len(targets)
        for target, width in zip(targets, CFG.cover_target_widths):
            while True:
                pose = rng.uniform(width / 2, 1.0 - width / 2)
                if not self._any_intersection(
                        pose, width, data, larger_gap=True):
                    break
            # [is_block, is_target, width, pose]
            data[target] = np.array([0.0, 1.0, width, pose])
        # [hand, pose_x, pose_z]
        # For the non-PyBullet environments, pose_x and pose_z are constant.
        data[self._robot] = np.array([0.5, self._workspace_x, self._workspace_z])
        state = State(data)
        # Allow some chance of holding a block in the initial state.
        if rng.uniform() < CFG.cover_initial_holding_prob:
            block = blocks[rng.choice(len(blocks))]
            pick_pose = state.get(block, "pose")
            if self._initial_pick_offsets:
                offset = rng.choice(self._initial_pick_offsets)
                assert -1.0 < offset < 1.0, \
                    "initial pick offset should be between -1 and 1"
                pick_pose += state.get(block, "width") * offset / 2.
            temp_env = CoverEnv()
            action = Action(np.array([pick_pose], dtype=np.float32))
            state = temp_env.simulate(state, action)
            assert self._Holding_holds(state, [block])
        return state

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
               (block_pose+block_width/2 >= target_pose+target_width/2) and \
               state.get(block, "grasp") == -1

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
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
    def _PickPlace_policy(state: State, memory: Dict,
                          objects: Sequence[Object], params: Array) -> Action:
        del state, memory, objects  # unused
        return Action(params)  # action is simply the parameter

    def _any_intersection(self,
                          pose: float,
                          width: float,
                          data: Dict[Object, Array],
                          block_only: bool = False,
                          larger_gap: bool = False,
                          excluded_object: Optional[Object] = None) -> bool:
        mult = 1.5 if larger_gap else 0.5
        for other in data:
            if block_only and other.type != self._block_type:
                continue
            if other == excluded_object:
                continue
            other_feats = data[other]
            distance = abs(other_feats[3] - pose)
            if distance <= (width + other_feats[2]) * mult:
                return True
        return False


class CoverEnvTypedOptions(CoverEnv):
    """Toy cover domain with options that have object arguments.

    This means we need two options (one for block, one for target).
    """

    def __init__(self) -> None:
        super().__init__()
        del self._PickPlace
        self._Pick: ParameterizedOption = utils.SingletonParameterizedOption(
            "Pick",
            self._Pick_policy,
            types=[self._block_type],
            params_space=Box(-0.1, 0.1, (1, )))
        self._Place: ParameterizedOption = utils.SingletonParameterizedOption(
            "Place",
            self._PickPlace_policy,  # use the parent class's policy
            types=[self._target_type],
            params_space=Box(0, 1, (1, )))

    @classmethod
    def get_name(cls) -> str:
        return "cover_typed_options"

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Pick, self._Place}

    def _Pick_policy(self, s: State, m: Dict, o: Sequence[Object],
                     p: Array) -> Action:
        del m  # unused
        _ = self  # unused
        # The pick parameter is a RELATIVE position, so we need to
        # add the pose of the object.
        pick_pose = s.get(o[0], "pose") + p[0]
        pick_pose = min(max(pick_pose, 0.0), 1.0)
        return Action(np.array([pick_pose], dtype=np.float32))


class CoverEnvHierarchicalTypes(CoverEnv):
    """Toy cover domain with hierarchical types, just for testing."""

    def __init__(self) -> None:
        super().__init__()
        # Change blocks to be of a derived type
        self._parent_block_type = self._block_type
        self._block_type = Type(
            "block_derived",
            ["is_block", "is_target", "width", "pose", "grasp"],
            parent=self._parent_block_type)

    @classmethod
    def get_name(cls) -> str:
        return "cover_hierarchical_types"

    @property
    def types(self) -> Set[Type]:
        return {
            self._block_type, self._parent_block_type, self._target_type,
            self._robot_type
        }


class CoverEnvRegrasp(CoverEnv):
    """A cover environment that is not always downward refinable, because the
    grasp on the initially held object sometimes requires placing and
    regrasping.

    This environment also has two different oracle NSRTs for placing, one for
    placing a target and one for placing on the table.

    This environment also has a Clear predicate, to prevent placing on already
    covered targets.

    Finally, to allow placing on the table, we need to change the allowed
    hand regions. We implement it so that there is a relatively small hand
    region centered at each target, but then everywhere else is allowed.
    """
    _allow_free_space_placing: ClassVar[bool] = True
    _initial_pick_offsets: ClassVar[List[float]] = [-0.95, 0.0, 0.95]

    def __init__(self) -> None:
        super().__init__()
        # Add a Clear predicate to prevent attempts at placing on already
        # covered targets.
        self._Clear = Predicate("Clear", [self._target_type],
                                self._Clear_holds)

    @classmethod
    def get_name(cls) -> str:
        return "cover_regrasp"

    @property
    def predicates(self) -> Set[Predicate]:
        return super().predicates | {self._Clear}

    def _get_hand_regions(self, state: State) -> List[Tuple[float, float]]:
        hand_regions = []
        # Construct the allowed hand regions from left to right.
        left_bound = 0.0
        targets = state.get_objects(self._target_type)
        for targ in sorted(targets, key=lambda t: state.get(t, "pose")):
            w = state.get(targ, "width")
            targ_left = state.get(targ, "pose") - w / 2
            targ_right = state.get(targ, "pose") + w / 2
            hand_regions.append((left_bound, targ_left - w))
            hand_regions.append((targ_left + w / 3, targ_right - w / 3))
            left_bound = targ_right + w
        hand_regions.append((left_bound, 1.0))
        return hand_regions

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        assert len(objects) == 1
        target = objects[0]
        for b in state:
            if b.type != self._block_type:
                continue
            if self._Covers_holds(state, [b, target]):
                return False
        return True


class CoverMultistepOptions(CoverEnvTypedOptions):
    """Cover domain with a lower level action space. Useful for using and
    learning multistep options.

    The action space is (dx, dy, dgrip). The last dimension
    controls the gripper "magnet" or "vacuum". The state space is updated to
    track x, y, grip.

    The robot can move anywhere as long as it, and the block it may be holding,
    does not collide with another block. Picking up a block is allowed when the
    robot gripper is empty, when the robot is in the allowable hand region, and
    when the robot is sufficiently close to the block in the y-direction.
    Placing is allowed anywhere. Collisions are handled in simulate().
    """
    grasp_height_tol: ClassVar[float] = 1e-2
    grasp_thresh: ClassVar[float] = 0.0
    initial_block_y: ClassVar[float] = 0.1
    block_height: ClassVar[float] = 0.1
    target_height: ClassVar[float] = 0.1  # Only for rendering purposes.
    placing_height: ClassVar[
        float] = 0.1  # A block's base must be below this to be placed.
    initial_robot_y: ClassVar[float] = 0.4
    collision_threshold: ClassVar[float] = 1e-5

    def __init__(self) -> None:
        super().__init__()
        # Need to now include y and gripper info in state.
        # Removing "pose" because that's ambiguous.
        # Also adding height to blocks.
        # The y position corresponds to the top of the block.
        # The x position corresponds to the center of the block.
        self._block_type = Type(
            "block",
            ["is_block", "is_target", "width", "x", "grasp", "y", "height"])
        # Targets don't need y because they're constant.
        self._target_type = Type("target",
                                 ["is_block", "is_target", "width", "x"])
        # Also removing "hand" because that's ambiguous.
        self._robot_type = Type("robot", ["x", "y", "grip", "holding"])
        self._block_hand_region_type = Type("block_hand_region",
                                            ["lb", "ub", "block_idx"])
        self._target_hand_region_type = Type("target_hand_region",
                                             ["lb", "ub"])

        # Need to override predicate creation because the types are
        # now different (in terms of equality).
        self._IsBlock = Predicate("IsBlock", [self._block_type],
                                  self._IsBlock_holds)
        self._IsTarget = Predicate("IsTarget", [self._target_type],
                                   self._IsTarget_holds)
        self._Covers = Predicate("Covers",
                                 [self._block_type, self._target_type],
                                 self._Covers_holds)
        self._HandEmpty = Predicate("HandEmpty", [], self._HandEmpty_holds)
        self._Holding = Predicate("Holding",
                                  [self._block_type, self._robot_type],
                                  self._Holding_holds)
        # Need to override static object creation because the types are now
        # different (in terms of equality).
        self._robot = Object("robby", self._robot_type)
        # Override the original options to make them multi-step.
        self._Pick = ParameterizedOption("Pick",
                                         types=[self._block_type],
                                         params_space=Box(-1.0, 1.0, (1, )),
                                         policy=self._Pick_policy,
                                         initiable=self._Pick_initiable,
                                         terminal=self._Pick_terminal)
        # Note: there is a change here -- the parameter space is now
        # relative to the target. In the parent env, the parameter
        # space is absolute, and the state of the target is not used.
        self._Place = ParameterizedOption("Place",
                                          types=[self._target_type],
                                          params_space=Box(-1.0, 1.0, (1, )),
                                          policy=self._Place_policy,
                                          initiable=self._Place_initiable,
                                          terminal=self._Place_terminal)
        # We also add two ground truth options that correspond to the options
        # learned by the _SimpleOptionLearner. The parameter for these options
        # is a concatenation of several vectors, where each vector corresponds
        # to a sampled state vector in the option's terminal state for an object
        # whose state changes after executing the option.
        self._LearnedEquivalentPick = ParameterizedOption(
            "LearnedEquivalentPick",
            types=[self._block_type, self._robot_type],
            params_space=Box(-np.inf, np.inf, (11, )),
            policy=self._Pick_learned_equivalent_policy,
            initiable=self._Pick_learned_equivalent_initiable,
            terminal=self._Pick_learned_equivalent_terminal)
        self._LearnedEquivalentPlace = ParameterizedOption(
            "LearnedEquivalentPlace",
            types=[self._block_type, self._robot_type, self._target_type],
            params_space=Box(-np.inf, np.inf, (11, )),
            policy=self._Place_learned_equivalent_policy,
            initiable=self._Place_learned_equivalent_initiable,
            terminal=self._Place_learned_equivalent_terminal)

    @classmethod
    def get_name(cls) -> str:
        return "cover_multistep_options"

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {
            self._Pick, self._Place, self._LearnedEquivalentPick,
            self._LearnedEquivalentPlace
        }

    @property
    def action_space(self) -> Box:
        # This is the main difference with respect to the parent
        # env. The action space now is (dx, dy, dgrip). The
        # last dimension controls the gripper "magnet" or "vacuum".
        # Note that the bounds are relatively low, which necessitates
        # multi-step options.
        lb, ub = CFG.cover_multistep_action_limits
        return Box(lb, ub, (3, ))

    def simulate(self, state: State, action: Action) -> State:
        # Since the action space is lower level, we need to write
        # a lower level simulate function.
        assert self.action_space.contains(action.arr)

        # Get data needed to collision check the trajectory.
        dx, dy, dgrip = action.arr
        next_state = state.copy()
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        grip = state.get(self._robot, "grip")
        blocks = state.get_objects(self._block_type)
        held_block = None
        for block in blocks:
            if state.get(block, "grasp") != -1:
                assert held_block is None
                held_block = block
                hx, hy = state.get(held_block, "x"), \
                           state.get(held_block, "y")
                hw, hh = state.get(held_block, "width"), \
                         state.get(held_block, "height")

        # Ensure neither the gripper nor the possible held block go below the
        # y-axis.
        if y + dy < 0 - self.collision_threshold:
            return state.copy()
        if held_block is not None:
            if hy - hh + dy < 0 - self.collision_threshold:
                return state.copy()

        # Ensure neither the gripper nor the possible held block collide with
        # another block during the trajectory defined by dx, dy.
        p1 = (x, y)
        p2 = (x + dx, y + dy)
        for block in blocks:
            bx, by = state.get(block, "x"), state.get(block, "y")
            bw, bh = state.get(block, "width"), state.get(block, "height")
            ct = self.collision_threshold
            # These segments defines a slightly smaller rectangle to prevent
            # floating point arithmetic making us declare a false positive.
            segments = [
                ((bx - bw / 2 + ct, by - ct), (bx + bw / 2 - ct,
                                               by - ct)),  # top
                ((bx + bw / 2 - ct, by - ct), (bx + bw / 2 - ct,
                                               by - bh)),  # right
                ((bx - bw / 2 + ct, by - ct), (bx - bw / 2 + ct, by - bh)
                 )  # left
            ]
            # Check if the robot collides with a block.
            if held_block is not None and block == held_block:
                continue
            if any(utils.intersects(p1, p2, p3, p4) for p3, p4 in segments):
                return state.copy()
            # Check if the held_block collides with a block.
            # For each of the four vertices of our held block, construct the
            # line segment from its old location to its new location, and check
            # if this line segment intersects any segment of the block. Also
            # check if the blocks overlap.
            if held_block is None:
                continue
            # Check translations.
            vertices = [(hx-hw/2, hy), (hx+hw/2, hy), \
                        (hx-hw/2, hy-hh), (hx+hw/2, hy-hh)]
            translations = [((vx, vy), (vx+dx, vy+dy)) \
                            for vx, vy in vertices]
            if any(utils.intersects(p1, p2, p3, p4) \
                for p1, p2 in translations for p3, p4 in segments):
                return state.copy()
            # Check overlap
            l1, r1 = (hx - hw / 2 + dx, hy + dy), (hx + hw / 2 + dx,
                                                   hy - hh + dy)
            l2, r2 = (bx - bw / 2, by), (bx + bw / 2, by - bh)
            if utils.overlap(l1, r1, l2, r2):
                return state.copy()

        # No collisions; update robot and possible held block state based on
        # action.
        x += dx
        y += dy
        grip = dgrip  # desired grip; set directly
        next_state.set(self._robot, "x", x)
        next_state.set(self._robot, "y", y)
        next_state.set(self._robot, "grip", grip)
        if held_block is not None:
            hx = hx + dx
            hy = hy + dy
            next_state.set(held_block, "x", hx)
            next_state.set(held_block, "y", hy)

        # Check if we are above a block.
        above_block = None
        for block in blocks:
            block_x_lb = state.get(block, "x") - state.get(block, "width") / 2
            block_x_ub = state.get(block, "x") + state.get(block, "width") / 2
            if state.get(block, "grasp") == -1 and \
               block_x_lb <= x <= block_x_ub:
                assert above_block is None
                above_block = block

        # If we're not holding anything and we're close enough to a block, grasp
        # it if the gripper is on and we are in the allowed grasping region.
        # Note: unlike parent env, we also need to check the grip.
        if held_block is None and above_block is not None and \
            grip > self.grasp_thresh and any(hand_lb <= x <= hand_rb for
            hand_lb, hand_rb in self._get_hand_regions_block(state)):
            by = state.get(above_block, "y")
            by_ub = by + self.grasp_height_tol
            by_lb = by - self.grasp_height_tol
            if by_lb <= y <= by_ub:
                next_state.set(self._robot, "y", by)
                next_state.set(above_block, "grasp", 1)
                next_state.set(self._robot, "holding", 1)

        # If we are holding something and we're not above a block, place it if
        # the gripper is off and we are low enough. Placing anywhere is allowed
        # but if we are over a target, we must be in its hand region. Possible
        # overlaps with other blocks is handled by the collision checker.
        # Note: unlike parent env, we also need to check the grip.
        if held_block is not None and above_block is None and \
            grip < self.grasp_thresh and (hy-hh) < self.placing_height:
            # Tentatively set the next state and check whether the placement
            # would cover some target.
            next_state.set(held_block, "y", self.initial_block_y)
            next_state.set(held_block, "grasp", -1)
            next_state.set(self._robot, "holding", -1)
            targets = state.get_objects(self._target_type)
            place_would_cover = any(
                self._Covers_holds(next_state, [held_block, targ])
                for targ in targets)
            # If the place would cover, but we were outside of an allowed
            # hand region, then disallow the place. Otherwise, keep the new
            # next state (place succeeded).
            if place_would_cover and not any(hand_lb <= x <= hand_rb for \
                hand_lb, hand_rb in self._get_hand_regions_target(state)):
                return state.copy()
        return next_state

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        # Need to override rendering to account for new state features.
        fig, ax = plt.subplots(1, 1)
        # Draw main line
        plt.plot([-0.2, 1.2], [-0.001, -0.001], color="black", linewidth=0.4)
        # Draw hand regions
        block_hand_regions = self._get_hand_regions_block(state)
        target_hand_regions = self._get_hand_regions_target(state)
        hand_regions = block_hand_regions + target_hand_regions
        for i, (hand_lb, hand_rb) in enumerate(hand_regions):
            if i == 0:
                label = "Allowed hand region"
            else:
                label = None
            plt.plot([hand_lb, hand_rb], [-0.08, -0.08],
                     color="red",
                     alpha=0.5,
                     lw=4.,
                     label=label)
        # Draw hand
        plt.scatter(state.get(self._robot, "x"),
                    state.get(self._robot, "y"),
                    color="r",
                    s=100,
                    alpha=1.,
                    zorder=10,
                    label="Hand")
        plt.plot([state.get(self._robot, "x"),
                  state.get(self._robot, "x")],
                 [1, state.get(self._robot, "y")],
                 color="r",
                 alpha=1.,
                 zorder=10,
                 label=None)

        lw = 2
        cs = ["blue", "purple", "green", "yellow"]
        block_alpha = 0.75
        targ_alpha = 0.25

        # Draw blocks
        for i, block in enumerate(state.get_objects(self._block_type)):
            c = cs[i]
            bx, by = state.get(block, "x"), state.get(block, "y")
            bw = state.get(block, "width")
            bh = state.get(block, "height")
            if state.get(block, "grasp") != -1:
                lcolor = "red"
                suffix = " (grasped)"
            else:
                lcolor = "gray"
                suffix = ""
            rect = plt.Rectangle((bx - bw / 2., by - bh),
                                 bw,
                                 bh,
                                 linewidth=lw,
                                 edgecolor=lcolor,
                                 facecolor=c,
                                 alpha=block_alpha,
                                 label=f"block{i}" + suffix)
            ax.add_patch(rect)
        # Draw targets
        for i, targ in enumerate(state.get_objects(self._target_type)):
            c = cs[i]
            rect = plt.Rectangle(
                (state.get(targ, "x") - state.get(targ, "width") / 2., 0.0),
                state.get(targ, "width"),
                self.target_height,
                linewidth=0,
                edgecolor=lcolor,
                facecolor=c,
                alpha=targ_alpha,
                label=f"target{i}")
            ax.add_patch(rect)
        grip = state.get(self._robot, "grip")
        plt.title(f"Grip: {grip:.3f}")
        plt.xlim(-0.2, 1.2)
        plt.ylim(-0.25, 1)
        plt.legend()
        if caption is not None:
            plt.suptitle(caption, wrap=True)
        plt.tight_layout()
        img = utils.fig2data(fig)
        plt.close()
        return [img]

    def _create_initial_state(self, blocks: List[Object],
                              targets: List[Object],
                              rng: np.random.Generator) -> State:
        """Creates initial state by (1) placing targets and blocks in random
        locations such that each target has enough space on either side to
        ensure no covering placement will cause a collision (note that this is
        not necessary to make the task solvable; but we can do this and instead
        sufficiently tune the difficulty through hand region specification),
        and (2) choosing hand region intervals on the targets and blocks such
        that the problem is solvable."""
        assert len(blocks) == CFG.cover_num_blocks
        assert len(targets) == CFG.cover_num_targets

        data: Dict[Object, Array] = {}

        # Create hand regions for each block and target.
        block_hand_region_objs = []
        target_hand_region_objs = []
        for i in range(CFG.cover_num_blocks):
            block_hand_region_objs.append(
                Object(f"block{i}_hand_region", self._block_hand_region_type))
        for i in range(CFG.cover_num_targets):
            target_hand_region_objs.append(
                Object(f"target{i}_hand_region",
                       self._target_hand_region_type))

        # Place targets and blocks
        counter = 0
        while True:
            overlap = False
            counter += 1
            if counter > CFG.cover_multistep_max_tb_placements:
                raise RuntimeError("Reached maximum number of " \
                "placements of targets and blocks.")
            block_placements = []
            for block, bw in zip(blocks, CFG.cover_block_widths):
                xb = rng.uniform(bw / 2, 1 - bw / 2)
                left_pt = xb - bw / 2
                right_pt = xb + bw / 2
                block_placements.append((left_pt, xb, right_pt))
            target_placements = []
            for target, tw, bw in zip(targets, CFG.cover_target_widths,
                                      CFG.cover_block_widths):
                xt = rng.uniform(bw - tw / 2, 1 - bw + tw / 2)
                left_pt = xt + tw / 2 - bw
                right_pt = xt - tw / 2 + bw
                target_placements.append((left_pt, xt, right_pt))
            # check for overlap
            all_placements = target_placements + block_placements
            all_placements.sort(key=lambda x: x[0])
            for i in range(1, len(all_placements)):
                curr = all_placements[i]
                prev = all_placements[i - 1]
                if curr[0] < prev[2]:
                    overlap = True
            if not overlap:
                break

        # Make the targets, blocks, and robot objects
        for target, width, placement in zip(targets, CFG.cover_target_widths,
                                            target_placements):
            _, x, _ = placement
            # [is_block, is_target, width, x]
            data[target] = np.array([0.0, 1.0, width, x])
        for block, width, placement in zip(blocks, CFG.cover_block_widths,
                                           block_placements):
            _, x, _ = placement
            # [is_block, is_target, width, x, grasp, y, height]
            data[block] = np.array([
                1.0, 0.0, width, x, -1.0, self.initial_block_y,
                self.block_height
            ])
        # [x, y, grip, holding]
        data[self._robot] = np.array([0.0, self.initial_robot_y, -1.0, -1.0])

        # Make the hand regions
        # Sample a hand region interval in each target
        target_hand_regions = []
        for i, target in enumerate(targets):
            target_hr = target_hand_region_objs[i]
            tw = CFG.cover_target_widths[i]
            _, x, _ = target_placements[i]
            region_length = tw * CFG.cover_multistep_thr_percent
            if CFG.cover_multistep_bimodal_goal:
                if rng.uniform(0, 1) < 0.5:
                    left_pt = x - tw / 2
                    region = [left_pt, left_pt + region_length]
                else:
                    right_pt = x + tw / 2
                    region = [right_pt - region_length, right_pt]
            else:
                left_pt = rng.uniform(x - tw / 2, x + tw / 2 - region_length)
                region = [left_pt, left_pt + region_length]
            data[target_hr] = np.array(region)
            target_hand_regions.append(region)

        # Sample a hand region interval in each block
        for i, block in enumerate(blocks):
            block_hr = block_hand_region_objs[i]
            thr_left, thr_right = target_hand_regions[i]
            bw = CFG.cover_block_widths[i]
            tw = CFG.cover_target_widths[i]
            _, bx, _ = block_placements[i]
            _, tx, _ = target_placements[i]
            region_length = bw * CFG.cover_multistep_bhr_percent

            # The hand region we assign must not make it impossible to
            # cover the block's target.
            # To check this, we perform the following operation:
            # "Place" the block in the leftmost position that still covers
            # the target. Move the block to the right until it reaches the
            # rightmost position that still covers the target. During this,
            # check that there is nonzero overlap between the interval
            # spanned by the moving block's hand region, and the target's
            # hand region. In other words, make sure that there is at least
            # one placement of the block which covers the target, and in
            # which the block's hand region and the target's hand region
            # have nonzero overlap.
            # A proxy for this operation, which we do below, is to check:
            # (1) That in the block's rightmost covering placement, its
            # interval IS NOT completely to the left of the target's
            # hand region, and
            # (2) That in the block's leftmost covering placement, its
            # interval IS NOT completely to the right of the target's
            # hand region.
            counter = 0
            while True:
                counter += 1
                if counter > CFG.cover_multistep_max_hr_placements:
                    raise RuntimeError("Reached maximum number of " \
                    "placements of hand regions.")
                # Sample hand region
                left_pt = rng.uniform(bx - bw / 2, bx + bw / 2 - region_length)
                region = [left_pt, left_pt + region_length]
                # Need to make hand region relative to center of block for
                # the hand region to move with the block for use by
                # _get_hand_regions()
                relative_region = [region[0] - bx, region[1] - bx]
                # Perform the valid interval check
                relative_r = region[1] - (bx - bw / 2)  # for (1)
                relative_l = bx + bw / 2 - region[0]  # for (2)
                if relative_l >= (tx + tw/2 - thr_right) and \
                    relative_r >= (thr_left-(tx - tw/2)):
                    break
            # Store the block index (i) in the hand region object.
            data[block_hr] = np.array(relative_region + [i])

        return State(data)

    def _get_hand_regions_block(self, state: State) \
        -> List[Tuple[float, float]]:
        blocks = state.get_objects(self._block_type)
        block_hand_regions = state.get_objects(self._block_hand_region_type)
        hand_regions = []
        for block_hr in block_hand_regions:
            block_idx_flt = state.get(block_hr, "block_idx")
            assert block_idx_flt.is_integer()
            block_idx = int(block_idx_flt)
            block = blocks[block_idx]
            hand_regions.append(
                (state.get(block, "x") + state.get(block_hr, "lb"),
                 state.get(block, "x") + state.get(block_hr, "ub")))
        return hand_regions

    def _get_hand_regions_target(self, state: State) \
        -> List[Tuple[float, float]]:
        hand_regions = []
        target_hand_regions = state.get_objects(self._target_hand_region_type)
        for target_hr in target_hand_regions:
            hand_regions.append((state.get(target_hr,
                                           "lb"), state.get(target_hr, "ub")))
        return hand_regions

    def _Pick_initiable(self, s: State, m: Dict, o: Sequence[Object],
                        p: Array) -> bool:
        # Pick is initiable if the hand is empty.
        del m, o, p  # unused
        return self._HandEmpty_holds(s, [])

    def _Pick_learned_equivalent_initiable(self, s: State, m: Dict,
                                           o: Sequence[Object],
                                           p: Array) -> bool:
        # Convert the relative parameters into absolute parameters.
        m["params"] = p
        m["absolute_params"] = s.vec(o) + p
        return self._Pick_initiable(s, m, o, p)

    def _Pick_policy(self, s: State, m: Dict, o: Sequence[Object],
                     p: Array) -> Action:
        del m  # unused
        # The object is the one we want to pick.
        assert len(o) == 1
        obj = o[0]
        # The parameter is a relative x position to pick at.
        assert len(p) == 1
        rel_x = p.item()
        assert obj.type == self._block_type
        desired_x = s.get(obj, "x") + rel_x * s.get(obj, "width") / 2.
        x = s.get(self._robot, "x")
        at_desired_x = abs(desired_x - x) < 1e-5
        y = s.get(self._robot, "y")
        desired_y = s.get(obj, "y")
        at_desired_y = abs(desired_y - y) < 1e-5
        lb, ub = CFG.cover_multistep_action_limits
        # If we're already above the object and prepared to pick,
        # then execute the pick (turn up the magnet).
        if at_desired_x and at_desired_y:
            return Action(np.array([0., 0., 0.1], dtype=np.float32))
        # If we're above the object but not yet close enough, move down.
        if at_desired_x:
            delta_y = np.clip(desired_y - y, lb, ub)
            return Action(np.array([0., delta_y, 0.], dtype=np.float32))
        # If we're not above the object, but we're at a safe height,
        # then move left/right.
        if y >= self.initial_robot_y:
            delta_x = np.clip(desired_x - x, lb, ub)
            return Action(np.array([delta_x, 0., 0.], dtype=np.float32))
        # If we're not above the object, and we're not at a safe height,
        # then move up.
        delta_y = np.clip(self.initial_robot_y + 1e-2 - y, lb, ub)
        return Action(np.array([0., delta_y, 0.], dtype=np.float32))

    def _Pick_learned_equivalent_policy(
            self,
            s: State,
            m: Dict,  # type: ignore
            o: Sequence[Object],
            p: Array) -> Action:
        assert np.allclose(p, m["params"])
        del p
        absolute_params = m["absolute_params"]
        # The object is the one we want to pick.
        assert len(o) == 2
        obj = o[0]
        assert len(
            absolute_params) == self._block_type.dim + self._robot_type.dim
        assert obj.type == self._block_type
        x = s.get(self._robot, "x")
        y = s.get(self._robot, "y")
        by = s.get(obj, "y")
        desired_x = absolute_params[7]
        desired_y = by + 1e-3
        at_desired_x = abs(desired_x - x) < 1e-5
        at_desired_y = abs(desired_y - y) < 1e-5

        lb, ub = CFG.cover_multistep_action_limits
        # If we're already above the object and prepared to pick,
        # then execute the pick (turn up the magnet).
        if at_desired_x and at_desired_y:
            return Action(np.array([0., 0., 1.0], dtype=np.float32))
        # If we're above the object but not yet close enough, move down.
        if at_desired_x:
            delta_y = np.clip(desired_y - y, lb, ub)
            return Action(np.array([0., delta_y, -1.0], dtype=np.float32))
        # If we're not above the object, but we're at a safe height,
        # then move left/right.
        if y >= self.initial_robot_y:
            delta_x = np.clip(desired_x - x, lb, ub)
            return Action(np.array([delta_x, 0., -1.0], dtype=np.float32))
        # If we're not above the object, and we're not at a safe height,
        # then move up.
        delta_y = np.clip(self.initial_robot_y + 1e-2 - y, lb, ub)
        return Action(np.array([0., delta_y, -1.0], dtype=np.float32))

    def _Pick_terminal(self, s: State, m: Dict, o: Sequence[Object],
                       p: Array) -> bool:
        del m, p  # unused
        block, = o
        # Pick is done when we're holding the desired object.
        return self._Holding_holds(s, [block, self._robot])

    def _Pick_learned_equivalent_terminal(self, s: State, m: Dict,
                                          o: Sequence[Object],
                                          p: Array) -> bool:
        assert np.allclose(p, m["params"])
        # Pick is done when we're holding the desired object.
        return self._Holding_holds(s, o)

    def _Place_initiable(self, s: State, m: Dict, o: Sequence[Object],
                         p: Array) -> bool:
        # Place is initiable if we're holding something.
        # Also may want to eventually check that the target is clear.
        del m, o, p  # unused
        return not self._HandEmpty_holds(s, [])

    def _Place_learned_equivalent_initiable(self, s: State, m: Dict,
                                            o: Sequence[Object],
                                            p: Array) -> bool:
        block, robot, _ = o
        assert block.is_instance(self._block_type)
        assert robot.is_instance(self._robot_type)
        # Convert the relative parameters into absolute parameters.
        m["params"] = p
        # Only the block and robot are changing.
        m["absolute_params"] = s.vec([block, robot]) + p
        # Place is initiable if we're holding the object.
        return self._Holding_holds(s, [block, robot])

    def _Place_policy(self, s: State, m: Dict, o: Sequence[Object],
                      p: Array) -> Action:
        del m  # unused
        # The object is the one we want to place at.
        assert len(o) == 1
        obj = o[0]
        # The parameter is a relative x position to place at.
        assert len(p) == 1
        rel_x = p.item()
        assert obj.type == self._target_type
        desired_x = s.get(obj, "x") + rel_x * s.get(obj, "width") / 2.
        x = s.get(self._robot, "x")
        at_desired_x = abs(desired_x - x) < 1e-5
        y = s.get(self._robot, "y")
        desired_y = self.block_height + 1e-3
        at_desired_y = abs(desired_y - y) < 1e-5
        lb, ub = CFG.cover_multistep_action_limits
        # If we're already above the object and prepared to place,
        # then execute the place (turn down the magnet).
        if at_desired_x and at_desired_y:
            return Action(np.array([0., 0., -0.1], dtype=np.float32))
        # If we're above the object but not yet close enough, move down.
        if at_desired_x:
            delta_y = np.clip(desired_y - y, lb, ub)
            return Action(np.array([0., delta_y, 0.1], dtype=np.float32))
        # If we're not above the object, but we're at a safe height,
        # then move left/right.
        if y >= self.initial_robot_y:
            delta_x = np.clip(desired_x - x, lb, ub)
            return Action(np.array([delta_x, 0., 0.1], dtype=np.float32))
        # If we're not above the object, and we're not at a safe height,
        # then move up.
        delta_y = np.clip(self.initial_robot_y + 1e-2 - y, lb, ub)
        return Action(np.array([0., delta_y, 0.1], dtype=np.float32))

    def _Place_learned_equivalent_policy(
            self,
            s: State,  # type: ignore
            m: Dict,
            o: Sequence[Object],
            p: Array) -> Action:
        assert np.allclose(p, m["params"])
        del p
        absolute_params = m["absolute_params"]
        # The object is the one we want to place at.
        assert len(o) == 3
        obj = o[0]
        assert len(
            absolute_params) == self._block_type.dim + self._robot_type.dim
        assert obj.type == self._block_type
        x = s.get(self._robot, "x")
        y = s.get(self._robot, "y")
        bh = s.get(obj, "height")
        desired_x = absolute_params[7]
        desired_y = bh + 1e-3

        at_desired_x = abs(desired_x - x) < 1e-5
        at_desired_y = abs(desired_y - y) < 1e-5

        lb, ub = CFG.cover_multistep_action_limits
        # If we're already above the object and prepared to place,
        # then execute the place (turn down the magnet).
        if at_desired_x and at_desired_y:
            return Action(np.array([0., 0., -1.0], dtype=np.float32))
        # If we're above the object but not yet close enough, move down.
        if at_desired_x:
            delta_y = np.clip(desired_y - y, lb, ub)
            return Action(np.array([0., delta_y, 1.0], dtype=np.float32))
        # If we're not above the object, but we're at a safe height,
        # then move left/right.
        if y >= self.initial_robot_y:
            delta_x = np.clip(desired_x - x, lb, ub)
            return Action(np.array([delta_x, 0., 1.0], dtype=np.float32))
        # If we're not above the object, and we're not at a safe height,
        # then move up.
        delta_y = np.clip(self.initial_robot_y + 1e-2 - y, lb, ub)
        return Action(np.array([0., delta_y, 1.0], dtype=np.float32))

    def _Place_terminal(self, s: State, m: Dict, o: Sequence[Object],
                        p: Array) -> bool:
        del m, o, p  # unused
        # Place is done when the hand is empty.
        return self._HandEmpty_holds(s, [])

    def _Place_learned_equivalent_terminal(self, s: State, m: Dict,
                                           o: Sequence[Object],
                                           p: Array) -> bool:
        del o  # unused
        assert np.allclose(p, m["params"])
        # Place is done when the hand is empty.
        return self._HandEmpty_holds(s, [])

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        block, robot = objects
        return state.get(block, "grasp") != -1 and \
            state.get(robot, "holding") != -1

    @staticmethod
    def _Covers_holds(state: State, objects: Sequence[Object]) -> bool:
        # Overriding because of the change from "pose" to "x" and because
        # block's x-position is updated in every step of simulate and not just
        # at the end of a place() operation so we cannot allow the predicate to
        # hold when the block is in the air.
        block, target = objects
        block_pose = state.get(block, "x")
        block_width = state.get(block, "width")
        target_pose = state.get(target, "x")
        target_width = state.get(target, "width")
        by, bh = state.get(block, "y"), state.get(block, "height")
        return (block_pose-block_width/2 <= target_pose-target_width/2) and \
               (block_pose+block_width/2 >= target_pose+target_width/2) and \
               (by - bh == 0)


class CoverMultistepOptionsFixedTasks(CoverMultistepOptions):
    """A variation of CoverMultistepOptions where there is only one possible
    initial state.

    This environment is useful for debugging option learning.

    Note that like the parent env, there are three possible goals:
    Cover(block0, target0), Cover(block1, target1), or both.
    """

    @classmethod
    def get_name(cls) -> str:
        return "cover_multistep_options_fixed_tasks"

    def _create_initial_state(self, blocks: List[Object],
                              targets: List[Object],
                              rng: np.random.Generator) -> State:
        # Force one initial state by overriding the rng and using an identical
        # one on every call, so this method becomes deterministic.
        del rng
        zero_rng = np.random.default_rng(0)
        return super()._create_initial_state(blocks, targets, zero_rng)


class CoverMultistepOptionsHolding(CoverMultistepOptions):
    """A variation of CoverMultistepOptions where the goals only involve
    Holding.

    This environment is useful for debugging option learning.
    """

    @classmethod
    def get_name(cls) -> str:
        return "cover_multistep_options_holding"

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Holding}

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[Task]:
        tasks = []
        blocks, targets = self._create_blocks_and_targets()
        for _ in range(num):
            init = self._create_initial_state(blocks, targets, rng)
            assert init.get_objects(self._block_type) == blocks
            assert init.get_objects(self._target_type) == targets
            goal_block_idx = rng.choice(len(blocks))
            goal_block = blocks[goal_block_idx]
            goal = {GroundAtom(self._Holding, [goal_block, self._robot])}
            tasks.append(Task(init, goal))
        return tasks
