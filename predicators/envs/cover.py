"""Toy cover domain.

This environment IS downward refinable (low-level search won't ever
fail), but it still requires backtracking.
"""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type


class CoverEnv(BaseEnv):
    """Toy cover domain."""

    _allow_free_space_placing: ClassVar[bool] = False
    _initial_pick_offsets: ClassVar[List[float]] = []  # see CoverEnvRegrasp

    workspace_x: ClassVar[float] = 1.35
    workspace_z: ClassVar[float] = 0.65

    # # Types
    # _block_type = Type("block",
    #                    ["is_block", "is_target", "width", "pose_y_norm", "grasp"])
    # _target_type = Type("target", ["is_block", "is_target", "width", "pose_y_norm"])
    # _robot_type = Type("robot", ["pose_y_norm", "pose_x", "pose_z"])
    # _table_type = Type("table", [])

    # Types
    bbox_features = ["bbox_left", "bbox_right", "bbox_upper", "bbox_lower"]
    _block_type = Type("block",
                    ["is_block", "is_target", "width", "pose_y_norm", "grasp"] +
                        bbox_features)
    _target_type = Type("target", 
                    ["is_block", "is_target", "width", "pose_y_norm"] +
                        bbox_features)
    _robot_type = Type("robot", ["pose_y_norm", "pose_x", "pose_z", "fingers"] +
                        bbox_features)
    _table_type = Type("table", bbox_features)


    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._IsBlock = Predicate("IsBlock", [self._block_type],
                                self._IsBlock_holds)
        self._IsTarget = Predicate("IsTarget", [self._target_type],
                                self._IsTarget_holds)
        self._Covers = Predicate("Covers",
                                [self._block_type, self._target_type],
                                self._Covers_holds)
        # IsClear(y) is Forall x:box. not Covers(x, y) -- quantify all but one
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)
        # HandEmpty() is Forall x. not Holding(x) -- quantify over all vars
        self._HandEmpty = Predicate("HandEmpty", [], self._HandEmpty_holds)

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._table = Object("table", self._table_type)

    @classmethod
    def get_name(cls) -> str:
        return "cover"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        pose = action.arr.item()
        next_state = state.copy()
        hand_regions = self._get_hand_regions(state)
        # If we're not in any hand region, noop.
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
            block_lb = state.get(block, "pose_y_norm") - state.get(block, 
                        "width") / 2
            block_ub = state.get(block, "pose_y_norm") + state.get(block, 
                        "width") / 2
            if state.get(block,
                         "grasp") == -1 and block_lb <= pose <= block_ub:
                assert above_block is None
                above_block = block
        # If we're not holding anything and we're above a block, grasp it.
        # The grasped block's pose stays the same.
        if held_block is None and above_block is not None:
            grasp = pose - state.get(above_block, "pose_y_norm")
            next_state.set(self._robot, "pose_y_norm", pose)
            if "hand_empty" in self._robot_type.feature_names:
                # See CoverEnvHandEmpty
                next_state.set(self._robot, "hand_empty", 0)
            if "fingers" in self._robot_type.feature_names:
                next_state.set(self._robot, "fingers", 0)
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
                any(state.get(targ, "pose_y_norm")-state.get(targ, "width")/2
                    <= pose <=
                    state.get(targ, "pose_y_norm")+state.get(targ, "width")/2
                    for targ in targets):
                next_state.set(self._robot, "pose_y_norm", pose)
                if "hand_empty" in self._robot_type.feature_names:
                    # See CoverEnvHandEmpty
                    next_state.set(self._robot, "hand_empty", 1)
                if "fingers" in self._robot_type.feature_names:
                    next_state.set(self._robot, "fingers", 1)
                next_state.set(held_block, "pose_y_norm", new_pose)
                next_state.set(held_block, "grasp", -1)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng,
                                is_train=True)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng,
                                is_train=False)

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
    def action_space(self) -> Box:
        return Box(0, 1, (1, ))  # same as option param space

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
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
                     lw=1.,
                     label=label)
        # Draw hand
        plt.scatter(state.get(self._robot, "pose_y_norm"),
                    0.05,
                    color="r",
                    s=100,
                    alpha=1.,
                    zorder=10,
                    label="pose_y_norm")
        lw = 3
        height = 0.1
        cs = ["blue", "purple", "green", "yellow"]
        block_alpha = 0.75
        targ_alpha = 0.25
        # Draw blocks
        for i, block in enumerate(state.get_objects(self._block_type)):
            c = cs[i % len(cs)]
            if state.get(block, "grasp") != -1:
                lcolor = "red"
                pose = state.get(self._robot, "pose_y_norm") - state.get(
                    block, "grasp")
                suffix = " (grasped)"
            else:
                lcolor = "gray"
                pose = state.get(block, "pose_y_norm")
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
            c = cs[i % len(cs)]
            lcolor = "gray"
            rect = plt.Rectangle(
                (state.get(targ, "pose_y_norm") - state.get(targ, "width") / 2.,
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
        if len(list(state)) < 8:  # disable legend if there are many objects
            plt.legend()
        if caption is not None:
            plt.suptitle(caption, wrap=True)
        plt.tight_layout()
        return fig

    def _get_hand_regions(self, state: State) -> List[Tuple[float, float]]:
        hand_regions = []
        for block in state.get_objects(self._block_type):
            hand_regions.append(
                (state.get(block, "pose_y_norm") - state.get(block, "width") / 2,
                 state.get(block, "pose_y_norm") + state.get(block, "width") / 2))
        for targ in state.get_objects(self._target_type):
            hand_regions.append(
                (state.get(targ, "pose_y_norm") - state.get(targ, "width") / 10,
                 state.get(targ, "pose_y_norm") + state.get(targ, "width") / 10))
        return hand_regions

    def _create_blocks_and_targets(self, is_train: Optional[bool] = True
                                   ) -> Tuple[List[Object], List[Object]]:
        blocks = []
        targets = []
        num_blocks = CFG.cover_num_blocks_train if is_train else\
                     CFG.cover_num_blocks_test
        num_targets = CFG.cover_num_blocks_train if is_train else\
                      CFG.cover_num_blocks_test

        for i in range(num_blocks):
            blocks.append(Object(f"block{i}", self._block_type))
        for i in range(num_targets):
            targets.append(Object(f"target{i}", self._target_type))
        return blocks, targets

    def _get_tasks(self, num: int,
                   rng: np.random.Generator,
                   is_train: Optional[bool] = True) -> List[EnvironmentTask]:
        tasks = []
        # Create blocks and targets.
        blocks, targets = self._create_blocks_and_targets(is_train)
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
            init = self._create_initial_state(blocks, targets, rng, is_train)
            assert init.get_objects(self._block_type) == blocks
            assert init.get_objects(self._target_type) == targets
            tasks.append(EnvironmentTask(init, goals[i % len(goals)]))
        return tasks

    def _create_initial_state(self, blocks: List[Object],
                              targets: List[Object],
                              rng: np.random.Generator,
                              is_train: Optional[bool] = True
                              ) -> State:
        data: Dict[Object, Array] = {}
        assert len(CFG.cover_block_widths) >= len(blocks)
        for block, width in zip(blocks, CFG.cover_block_widths):
            while True:
                pose = rng.uniform(width / 2, 1.0 - width / 2)
                if not self._any_intersection(pose, width, data):
                    break
            # [is_block, is_target, width, pose, grasp]
            data[block] = np.array([1.0, 0.0, width, pose, -1.0])
        assert len(CFG.cover_target_widths) >= len(targets)
        for target, width in zip(targets, CFG.cover_target_widths):
            while True:
                pose = rng.uniform(width / 2, 1.0 - width / 2)
                if not self._any_intersection(
                        pose, width, data, larger_gap=True):
                    break
            # [is_block, is_target, width, pose]
            data[target] = np.array([0.0, 1.0, width, pose])
        # For the non-PyBullet environments, pose_x and pose_z are constant.
        if "hand_empty" in self._robot_type.feature_names or\
           "fingers" in self._robot_type.feature_names:
            # [hand, pose_x, pose_z, hand_empty]
            data[self._robot] = np.array(
                [0.5, self.workspace_x, self.workspace_z, 1])
        else:
            # [hand, pose_x, pose_z]
            data[self._robot] = np.array(
                [0.5, self.workspace_x, self.workspace_z])
        data[self._table] = np.array([], dtype=np.float32)
        state = State(data)
        # Allow some chance of holding a block in the initial state.
        if rng.uniform() < CFG.cover_initial_holding_prob:
            block = blocks[rng.choice(len(blocks))]
            block_pose = state.get(block, "pose_y_norm")
            pick_pose = block_pose
            if self._initial_pick_offsets:
                offset = rng.choice(self._initial_pick_offsets)
                assert -1.0 < offset < 1.0, \
                    "initial pick offset should be between -1 and 1"
                pick_pose += state.get(block, "width") * offset / 2.
            state.set(self._robot, "pose_y_norm", pick_pose)
            if "hand_empty" in self._robot_type.feature_names:
                # See CoverEnvHandEmpty
                state.set(self._robot, "hand_empty", 0)
            if "fingers" in self._robot_type.feature_names:
                state.set(self._robot, "fingers", 0)
            state.set(block, "grasp", pick_pose - block_pose)
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
        block_pose = state.get(block, "pose_y_norm")
        block_width = state.get(block, "width")
        target_pose = state.get(target, "pose_y_norm")
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


class CoverEnvHandEmpty(CoverEnv):
    """Toy cover domain where the robot has a feature indicating whether its
    hand is empty or not.

    This allows us to learn all the predicates (Cover, Holding,
    HandEmpty) with the assumption that the predicates are a function of
    only their argument's states.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Add attribute.
        self._robot_type = Type("robot",
                                ["pose_y_norm", "pose_x", "pose_z", "hand_empty"])
        # Override HandEmpty predicate.
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        # Create new robot because of new robot type
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "cover_handempty"

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "hand_empty") == 1


class CoverEnvTypedOptions(CoverEnv):
    """Toy cover domain with options that have object arguments.

    This means we need two options (one for block, one for target).
    """

    @classmethod
    def get_name(cls) -> str:
        return "cover_typed_options"


class CoverEnvHierarchicalTypes(CoverEnv):
    """Toy cover domain with hierarchical types, just for testing."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Change blocks to be of a derived type
        self._parent_block_type = self._block_type
        self._block_type = Type(
            "block_derived",
            ["is_block", "is_target", "width", "pose_y_norm", "grasp"],
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

    This environment also has two different oracle NSRTs for placing,
    one for placing a target and one for placing on the table.

    This environment also has a Clear predicate, to prevent placing on
    already covered targets.

    Finally, to allow placing on the table, we need to change the
    allowed hand regions. We implement it so that there is a relatively
    small hand region centered at each target, but then everywhere else
    is allowed.
    """
    _allow_free_space_placing: ClassVar[bool] = True
    _initial_pick_offsets: ClassVar[List[float]] = [-0.95, 0.0, 0.95]

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

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
        for targ in sorted(targets, key=lambda t: state.get(t, "pose_y_norm")):
            w = state.get(targ, "width")
            targ_left = state.get(targ, "pose_y_norm") - w / 2
            targ_right = state.get(targ, "pose_y_norm") + w / 2
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

    The action space is (dx, dy, dgrip). The last dimension controls the
    gripper "magnet" or "vacuum". The state space is updated to track x,
    y, grip.

    The robot can move anywhere as long as it, and the block it may be
    holding, does not collide with another block. Picking up a block is
    allowed when the robot gripper is empty, when the robot is in the
    allowable hand region, and when the robot is sufficiently close to
    the block in the y-direction. Placing is allowed anywhere.
    Collisions are handled in simulate().
    """
    grasp_thresh: ClassVar[float] = 0.0
    initial_block_y: ClassVar[float] = 0.1
    block_height: ClassVar[float] = 0.1
    target_height: ClassVar[float] = 0.1  # Only for rendering purposes.
    initial_robot_y: ClassVar[float] = 0.4
    grip_lb: ClassVar[float] = -1.0
    grip_ub: ClassVar[float] = 1.0
    snap_tol: ClassVar[float] = 1e-2

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        # Need to now include y and gripper info in state.
        # Removing "pose_y_norm" because that's ambiguous.
        # Also adding height to blocks.
        # The y position corresponds to the top of the block.
        # The x position corresponds to the center of the block.
        self._block_type = Type(
            "block",
            ["is_block", "is_target", "width", "x", "grasp", "y", "height"])
        # Targets don't need y because they're constant.
        self._target_type = Type("target",
                                 ["is_block", "is_target", "width", "x"])
        # Also removing "pose_y_norm" because that's ambiguous.
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

    @classmethod
    def get_name(cls) -> str:
        return "cover_multistep_options"

    @property
    def action_space(self) -> Box:
        # This is the main difference with respect to the parent
        # env. The action space now is (dx, dy, dgrip). The
        # last dimension controls the gripper "magnet" or "vacuum".
        # Note that the bounds are relatively low, which necessitates
        # multi-step options. The action limits are for dx, dy only;
        # dgrip is constrained separately based on the grip limits.
        lb, ub = CFG.cover_multistep_action_limits
        return Box(np.array([lb, lb, self.grip_lb], dtype=np.float32),
                   np.array([ub, ub, self.grip_ub], dtype=np.float32))

    def simulate(self, state: State, action: Action) -> State:
        # Since the action space is lower level, we need to write
        # a lower level simulate function.
        assert self.action_space.contains(action.arr)

        dx, dy, dgrip = action.arr
        next_state = state.copy()
        x = state.get(self._robot, "x")
        y = state.get(self._robot, "y")
        grip = state.get(self._robot, "grip")
        blocks = state.get_objects(self._block_type)
        # Detect if a block is held, and if so, record that block's features.
        held_block = None
        for block in blocks:
            if state.get(block, "grasp") != -1:
                assert held_block is None
                held_block = block
                hx, hy = state.get(held_block, "x"), \
                           state.get(held_block, "y")
                hw, hh = state.get(held_block, "width"), \
                         state.get(held_block, "height")
                # Note: the block (x, y) is the middle-top of the block. The
                # Rectangle expects the lower left corner as (x, y).
                held_rect = utils.Rectangle(x=(hx - hw / 2),
                                            y=(hy - hh),
                                            width=hw,
                                            height=hh,
                                            theta=0)
                next_held_rect = utils.Rectangle(x=(held_rect.x + dx),
                                                 y=(held_rect.y + dy),
                                                 width=held_rect.width,
                                                 height=held_rect.height,
                                                 theta=held_rect.theta)
                # Compute line segments corresponding to the movement of each
                # of the held object vertices.
                held_move_segs = [
                    utils.LineSegment(x1, y1, x2, y2) for (x1, y1), (
                        x2,
                        y2) in zip(held_rect.vertices, next_held_rect.vertices)
                ]

        # Prevent the robot from going below the top of the blocks.
        y_min_robot = self.block_height
        if y + dy < y_min_robot:
            dy = y_min_robot - y
        # Prevent the robot from going above the initial robot position.
        if y + dy > self.initial_robot_y:
            dy = self.initial_robot_y - y
        # If the robot is holding a block that is close to the floor, and if
        # the robot is moving down to place, snap the robot so that the block
        # is exactly on the floor.
        at_place_height = False
        if held_block is not None and dy < 0:
            held_block_bottom = hy - hh
            # Always snap if below the floor.
            y_floor = 0
            if held_block_bottom + dy < y_floor or \
                abs(held_block_bottom + dy - y_floor) < self.snap_tol:
                dy = y_floor - held_block_bottom
                at_place_height = True
        # If the robot is not holding anything and is moving down, and if
        # the robot is close enough to the top of the block, snap it so that
        # the robot is exactly on top of the block.
        block_to_be_picked = None
        if held_block is None and dy < 0:
            for block in blocks:
                bx_lb = state.get(block, "x") - state.get(block, "width") / 2
                bx_ub = state.get(block, "x") + state.get(block, "width") / 2
                if bx_lb <= x <= bx_ub:
                    above_block_top = state.get(block, "y")
                    if abs(y + dy - above_block_top) < self.snap_tol:
                        block_to_be_picked = block
                        dy = above_block_top - y
                        break
        # Ensure that blocks do not collide with other blocks.
        if held_block is not None:
            for block in blocks:
                if block == held_block:
                    continue
                bx, by = state.get(block, "x"), state.get(block, "y")
                bw, bh = state.get(block, "width"), state.get(block, "height")
                rect = utils.Rectangle(x=(bx - bw / 2),
                                       y=(by - bh),
                                       width=bw,
                                       height=bh,
                                       theta=0)
                # Check the line segments corresponding to the movement of each
                # of the held object vertices.
                if any(seg.intersects(rect) for seg in held_move_segs):
                    return state.copy()
                # Check for overlap between the held object and this block.
                if rect.intersects(next_held_rect):
                    return state.copy()

        # Update the robot state.
        x += dx
        y += dy
        # Set desired grip directly and clip it.
        grip = np.clip(dgrip, self.grip_lb, self.grip_ub)
        next_state.set(self._robot, "x", x)
        next_state.set(self._robot, "y", y)
        next_state.set(self._robot, "grip", grip)
        if held_block is not None:
            hx = hx + dx
            hy = hy + dy
            next_state.set(held_block, "x", hx)
            next_state.set(held_block, "y", hy)

        # If we're not holding anything and we're close enough to a block, grasp
        # it if the gripper is on and we are in the allowed grasping region.
        # Note: unlike parent env, we also need to check the grip.
        if block_to_be_picked is not None and grip > self.grasp_thresh and \
            any(hand_lb <= x <= hand_rb
                for hand_lb, hand_rb in self._get_hand_regions_block(state)):
            by = state.get(block_to_be_picked, "y")
            assert abs(y - by) < 1e-7  # due to snapping
            next_state.set(block_to_be_picked, "grasp", 1)
            next_state.set(self._robot, "holding", 1)

        # If we are holding something and we're not above a block, place it if
        # the gripper is off and we are low enough. Placing anywhere is allowed
        # but if we are over a target, we must be in its hand region.
        # Note: unlike parent env, we also need to check the grip.
        if held_block is not None and block_to_be_picked is None and \
            grip < self.grasp_thresh and at_place_height:
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

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
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
                    label="pose_y_norm")
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
            c = cs[i % len(cs)]
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
            c = cs[i % len(cs)]
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
        return fig

    def _create_initial_state(self, blocks: List[Object],
                              targets: List[Object],
                              rng: np.random.Generator,
                              is_train: Optional[bool] = True
                              ) -> State:
        """Creates initial state by (1) placing targets and blocks in random
        locations such that each target has enough space on either side to
        ensure no covering placement will cause a collision (note that this is
        not necessary to make the task solvable; but we can do this and instead
        sufficiently tune the difficulty through hand region specification),
        and (2) choosing hand region intervals on the targets and blocks such
        that the problem is solvable."""
        if is_train:
            assert len(blocks) == CFG.cover_num_blocks_train
            assert len(targets) == CFG.cover_num_targets_train
        else:
            assert len(blocks) == CFG.cover_num_blocks_test
            assert len(targets) == CFG.cover_num_targets_test

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

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        block, robot = objects
        return state.get(block, "grasp") != -1 and \
            state.get(robot, "holding") != -1

    @staticmethod
    def _Covers_holds(state: State, objects: Sequence[Object]) -> bool:
        # Overriding because of the change from "pose_y_norm" to "x" and because
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


class CoverEnvPlaceHard(CoverEnv):
    """A cover environment where the only thing that's hard is placing.
    Specifically, there is only one block and one target, and the default grasp
    sampler always picks up the block directly in the middle. The robot is
    allowed to place anywhere, and the default sampler tries placing in a
    region that's 2x bigger than the target, often missing the target. The only
    thing that needs to be learned is how to place to correctly cover the
    target.

    This environment is specifically useful for testing various aspects
    of different sampler learning approaches.
    """
    _allow_free_space_placing: ClassVar[bool] = True

    # Repeating Types and Predicates for LLM input.
    # Types
    _block_type = Type("block",
                       ["is_block", "is_target", "width", "pose_y_norm", "grasp"])
    _target_type = Type("target", ["is_block", "is_target", "width", "pose_y_norm"])
    _robot_type = Type("robot", ["pose_y_norm", "pose_x", "pose_z"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

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

    @classmethod
    def get_name(cls) -> str:
        return "cover_place_hard"

    def _get_hand_regions(self, state: State) -> List[Tuple[float, float]]:
        # Allow placing anywhere!
        return [(0.0, 1.0)]


class BumpyCoverEnv(CoverEnvRegrasp):
    """A variation on the cover regrasp environment where some blocks are
    'bumpy', as indicated by a new feature of blocks.

    If they are bumpy, then the allowed hand region for picking the
    blocks is complicated, making it hard to sample a good grasp. Non-
    bumpy blocks have a simple allowed hand region. This environment is
    intended for use with reset-free sampler learning, so it's important
    that the blocks can be picked up off the targets after they have
    already been placed, which is why this environment inherits from
    CoverEnvRegrasp.
    """
    _allow_free_space_placing: ClassVar[bool] = False
    _bumps_regional: ClassVar[bool] = False

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Need to include "bumpy" feature to distinguish bumpy from smooth
        # blocks. Otherwise the sampler couldn't possibly know which is which.
        self._block_type = Type(
            "block",
            ["is_block", "is_target", "width", "pose_y_norm", "grasp", "bumpy"])

        # Need to override predicate creation  for blocks because the types are
        # now different (in terms of equality).
        self._IsBlock = Predicate("IsBlock", [self._block_type],
                                  self._IsBlock_holds)
        self._Covers = Predicate("Covers",
                                 [self._block_type, self._target_type],
                                 self._Covers_holds)
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)

    @classmethod
    def get_name(cls) -> str:
        return "bumpy_cover"

    def _create_initial_state(self, blocks: List[Object],
                              targets: List[Object],
                              rng: np.random.Generator) -> State:
        data: Dict[Object, Array] = {}
        assert len(CFG.cover_target_widths) == len(targets)
        for target, width in zip(targets, CFG.cover_target_widths):
            target_ub = 1.0
            # If there is a special bumpy region, keep targets away from it
            # to make things simpler.
            if self._bumps_regional:
                target_ub = CFG.bumpy_cover_bumpy_region_start - width / 2
            while True:
                pose = rng.uniform(width / 2, target_ub - width / 2)
                if not self._any_intersection(
                        pose, width, data, larger_gap=True):
                    break
            # [is_block, is_target, width, pose]
            data[target] = np.array([0.0, 1.0, width, pose])
        assert len(CFG.cover_block_widths) == len(blocks)
        want_block_in_bumpy = rng.uniform() < CFG.bumpy_cover_init_bumpy_prob
        for i, (block, width) in enumerate(zip(blocks,
                                               CFG.cover_block_widths)):
            if i == 0:
                bumpy = 1.0
            else:
                bumpy = 0.0
            while True:
                if self._bumps_regional and want_block_in_bumpy and bumpy:
                    lb = max(width / 2, CFG.bumpy_cover_bumpy_region_start)
                    ub = 1.0 - width / 2
                    want_block_in_bumpy = False
                else:
                    lb = width / 2
                    ub = CFG.bumpy_cover_bumpy_region_start - width / 2
                pose = rng.uniform(lb, ub)
                if not self._any_intersection(pose, width, data):
                    break
            # [is_block, is_target, width, pose, grasp, bumpy]
            data[block] = np.array([1.0, 0.0, width, pose, -1.0, bumpy])
        # [hand, pose_x, pose_z]
        data[self._robot] = np.array([0.5, self.workspace_x, self.workspace_z])
        state = State(data)
        return state

    def _get_hand_regions(self, state: State) -> List[Tuple[float, float]]:
        hand_regions = []
        for block in state.get_objects(self._block_type):
            pose = state.get(block, "pose_y_norm")
            bumpy = abs(state.get(block, "bumpy") - 1.0) < 1e-3
            in_bumpy_region = not self._bumps_regional or \
                pose > CFG.bumpy_cover_bumpy_region_start
            if bumpy and in_bumpy_region:
                # Evenly spaced intervals.
                start = pose - state.get(block, "width") / 2
                end = pose + state.get(block, "width") / 2
                skip = (1 + CFG.bumpy_cover_spaces_per_bump)
                num_points = skip * CFG.bumpy_cover_num_bumps
                points = np.linspace(start, end, num=num_points)
                for left, right in zip(points[::skip], points[1::skip]):
                    hand_regions.append((left, right))
            else:
                hand_regions.append((pose - state.get(block, "width") / 2,
                                     pose + state.get(block, "width") / 2))
        for targ in state.get_objects(self._target_type):
            center = state.get(targ, "pose_y_norm")
            if CFG.bumpy_cover_right_targets:
                center += 3 * state.get(targ, "width") / 4
            left = center - state.get(targ, "width") / 2
            right = center + state.get(targ, "width") / 2
            hand_regions.append((left, right))
        return hand_regions


class RegionalBumpyCoverEnv(BumpyCoverEnv):
    """Variation of bumpy cover where bumpy appear only in a region.

    Unlike the parent class, blocks can be placed anywhere once held.
    The focus is completely on picking bumpy objects.
    """

    _allow_free_space_placing: ClassVar[bool] = True
    _bumps_regional: ClassVar[bool] = True

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        assert not CFG.bumpy_cover_right_targets, \
            ("Right targets are meaningless in the regional variation because "
            "the agent can place anywhere. The only hard part is picking.")

        self._InBumpyRegion = Predicate("InBumpyRegion", [self._block_type],
                                        self._InBumpyRegion_holds)

        self._InSmoothRegion = Predicate("InSmoothRegion", [self._block_type],
                                         self._InSmoothRegion_holds)

    @classmethod
    def get_name(cls) -> str:
        return "regional_bumpy_cover"

    @property
    def predicates(self) -> Set[Predicate]:
        return super().predicates | {self._InBumpyRegion, self._InSmoothRegion}

    def _get_hand_regions(self, state: State) -> List[Tuple[float, float]]:
        # If a block is held, allow placing anywhere in this environment.
        if not self._HandEmpty_holds(state, []):
            return [(0.0, 1.0)]
        # Otherwise, choose hand regions carefully.
        return super()._get_hand_regions(state)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig = super().render_state_plt(state, task, action, caption)
        x = CFG.bumpy_cover_bumpy_region_start
        plt.plot([x, x], [-100, 100], color="gray", label="bump region lb")
        if len(list(state)) < 8:  # disable legend if there are many objects
            plt.legend()
        return fig

    def _InBumpyRegion_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, objects):
            return False
        block, = objects
        return state.get(block, "pose_y_norm") > CFG.bumpy_cover_bumpy_region_start

    def _InSmoothRegion_holds(self, state: State,
                              objects: Sequence[Object]) -> bool:
        if self._Holding_holds(state, objects):
            return False
        return not self._InBumpyRegion_holds(state, objects)
