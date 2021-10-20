"""Blocks domain. This environment IS downward refinable and DOESN'T
require any backtracking. Its simplicity makes it a good testbed for
predicate invention.
"""

from typing import List, Set, Sequence, Dict, Tuple, Optional
import numpy as np
import ipdb
from gym.spaces import Box
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


class BlocksEnv(BaseEnv):
    """Blocks domain.
    """
    # Parameters that aren't important enough to need to clog up settings.py
    table_height = 0.2
    x_lb = 1.3
    x_ub = 1.4
    y_lb = 0.15
    y_ub = 20.85
    held_tol = 0.5
    clear_tol = 0.5
    open_fingers = 0.8
    pick_tol = 0.08
    assert pick_tol < CFG.blocks_block_size
    lift_amt = 1.0

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._block_type = Type(
            "block", ["pose_x", "pose_y", "pose_z", "held", "clear"])
        self._robot_type = Type("robot", ["fingers"])
        # Predicates
        self._On = Predicate(
            "On", [self._block_type, self._block_type], self._On_holds)
        self._OnTable = Predicate(
            "OnTable", [self._block_type], self._OnTable_holds)
        self._GripperOpen = Predicate(
            "GripperOpen", [self._robot_type], self._GripperOpen_holds)
        self._Holding = Predicate(
            "Holding", [self._block_type], self._Holding_holds)
        self._Clear = Predicate(
            "Clear", [self._block_type], self._Clear_holds)
        # Options
        self._Pick = ParameterizedOption(
            # variables: [object to pick]
            # params: [delta x, delta y, delta z]
            "Pick", types=[self._block_type], params_space=Box(-1, 1, (3,)),
            _policy=self._Pick_policy,
            _initiable=self._Pick_initiable,
            _terminal=self._Pick_terminal)
        self._Stack = ParameterizedOption(
            # variables: [object to stack currently held object onto]
            # params: [delta x, delta y, delta z]
            "Stack", types=[self._block_type],
            params_space=Box(-1, 1, (3,)),
            _policy=self._Stack_policy,
            _initiable=self._Stack_initiable,
            _terminal=self._Stack_terminal)
        self._PutOnTable = ParameterizedOption(
            # params: [x, y] (normalized coordinates on the table surface)
            "PutOnTable", types=[],
            params_space=Box(0, 1, (2,)),
            _policy=self._PutOnTable_policy,
            _initiable=self._PutOnTable_initiable,
            _terminal=self._PutOnTable_terminal)
        # Objects
        self._robot = Object("robby", self._robot_type)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        _, _, z, fingers = action.arr
        # Infer which transition function to follow
        if fingers < 0.5:
            fn_name = "pick"
            transition_fn = self._transition_pick
        elif z < self.table_height + CFG.blocks_block_size:
            fn_name = "putontable"
            transition_fn = self._transition_putontable
        else:
            fn_name = "stack"
            transition_fn = self._transition_stack
        next_state = transition_fn(state, action)
        return next_state

    def _transition_pick(self, state: State, action: Action) -> State:
        next_state = state.copy()
        # Can only pick if fingers are open
        if state.get(self._robot, "fingers") < self.open_fingers:
            return next_state
        x, y, z, fingers = action.arr
        block = self._get_block_at_xyz(state, x, y, z)
        if block is None:  # no block at this pose
            return next_state
        # Can only pick if object is clear
        if state.get(block, "clear") < self.clear_tol:
            return next_state
        # Execute pick
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z+self.lift_amt)
        next_state.set(block, "held", 1.0)
        next_state.set(block, "clear", 0.0)
        next_state.set(self._robot, "fingers", fingers)
        # Update clear bit of block below, if there is one
        cur_x = state.get(block, "pose_x")
        cur_y = state.get(block, "pose_y")
        cur_z = state.get(block, "pose_z")
        poss_below_block = self._get_highest_block_below(
            state, cur_x, cur_y, cur_z)
        assert poss_below_block != block
        if poss_below_block is not None:
            next_state.set(poss_below_block, "clear", 1.0)
        return next_state

    def _transition_putontable(self, state: State, action: Action) -> State:
        next_state = state.copy()
        # Can only putontable if fingers are closed
        if state.get(self._robot, "fingers") >= self.open_fingers:
            return next_state
        block = self._get_held_block(state)
        if block is None:  # no currently held block
            return next_state
        x, y, z, fingers = action.arr
        # Check that table surface is clear at this pose
        poses = [[state.get(b, "pose_x"),
                  state.get(b, "pose_y"),
                  state.get(b, "pose_z")] for b in state
                 if b.type == self._block_type]
        existing_xys = {(float(p[0]), float(p[1])) for p in poses}
        if not self._table_xy_is_clear(x, y, existing_xys):
            return next_state
        # Execute putontable
        next_state.set(block, "pose_x", x)
        next_state.set(block, "pose_y", y)
        next_state.set(block, "pose_z", z)
        next_state.set(block, "held", 0.0)
        next_state.set(block, "clear", 1.0)
        next_state.set(self._robot, "fingers", fingers)
        return next_state

    def _transition_stack(self, state: State, action: Action) -> State:
        next_state = state.copy()
        # Can only stack if fingers are closed
        if state.get(self._robot, "fingers") >= self.open_fingers:
            return next_state
        # Check that both blocks exist
        block = self._get_held_block(state)
        if block is None:  # no currently held block
            return next_state
        x, y, z, fingers = action.arr
        other_block = self._get_highest_block_below(state, x, y, z)
        if other_block is None:  # no block to stack onto
            return next_state
        # Can't stack onto yourself!
        if block == other_block:
            return next_state
        # Need block we're stacking onto to be clear
        if state.get(other_block, "clear") < self.clear_tol:
            return next_state
        # Execute stack by snapping into place
        cur_x = state.get(other_block, "pose_x")
        cur_y = state.get(other_block, "pose_y")
        cur_z = state.get(other_block, "pose_z")
        next_state.set(block, "pose_x", cur_x)
        next_state.set(block, "pose_y", cur_y)
        next_state.set(block, "pose_z", cur_z+CFG.blocks_block_size)
        next_state.set(block, "held", 0.0)
        next_state.set(block, "clear", 1.0)
        next_state.set(other_block, "clear", 0.0)
        next_state.set(self._robot, "fingers", fingers)
        return next_state

    def get_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               possible_num_blocks=CFG.blocks_num_blocks_train,
                               rng=self._train_rng)

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               possible_num_blocks=CFG.blocks_num_blocks_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._On, self._OnTable, self._GripperOpen, self._Holding,
                self._Clear}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._robot_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Pick, self._Stack, self._PutOnTable}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers]
        return Box(-100.0, 100.0, (4,))

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> Image:
        raise NotImplementedError

    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for _ in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks)
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            atoms = utils.abstract(init_state, self.predicates)
            while True:  # repeat until goal is not satisfied
                goal = self._sample_goal_from_piles(num_blocks, piles, rng)
                if not goal.issubset(atoms):
                    break
            tasks.append(Task(init_state, goal))
        return tasks

    def _sample_initial_piles(self, num_blocks: int, rng: np.random.Generator
                              ) -> List[List[Object]]:
        piles: List[List[Object]] = []
        for block_num in range(num_blocks):
            block = Object(f"block{block_num}", self._block_type)
            # If coin flip, start new pile
            if block_num == 0 or rng.uniform() < 0.2:
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
            z = self.table_height + CFG.blocks_block_size * (0.5 + pile_j)
            max_j = max(j for i, j in block_to_pile_idx.values() if i == pile_i)
            # [pose_x, pose_y, pose_z, held, clear]
            data[block] = np.array([x, y, z, 0.0, int(pile_j == max_j)*1.0])
        # [fingers]
        data[self._robot] = np.array([1.0])  # fingers start off open
        return State(data)

    def _sample_goal_from_piles(self, num_blocks: int,
                                piles: List[List[Object]],
                                rng: np.random.Generator) -> Set[GroundAtom]:
        # Sample goal pile that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_blocks, rng)
            if goal_piles != piles:
                break
        # Create goal from piles
        goal_atoms = set()
        for pile in goal_piles:
            goal_atoms.add(GroundAtom(self._OnTable, [pile[0]]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_atoms.add(GroundAtom(self._On, [block1, block2]))
        return goal_atoms

    def _sample_initial_pile_xy(self, rng: np.random.Generator,
                                existing_xys: Set[Tuple[float, float]]
                                ) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.x_lb, self.x_ub)
            y = rng.uniform(self.y_lb, self.y_ub)
            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    @staticmethod
    def _table_xy_is_clear(x: float, y: float,
                           existing_xys: Set[Tuple[float, float]]) -> bool:
        if all(abs(x-other_x) > 2*CFG.blocks_block_size
               for other_x, _ in existing_xys):
            return True
        if all(abs(y-other_y) > 2*CFG.blocks_block_size
               for _, other_y in existing_xys):
            return True
        return False

    @staticmethod
    def _On_holds(state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        cls = BlocksEnv
        if state.get(block1, "held") >= cls.held_tol or \
           state.get(block2, "held") >= cls.held_tol:
            return False
        x1 = state.get(block1, "pose_x")
        y1 = state.get(block1, "pose_y")
        z1 = state.get(block1, "pose_z")
        x2 = state.get(block2, "pose_x")
        y2 = state.get(block2, "pose_y")
        z2 = state.get(block2, "pose_z")
        return np.allclose([x1, y1, z1], [x2, y2, z2+CFG.blocks_block_size],
                           atol=cls.pick_tol)

    @staticmethod
    def _OnTable_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        z = state.get(block, "pose_z")
        cls = BlocksEnv
        desired_z = cls.table_height + CFG.blocks_block_size * 0.5
        return (state.get(block, "held") < cls.held_tol) and \
            (desired_z-cls.pick_tol < z < desired_z+cls.pick_tol)

    @staticmethod
    def _GripperOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") >= BlocksEnv.open_fingers

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return self._get_held_block(state) == block

    @staticmethod
    def _Clear_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        return state.get(block, "clear") >= BlocksEnv.clear_tol

    @staticmethod
    def _Pick_policy(state: State, objects: Sequence[Object],
                     params: Array) -> Action:
        block, = objects
        block_pose = np.array([state.get(block, "pose_x"),
                               state.get(block, "pose_y"),
                               state.get(block, "pose_z")])
        return Action(np.r_[block_pose+params, 0.0].astype(np.float32))

    @staticmethod
    def _Pick_initiable(state: State, objects: Sequence[Object],
                        params: Array) -> bool:
        del state, objects, params  # unused
        return True  # can be run from anywhere

    @staticmethod
    def _Pick_terminal(state: State, objects: Sequence[Object],
                       params: Array) -> bool:
        del state, objects, params  # unused
        return True  # always 1 timestep

    @staticmethod
    def _Stack_policy(state: State, objects: Sequence[Object],
                      params: Array) -> Action:
        block, = objects
        block_pose = np.array([state.get(block, "pose_x"),
                               state.get(block, "pose_y"),
                               state.get(block, "pose_z")])
        return Action(np.r_[block_pose+params, 1.0].astype(np.float32))

    @staticmethod
    def _Stack_initiable(state: State, objects: Sequence[Object],
                         params: Array) -> bool:
        del state, objects, params  # unused
        return True  # can be run from anywhere

    @staticmethod
    def _Stack_terminal(state: State, objects: Sequence[Object],
                        params: Array) -> bool:
        del state, objects, params  # unused
        return True  # always 1 timestep

    def _PutOnTable_policy(self, state: State, objects: Sequence[Object],
                           params: Array) -> Action:
        del state, objects  # unused
        # Un-normalize parameters to actual table coordinates
        x_norm, y_norm = params
        x = self.x_lb + (self.x_ub - self.x_lb) * x_norm
        y = self.y_lb + (self.y_ub - self.y_lb) * y_norm
        z = self.table_height + 0.5*CFG.blocks_block_size
        return Action(np.array([x, y, z, 1.0], dtype=np.float32))

    @staticmethod
    def _PutOnTable_initiable(state: State, objects: Sequence[Object],
                              params: Array) -> bool:
        del state, objects, params  # unused
        return True  # can be run from anywhere

    @staticmethod
    def _PutOnTable_terminal(state: State, objects: Sequence[Object],
                             params: Array) -> bool:
        del state, objects, params  # unused
        return True  # always 1 timestep

    def _get_held_block(self, state: State) -> Optional[Object]:
        for block in state:
            if block.type != self._block_type:
                continue
            if state.get(block, "held") >= self.held_tol:
                return block
        return None

    def _get_block_at_xyz(self, state: State, x: float, y: float,
                          z: float) -> Optional[Object]:
        close_blocks = []
        for block in state:
            if block.type != self._block_type:
                continue
            block_pose = np.array([state.get(block, "pose_x"),
                                   state.get(block, "pose_y"),
                                   state.get(block, "pose_z")])
            if np.allclose([x, y, z], block_pose, atol=self.pick_tol):
                dist = np.linalg.norm(np.array([x, y, z])-  # type: ignore
                                      block_pose)
                close_blocks.append((block, dist))
        if not close_blocks:
            return None
        return min(close_blocks, key=lambda x: x[1])[0]  # min distance

    def _get_highest_block_below(self, state: State, x: float, y: float,
                                 z: float) -> Optional[Object]:
        blocks_here = []
        for block in state:
            if block.type != self._block_type:
                continue
            block_pose = np.array([state.get(block, "pose_x"),
                                   state.get(block, "pose_y")])
            block_z = state.get(block, "pose_z")
            if np.allclose([x, y], block_pose, atol=self.pick_tol) and \
               block_z < z:
                blocks_here.append((block, block_z))
        if not blocks_here:
            return None
        return max(blocks_here, key=lambda x: x[1])[0]  # highest z
