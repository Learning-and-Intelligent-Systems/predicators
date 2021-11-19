"""Boring room vs. playroom domain.
"""

from typing import List, Set, Sequence, Dict, Tuple, Optional
import numpy as np
from gym.spaces import Box
from matplotlib import pyplot as plt
from matplotlib import patches
from predicators.src.envs import BlocksEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


class PlayroomEnv(BlocksEnv):
    """Boring room vs. playroom domain.
    """
    # Parameters that aren't important enough to need to clog up settings.py
    x_lb = 0.0
    y_lb = 0.0
    x_ub = 140.0
    y_ub = 30.0
    table_tol = 1.0
    table_x_lb = 10.0
    table_y_lb = 10.0
    table_x_ub = 20.0
    table_y_ub = 20.0
    door_r = 5.0  # half of width
    door_button_z = 1.0  # relative to ground
    door_tol = 1.0
    dial_on = 0.5
    dial_r = 3.0
    dial_tol = 1.0
    dial_button_tol = 0.4
    pick_tol = 0.4
    assert pick_tol < CFG.playroom_block_size

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._block_type = Type(
            "block", ["pose_x", "pose_y", "pose_z", "held", "clear"])
        self._robot_type = Type("robot", ["pose_x", "pose_y", "rotation",
                                          "fingers"])
        self._door_type = Type("door", ["pose_x", "pose_y", "open"])
        self._dial_type = Type("dial", ["pose_x", "pose_y", "level"])
        # Predicates
        # Still need to add new predicates
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
            # variables: [robot, object to pick]
            # params: [delta x, delta y, delta z]
            "Pick", types=[self._robot_type, self._block_type],
            params_space=Box(-1, 1, (3,)),
            _policy=self._Pick_policy,
            _initiable=self._Pick_initiable,
            _terminal=self._Pick_terminal)
        self._Stack = ParameterizedOption(
            # variables: [robot, object on which to stack currently-held-object]
            # params: [delta x, delta y, delta z]
            "Stack", types=[self._robot_type, self._block_type],
            params_space=Box(-1, 1, (3,)),
            _policy=self._Stack_policy,
            _initiable=self._Stack_initiable,
            _terminal=self._Stack_terminal)
        self._PutOnTable = ParameterizedOption(
            # variables: [robot]
            # params: [x, y] (normalized coordinates on the table surface)
            "PutOnTable", types=[self._robot_type],
            params_space=Box(0, 1, (2,)),
            _policy=self._PutOnTable_policy,
            _initiable=self._PutOnTable_initiable,
            _terminal=self._PutOnTable_terminal)
        # Still need to add new options
        # Objects
        self._robot = Object("robby", self._robot_type)
        self._door1 = Object("door1", self._door_type)
        self._door2 = Object("door2", self._door_type)
        self._door3 = Object("door3", self._door_type)
        self._door4 = Object("door4", self._door_type)
        self._door5 = Object("door5", self._door_type)
        self._door6 = Object("door6", self._door_type)
        self._doors = (self._door1, self._door2, self._door3,
                       self._door4, self._door5, self._door6)
        self._dial = Object("dial", self._dial_type)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        x, y, z, _, fingers = action.arr
        # Update robot position
        if not self._is_valid_loc(x, y):
            return state.copy()
        if self._robot_can_move(state, action):
            state = self._transition_move(state, action)

        x = state.get(self._robot, "pose_x")
        y = state.get(self._robot, "pose_y")
        # Interact with blocks
        if (self.table_x_lb < x < self.table_x_ub) \
            and (self.table_y_lb < y < self.table_y_ub) \
            and self._robot_is_facing_table(action):
            if fingers < 0.5:
                return self._transition_pick(state, action)
            if z < self.table_height + CFG.playroom_block_size:
                return self._transition_putontable(state, action)
            return self._transition_stack(state, action)
        # Interact with some door
        if any(self._NextToDoor_holds(state, (self._robot, door))
               for door in self._doors):
            door = self._get_door_next_to(state)
            door_x = state.get(door, "pose_x")
            door_y = state.get(door, "pose_y")
            if (door_x-self.door_tol < x < door_x+self.door_tol) \
                and (door_y-self.door_tol < y < door_y+self.door_tol) \
                and (self.door_button_z-self.door_tol < z
                        < self.door_button_z+self.door_tol) \
                and fingers >= self.open_fingers \
                and self._robot_is_facing_door(state, action, door):
                return self._transition_door(state, door)
        # Interact with dial
        dial_x = state.get(self._dial, "pose_x")
        dial_y = state.get(self._dial, "pose_y")
        if (dial_x-self.dial_button_tol < x < dial_x+self.dial_button_tol) \
            and (dial_y-self.dial_button_tol < y
                    < dial_y+self.dial_button_tol) \
            and fingers >= self.open_fingers \
            and self._robot_is_facing_dial(state, action):
            return self._transition_dial(state)

        return state.copy()

    def _transition_pick(self, state: State, action: Action) -> State:
        next_state = state.copy()
        # Can only pick if fingers are open
        if state.get(self._robot, "fingers") < self.open_fingers:
            return next_state
        x, y, z, _, fingers = action.arr
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
        assert block is not None
        x, y, z, _, fingers = action.arr
        # Check that table surface is clear at this pose
        poses = [[state.get(b, "pose_x"),
                  state.get(b, "pose_y"),
                  state.get(b, "pose_z")] for b in state
                 if b.is_instance(self._block_type)]
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
        assert block is not None
        x, y, z, _, fingers = action.arr
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
        next_state.set(block, "pose_z", cur_z+CFG.playroom_block_size)
        next_state.set(block, "held", 0.0)
        next_state.set(block, "clear", 1.0)
        next_state.set(other_block, "clear", 0.0)
        next_state.set(self._robot, "fingers", fingers)
        return next_state

    def _transition_move(self, state: State, action: Action) -> State:
        x, y, _, rotation, _ = action.arr
        next_state = state.copy()
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        next_state.set(self._robot, "rotation", rotation)
        return next_state

    def _transition_door(self, state: State, door: Object) -> State:
        # opens/closes a door that the robot is next to and facing
        assert door.type == self._door_type
        next_state = state.copy()
        if state.get(door, "open") < 0.5:
            next_state.set(door, "open", 1.0)
        else:
            next_state.set(door, "open", 0.0)
        return next_state

    def _transition_dial(self, state: State) -> State:
        next_state = state.copy()
        if state.get(self._dial, "level") < self.dial_on:
            next_state.set(self._dial, "level", 1.0)
        else:
            next_state.set(self._dial, "level", 0.0)
        return next_state

    def get_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                    possible_num_blocks=CFG.playroom_num_blocks_train,
                    rng=self._train_rng)

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                    possible_num_blocks=CFG.playroom_num_blocks_test,
                    rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        # To update later
        return {self._On, self._OnTable, self._GripperOpen, self._Holding,
                self._Clear}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # To update later
        return {self._On, self._OnTable}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._robot_type, self._door_type,
                self._dial_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        # To update later
        return {self._Pick, self._Stack, self._PutOnTable}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, rotation, fingers]
        # x, y, z location for the robot's disembodied hand
        # robot's heading is the angle (rotation * pi/2) in standard position
        lowers = np.array([self.x_lb, self.y_lb, 0.0, -2.0, 0.0],
                          dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 2.0, 1.0],
                          dtype=np.float32)
        return Box(lowers, uppers)

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> List[Image]:
        r = CFG.playroom_block_size * 0.5  # block radius

        fig, ax = plt.subplots(1, 1, figsize=(20, 8))
        ax.set_xlabel("x", fontsize=24)
        ax.set_ylabel("y", fontsize=24)
        ax.set_xlim((self.x_lb - 5, self.x_ub + 5))
        ax.set_ylim((self.y_lb - 5, self.y_ub + 5))

        # Draw rooms and hallway
        boring_room = patches.Rectangle(
                (self.x_lb, self.y_lb), 30, 30, zorder=0, linewidth=1,
                edgecolor='black', facecolor='white')
        ax.add_patch(boring_room)
        playroom = patches.Rectangle(
                (110, self.y_lb), 30, 30, zorder=0, linewidth=1,
                edgecolor='black', facecolor='white')
        ax.add_patch(playroom)
        hallway = patches.Rectangle(
                (30, 10), 80, 10, zorder=0, linewidth=1,
                edgecolor='black', facecolor='white')
        ax.add_patch(hallway)

        # Draw doors
        for door in self._doors:
            x = state.get(door, "pose_x")
            y = state.get(door, "pose_y")
            if state.get(door, "open") < 0.5:  # door closed
                door = patches.Rectangle(
                        (x-1.0, y-5.0), 1, 10, zorder=1, linewidth=1,
                        edgecolor='black', facecolor='brown')
                ax.add_patch(door)
            else:
                door = patches.Rectangle(
                        (x-1.0, y-5.0), 1, 1, zorder=1, linewidth=1,
                        edgecolor='black', facecolor='brown')
                ax.add_patch(door)

        # Draw dial
        dial_x = state.get(self._dial, "pose_x")
        dial_y = state.get(self._dial, "pose_y")
        dial_face = patches.Circle((dial_x, dial_y), radius=self.dial_r,
                                   edgecolor='black', facecolor='black')
        ax.add_patch(dial_face)
        level = state.get(self._dial, "level")
        dx = self.dial_r*np.sin(level*2*np.pi)
        dy = self.dial_r*np.cos(level*2*np.pi)
        dial_arrow = patches.Arrow(dial_x, dial_y, dx, dy, edgecolor='red',
                                   facecolor='red')
        ax.add_patch(dial_arrow)

        # Draw table and blocks
        table = patches.Rectangle(
                (10, 10), 10, 10, zorder=self.table_height,
                linewidth=1, edgecolor='black', facecolor='brown')
        ax.add_patch(table)
        colors = ["red", "blue", "green", "orange", "purple", "yellow",
                  "brown", "cyan"]
        blocks = [o for o in state if o.is_instance(self._block_type)]
        held = "None"
        for i, block in enumerate(sorted(blocks)):
            x = state.get(block, "pose_x")
            y = state.get(block, "pose_y")
            z = state.get(block, "pose_z")
            c = colors[i % len(colors)]  # block color
            if state.get(block, "held") > self.held_tol:
                assert held == "None"
                held = f"{block.name} ({c})"
            rect = patches.Rectangle(
                (x - r, y - r), 2*r, 2*r, zorder=self.table_height+z,
                linewidth=1, edgecolor='black', facecolor=c)
            ax.add_patch(rect)

        # Draw robot
        robot_x = state.get(self._robot, "pose_x")
        robot_y = state.get(self._robot, "pose_y")
        fingers = state.get(self._robot, "fingers")
        robby = patches.Circle((robot_x, robot_y), radius=1,
                               edgecolor='black', facecolor='yellow')
        ax.add_patch(robby)
        rotation = state.get(self._robot, "rotation")
        dx, dy = np.cos(rotation*np.pi/2), np.sin(rotation*np.pi/2)
        robot_arrow = patches.Arrow(robot_x, robot_y, dx, dy,
            edgecolor='black', facecolor='black', width=0.5)
        ax.add_patch(robot_arrow)

        plt.suptitle(f"Held: {held}, Fingers: {fingers}", fontsize=36)
        plt.tight_layout()
        img = utils.fig2data(fig)

        return [img]

    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[Task]:
        # to change later: right now tasks only vary by block placement
        tasks = []
        for _ in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks)
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            atoms = utils.abstract(init_state, self.predicates)
            while True:  # repeat until goal is not satisfied
                # To change later
                goal = self._sample_goal_from_piles(num_blocks, piles, rng)
                if not goal.issubset(atoms):
                    break
            tasks.append(Task(init_state, goal))
        return tasks

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
            z = self.table_height + CFG.playroom_block_size * (0.5 + pile_j)
            max_j = max(j for i, j in block_to_pile_idx.values() if i == pile_i)
            # [pose_x, pose_y, pose_z, held, clear]
            data[block] = np.array([x, y, z, 0.0, int(pile_j == max_j)*1.0])
        # [pose_x, pose_y, rotation, fingers], fingers start off open
        data[self._robot] = np.array([5.0, 5.0, 0.0, 1.0])
        # [pose_x, pose_y, open], all doors start off closed
        data[self._door1] = np.array([30.0, 15.0, 0.0])
        data[self._door2] = np.array([50.0, 15.0, 0.0])
        data[self._door3] = np.array([60.0, 15.0, 0.0])
        data[self._door4] = np.array([80.0, 15.0, 0.0])
        data[self._door5] = np.array([100.0, 15.0, 0.0])
        data[self._door6] = np.array([110.0, 15.0, 0.0])
        # [pose_x, pose_y, level], light starts off off
        data[self._dial] = np.array([125.0, 15.0, 0.0])
        return State(data)

    def _sample_initial_pile_xy(self, rng: np.random.Generator,
                                existing_xys: Set[Tuple[float, float]]
                                ) -> Tuple[float, float]:
        while True:
            x = rng.uniform(self.table_x_lb, self.table_x_ub)
            y = rng.uniform(self.table_y_lb, self.table_y_ub)
            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    @staticmethod
    def _table_xy_is_clear(x: float, y: float,
                           existing_xys: Set[Tuple[float, float]]) -> bool:
        if all(abs(x-other_x) > 2*CFG.playroom_block_size
               for other_x, _ in existing_xys):
            return True
        if all(abs(y-other_y) > 2*CFG.playroom_block_size
               for _, other_y in existing_xys):
            return True
        return False

    @staticmethod
    def _is_valid_loc(x: float, y: float) -> bool:
        return (0 <= x <= 30 and 0 <= y <= 30) or \
               (30 <= x <= 110 and 10 <= y <= 20) or \
               (110 <= x <= 140 and 0 <= y <= 30)

    @staticmethod
    def _robot_is_facing_table(action: Action) -> bool:
        x, y, _, rotation, _ = action.arr
        cls = PlayroomEnv
        table_x = (cls.table_x_lb+cls.table_x_ub)/2
        table_y = (cls.table_y_lb+cls.table_y_ub)/2
        theta = np.arctan2(table_y-y, table_x-x)
        if np.pi*3/4 >= theta >= np.pi/4:  # N
            if 1.5 >= rotation >= 0.5:
                return True
        elif np.pi/4 >= theta >= -np.pi/4:  # E
            if 0.5 >= rotation >= -0.5:
                return True
        elif -np.pi/4 >= theta >= -np.pi*3/4:  # S
            if -0.5 >= rotation >= -1.5:
                return True
        else:  # W
            if rotation >= 1.5 or rotation <= -1.5:
                return True
        return False

    @staticmethod
    def _On_holds(state: State, objects: Sequence[Object]) -> bool:
        block1, block2 = objects
        cls = PlayroomEnv
        if state.get(block1, "held") >= cls.held_tol or \
           state.get(block2, "held") >= cls.held_tol:
            return False
        x1 = state.get(block1, "pose_x")
        y1 = state.get(block1, "pose_y")
        z1 = state.get(block1, "pose_z")
        x2 = state.get(block2, "pose_x")
        y2 = state.get(block2, "pose_y")
        z2 = state.get(block2, "pose_z")
        return np.allclose([x1, y1, z1], [x2, y2, z2+CFG.playroom_block_size],
                           atol=cls.pick_tol)

    @staticmethod
    def _OnTable_holds(state: State, objects: Sequence[Object]) -> bool:
        block, = objects
        z = state.get(block, "pose_z")
        cls = PlayroomEnv
        desired_z = cls.table_height + CFG.playroom_block_size * 0.5
        return (state.get(block, "held") < cls.held_tol) and \
            (desired_z-cls.pick_tol < z < desired_z+cls.pick_tol)

    # To pull in when I add all the predicates
    # @staticmethod
    # def _NextToTable_holds(state: State, objects: Sequence[Object]
    #                        ) -> bool:
    #     # for now this also includes all table coords
    #     robot, = objects
    #     x, y = state.get(robot, "pose_x"), state.get(robot, "pose_y")
    #     cls = PlayroomEnv
    #     return (cls.table_x_lb-cls.table_tol < x
    #             < cls.table_x_ub+cls.table_tol) and \
    #            (cls.table_y_lb-cls.table_tol < y
    #             < cls.table_y_ub+cls.table_tol)

    @staticmethod
    def _NextToDoor_holds(state: State, objects: Sequence[Object]
                          ) -> bool:
        robot, door = objects
        x, y = state.get(robot, "pose_x"), state.get(robot, "pose_y")
        door_x, door_y = state.get(door, "pose_x"), state.get(door, "pose_y")
        cls = PlayroomEnv
        return (door_x-cls.door_tol < x < door_x+cls.door_tol) \
                and (door_y-cls.door_r-cls.door_tol < y
                     < door_y+cls.door_r+cls.door_tol)

    # To pull in when I add all the predicates
    # @staticmethod
    # def _NextToDial_holds(state: State, objects: Sequence[Object]
    #                       ) -> bool:
    #     robot, dial = objects
    #     x, y = state.get(robot, "pose_x"), state.get(robot, "pose_y")
    #     dial_x, dial_y = state.get(dial, "pose_x"), state.get(dial, "pose_y")
    #     cls = PlayroomEnv
    #     return (dial_x-cls.dial_r-cls.dial_tol < x
    #             < dial_x+cls.dial_r+cls.dial_tol) and \
    #            (dial_y-cls.dial_r-cls.dial_tol < y
    #             < dial_y+cls.dial_r+cls.dial_tol)

    def _Pick_policy(self, state: State, objects: Sequence[Object],
                     params: Array) -> Action:
        robot, block = objects
        block_pose = np.array([state.get(block, "pose_x"),
                               state.get(block, "pose_y"),
                               state.get(block, "pose_z")])
        rotation = state.get(robot, "rotation")
        arr = np.r_[block_pose+params, rotation, 0.0].astype(np.float32)
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    def _Stack_policy(self, state: State, objects: Sequence[Object],
                      params: Array) -> Action:
        robot, block = objects
        block_pose = np.array([state.get(block, "pose_x"),
                               state.get(block, "pose_y"),
                               state.get(block, "pose_z")])
        rotation = state.get(robot, "rotation")
        arr = np.r_[block_pose+params, rotation, 1.0].astype(np.float32)
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    def _PutOnTable_policy(self, state: State, objects: Sequence[Object],
                           params: Array) -> Action:
        robot, = objects
        # Un-normalize parameters to actual table coordinates
        x_norm, y_norm = params
        x = self.table_x_lb + (self.table_x_ub - self.table_x_lb) * x_norm
        y = self.table_y_lb + (self.table_y_ub - self.table_y_lb) * y_norm
        z = self.table_height + 0.5*CFG.playroom_block_size
        rotation = state.get(robot, "rotation")
        arr = np.array([x, y, z, rotation, 1.0], dtype=np.float32)
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    def _robot_is_facing_dial(self, state: State, action: Action) -> bool:
        x, y, _, rotation, _ = action.arr
        dial_x = state.get(self._dial, "pose_x")
        dial_y = state.get(self._dial, "pose_y")
        theta = np.arctan2(dial_y-y, dial_x-x)
        if np.pi*3/4 >= theta > np.pi/4:  # N
            if 1.5 >= rotation > 0.5:
                return True
        elif np.pi/4 >= theta > -np.pi/4:  # E
            if 0.5 >= rotation > -0.5:
                return True
        elif -np.pi/4 >= theta > -np.pi*3/4:  # S
            if -0.5 >= rotation > -1.5:
                return True
        else:  # W
            if rotation > 1.5 or rotation <= -1.5:
                return True
        return False

    def _get_door_next_to(self, state: State) -> Object:
        # cannot be next to multiple doors at once
        for door in self._doors:
            if self._NextToDoor_holds(state, (self._robot, door)):
                return door
        # usage should ensure this is never reached
        raise RuntimeError("Robot not next to any door")  # pragma: no cover

    def _robot_is_facing_door(self, state: State, action: Action, door: Object
                              ) -> bool:
        assert door.type == self._door_type
        door_x = state.get(door, "pose_x")
        x, _, _, rotation, _ = action.arr
        return (x < door_x and 0.5 >= rotation >= -0.5) \
            or (x >= door_x and (rotation >= 1.5 or rotation <= -1.5))

    def _robot_can_move(self, state: State, action: Action) -> bool:
        # Any doors along the path must be open
        prev_x = state.get(self._robot, "pose_x")
        x = action.arr[0]
        for door in self._doors:
            if x <= state.get(door, "pose_x") <= prev_x \
               or prev_x <= state.get(door, "pose_x") <= x:
                if state.get(door, "open") < 0.5:
                    return False
        return True
