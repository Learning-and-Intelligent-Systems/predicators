"""Boring room vs playroom domain and variants."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators import utils
from predicators.envs.blocks import BlocksEnv, BlocksEnvClear
from predicators.settings import CFG
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type


class PlayroomSimpleEnv(BlocksEnv):
    """Simple version of the boring room vs playroom domain."""
    # Parameters that aren't important enough to need to clog up settings.py
    x_lb: ClassVar[float] = 0.0
    y_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 140.0
    y_ub: ClassVar[float] = 30.0
    open_fingers: ClassVar[float] = 0.8
    table_tol: ClassVar[float] = 1.0
    table_x_lb: ClassVar[float] = 10.0
    table_y_lb: ClassVar[float] = 10.0
    table_x_ub: ClassVar[float] = 20.0
    table_y_ub: ClassVar[float] = 20.0
    dial_on_thresh: ClassVar[float] = 0.5
    dial_r: ClassVar[float] = 3.0
    dial_button_z: ClassVar[float] = 1.0
    dial_tol: ClassVar[float] = 0.5
    dial_button_tol: ClassVar[float] = 0.4
    pick_tol: ClassVar[float] = 0.4
    on_tol: ClassVar[float] = pick_tol
    pick_z: ClassVar[float] = 1.5

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._block_type = Type(
            "block", ["pose_x", "pose_y", "pose_z", "held", "clear"])
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "rotation", "fingers"])
        self._dial_type = Type("dial", ["pose_x", "pose_y", "level"])
        # Predicates
        self._On = Predicate("On", [self._block_type, self._block_type],
                             self._On_holds)
        self._OnTable = Predicate("OnTable", [self._block_type],
                                  self._OnTable_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding", [self._block_type],
                                  self._Holding_holds)
        self._Clear = Predicate("Clear", [self._block_type], self._Clear_holds)
        self._NextToTable = Predicate("NextToTable", [self._robot_type],
                                      self._NextToTable_holds)
        self._NextToDial = Predicate("NextToDial",
                                     [self._robot_type, self._dial_type],
                                     self._NextToDial_holds)
        self._LightOn = Predicate("LightOn", [self._dial_type],
                                  self._LightOn_holds)
        self._LightOff = Predicate("LightOff", [self._dial_type],
                                   self._LightOff_holds)
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._dial = Object("dial", self._dial_type)
        # Hyperparameters from CFG.
        self._num_blocks_train = CFG.playroom_num_blocks_train
        self._num_blocks_test = CFG.playroom_num_blocks_test
        # Override block size from parent class.
        self._block_size = 0.5

    @classmethod
    def get_name(cls) -> str:
        return "playroom_simple"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        x, y, z, _, fingers = action.arr
        was_next_to_table = self._NextToTable_holds(state, (self._robot, ))
        was_next_to_dial = self._NextToDial_holds(state,
                                                  (self._robot, self._dial))
        if not self._is_valid_loc(x, y) or not self._robot_can_move(
                state, action):
            return state.copy()
        # Update robot position
        state = self._transition_move(state, action)
        x = state.get(self._robot, "pose_x")
        y = state.get(self._robot, "pose_y")
        # Interact with blocks if robot was already next to table
        if was_next_to_table \
            and (self.table_x_lb < x < self.table_x_ub) \
            and (self.table_y_lb < y < self.table_y_ub):
            if fingers < 0.5:
                return self._transition_pick(state, x, y, z)
            if z < self.table_height + self._block_size:
                return self._transition_putontable(state, x, y, z)
            return self._transition_stack(state, x, y, z)
        # Interact with dial if robot was already next to dial
        dial_x = state.get(self._dial, "pose_x")
        dial_y = state.get(self._dial, "pose_y")
        if was_next_to_dial \
            and (dial_x-self.dial_button_tol < x
                    < dial_x+self.dial_button_tol) \
            and (dial_y-self.dial_button_tol < y
                    < dial_y+self.dial_button_tol) \
            and (self.dial_button_z-self.dial_button_tol < z
                    < self.dial_button_z+self.dial_button_tol) \
            and fingers >= self.open_fingers:
            return self._transition_dial(state)

        return state.copy()

    def _transition_move(self, state: State, action: Action) -> State:
        x, y, _, rotation, _ = action.arr
        next_state = state.copy()
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        next_state.set(self._robot, "rotation", rotation)
        return next_state

    def _transition_dial(self, state: State) -> State:
        next_state = state.copy()
        if state.get(self._dial, "level") < self.dial_on_thresh:
            next_state.set(self._dial, "level", 1.0)
        else:
            next_state.set(self._dial, "level", 0.0)
        return next_state

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnTable, self._GripperOpen, self._Holding,
            self._Clear, self._NextToTable, self._NextToDial, self._LightOn,
            self._LightOff
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On, self._OnTable, self._LightOn, self._LightOff}

    @property
    def types(self) -> Set[Type]:
        return {self._block_type, self._robot_type, self._dial_type}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, rotation, fingers]
        # x, y, z location for the robot's disembodied hand
        # robot's heading is the angle (rotation * pi) in standard position
        lowers = np.array([self.x_lb, self.y_lb, 0.0, -1.0, 0.0],
                          dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 1.0, 1.0],
                          dtype=np.float32)
        return Box(lowers, uppers)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        r = self._block_size * 0.5  # block radius

        fig = plt.figure(figsize=(20, 16))
        ax = plt.subplot(211)
        ax.set_xlabel("x", fontsize=24)
        ax.set_ylabel("y", fontsize=24)
        ax.set_xlim((self.x_lb - 5, self.x_ub + 5))
        ax.set_ylim((self.y_lb - 5, self.y_ub + 5))

        # Draw rooms and hallway
        boring_room = patches.Rectangle((self.x_lb, self.y_lb),
                                        30,
                                        30,
                                        zorder=0,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor='white')
        ax.add_patch(boring_room)
        playroom = patches.Rectangle((110, self.y_lb),
                                     30,
                                     30,
                                     zorder=0,
                                     linewidth=1,
                                     edgecolor='black',
                                     facecolor='white')
        ax.add_patch(playroom)
        hallway = patches.Rectangle((30, 10),
                                    80,
                                    10,
                                    zorder=0,
                                    linewidth=1,
                                    edgecolor='black',
                                    facecolor='white')
        ax.add_patch(hallway)

        # Draw doors if playroom env
        if isinstance(self, PlayroomEnv):
            for door in self._doors:
                x = state.get(door, "pose_x")
                y = state.get(door, "pose_y")
                if state.get(door, "open") < self.door_open_thresh:
                    door = patches.Rectangle((x - 1.0, y - 5.0),
                                             1,
                                             10,
                                             zorder=1,
                                             linewidth=1,
                                             edgecolor='black',
                                             facecolor='brown')
                    ax.add_patch(door)
                else:
                    door = patches.Rectangle((x - 1.0, y - 5.0),
                                             1,
                                             1,
                                             zorder=1,
                                             linewidth=1,
                                             edgecolor='black',
                                             facecolor='brown')
                    ax.add_patch(door)

        # Draw dial
        dial_x = state.get(self._dial, "pose_x")
        dial_y = state.get(self._dial, "pose_y")
        dial_face = patches.Circle((dial_x, dial_y),
                                   radius=self.dial_r,
                                   edgecolor='black',
                                   facecolor='black')
        ax.add_patch(dial_face)
        level = state.get(self._dial, "level")
        dx = self.dial_r * np.sin(level * 2 * np.pi)
        dy = self.dial_r * np.cos(level * 2 * np.pi)
        dial_arrow = patches.Arrow(dial_x,
                                   dial_y,
                                   dx,
                                   dy,
                                   edgecolor='red',
                                   facecolor='red')
        ax.add_patch(dial_arrow)

        # Draw table and blocks
        table = patches.Rectangle((10, 10),
                                  10,
                                  10,
                                  zorder=self.table_height,
                                  linewidth=1,
                                  edgecolor='black',
                                  facecolor='brown')
        ax.add_patch(table)
        colors = [
            "red", "blue", "green", "orange", "purple", "yellow", "brown",
            "cyan"
        ]
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
            rect = patches.Rectangle((x - r, y - r),
                                     2 * r,
                                     2 * r,
                                     zorder=self.table_height + z,
                                     linewidth=1,
                                     edgecolor='black',
                                     facecolor=c)
            ax.add_patch(rect)

        # Draw robot
        robot_x = state.get(self._robot, "pose_x")
        robot_y = state.get(self._robot, "pose_y")
        fingers = state.get(self._robot, "fingers")
        robby = patches.Circle((robot_x, robot_y),
                               radius=1,
                               edgecolor='black',
                               facecolor='yellow')
        ax.add_patch(robby)
        rotation = state.get(self._robot, "rotation")
        dx, dy = np.cos(rotation * np.pi), np.sin(rotation * np.pi)
        robot_arrow = patches.Arrow(robot_x,
                                    robot_y,
                                    dx,
                                    dy,
                                    edgecolor='black',
                                    facecolor='black',
                                    width=0.5)
        ax.add_patch(robot_arrow)

        # Concatenate with table view of blocks
        xz_ax, yz_ax = plt.subplot(223), plt.subplot(224)
        xz_ax.set_xlabel("x", fontsize=24)
        xz_ax.set_ylabel("z", fontsize=24)
        xz_ax.set_xlim((self.table_x_lb - 2 * r, self.table_x_ub + 2 * r))
        xz_ax.set_ylim((self.table_height, r * 16 + 0.1))
        yz_ax.set_xlabel("y", fontsize=24)
        yz_ax.set_ylabel("z", fontsize=24)
        yz_ax.set_xlim((self.table_y_lb - 2 * r, self.table_y_ub + 2 * r))
        yz_ax.set_ylim((self.table_height, r * 16 + 0.1))

        colors = [
            "red", "blue", "green", "orange", "purple", "yellow", "brown",
            "cyan"
        ]
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

            # xz axis
            xz_rect = patches.Rectangle((x - r, z - r),
                                        2 * r,
                                        2 * r,
                                        zorder=-y,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor=c)
            xz_ax.add_patch(xz_rect)

            # yz axis
            yz_rect = patches.Rectangle((y - r, z - r),
                                        2 * r,
                                        2 * r,
                                        zorder=-x,
                                        linewidth=1,
                                        edgecolor='black',
                                        facecolor=c)
            yz_ax.add_patch(yz_rect)

        title = f"Held: {held}, Fingers: {fingers}"
        if caption is not None:
            title += f"; {caption}"
        plt.suptitle(title, fontsize=24, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num_tasks: int, possible_num_blocks: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        # Initial states vary by block placement, and light is randomly on/off.
        # Goals involve goal piles and light different from the initial state.
        tasks = []
        for _ in range(num_tasks):
            num_blocks = rng.choice(possible_num_blocks)
            piles = self._sample_initial_piles(num_blocks, rng)
            init_state = self._sample_state_from_piles(piles, rng)
            light_is_on = init_state.get(self._dial, "level") > 0.5
            while True:  # repeat until goal is not satisfied
                goal = self._sample_goal(num_blocks, piles, light_is_on, rng)
                if not all(goal_atom.holds(init_state) for goal_atom in goal):
                    break
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    def _sample_state_from_piles(self, piles: List[List[Object]],
                                 rng: np.random.Generator) -> State:
        """No doors or regions."""
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
            z = self.table_height + self._block_size * (0.5 + pile_j)
            max_j = max(j for i, j in block_to_pile_idx.values()
                        if i == pile_i)
            # [pose_x, pose_y, pose_z, held, clear]
            data[block] = np.array([x, y, z, 0.0, int(pile_j == max_j) * 1.0])
        # [pose_x, pose_y, rotation, fingers], fingers start off open
        data[self._robot] = np.array([10.0, 15.0, 0.0, 1.0])
        # [pose_x, pose_y, level], light starts on/off randomly
        data[self._dial] = np.array([125.0, 15.0, rng.uniform(0.0, 1.0)])
        return State(data)

    def _sample_goal(self, num_blocks: int, piles: List[List[Object]],
                     light_is_on: bool,
                     rng: np.random.Generator) -> Set[GroundAtom]:
        # Samples goal pile and light on/off that is different from initial
        while True:
            goal_piles = self._sample_initial_piles(num_blocks, rng)
            if goal_piles != piles:
                break
        goal_atoms = set()
        for pile in goal_piles:
            goal_atoms.add(GroundAtom(self._OnTable, [pile[0]]))
            if len(pile) == 1:
                continue
            for block1, block2 in zip(pile[1:], pile[:-1]):
                goal_atoms.add(GroundAtom(self._On, [block1, block2]))
        if light_is_on:
            goal_atoms.add(GroundAtom(self._LightOff, [self._dial]))
        else:
            goal_atoms.add(GroundAtom(self._LightOn, [self._dial]))
        return goal_atoms

    def _sample_initial_pile_xy(
            self, rng: np.random.Generator,
            existing_xys: Set[Tuple[float, float]]) -> Tuple[float, float]:
        # Differs from blocks because lower and upper bounds are set by table
        while True:
            x = rng.uniform(self.table_x_lb, self.table_x_ub)
            y = rng.uniform(self.table_y_lb, self.table_y_ub)
            if self._table_xy_is_clear(x, y, existing_xys):
                return (x, y)

    @staticmethod
    def _is_valid_loc(x: float, y: float) -> bool:
        return (0 <= x <= 30 and 0 <= y <= 30) or \
               (30 <= x <= 110 and 10 <= y <= 20) or \
               (110 <= x <= 140 and 0 <= y <= 30)

    @staticmethod
    def _NextToTable_holds(state: State, objects: Sequence[Object]) -> bool:
        # Being "in" the table also counts as next to table
        robot, = objects
        x, y = state.get(robot, "pose_x"), state.get(robot, "pose_y")
        cls = PlayroomSimpleEnv
        return (cls.table_x_lb-cls.table_tol < x
                < cls.table_x_ub+cls.table_tol) and \
               (cls.table_y_lb-cls.table_tol < y
                < cls.table_y_ub+cls.table_tol)

    @staticmethod
    def _NextToDial_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, dial = objects
        x, y = state.get(robot, "pose_x"), state.get(robot, "pose_y")
        dial_x, dial_y = state.get(dial, "pose_x"), state.get(dial, "pose_y")
        cls = PlayroomSimpleEnv
        return (dial_x-cls.dial_r-cls.dial_tol < x
                < dial_x+cls.dial_r+cls.dial_tol) and \
               (dial_y-cls.dial_r-cls.dial_tol < y
                < dial_y+cls.dial_r+cls.dial_tol)

    @staticmethod
    def _LightOn_holds(state: State, objects: Sequence[Object]) -> bool:
        dial, = objects
        return state.get(dial, "level") >= PlayroomSimpleEnv.dial_on_thresh

    def _LightOff_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return not self._LightOn_holds(state, objects)

    def _robot_can_move(self, state: State, action: Action) -> bool:
        """No region or door stuff."""
        x, y, _, _, _ = action.arr
        next_state = state.copy()
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        # Robot must end up next to something
        if not (self._NextToTable_holds(next_state, (self._robot, ))
                or self._NextToDial_holds(next_state,
                                          (self._robot, self._dial))):
            return False
        return True


class PlayroomSimpleEnvClear(PlayroomSimpleEnv, BlocksEnvClear):
    """Simple version of the boring room vs playroom domain where each block
    has a feature indicating whether it is clear or not.

    This allows us to learn all the predicates with the assumption that
    the predicates are a function of only their argument's states.
    """

    @classmethod
    def get_name(cls) -> str:
        return "playroom_simple_clear"


class PlayroomEnv(PlayroomSimpleEnv):
    """Boring room vs playroom domain with hallway and doors."""
    # Additional parameters
    door_open_thresh: ClassVar[float] = 0.5
    door_r: ClassVar[float] = 5.0  # half of width
    door_button_z: ClassVar[float] = 3.0
    door_tol: ClassVar[float] = 0.5

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Additional types
        self._door_type = Type("door", ["id", "pose_x", "pose_y", "open"])
        self._region_type = Type("region",
                                 ["id", "x_lb", "y_lb", "x_ub", "y_ub"])
        # Additional predicates
        self._NextToDoor = Predicate("NextToDoor",
                                     [self._robot_type, self._door_type],
                                     self._NextToDoor_holds)
        self._InRegion = Predicate("InRegion",
                                   [self._robot_type, self._region_type],
                                   self._InRegion_holds)
        self._Borders = Predicate(
            "Borders", [self._door_type, self._region_type, self._door_type],
            self._Borders_holds)
        self._Connects = Predicate(
            "Connects",
            [self._door_type, self._region_type, self._region_type],
            self._Connects_holds)
        self._IsBoringRoom = Predicate("IsBoringRoom", [self._region_type],
                                       self._IsBoringRoom_holds)
        self._IsPlayroom = Predicate("IsPlayroom", [self._region_type],
                                     self._IsPlayroom_holds)
        self._IsBoringRoomDoor = Predicate("IsBoringRoomDoor",
                                           [self._door_type],
                                           self._IsBoringRoomDoor_holds)
        self._IsPlayroomDoor = Predicate("IsPlayroomDoor", [self._door_type],
                                         self._IsPlayroomDoor_holds)
        self._DoorOpen = Predicate("DoorOpen", [self._door_type],
                                   self._DoorOpen_holds)
        self._DoorClosed = Predicate("DoorClosed", [self._door_type],
                                     self._DoorClosed_holds)
        # Additional static objects (always exist no matter the settings).
        self._door1 = Object("door1", self._door_type)
        self._door2 = Object("door2", self._door_type)
        self._door3 = Object("door3", self._door_type)
        self._door4 = Object("door4", self._door_type)
        self._door5 = Object("door5", self._door_type)
        self._door6 = Object("door6", self._door_type)
        self._doors = (self._door1, self._door2, self._door3, self._door4,
                       self._door5, self._door6)
        self._region1 = Object("region1", self._region_type)
        self._region2 = Object("region2", self._region_type)
        self._region3 = Object("region3", self._region_type)
        self._region4 = Object("region4", self._region_type)
        self._region5 = Object("region5", self._region_type)
        self._region6 = Object("region6", self._region_type)
        self._region7 = Object("region7", self._region_type)

    @classmethod
    def get_name(cls) -> str:
        return "playroom"

    def simulate(self, state: State, action: Action) -> State:
        """Allows for interaction with doors."""
        assert self.action_space.contains(action.arr)
        x, y, z, _, fingers = action.arr
        was_next_to_table = self._NextToTable_holds(state, (self._robot, ))
        was_next_to_door = {
            door: PlayroomEnv._NextToDoor_holds(state, (self._robot, door))
            for door in self._doors
        }
        prev_region = self._get_region_in(state,
                                          state.get(self._robot, "pose_x"))
        was_next_to_dial = self._NextToDial_holds(state,
                                                  (self._robot, self._dial))
        if not self._is_valid_loc(x, y) or not self._robot_can_move(
                state, action):
            return state.copy()
        # Update robot position
        state = self._transition_move(state, action)
        x = state.get(self._robot, "pose_x")
        y = state.get(self._robot, "pose_y")
        # Interact with blocks if robot was already next to table
        if was_next_to_table \
            and (self.table_x_lb < x < self.table_x_ub) \
            and (self.table_y_lb < y < self.table_y_ub):
            if fingers < 0.5:
                return self._transition_pick(state, x, y, z)
            if z < self.table_height + self._block_size:
                return self._transition_putontable(state, x, y, z)
            return self._transition_stack(state, x, y, z)
        # Interact with some door
        if any(
                PlayroomEnv._NextToDoor_holds(state, (self._robot, door))
                for door in self._doors):
            door = self._get_door_next_to(state)
            current_region = self._get_region_in(state, x)
            # Robot was already next to this door and did not go through it
            if was_next_to_door[door] and prev_region == current_region:
                door_x = state.get(door, "pose_x")
                door_y = state.get(door, "pose_y")
                if (door_x-self.door_tol < x < door_x+self.door_tol) \
                    and (door_y-self.door_tol < y < door_y+self.door_tol) \
                    and (self.door_button_z-self.door_tol < z
                            < self.door_button_z+self.door_tol) \
                    and fingers >= self.open_fingers:
                    return self._transition_door(state, door)
        # Interact with dial if robot was already next to dial
        dial_x = state.get(self._dial, "pose_x")
        dial_y = state.get(self._dial, "pose_y")
        if was_next_to_dial \
            and (dial_x-self.dial_button_tol < x
                    < dial_x+self.dial_button_tol) \
            and (dial_y-self.dial_button_tol < y
                    < dial_y+self.dial_button_tol) \
            and (self.dial_button_z-self.dial_button_tol < z
                    < self.dial_button_z+self.dial_button_tol) \
            and fingers >= self.open_fingers:
            return self._transition_dial(state)

        return state.copy()

    def _transition_door(self, state: State, door: Object) -> State:
        # opens/closes a door that the robot is next to
        assert door.type == self._door_type
        next_state = state.copy()
        if state.get(door, "open") < self.door_open_thresh:
            next_state.set(door, "open", 1.0)
        else:
            next_state.set(door, "open", 0.0)
        return next_state

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnTable, self._GripperOpen, self._Holding,
            self._Clear, self._NextToTable, self._NextToDoor, self._NextToDial,
            self._InRegion, self._Borders, self._Connects, self._IsBoringRoom,
            self._IsPlayroom, self._IsBoringRoomDoor, self._IsPlayroomDoor,
            self._DoorOpen, self._DoorClosed, self._LightOn, self._LightOff
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._block_type, self._robot_type, self._door_type,
            self._dial_type, self._region_type
        }

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
            z = self.table_height + self._block_size * (0.5 + pile_j)
            max_j = max(j for i, j in block_to_pile_idx.values()
                        if i == pile_i)
            # [pose_x, pose_y, pose_z, held, clear]
            data[block] = np.array([x, y, z, 0.0, int(pile_j == max_j) * 1.0])
        # [pose_x, pose_y, rotation, fingers], fingers start off open
        data[self._robot] = np.array([10.0, 15.0, 0.0, 1.0])
        # [pose_x, pose_y, open], all doors start off open except door1
        data[self._door1] = np.array([1, 30.0, 15.0, 0.0])
        data[self._door2] = np.array([2, 50.0, 15.0, 1.0])
        data[self._door3] = np.array([3, 60.0, 15.0, 1.0])
        data[self._door4] = np.array([4, 80.0, 15.0, 1.0])
        data[self._door5] = np.array([5, 100.0, 15.0, 1.0])
        data[self._door6] = np.array([6, 110.0, 15.0, 1.0])
        # [pose_x, pose_y, level], light starts on/off randomly
        data[self._dial] = np.array([125.0, 15.0, rng.uniform(0.0, 1.0)])
        # [id, x_lb, y_lb, x_ub, y_ub], regions left to right
        data[self._region1] = np.array([1, 0.0, 0.0, 30.0, 30.0])
        data[self._region2] = np.array([2, 30.0, 10.0, 50.0, 20.0])
        data[self._region3] = np.array([3, 50.0, 10.0, 60.0, 20.0])
        data[self._region4] = np.array([4, 60.0, 10.0, 80.0, 20.0])
        data[self._region5] = np.array([5, 80.0, 10.0, 100.0, 20.0])
        data[self._region6] = np.array([6, 100.0, 10.0, 110.0, 20.0])
        data[self._region7] = np.array([7, 110.0, 0.0, 140.0, 30.0])
        return State(data)

    @staticmethod
    def _NextToDoor_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, door = objects
        x, y = state.get(robot, "pose_x"), state.get(robot, "pose_y")
        door_x, door_y = state.get(door, "pose_x"), state.get(door, "pose_y")
        cls = PlayroomEnv
        return (door_x-cls.door_tol < x < door_x+cls.door_tol) \
                and (door_y-cls.door_r-cls.door_tol < y
                     < door_y+cls.door_r+cls.door_tol)

    @staticmethod
    def _InRegion_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, region = objects
        x, y = state.get(robot, "pose_x"), state.get(robot, "pose_y")
        x_lb, y_lb = state.get(region, "x_lb"), state.get(region, "y_lb")
        x_ub, y_ub = state.get(region, "x_ub"), state.get(region, "y_ub")
        return x_lb <= x <= x_ub and y_lb <= y <= y_ub

    @staticmethod
    def _IsBoringRoom_holds(state: State, objects: Sequence[Object]) -> bool:
        region, = objects
        return state.get(region, "id") == 1

    @staticmethod
    def _IsPlayroom_holds(state: State, objects: Sequence[Object]) -> bool:
        region, = objects
        return state.get(region, "id") == 7

    @staticmethod
    def _Borders_holds(state: State, objects: Sequence[Object]) -> bool:
        door1, region, door2 = objects
        return (state.get(door1, "pose_x") == state.get(region, "x_lb") and \
               state.get(door2, "pose_x") == state.get(region, "x_ub")) or \
               (state.get(door2, "pose_x") == state.get(region, "x_lb") and \
               state.get(door1, "pose_x") == state.get(region, "x_ub"))

    @staticmethod
    def _Connects_holds(state: State, objects: Sequence[Object]) -> bool:
        door, from_region, to_region = objects
        door_x = state.get(door, "pose_x")
        return (door_x == state.get(from_region, "x_ub") and \
               door_x == state.get(to_region, "x_lb")) or \
               (door_x == state.get(to_region, "x_ub") and \
               door_x == state.get(from_region, "x_lb"))

    @staticmethod
    def _IsBoringRoomDoor_holds(state: State,
                                objects: Sequence[Object]) -> bool:
        door, = objects
        return state.get(door, "pose_x") == 30.0

    @staticmethod
    def _IsPlayroomDoor_holds(state: State, objects: Sequence[Object]) -> bool:
        door, = objects
        return state.get(door, "pose_x") == 110.0

    @staticmethod
    def _DoorOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        door, = objects
        return state.get(door, "open") >= PlayroomEnv.door_open_thresh

    @staticmethod
    def _DoorClosed_holds(state: State, objects: Sequence[Object]) -> bool:
        return not PlayroomEnv._DoorOpen_holds(state, objects)

    def _get_door_next_to(self, state: State) -> Object:
        # cannot be next to multiple doors at once
        for door in self._doors:
            if self._NextToDoor_holds(state, (self._robot, door)):
                return door
        raise RuntimeError("Robot not next to any door")

    def _get_region_in(self, state: State, x: float) -> int:
        # return the id of the region that x-coordinate `x` is located in
        for obj in state:
            if obj.type == self._region_type and \
                    state.get(obj, "x_lb") <= x <= state.get(obj, "x_ub"):
                return state.get(obj, "id")
        raise RuntimeError(f"x-coord {x} not part of any region")

    def _robot_can_move(self, state: State, action: Action) -> bool:
        """Includes extra checks for allowable movement between regions and
        through doors."""
        prev_x = state.get(self._robot, "pose_x")
        x, y, _, _, _ = action.arr
        prev_region = self._get_region_in(state, prev_x)
        region = self._get_region_in(state, x)
        next_state = state.copy()
        next_state.set(self._robot, "pose_x", x)
        next_state.set(self._robot, "pose_y", y)
        # Robot must end up next to something
        if not (self._NextToTable_holds(next_state, (self._robot, ))
                or self._NextToDial_holds(next_state,
                                          (self._robot, self._dial))
                or any(
                    self._NextToDoor_holds(next_state, (self._robot, door))
                    for door in self._doors)):
            return False
        if region == prev_region:  # Robot can stay in same region
            return True
        if abs(region - prev_region) > 1:  # Robot may not "skip over" regions
            return False
        # The only remaining possibility is that the robot moves to an
        # adjacent region, which can only happen if it was next to a door.
        if not any(
                self._NextToDoor_holds(state, (self._robot, door))
                for door in self._doors):
            return False
        # Any doors along the path must be open
        for door in self._doors:
            door_x = state.get(door, "pose_x")
            if x <= door_x <= prev_x or prev_x <= door_x <= x:
                if state.get(door, "open") < self.door_open_thresh:
                    raise utils.EnvironmentFailure(
                        "collision", {"offending_objects": {door}})
        door = self._get_door_next_to(state)
        # After the robot moves through the door, it must still be next to
        # that same door.
        if any(
                self._NextToDoor_holds(next_state, (self._robot, door))
                for door in self._doors):
            return door == self._get_door_next_to(next_state)
        return False


class PlayroomHardEnv(PlayroomEnv):
    """Boring room vs playroom domain where the light on/off function is
    arbitrary and hard."""
    dial_on_intervals: ClassVar[List[Tuple[float, float]]] = [(0.0, 0.05),
                                                              (0.3, 0.65),
                                                              (0.9, 1.0)]

    @classmethod
    def get_name(cls) -> str:
        return "playroom_hard"

    def _transition_dial(self, state: State) -> State:
        next_state = state.copy()
        level = state.get(self._dial, "level")
        light_is_on = False
        for (low, high) in self.dial_on_intervals:
            if low <= level <= high:
                light_is_on = True
        if not light_is_on:
            next_state.set(self._dial, "level", 0.5)
        else:
            next_state.set(self._dial, "level", 0.1)
        return next_state

    @staticmethod
    def _LightOn_holds(state: State, objects: Sequence[Object]) -> bool:
        dial, = objects
        level = state.get(dial, "level")
        for (low, high) in PlayroomHardEnv.dial_on_intervals:
            if low <= level <= high:
                return True
        return False
