"""A simple gridworld environment inspired by https://github.com/portal-cornell/robotouille.

The environment also uses a lot of assets from robotouille.
"""

import logging
from typing import List, Optional, Sequence, Set, Callable, Tuple

# import pygame
import numpy as np
from gym.spaces import Box
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type, Observation

# - Start with only VLM predicates
# - Then do KNOWN predicates + VLM predicates
# - Then do predicates generated from the grammar + VLM predicates
# - Towards this, look into where the ground atom dataset is created, after that it's just the normal grammar search approach

# - Want to be able to run this as a VLM env, or a normal env
# - Want to give oracle's predicates access to full state, but not non-oracle's (maybe can use a different perceiver -- the perceiver we use with non-oracle actually throws out info that is there)
# - Need to keep track of certain state outside of the actual state (e.g. is_cooked for patty)
# - Make subclass of state that has the observation that produced it as a field
# - There should be a texture-based rendering (using pygame) and a non-texture-based rendering, using matplotlib?

# Operators
# - Pick(robot, item)
# - Place(robot, item)
# - Cook(robot, item, grill)
# - Cut(robot, item, cutting board)

class GridWorldEnv(BaseEnv):
    """TODO"""

    # TODO
    ASSETS_DIRECTORY = ""

    # Types
    _object_type = Type("object", [])
    _item_type = Type("item", [], _object_type)
    _station_type = Type("station", [], _object_type)

    _robot_type = Type("robot", ["row", "col", "fingers", "dir"])
    _cell_type = Type("cell", ["row", "col"])

    _patty_type = Type("patty", ["row", "col", "z"], _item_type)
    _tomato_type = Type("tomato", ["row", "col", "z"], _item_type)
    _cheese_type = Type("cheese", ["row", "col", "z"], _item_type)
    _bottom_bun_type = Type("bottom_bun", ["row", "col", "z"], _item_type)
    _top_bun_type = Type("top_bun", ["row", "col", "z"], _item_type)

    _grill_type = Type("grill", ["row", "col", "z"], _station_type)
    _cutting_board_type = Type("cutting_board", ["row", "col", "z"], _station_type)

    dir_to_int = {"up": 0, "left": 1, "down": 2, "right": 3}
    int_to_dir = {0: "up", 1: "left", 2: "down", 3: "right"}


    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        self.num_rows = CFG.gridworld_num_rows
        self.num_cols = CFG.gridworld_num_cols
        self.num_cells = self.num_rows * self.num_cols

        # Predicates
        # self._RobotInCell = Predicate("RobotInCell", [self._robot_type, self._cell_type], self._In_holds)
        self._Adjacent = Predicate("Adjacent", [self._robot_type, self._item_type], self.Adjacent_holds)
        self._Facing = Predicate("Facing", [self._robot_type, self._object_type], self.Facing_holds)
        self._IsCooked = Predicate("IsCooked", [self._patty_type], self._IsCooked_holds)
        self._IsSliced = Predicate("IsSliced", [self._tomato_type], self._IsSliced_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type], self._HandEmpty_holds)
        self._Holding = Predicate("Holding", [self._robot_type, self._item_type], self._Holding_holds)
        self._On = Predicate("On", [self._object_type, self._object_type], self._On_holds)
        self._GoalHack = Predicate("GoalHack", [self._bottom_bun_type, self._patty_type, self._cheese_type, self._tomato_type, self._top_bun_type], self._GoalHack_holds)

        # Static objects (exist no matter the settings)
        self._robot = Object("robby", self._robot_type)
        self._grill = Object("grill", self._grill_type)
        self._cutting_board = Object("cutting_board", self._cutting_board_type)
        self._cells = [
            Object(f"cell{i}", self._cell_type) for i in range(self.num_cells)
        ]

    @classmethod
    def get_name(cls) -> str:
        return "gridworld"

    @property
    def types(self) -> Set[Type]:
        return {
            self._object_type,
            self._item_type,
            self._station_type,
            self._robot_type,
            self._cell_type,
            self._patty_type,
            self._tomato_type,
            self._cheese_type,
            self._bottom_bun_type,
            self._top_bun_type,
            self._grill_type,
            self._cutting_board_type
        }

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []

        # Add robot, grill, and cutting board
        state_dict = {}
        hidden_state = {}
        state_dict[self._robot] = {"row": 2, "col": 2, "fingers": 0.0, "dir": 3}
        state_dict[self._grill] = {"row": 2, "col": 3, "z": 0}
        state_dict[self._cutting_board] = {"row": 1, "col": 3, "z": 0}

        # # Add cells
        # counter = 0
        # for i in range(self.num_rows):
        #     for j in range(self.num_cols):
        #         # get the next cell
        #         cell = self._cells[counter]
        #         state_dict[cell] = {
        #             "row": i,
        #             "col": j
        #         }
        #         counter += 1

        # Add patty
        patty = Object("patty", self._patty_type)
        state_dict[patty] = {"row": 0, "col": 0, "z": 0}
        hidden_state[patty] = {"is_cooked": 0.0, "is_held": 0.0}

        # Add tomato
        tomato = Object("tomato", self._tomato_type)
        state_dict[tomato] = {"row": 0, "col": 1, "z": 0}
        hidden_state[tomato] = {"is_sliced": 0.0, "is_held": 0.0}

        # Add cheese
        cheese = Object("cheese", self._cheese_type)
        state_dict[cheese] = {"row": 3, "col": 0, "z": 0}
        hidden_state[cheese] = {"is_held": 0.0}

        # Add top bun
        top_bun = Object("top_bun", self._top_bun_type)
        state_dict[top_bun] = {"row": 3, "col": 1, "z": 0}
        hidden_state[top_bun] = {"is_held": 0.0}

        # Add bottom bun
        bottom_bun = Object("bottom_bun", self._bottom_bun_type)
        state_dict[bottom_bun] = {"row": 0, "col": 2, "z": 0}
        hidden_state[bottom_bun] = {"is_held": 0.0}

        # # Get the top right cell and make the goal for the agent to go there.
        # top_right_cell = self._cells[-1]
        # goal = {GroundAtom(self._RobotInCell, [self._robot, top_right_cell])}
        goal = {
            # GroundAtom(
            #     self._GoalHack,
            #     [bottom_bun, patty, cheese, tomato, top_bun]
            # )

            # GroundAtom(self._On, [patty, bottom_bun]),
            # GroundAtom(self._On, [cheese, patty]),
            # GroundAtom(self._On, [tomato, cheese]),
            # GroundAtom(self._On, [top_bun, tomato]),
            # GroundAtom(self._IsCooked, [patty]),
            # GroundAtom(self._IsSliced, [tomato])

            GroundAtom(self._Holding, [self._robot, tomato])
        }

        for i in range(num):
            state = utils.create_state_from_dict(state_dict)
            state.simulator_state = hidden_state
            # Note: this takes in Observation, GoalDescription, whose types is Any
            tasks.append(EnvironmentTask(state, goal))
        return tasks

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    def _In_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, cell = objects
        robot_row, robot_col = state.get(robot, "row"), state.get(robot, "col")
        cell_row, cell_col = state.get(cell, "row"), state.get(cell, "col")
        return robot_row == cell_row and robot_col == cell_col

    @classmethod
    def Adjacent_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        robot, item = objects
        robot_col, robot_row = cls.get_position(robot, state)
        item_col, item_row = cls.get_position(item, state)
        return cls.is_adjacent(item_col, item_row, robot_col, robot_row)

    @classmethod
    def Facing_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        robot, obj = objects
        robot_col, robot_row = cls.get_position(robot, state)
        # robot_dir = self._hidden_state[self._robot]["dir"]
        try:
            robot_dir = state.get(robot, "dir")
        except:
            import pdb; pdb.set_trace()
        obj_col, obj_row = cls.get_position(obj, state)
        facing_left = robot_row == obj_row and robot_col - obj_col == 1 and robot_dir == 1
        facing_right = robot_row == obj_row and robot_col - obj_col == -1 and robot_dir == 3
        facing_down = robot_row - obj_row == 1 and robot_col == obj_col and robot_dir == 2
        facing_up = robot_row - obj_row == -1 and robot_col == obj_col and robot_dir == 0
        return facing_left or facing_right or facing_down or facing_up

    def _IsCooked_holds(self, state: State, objects: Sequence[Object]) -> bool:
        patty, = objects
        return state.simulator_state[patty]["is_cooked"] > 0.5

    def _IsSliced_holds(self, state: State, objects: Sequence[Object]) -> bool:
        tomato, = objects
        return state.simulator_state[tomato]["is_sliced"] > 0.5

    def _HandEmpty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") < 0.5

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, item = objects
        return not self._HandEmpty_holds(state, [robot]) and self._hidden_state[item]["is_held"] > 0.5

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        a, b = objects
        a_x, a_y = self.get_position(a, state)
        b_x, b_y = self.get_position(b, state)
        a_z = state.get(a, "z")
        b_z = state.get(b, "z")

        return a_x==b_x and a_y==b_y and a_z - 1 == b_z

    def _GoalHack_holds(self, state: State, objects: Sequence[Object]) -> bool:
        bottom, patty, cheese, tomato, top = objects
        atoms = [
            self._On_holds(state, [patty, bottom]),
            self._On_holds(state, [cheese, patty]),
            self._On_holds(state, [tomato, cheese]),
            self._On_holds(state, [top, tomato]),
            self._IsCooked_holds(state, [patty]),
            self._IsSliced_holds(state, [tomato])
        ]
        return all(atoms)

    @staticmethod
    def get_position(object: Object, state: State) -> Tuple[int, int]:
        col = state.get(object, "col")
        row = state.get(object, "row")
        return col, row

    @staticmethod
    def is_adjacent(col_1, row_1, col_2, row_2):
        adjacent_vertical = col_1 == col_2 and abs(row_1 - row_2) == 1
        adjacent_horizontal = row_1 == row_2 and abs(col_1 - col_2) == 1
        return adjacent_vertical or adjacent_horizontal

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Adjacent, self._Facing, self._IsCooked, self._IsSliced, self._HandEmpty, self._Holding,  self._On, self._GoalHack}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # return {self._IsCooked, self._IsSliced, self._On, self._GoalHack}
        return {self._Holding}

    @property
    def action_space(self) -> Box:
        # dx (column), dy (row), cutcook, pickplace
        # We expect dx and dy to be one of -1, 0, or 1.
        # We expect interact to be either 0 or 1.
        return Box(low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 3.0, 1.0, 1.0]), dtype=np.float32)

    @staticmethod
    def _get_robot_direction(dx: float, dy: float) -> str:
        if dx < 0:
            return "left"
        elif dx > 0:
            return "right"
        elif dy < 0:
            return "down"
        elif dy > 0:
            return "up"
        return "no_change"

    @staticmethod
    def _get_cell_in_direction(row, col, direction) -> Tuple[int, int]:
        if direction == "left":
            return (row, col - 1)
        elif direction == "right":
            return (row, col + 1)
        elif direction == "up":
            return (row + 1, col)
        elif direction == "down":
            return (row - 1, col)
        return (row, col)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        dcol, drow, turn, interact, pickplace = action.arr

        robot_col, robot_row = self.get_position(self._robot, state)
        new_col = np.clip(robot_col + dcol, 0, self.num_cols - 1)
        new_row = np.clip(robot_row + drow, 0, self.num_rows - 1)

        # compute robot direction
        direction = self._get_robot_direction(dcol, drow)
        if direction != "no_change":
            # We'll need to be facing an object to pick it up or interact with
            # it.
            next_state.set(self._robot, "dir", self.dir_to_int[direction])
        elif turn in [0, 1, 2, 3]:
            next_state.set(self._robot, "dir", turn)

        # get the objects we can interact with
        items = [obj for obj in state if obj.is_instance(self._item_type)]

        # check for collision
        other_objects = []
        for object in state:
            if not object.is_instance(self._robot_type) and not object.is_instance(self._cell_type):
                other_objects.append(object)
        for obj in other_objects:
            if obj in items:
                if state.simulator_state[obj]["is_held"] > 0.5:
                    continue
            obj_col, obj_row = self.get_position(obj, state)
            if abs(new_col - obj_col) < 1e-3 and abs(new_row - obj_row) < 1e-3:
                return next_state

        next_state.set(self._robot, "col", new_col)
        next_state.set(self._robot, "row", new_row)

        # also move held object, if there is one
        for item in items:
            if state.simulator_state[item]["is_held"] > 0.5:
                next_state.set(item, "col", new_col)
                next_state.set(item, "row", new_row)

        # handle interaction
        for item in items:
            item_x, item_y = self.get_position(item, state)
            board_x, board_y = self.get_position(self._cutting_board, state)
            grill_x, grill_y = self.get_position(self._grill, state)
            if self.Facing_holds(state, [self._robot, item]):
                if interact > 0.5:
                    if item.is_instance(self._patty_type) and grill_x==item_x and grill_y==item_y:
                        next_state.simulator_state[item]["is_cooked"] = 1.0
                    elif item.is_instance(self._tomato_type) and board_x==item_x and board_y==item_y:
                        next_state.simulator_state[item]["is_sliced"] = 1.0

        # handle pick
        if pickplace > 0.5 and self._HandEmpty_holds(state, [self._robot]):
            # get all items we are facing
            facing_items = []
            for item in items:
                if self.Facing_holds(state, [self._robot, item]):
                    item_z = state.get(item, "z")
                    facing_items.append((item, item_z))
            if len(facing_items) > 0:
                # We'll pick up the item that is "on top".
                on_top = max(facing_items, key=lambda x: x[1])[0]
                next_state.simulator_state[on_top]["is_held"] = 1.0
                next_state.set(on_top, "col", robot_col)
                next_state.set(on_top, "row", robot_row)
                next_state.set(on_top, "z", 0)
                next_state.set(self._robot, "fingers", 1.0)

        # handle place
        if pickplace > 0.5 and not self._HandEmpty_holds(state, [self._robot]):
            held_item = [item for item in items if state.simulator_state[item]["is_held"] > 0.5][0]
            place_row, place_col = self._get_cell_in_direction(robot_row, robot_col, self.int_to_dir[state.get(self._robot, "dir")])
            if 0 <= place_row <= self.num_rows and 0 <= place_col <= self.num_cols:
                next_state.set(self._robot, "fingers", 0.0)
                next_state.simulator_state[held_item]["is_held"] = 0.0
                next_state.set(held_item, "col", place_col)
                next_state.set(held_item, "row", place_row)
                # If any other objects are at this location, then this must go
                # on top of them.
                # get objects at this location
                objects_at_loc = []
                for obj in state:
                    if obj.is_instance(self._item_type) or obj.is_instance(self._station_type):
                        x, y = self.get_position(obj, state)
                        if x == place_col and y == place_row:
                            objects_at_loc.append((obj, state.get(obj, "z")))
                if len(objects_at_loc) > 0:
                    new_z = max(objects_at_loc, key=lambda x: x[1])[1] + 1
                else:
                    new_z = 0
                next_state.set(held_item, "z", new_z)

        # print("Action that was taken: ", action)
        # print("Hidden state: ", self._hidden_state)
        # print("New state: ", next_state)
        return next_state

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        # import pdb; pdb.set_trace()
        # # Reset the hidden state.
        # for k, v in self._hidden_state.items():
        #     if k.is_instance(self._patty_type):
        #         v["is_cooked"] = 0.0
        #         v["is_held"] = 0.0
        #     elif k.is_instance(self._tomato_type):
        #         v["is_sliced"] = 0.0
        #         v["is_held"] = 0.0

        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = self._current_task.init_obs
        # Copy to prevent external changes to the environment's state.
        # This default implementation of reset assumes that observations are
        # states. Subclasses with different states should override.
        assert isinstance(self._current_observation, State)
        return self._current_observation.copy()

    #
    # def step(self, action: Action) -> Observation:
    #     pass

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        figsize = (self.num_cols * 2, self.num_rows * 2)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        plt.suptitle(caption, wrap=True)

        # Plot vertical lines
        for i in range(self.num_cols + 1):
            ax.axvline(x=i, color="k", linestyle="-")

        # Plot horizontal lines
        for i in range(self.num_rows + 1):
            ax.axhline(y=i, color="k", linestyle="-")

        # Draw robot
        robot_col = state.get(self._robot, "col")
        robot_row = state.get(self._robot, "row")
        # ax.plot(robot_col + 0.5, robot_row + 0.5, 'rs', markersize=20)
        robot_direction = self.int_to_dir[state.get(self._robot, "dir")]
        robot_img = mpimg.imread(f"predicators/envs/assets/imgs/robot_{robot_direction}.png")
        x, y = robot_col, robot_row
        image_size = (0.8, 0.8)
        # ax.imshow(robot_img, extent=[robot_col, robot_col + 1, robot_row, robot_row + 1])
        ax.imshow(robot_img, extent=[x + (1 - image_size[0]) / 2, x + (1 + image_size[0]) / 2, y + (1 - image_size[1]) / 2, y + (1 + image_size[1]) / 2])

        # Draw grill
        grill_img = mpimg.imread("predicators/envs/assets/imgs/grill.png")
        grill_col, grill_row = self.get_position(self._grill, state)
        x, y = grill_col, grill_row
        ax.imshow(grill_img, extent=[x, x+1, y, y+1])

        # Draw cutting board
        cutting_board_img = mpimg.imread("predicators/envs/assets/imgs/cutting_board.png")
        cutting_board_col, cutting_board_row = self.get_position(self._cutting_board, state)
        x, y = cutting_board_col, cutting_board_row
        ax.imshow(cutting_board_img, extent=[x, x+1, y, y+1])

        # Draw items
        type_to_img = {
            self._cheese_type: mpimg.imread("predicators/envs/assets/imgs/cheese.png"),
            self._top_bun_type: mpimg.imread("predicators/envs/assets/imgs/top_bun.png"),
            self._bottom_bun_type: mpimg.imread("predicators/envs/assets/imgs/bottom_bun.png")
        }
        held_img_size = (0.3, 0.3)
        offset = held_img_size[1] * (1/3)
        patty = [object for object in state if object.is_instance(self._patty_type)][0]
        tomato = [obj for obj in state if obj.is_instance(self._tomato_type)][0]
        cheese = [obj for obj in state if obj.is_instance(self._cheese_type)][0]
        top_bun = [obj for obj in state if obj.is_instance(self._top_bun_type)][0]
        bottom_bun = [obj for obj in state if obj.is_instance(self._bottom_bun_type)][0]
        items = [patty, tomato, cheese, top_bun, bottom_bun]
        for item in items:
            img = None
            if "is_cooked" in state.simulator_state[item]:
                raw_patty_img = mpimg.imread("predicators/envs/assets/imgs/raw_patty.png")
                cooked_patty_img = mpimg.imread("predicators/envs/assets/imgs/cooked_patty.png")
                img = cooked_patty_img if self._IsCooked_holds(state, [item]) else raw_patty_img
            elif "is_sliced" in state.simulator_state[item]:
                whole_tomato_img = mpimg.imread("predicators/envs/assets/imgs/whole_tomato.png")
                sliced_tomato_img = mpimg.imread("predicators/envs/assets/imgs/sliced_tomato.png")
                img = sliced_tomato_img if self._IsSliced_holds(state, [tomato]) else whole_tomato_img
            else:
                img = type_to_img[item.type]
            zorder = state.get(item, "z")
            is_held = state.simulator_state[item]["is_held"] > 0.5
            x, y = self.get_position(item, state)
            if is_held:
                extent = [x + (1 - held_img_size[0]) * (1/2), x + (1 + held_img_size[0]) * (1/2), y + offset, y + held_img_size[1] + offset]
            elif zorder > 0:
                offset = 0.1 * zorder
                image_size = (0.7, 0.7)
                extent = [x + (1 - image_size[0]) * (1/2), x + (1 + image_size[0]) * (1/2), y + (1 - image_size[1]) / 2 + offset, y + (1 + image_size[1]) / 2 + offset]
            else:
                # extent = [x, x+1, y, y+1]
                image_size = (0.7, 0.7)
                extent = [x + (1 - image_size[0]) * (1/2), x + (1 + image_size[0]) * (1/2), y + (1 - image_size[1]) / 2, y + (1 + image_size[1]) / 2]
            ax.imshow(img, extent=extent, zorder=zorder)

        # Draw background
        floor_img = mpimg.imread("predicators/envs/assets/imgs/floorwood.png")
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                x, y = j, i
                extent = [
                    x, x+1, y, y+1
                ]
                ax.imshow(floor_img, extent=extent, zorder=-1)

        ax.set_xlim(0, self.num_cols)
        ax.set_ylim(0, self.num_rows)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        return fig

    def get_event_to_action_fn(self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:
            logging.info("Controls: arrow keys to move, (e) to interact, (f) to pickplace")
            dcol, drow, turn, interact, pickplace = 0, 0, -1, 0, 0
            if event.key == "w":
                turn = 0
            elif event.key == "a":
                turn = 1
            elif event.key == "s":
                turn = 2
            elif event.key == "d":
                turn = 3
            elif event.key == "left":
                drow = 0
                dcol = -1
            elif event.key == "right":
                drow = 0
                dcol = 1
            elif event.key == "down":
                drow = -1
                dcol = 0
            elif event.key == "up":
                drow = 1
                dcol = 0
            elif event.key == "e":
                interact = 1
            elif event.key == "f":
                pickplace = 1
            action = Action(np.array([dcol, drow, turn, interact, pickplace], dtype=np.float32))
            return action

        return _event_to_action