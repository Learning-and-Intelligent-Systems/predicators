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

# Options
# -

class GridWorld(BaseEnv):
    """TODO"""

    ASSETS_DIRECTORY = ""

    # Types
    _object_type = Type("object", [])
    _item_type = Type("item", [], _object_type)
    _station_type = Type("station", [], _object_type)

    _robot_type = Type("robot", ["row", "col", "fingers"])
    _cell_type = Type("cell", ["row", "col"])

    _patty_type = Type("patty", ["row", "col"], _item_type)
    _tomato_type = Type("tomato", ["row", "col"], _item_type)

    _grill_type = Type("grill", ["row", "col"], _station_type)
    _cutting_board_type = Type("cutting_board", ["row", "col"], _station_type)


    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        self.num_rows = CFG.gridworld_num_rows
        self.num_cols = CFG.gridworld_num_cols
        self.num_cells = self.num_rows * self.num_cols

        # Predicates
        self._RobotInCell = Predicate("RobotInCell", [self._robot_type, self._cell_type], self._In_holds)
        self._Adjacent = Predicate("Adjacent", [self._robot_type, self._item_type], self._Adjacent_holds)
        self._Facing = Predicate("Facing", [self._robot_type, self._object_type], self._Facing_holds)
        self._IsCooked = Predicate("IsCooked", [self._patty_type], self._IsCooked_holds)
        self._IsSliced = Predicate("IsSliced", [self._tomato_type], self._IsSliced_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type], self._HandEmpty_holds)

        # Static objects (exist no matter the settings)
        self._robot = Object("robby", self._robot_type)
        self._grill = Object("grill", self._grill_type)
        self._cutting_board = Object("cutting_board", self._cutting_board_type)
        self._cells = [
            Object(f"cell{i}", self._cell_type) for i in range(self.num_cells)
        ]

        self._hidden_state = {}

    @classmethod
    def get_name(cls) -> str:
        return "gridworld"

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._cell_type}

    def _get_tasks(self, num: int, rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []

        # Add robot, grill, and cutting board
        state_dict = {}
        state_dict[self._robot] = {"row": 0, "col": 0, "fingers": 0.0}
        self._hidden_state[self._robot] = {"dir": "down"}
        state_dict[self._grill] = {"row": 2, "col": 3}
        state_dict[self._cutting_board] = {"row": 1, "col": 3}

        # Add cells
        counter = 0
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                # get the next cell
                cell = self._cells[counter]
                state_dict[cell] = {
                    "row": i,
                    "col": j
                }
                counter += 1

        # Add patty
        patty = Object("patty", self._patty_type)
        state_dict[patty] = {
            "row": 1,
            "col": 0
        }
        self._hidden_state[patty] = {
            "is_cooked": 0.0
        }

        # Add tomato
        tomato = Object("tomato", self._tomato_type)
        state_dict[tomato] = {
            "row": 2,
            "col": 0
        }
        self._hidden_state[tomato] = {
            "is_sliced": 0.0
        }

        # Get the top right cell and make the goal for the agent to go there.
        top_right_cell = self._cells[-1]
        goal = {GroundAtom(self._RobotInCell, [self._robot, top_right_cell])}

        state = utils.create_state_from_dict(state_dict)
        for i in range(num):
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

    def _Adjacent_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, item = objects
        robot_col, robot_row = self._get_position(robot, state)
        item_col, item_row = self._get_position(item, state)
        return self._is_adjacent(item_col, item_row, robot_col, robot_row)

    def _Facing_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, obj = objects
        robot_col, robot_row = self._get_position(self._robot, state)
        robot_dir = self._hidden_state[self._robot]["dir"]
        obj_col, obj_row = self._get_position(obj, state)
        facing_left = robot_row == obj_row and robot_col - obj_col == 1 and robot_dir == "left"
        facing_right = robot_row == obj_row and robot_col - obj_col == -1 and robot_dir == "right"
        facing_down = robot_row - obj_row == 1 and robot_col == obj_col and robot_dir == "down"
        facing_up = robot_row - obj_row == -1 and robot_col == obj_col and robot_dir == "up"
        if facing_left or facing_right or facing_down or facing_up:
            return True
        return False

    def _IsCooked_holds(self, state: State, objects: Sequence[Object]) -> bool:
        patty, = objects
        if self._hidden_state[patty]["is_cooked"] > 0.5:
            return True
        return False

    def _IsSliced_holds(self, state: State, objects: Sequence[Object]) -> bool:
        tomato, = objects
        if self._hidden_state[tomato]["is_sliced"] > 0.5:
            return True
        return False

    def _HandEmpty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        if state.get(robot, "fingers") < 0.5:
            return True
        return False

    @staticmethod
    def _get_position(object: Object, state: State) -> Tuple[int, int]:
        col = state.get(object, "col")
        row = state.get(object, "row")
        return col, row

    @staticmethod
    def _is_adjacent(col_1, row_1, col_2, row_2):
        adjacent_vertical = col_1 == col_2 and abs(row_1 - row_2) == 1
        adjacent_horizontal = row_1 == row_2 and abs(col_1 - col_2) == 1
        if adjacent_vertical or adjacent_horizontal:
            return True
        return False

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._RobotInCell, self._Adjacent, self._IsCooked, self._IsSliced}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._RobotInCell}

    @property
    def action_space(self) -> Box:
        # dx (column), dy (row), cutcook, pickplace
        # We expect dx and dy to be one of -1, 0, or 1.
        # We expect interact to be either 0 or 1.
        return Box(low=np.array([-1.0, -1.0, 0.0, 0.0]), high=np.array([1.0, 1.0, 1.0, 1.0]), dtype=np.float32)

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
        dcol, drow, interact, pickplace = action.arr
        print("action: ", dcol, drow, interact, pickplace)

        robot_col, robot_row = self._get_position(self._robot, state)
        new_col = np.clip(robot_col + dcol, 0, self.num_cols - 1)
        new_row = np.clip(robot_row + drow, 0, self.num_rows - 1)

        # compute robot direction
        direction = self._get_robot_direction(dcol, drow)
        if direction != "no_change":
            self._hidden_state[self._robot]["dir"] = direction

        # get the objects we can interact with
        items = []
        for object in state:
            if object.is_instance(self._item_type):
                items.append(object)

        # check for collision
        other_objects = []
        for object in state:
            if not object.is_instance(self._robot_type) and not object.is_instance(self._cell_type):
                other_objects.append(object)
        for obj in other_objects:
            if obj in items:
                if self._hidden_state[obj]["is_held"] > 0.5:
                    continue
            obj_col, obj_row = self._get_position(obj, state)
            if abs(new_col - obj_col) < 1e-3 and abs(new_row - obj_row) < 1e-3:
                return next_state

        next_state.set(self._robot, "col", new_col)
        next_state.set(self._robot, "row", new_row)

        # also move held object, if there is one
        for item in items:
            if self._hidden_state[item]["is_held"] > 0.5:
                next_state.set(item, "col", new_col)
                next_state.set(item, "row", new_row)

        # handle interaction
        for item in items:
            if self._Facing_holds(state, [self._robot, item]):
                if interact > 0.5:
                    if item.is_instance(self._patty_type):
                        self._hidden_state[item]["is_cooked"] = 1.0
                    elif item.is_instance(self._tomato_type):
                        self._hidden_state[item]["is_sliced"] = 1.0

        # handle pickplace
        for item in items:
            is_held = self._hidden_state[item]["is_held"] > 0.5
            print("is held: ", is_held, item)
            if pickplace > 0.5 and self._Facing_holds(state, [self._robot, item]) and self._HandEmpty_holds(state, [self._robot]):
                # pick
                if pickplace > 0.5 and not is_held:
                    self._hidden_state[item]["is_held"] = 1.0
                    next_state.set(item, "col", robot_col)
                    next_state.set(item, "row", robot_row)
                    next_state.set(self._robot, "fingers", 1.0)

            # place
            elif pickplace > 0.5 and is_held:
                place_row, place_col = self._get_cell_in_direction(robot_row, robot_col, self._hidden_state[self._robot]["dir"])
                print("placing")
                print("place row, place col:", place_row, place_col)
                print("robot_row, robot_col:", robot_row, robot_col)
                if 0 <= place_row <= self.num_rows and 0 <= place_col <= self.num_cols:
                    self._hidden_state[item]["is_held"] = 0.0
                    next_state.set(item, "col", place_col)
                    next_state.set(item, "row", place_row)
                    next_state.set(self._robot, "fingers", 0.0)

        # print("Action that was taken: ", action)
        # print("Hidden state: ", self._hidden_state)
        # print("New state: ", next_state)
        return next_state

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        # Reset the hidden state.
        for k, v in self._hidden_state.items():
            v["is_held"] = 0.0
            if k.is_instance(self._patty_type):
                v["is_cooked"] = 0.0
            elif k.is_instance(self._tomato_type):
                v["is_sliced"] = 0.0

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
        robot_direction = self._hidden_state[self._robot]["dir"]
        robot_img = mpimg.imread(f"predicators/envs/assets/imgs/robot_{robot_direction}.png")
        x, y = robot_col, robot_row
        image_size = (0.8, 0.8)
        # ax.imshow(robot_img, extent=[robot_col, robot_col + 1, robot_row, robot_row + 1])
        ax.imshow(robot_img, extent=[x + (1 - image_size[0]) / 2, x + (1 + image_size[0]) / 2,
                                      y + (1 - image_size[1]) / 2, y + (1 + image_size[1]) / 2])

        # Draw grill
        grill_img = mpimg.imread("predicators/envs/assets/imgs/grill.png")
        grill_col, grill_row = self._get_position(self._grill, state)
        x, y = grill_col, grill_row
        ax.imshow(grill_img, extent=[x, x+1, y, y+1])

        # Draw cutting board
        cutting_board_img = mpimg.imread("predicators/envs/assets/imgs/cutting_board.png")
        cutting_board_col, cutting_board_row = self._get_position(self._cutting_board, state)
        x, y = cutting_board_col, cutting_board_row
        ax.imshow(cutting_board_img, extent=[x, x+1, y, y+1])

        # Draw patty
        light_brown = (0.72, 0.52, 0.04)
        dark_brown = (0.39, 0.26, 0.13)
        patty = [object for object in state if object.is_instance(self._patty_type)][0]
        patty_color = dark_brown if self._IsCooked_holds(state, [patty]) else light_brown
        patty_col = state.get(patty, "col")
        patty_row = state.get(patty, "row")
        raw_patty_img = mpimg.imread("predicators/envs/assets/imgs/raw_patty.png")
        cooked_patty_img = mpimg.imread("predicators/envs/assets/imgs/cooked_patty.png")
        patty_img = cooked_patty_img if self._IsCooked_holds(state, [patty]) else raw_patty_img
        ax.imshow(patty_img, extent=[patty_col, patty_col+1, patty_row, patty_row+1])
        # ax.plot(patty_col + 0.5, patty_row + 0.5, 'o', color=patty_color, markersize=20)

        # Draw tomato
        tomato = [obj for obj in state if obj.is_instance(self._tomato_type)][0]
        whole_tomato_img = mpimg.imread("predicators/envs/assets/imgs/whole_tomato.png")
        sliced_tomato_img = mpimg.imread("predicators/envs/assets/imgs/sliced_tomato.png")
        tomato_img = sliced_tomato_img if self._IsSliced_holds(state, [tomato]) else whole_tomato_img
        tomato_col, tomato_row = self._get_position(tomato, state)
        x, y = tomato_col, tomato_row
        ax.imshow(tomato_img, extent=[x, x+1, y, y+1])

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
            dcol, drow, interact, pickplace = 0, 0, 0, 0
            if event.key in ["left", "a"]:
                drow = 0
                dcol = -1
            elif event.key in ["right", "d"]:
                drow = 0
                dcol = 1
            elif event.key in ["down", "s"]:
                drow = -1
                dcol = 0
            elif event.key in ["up", "w"]:
                drow = 1
                dcol = 0
            elif event.key == "e":
                interact = 1
            elif event.key == "f":
                pickplace = 1
            action = Action(np.array([dcol, drow, interact, pickplace], dtype=np.float32))
            return action

        return _event_to_action