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

# - Want to be able to run this as a VLM env, or a normal env
# - Want to give oracle's predicates access to full state, but not non-oracle's (maybe can use a different perceiver -- the perceiver we use with non-oracle actually throws out info that is there)
# - Need to keep track of certain state outside of the actual state (e.g. is_cooked for patty)
# - Make subclass of state that has the observation that produced it as a field
# - There should be a texture-based rendering (using pygame) and a non-texture-based rendering, using matplotlib?

class GridWorld(BaseEnv):
    """TODO"""

    ASSETS_DIRECTORY = ""

    # Types
    _item_type = Type("item", [])
    _robot_type = Type("robot", ["row", "col"])
    _cell_type = Type("cell", ["row", "col"])
    _patty_type = Type("patty", ["row", "col"], _item_type)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        self.num_rows = CFG.gridworld_num_rows
        self.num_cols = CFG.gridworld_num_cols
        self.num_cells = self.num_rows * self.num_cols

        # Predicates
        self._RobotInCell = Predicate("RobotInCell", [self._robot_type, self._cell_type], self._In_holds)
        self._Adjacent = Predicate("AdjacentTo", [self._robot_type, self._item_type], self._Adjacent_holds)
        self._IsCooked = Predicate("IsCooked", [self._patty_type], self._IsCooked_holds)

        # Static objects (exist no matter the settings)
        self._robot = Object("robby", self._robot_type)
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
        state_dict = {
            self._robot: {
                "row": 0,
                "col": 0
            }
        }
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

        # Add items, stations, etc.
        patty = Object("patty", self._patty_type)
        state_dict[patty] = {
            "row": 1,
            "col": 0
        }
        self._hidden_state[patty] = {
            "is_cooked": 0.0
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

    def _IsCooked_holds(self, state: State, objects: Sequence[Object]) -> bool:
        patty, = objects
        if self._hidden_state[patty]["is_cooked"] > 0.5:
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
        return {self._RobotInCell, self._Adjacent}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._RobotInCell}

    @property
    def action_space(self) -> Box:
        # dx (column), dy (row), interact
        # We expect dx and dy to be one of -1, 0, or 1.
        # We expect interact to be either 0 or 1.
        return Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        dcol, drow, interact = action.arr

        robot_col = state.get(self._robot, "col")
        robot_row = state.get(self._robot, "row")
        new_col = np.clip(robot_col + dcol, 0, self.num_cols - 1)
        new_row = np.clip(robot_row + drow, 0, self.num_rows - 1)

        # check for collision
        other_objects = []
        for object in state:
            if not object.is_instance(self._robot_type) and not object.is_instance(self._cell_type):
                other_objects.append(object)
        for obj in other_objects:
            obj_col = state.get(obj, "col")
            obj_row = state.get(obj, "row")
            if abs(new_col - obj_col) < 1e-3 and abs(new_row - obj_row) < 1e-3:
                return next_state

        next_state.set(self._robot, "col", new_col)
        next_state.set(self._robot, "row", new_row)

        # handle interaction
        items = []
        for object in state:
            if object.is_instance(self._item_type):
                items.append(object)
        for item in items:
            item_col, item_row = self._get_position(item, state)
            if self._is_adjacent(item_col, item_row, new_col, new_row):
                if interact > 0.5:
                    self._hidden_state[item]["is_cooked"] = 1.0

        # print("Action that was taken: ", action)
        # print("Hidden state: ", self._hidden_state)
        # print("New state: ", next_state)
        return next_state

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        # Reset the hidden state.
        for k, v in self._hidden_state.items():
            if k.is_instance(self._patty_type):
                self._hidden_state[k]["is_cooked"] = 0.0

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

        # Draw patty as a brown circle.
        light_brown = (0.72, 0.52, 0.04)
        dark_brown = (0.39, 0.26, 0.13)
        patty = [object for object in state if object.is_instance(self._patty_type)][0]
        patty_color = dark_brown if self._IsCooked_holds(state, [patty]) else light_brown
        patty_col = state.get(patty, "col")
        patty_row = state.get(patty, "row")
        raw_patty_img = mpimg.imread("patty.png")
        cooked_patty_img = mpimg.imread("cookedpatty.png")
        patty_img = cooked_patty_img if self._IsCooked_holds(state, [patty]) else raw_patty_img
        ax.imshow(patty_img, extent=[patty_col, patty_col+1, patty_row, patty_row+1])
        # ax.plot(patty_col + 0.5, patty_row + 0.5, 'o', color=patty_color, markersize=20)

        # Draw robot as a red circle.
        robot_col = state.get(self._robot, "col")
        robot_row = state.get(self._robot, "row")
        # ax.plot(robot_col + 0.5, robot_row + 0.5, 'rs', markersize=20)

        # # Try drawing the robot as an image.
        robot_img = mpimg.imread("robot.png")
        x, y = robot_col, robot_row
        image_size = (0.8, 0.8)
        # ax.imshow(robot_img, extent=[robot_col, robot_col + 1, robot_row, robot_row + 1])
        ax.imshow(robot_img, extent=[x + (1 - image_size[0]) / 2, x + (1 + image_size[0]) / 2,
                                      y + (1 - image_size[1]) / 2, y + (1 + image_size[1]) / 2])

        # Draw background
        floor_img = mpimg.imread("floorwood.png")
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
            logging.info("Controls: arrow keys to move, (e) to interact")
            dcol, drow, interact = 0, 0, 0
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
            action = Action(np.array([dcol, drow, interact], dtype=np.float32))
            return action

        return _event_to_action