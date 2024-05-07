"""A simple gridworld environment inspired by https://github.com/portal-cornell/robotouille."""

import logging
from typing import List, Optional, Sequence, Set, Callable

# import pygame
import numpy as np
from gym.spaces import Box
import matplotlib
import matplotlib.pyplot as plt

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type

# - Want to be able to run this as a VLM env, or a normal env
# - Want to give oracle's predicates access to full state, but not non-oracle's (maybe can use a different perceiver -- the perceiver we use with non-oracle actually throws out info that is there)
# - Need to keep track of certain state outside of the actual state (e.g. is_cooked for patty)
# - Make subclass of state that has the observation that produced it as a field
# - There should be a texture-based rendering (using pygame) and a non-texture-based rendering, using matplotlib?

class GridWorld(BaseEnv):
    """TODO"""

    ASSETS_DIRECTORY = ""

    # Types
    _robot_type = Type("robot", ["row", "col"])
    _cell_type = Type("cell", ["row", "col"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        self.num_rows = CFG.gridworld_num_rows
        self.num_cols = CFG.gridworld_num_cols
        self.num_cells = self.num_rows * self.num_cols

        # Predicates
        self._RobotInCell = Predicate("RobotInCell", [self._robot_type, self._cell_type], self._In_holds)

        # Static objects (exist no matter the settings)
        self._robot = Object("robby", self._robot_type)
        self._cells = [
            Object(f"cell{i}", self._cell_type) for i in range(self.num_cells)
        ]

        self._cell_to_neighbors = {}

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

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._RobotInCell}

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
        next_state.set(self._robot, "col", new_col)
        next_state.set(self._robot, "row", new_row)
        return next_state

    # def reset(self, train_or_test: str, task_idx: int) -> Observation:
    #     pass
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

        # Draw robot as a red circle.
        robot_col = state.get(self._robot, "col")
        robot_row = state.get(self._robot, "row")
        ax.plot(robot_col + 0.5, robot_row + 0.5, 'ro', markersize=20)

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




