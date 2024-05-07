"""A simple gridworld environment inspired by https://github.com/portal-cornell/robotouille."""

from typing import List, Optional, Sequence, Set

import pygame
import numpy as np
from gym.spaces import Box

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
    self._robot_type = Type("robot", ["row", "col"])
    self._cell_type = Type("cell", ["row", "col"])

    def __init__(self, use_gui: bool = True):
        super().__init__(use_gui)

        self.num_rows = CFG.gridworld_num_rows
        self.num_cols = CFG.gridworld_num_cols
        self.num_cells = self.num_rows * self.num_cols

        # Predicates
        self._RobotInCell = Predicate("RobotInCell", [self._robot_type, self._cell_type], self._In_holds)

        # Static objects (exist no matter the settings)
        self._robot = Object("robot", [], self._object_type)
        self._cells = [
            Object(f"cell{i}", self._object_type) for i in range(self.num_cells)
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
        pass

    @property
    def goal_predicates(self) -> Set[Predicate]:
        pass

    @property
    def action_space(self) -> Box:
        # dx (column), dy (row), interact
        # We expect dx and dy to be one of -1, 0, or 1.
        # We expect interact to be either 0 or 1.
        return Box(low=np.array([-1.0, -1.0, 0.0]), high=np.array([1.0, 1.0, 1.0]), dtype=np.float32)

    def simulate(self, state: State, action: Action) -> State:
        pass

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        pass

    def step(self, action: Action) -> Observation:
        pass




