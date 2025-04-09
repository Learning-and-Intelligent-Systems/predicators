"""A simple gridworld environment where a robot prepares a burger, inspired by
https://github.com/portal-cornell/robotouille.

This environment uses assets from robotouille that were designed by
Nicole Thean (https://github.com/nicolethean).
"""

import copy
import io
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from PIL import Image

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, DefaultEnvironmentTask, \
    EnvironmentTask, GroundAtom, Object, Observation, Predicate, State, Task, \
    Type, Video


class BurgerEnv(BaseEnv):
    """A simple gridworld environment where a robot prepares a burger, inspired
    by https://github.com/portal-cornell/robotouille.

    This environment is designed to showcase a predicate invention approach that
    learns geometric predicates that operate on the object-centric state and
    vision-language model predicates that operate on the visual rendering of the
    state.

    One quirk of this environment is that we want certain parts of the state to
    only be accessible by the oracle approach. This is because we want to invent
    predicates like IsCooked as a VLM predicate, but not a geometric predicate,
    so no information about how cooked an object is should be in the state. The
    solution to this is to hide certain state inside State.simulator_state.
    After the demonstrations are created by the oracle approach, we can erase
    this simulator state before we pass the demonstrations to the predicate
    invention approach.

    Example command to see demos created:
    python predicators/main.py --env burger
    --approach grammar_search_invention --seed 0 --num_train_tasks 10
    --option_model_terminate_on_repeat False
    --sesame_max_skeletons_optimized 1000 --timeout 80
    --make_demo_videos
    --bilevel_plan_without_sim True
    --sesame_task_planner fdopt

    Note that the default task planner is too slow -- fast downward is required.
    """

    # Types
    _object_type = Type("object", [])
    _item_or_station_type = Type("item_or_station", [], _object_type)
    # _item_type = Type("item", [], _item_or_station_type)
    # _station_type = Type("station", [], _item_or_station_type)
    _item_type = Type("item", [], _object_type)
    _station_type = Type("station", [], _object_type)
    _robot_type = Type("robot", ["row", "col", "z", "fingers", "dir"],
                       _object_type)
    _patty_type = Type("patty", ["row", "col", "z"], _item_type)
    _tomato_type = Type("lettuce", ["row", "col", "z"], _item_type)
    _cheese_type = Type("cheese", ["row", "col", "z"], _item_type)
    _bottom_bun_type = Type("bottom_bun", ["row", "col", "z"], _item_type)
    _top_bun_type = Type("top_bun", ["row", "col", "z"], _item_type)
    _grill_type = Type("grill", ["row", "col", "z"], _station_type)
    _cutting_board_type = Type("cutting_board", ["row", "col", "z"],
                               _station_type)

    dir_to_enum = {"up": 0, "left": 1, "down": 2, "right": 3}
    enum_to_dir = {0: "up", 1: "left", 2: "down", 3: "right"}

    # Currently, a lot of the code for this env -- like the rendering -- assumes
    # a 5x5 grid.
    num_rows = 5
    num_cols = 5

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._Adjacent = Predicate("Adjacent",
                                   [self._robot_type, self._object_type],
                                   self.Adjacent_holds)
        self._AdjacentToNothing = Predicate("AdjacentToNothing",
                                            [self._robot_type],
                                            self._AdjacentToNothing_holds)
        self._Facing = Predicate("Facing",
                                 [self._robot_type, self._object_type],
                                 self.Facing_holds)
        self._AdjacentNotFacing = Predicate(
            "AdjacentNotFacing", [self._robot_type, self._object_type],
            self._AdjacentNotFacing_holds)
        self._IsCooked = Predicate("IsCooked", [self._patty_type],
                                   self._IsCooked_holds)
        self._IsSliced = Predicate("IsSliced", [self._tomato_type],
                                   self._IsSliced_holds)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._HandEmpty_holds)
        self._Holding = Predicate("Holding",
                                  [self._robot_type, self._item_type],
                                  self._Holding_holds)
        self._On = Predicate("On", [self._item_type, self._object_type],
                             self._On_holds)
        self._OnNothing = Predicate("OnNothing", [self._object_type],
                                    self._OnNothing_holds)
        # When an argument should be an item or a station but not a robot,
        # having the argument more narrowly scoped than the object type would be
        # helpful. But this is causing errors so we'll avoid it for now.
        # # self._Clear = Predicate("Clear", [self._item_or_station_type],
        #                         self._Clear_holds)
        # TODO(ashay): fix the errors associated with this.
        self._Clear = Predicate("Clear", [self._object_type],
                                self._Clear_holds)
        self._GoalHack = Predicate("GoalHack", [
            self._bottom_bun_type, self._patty_type, self._cheese_type,
            self._tomato_type, self._top_bun_type
        ], self._GoalHack_holds)

        # Static objects (exist no matter the settings)
        self._robot = Object("robot", self._robot_type)
        self._grill = Object("grill", self._grill_type)
        self._cutting_board = Object("cutting_board", self._cutting_board_type)

    @classmethod
    def get_name(cls) -> str:
        return "burger"

    @property
    def types(self) -> Set[Type]:
        return {
            self._object_type, self._item_type, self._station_type,
            self._robot_type, self._patty_type, self._tomato_type,
            self._cheese_type, self._bottom_bun_type, self._top_bun_type,
            self._grill_type, self._cutting_board_type
        }

    def get_edge_cells_for_object_placement(
            self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        """Selects edge cells such that if objects were placed in these cells,
        the robot would never find itself adjacent to more than one object.

        This helper function assumes that the grid is 5x5.

        Public for use by tests.
        """
        n_row = self.num_rows
        n_col = self.num_cols
        top = [(n_row - 1, col) for col in range(n_col)]
        left = [(row, 0) for row in range(n_row)]
        bottom = [(0, col) for col in range(n_col)]
        right = [(row, n_col - 1) for row in range(n_row)]
        corners = [(0, 0), (0, self.num_cols - 1), (self.num_rows - 1, 0),
                   (self.num_rows - 1, self.num_cols - 1)]

        # Pick edge cells for objects to be placed in such that the robot will
        # never be adjacent to two objects at the same time.
        # 1. Pick one edge to keep all its cells.
        # 2. Pick one edge to lose two cells.
        # 3. The cells we keep in the remaining two edges are determined by the
        # previous choices.
        # If this strategy is confusing to you, spend a few minutes drawing it
        # out on graph paper.

        # We don't consider placing objects in the corners because the robot
        # cannot interact with an object that is diagonally positioned.
        edges = [top, left, bottom, right]
        for i, edge in enumerate(edges):
            edges[i] = [c for c in edge if c not in corners]
        top, left, bottom, right = edges
        # Without loss of generality, have the top edge keep all its cells. To
        # generate other possibilities, we will later rotate the entire grid
        # with some probability.
        # Note that we can always keep cells that are in the "middle" of the
        # edge -- we only need to worry about cells at the ends of an edge.
        # If one edge keeps all its cells (call this edge A), then for the two
        # edges that are adjacent to A, we can't choose the cell in each of
        # these that is closest to A -- otherwise the robot could be adjacent
        # to objects at once. Since our grid has 4 edges, this implies that
        # one edge will have to lose two cells, and the others will lose one
        # cell.
        loses_two = edges[rng.choice([1, 2, 3])]
        if loses_two == left:
            left = left[1:len(left) - 1]
            bottom = bottom[:-1]
            right = right[:-1]
        elif loses_two == bottom:
            left = left[:-1]
            bottom = bottom[1:len(bottom) - 1]
            right = right[:-1]
        elif loses_two == right:
            left = left[:-1]
            bottom = bottom[1:]
            right = right[1:len(right) - 1]
        edges = [top, left, bottom, right]
        # Now, rotate the grid with some probability to cover the total set of
        # possibilities for object placements that satisfy our constraint. To
        # see why this rotation covers all the possibilities, draw it out on
        # graph paper.
        cells = top + left + bottom + right
        rotate = rng.choice([0, 1, 2, 3])
        # Rotate 0 degrees.
        if rotate == 0:
            ret = cells
        elif rotate == 1:
            # Rotate 90 degrees.
            ret = [(col, n_row - 1 - row) for row, col in cells]
        elif rotate == 2:
            # Rotate 180 degrees.
            ret = [(n_row - 1 - row, n_col - 1 - col) for row, col in cells]
        else:
            # Rotate 270 degrees.
            ret = [(col, row) for row, col in cells]

        return ret

    def _get_tasks(self, num: int, rng: np.random.Generator,
                   train_or_test: str) -> List[EnvironmentTask]:
        del train_or_test  # unused
        tasks = []
        state_dict = {}
        hidden_state = {}

        spots_for_objects = self.get_edge_cells_for_object_placement(rng)

        for _ in range(num):
            shuffled_spots = spots_for_objects.copy()
            rng.shuffle(shuffled_spots)

            # Add robot, grill, and cutting board
            r, c = shuffled_spots[0]
            state_dict[self._robot] = {
                "row": 2,  # assumes 5x5 grid
                "col": 2,  # assumes 5x5 grid
                "z": 0,
                "fingers": 0.0,
                "dir": 3
            }
            r, c = shuffled_spots[1]
            state_dict[self._grill] = {"row": r, "col": c, "z": 0}
            r, c = shuffled_spots[2]
            state_dict[self._cutting_board] = {"row": r, "col": c, "z": 0}

            # Add patty
            r, c = shuffled_spots[3]
            patty = Object("patty", self._patty_type)
            state_dict[patty] = {"row": r, "col": c, "z": 0}
            hidden_state[patty] = {"is_cooked": 0.0, "is_held": 0.0}

            # Add tomato
            r, c = shuffled_spots[4]
            tomato = Object("lettuce", self._tomato_type)
            state_dict[tomato] = {"row": r, "col": c, "z": 0}
            hidden_state[tomato] = {"is_sliced": 0.0, "is_held": 0.0}

            # Add cheese
            r, c = shuffled_spots[5]
            cheese = Object("cheese", self._cheese_type)
            state_dict[cheese] = {"row": r, "col": c, "z": 0}
            hidden_state[cheese] = {"is_held": 0.0}

            # Add top bun
            r, c = shuffled_spots[6]
            top_bun = Object("top_bun", self._top_bun_type)
            state_dict[top_bun] = {"row": r, "col": c, "z": 0}
            hidden_state[top_bun] = {"is_held": 0.0}

            # Add bottom bun
            r, c = shuffled_spots[7]
            bottom_bun = Object("bottom_bun", self._bottom_bun_type)
            state_dict[bottom_bun] = {"row": r, "col": c, "z": 0}
            hidden_state[bottom_bun] = {"is_held": 0.0}

            # Note that the test task differs from the train task only in the
            # positions of objects.
            goal = {
                GroundAtom(self._On, [patty, bottom_bun]),
                GroundAtom(self._On, [cheese, patty]),
                # GroundAtom(self._On, [tomato, cheese]),
                # GroundAtom(self._On, [top_bun, tomato]),
                GroundAtom(self._IsCooked, [patty]),
                GroundAtom(self._IsSliced, [tomato]),
            }

            alt_goal = {
                GroundAtom(self._On, [patty, bottom_bun]),
                GroundAtom(self._On, [cheese, patty]),
                GroundAtom(self._GoalHack,
                           [bottom_bun, patty, cheese, tomato, top_bun])
            }
            state = utils.create_state_from_dict(state_dict)
            state.simulator_state = {}
            state.simulator_state["state"] = hidden_state
            # A DefaultEnvironmentTask is a dummy environment task. Our render
            # function does not use the task argument, so this is ok.
            state.simulator_state["images"] = self.render_state(
                state, DefaultEnvironmentTask)
            # Recall that a EnvironmentTask consists of an Observation and a
            # GoalDescription, both of whose types are Any.
            tasks.append(EnvironmentTask(state, goal, alt_goal_desc=alt_goal))

        return tasks

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               rng=self._train_rng,
                               train_or_test="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               rng=self._test_rng,
                               train_or_test="test")

    @classmethod
    def Adjacent_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Public for use by oracle options."""
        robot, obj = objects
        rx, ry = cls.get_position(robot, state)
        ox, oy = cls.get_position(obj, state)
        return cls.is_adjacent(rx, ry, ox, oy)

    def _AdjacentToNothing_holds(self, state: State,
                                 objects: Sequence[Object]) -> bool:
        robot, = objects
        for obj in state:
            if obj.is_instance(self._item_type) or \
                obj.is_instance(self._station_type):
                if self.Adjacent_holds(state, [robot, obj]):
                    return False
        return True

    @classmethod
    def Facing_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        """Public for use by oracle options."""
        robot, obj = objects
        rx, ry = cls.get_position(robot, state)
        rdir = state.get(robot, "dir")
        ox, oy = cls.get_position(obj, state)
        facing_left = ry == oy and rx - ox == 1 and cls.enum_to_dir[
            rdir] == "left"
        facing_right = ry == oy and rx - ox == -1 and cls.enum_to_dir[
            rdir] == "right"
        facing_down = ry - oy == 1 and rx == ox and cls.enum_to_dir[
            rdir] == "down"
        facing_up = ry - oy == -1 and rx == ox and cls.enum_to_dir[rdir] == "up"
        return facing_left or facing_right or facing_down or facing_up

    def _AdjacentNotFacing_holds(self, state: State,
                                 objects: Sequence[Object]) -> bool:
        return self.Adjacent_holds(
            state, objects) and not self.Facing_holds(state, objects)

    def _IsCooked_holds(self, state: State, objects: Sequence[Object]) -> bool:
        patty, = objects
        assert state.simulator_state is not None
        assert "state" in state.simulator_state
        return state.simulator_state["state"][patty]["is_cooked"] > 0.5

    def _IsSliced_holds(self, state: State, objects: Sequence[Object]) -> bool:
        tomato, = objects
        assert state.simulator_state is not None
        assert "state" in state.simulator_state
        return state.simulator_state["state"][tomato]["is_sliced"] > 0.5

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") < 0.5

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, item = objects
        assert state.simulator_state is not None
        assert "state" in state.simulator_state
        return not self._HandEmpty_holds(state, [robot]) and \
            state.simulator_state["state"][item]["is_held"] > 0.5

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        a, b = objects
        assert state.simulator_state is not None
        assert "state" in state.simulator_state
        ax, ay = self.get_position(a, state)
        bx, by = self.get_position(b, state)
        az = state.get(a, "z")
        bz = state.get(b, "z")
        # If an object is held by the robot, the object is not on the robot.
        if a.is_instance(self._item_type):
            if state.simulator_state["state"][a]["is_held"] > 0.5:
                return False
        return ax == bx and ay == by and az - 1 == bz

    def _OnNothing_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        obj, = objects
        for other_obj in state:
            if other_obj.is_instance(self._item_type) or other_obj.is_instance(
                    self._station_type):
                if self._On_holds(state, [obj, other_obj]):
                    return False
        return True

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        assert state.simulator_state is not None
        assert "state" in state.simulator_state
        # A held object is not clear.
        if obj.is_instance(
                self._item_type
        ) and state.simulator_state["state"][obj]["is_held"] > 0.5:
            return False
        for other_obj in state:
            if other_obj.is_instance(self._item_type) or other_obj.is_instance(
                    self._station_type):
                if self._On_holds(state, [other_obj, obj]):
                    return False
        return True

    def _GoalHack_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # bottom, patty, cheese, tomato, top = objects
        bottom, patty, cheese, tomato, _ = objects
        atoms = [
            self._On_holds(state, [patty, bottom]),
            self._On_holds(state, [cheese, patty]),
            # self._On_holds(state, [tomato, cheese]),
            # self._On_holds(state, [top, tomato]),
            self._IsCooked_holds(state, [patty]),
            self._IsSliced_holds(state, [tomato])
        ]
        return all(atoms)

    @classmethod
    def get_position(cls, obj: Object, state: State) -> Tuple[int, int]:
        """Public for use by oracle options."""
        col = state.get(obj, "col")
        row = state.get(obj, "row")
        return col, row

    @classmethod
    def is_adjacent(cls, col_1: int, row_1: int, col_2: int,
                    row_2: int) -> bool:
        """Public for use by oracle options."""
        adjacent_vertical = col_1 == col_2 and abs(row_1 - row_2) == 1
        adjacent_horizontal = row_1 == row_2 and abs(col_1 - col_2) == 1
        return adjacent_vertical or adjacent_horizontal

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Adjacent, self._AdjacentToNothing, self._AdjacentNotFacing,
            self._Facing, self._IsCooked, self._IsSliced, self._HandEmpty,
            self._Holding, self._On, self._OnNothing, self._Clear,
            self._GoalHack
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On, self._IsCooked, self._IsSliced}

    @property
    def agent_goal_predicates(self) -> Set[Predicate]:
        return {self._On, self._GoalHack}

    @property
    def action_space(self) -> Box:
        # dx (column), dy (row), direction, cut/cook, pick/place
        # We expect dx and dy to be one of -1, 0, or 1.
        # We expect direction to be one of -1, 0, 1, 2, or 3. -1 signifies
        # "no change in direction", and 0, 1, 2, and 3 signify a direction.
        # We expect cut/cook and pick/place to be 0 or 1.
        return Box(low=np.array([-1.0, -1.0, -1.0, 0.0, 0.0]),
                   high=np.array([1.0, 1.0, 3.0, 1.0, 1.0]),
                   dtype=np.float32)

    @staticmethod
    def _get_robot_direction(dx: float, dy: float) -> str:
        if dx < 0:
            return "left"
        if dx > 0:
            return "right"
        if dy < 0:
            return "down"
        if dy > 0:
            return "up"
        return "no_change"

    @staticmethod
    def get_cell_in_direction(x: int, y: int,
                              direction: str) -> Tuple[int, int]:
        """Public for use by tests."""
        if direction == "left":
            return (x - 1, y)
        if direction == "right":
            return (x + 1, y)
        if direction == "up":
            return (x, y + 1)
        if direction == "down":
            return (x, y - 1)
        return (x, y)

    @classmethod
    def get_empty_cells(cls, state: State) -> Set[Tuple[int, int]]:
        """Public for use by oracle options."""
        cells = set()
        for y in range(cls.num_rows):
            for x in range(cls.num_cols):
                cells.add((x, y))

        for obj in state:
            x, y = cls.get_position(obj, state)
            if (x, y) in cells:
                cells.remove((x, y))

        return set(cells)

    def simulate(self, state: State, action: Action) -> State:
        held_item = None
        # We assume only one of <dcol, drow>, <direction>, <interact>,
        # <pickplace> is not "null" in each action.
        # If each one was null, the action would be <0, 0, -1, 0, 0>.
        assert state.simulator_state is not None
        assert "state" in state.simulator_state
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        assert next_state.simulator_state is not None
        assert "state" in next_state.simulator_state
        dcol, drow, dir_from_turning, interact, pickplace = action.arr

        rx, ry = self.get_position(self._robot, state)
        new_rx = np.clip(rx + dcol, 0, self.num_cols - 1)
        new_ry = np.clip(ry + drow, 0, self.num_rows - 1)

        # Compute the robot's direction.
        dir_from_movement = self._get_robot_direction(dcol, drow)
        if dir_from_movement != "no_change":
            next_state.set(self._robot, "dir",
                           self.dir_to_enum[dir_from_movement])
        elif dir_from_turning in [0, 1, 2, 3]:
            next_state.set(self._robot, "dir", dir_from_turning)

        # Get the objects we can interact with.
        items = [obj for obj in state if obj.is_instance(self._item_type)]

        # Check for collision.
        other_objects = [
            obj for obj in state if not obj.is_instance(self._robot_type)
        ]
        for obj in other_objects:
            if obj in items:
                if state.simulator_state["state"][obj]["is_held"] > 0.5:
                    continue
            ox, oy = self.get_position(obj, state)
            if abs(new_rx - ox) < 1e-3 and abs(new_ry - oy) < 1e-3:
                next_state.simulator_state["images"] = self.render_state(
                    next_state, DefaultEnvironmentTask)
                return next_state

        # No collision detected, so we can move the robot.
        next_state.set(self._robot, "col", new_rx)
        next_state.set(self._robot, "row", new_ry)

        # If an object was held, move it with the robot.
        for item in items:
            if state.simulator_state["state"][item]["is_held"] > 0.5:
                next_state.set(item, "col", new_rx)
                next_state.set(item, "row", new_ry)

        # Handle interaction (cutting or cooking).
        for item in items:
            if self.Facing_holds(state,
                                 [self._robot, item]) and interact > 0.5:
                if item.is_instance(self._patty_type) and self._On_holds(
                        state, [item, self._grill]):
                    next_state.simulator_state["state"][item][
                        "is_cooked"] = 1.0
                elif item.is_instance(self._tomato_type) and self._On_holds(
                        state, [item, self._cutting_board]):
                    next_state.simulator_state["state"][item][
                        "is_sliced"] = 1.0

        # Handle picking.
        if pickplace > 0.5 and self._HandEmpty_holds(state, [self._robot]):
            facing_items = []
            for item in items:
                if self.Facing_holds(state, [self._robot, item]):
                    facing_items.append((item, state.get(item, "z")))
            if len(facing_items) > 0:
                # We'll pick up the item that is "on top".
                on_top = max(facing_items, key=lambda x: x[1])[0]
                next_state.simulator_state["state"][on_top]["is_held"] = 1.0
                next_state.set(on_top, "col", rx)
                next_state.set(on_top, "row", ry)
                next_state.set(on_top, "z", 1)
                next_state.set(self._robot, "fingers", 1.0)

        # Handle placing.
        if pickplace > 0.5 and not self._HandEmpty_holds(state, [self._robot]):
            held_item = [
                item for item in items
                if state.simulator_state["state"][item]["is_held"] > 0.5
            ][0]
            px, py = self.get_cell_in_direction(
                rx, ry, self.enum_to_dir[state.get(self._robot, "dir")])
            if 0 <= py <= self.num_rows and 0 <= px <= self.num_cols:
                next_state.set(self._robot, "fingers", 0.0)
                next_state.simulator_state["state"][held_item]["is_held"] = 0.0
                next_state.set(held_item, "col", px)
                next_state.set(held_item, "row", py)
                # If any other objects are at this location, then this must go
                # on top of them. Otherwise, we are placing on the ground.
                objects_at_loc = []
                for obj in other_objects:
                    ox, oy = self.get_position(obj, state)
                    if ox == px and oy == py:
                        objects_at_loc.append((obj, state.get(obj, "z")))
                if len(objects_at_loc) > 0:
                    new_z = max(objects_at_loc, key=lambda x: x[1])[1] + 1
                else:
                    new_z = 0
                next_state.set(held_item, "z", new_z)

        # Update the image
        assert next_state.simulator_state is not None
        next_state.simulator_state["images"] = self.render_state(
            next_state, DefaultEnvironmentTask)

        return next_state

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        figsize = (self.num_cols * 2, self.num_rows * 2)
        # The DPI has to be sufficiently high otherwise when the matplotlib
        # figure gets converted to a PIL image, text in the image can become
        # blurry.
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=216)
        fontsize = 14

        # Plot vertical lines
        for i in range(self.num_cols + 1):
            ax.axvline(x=i, color="k", linestyle="-")

        # Plot horizontal lines
        for i in range(self.num_rows + 1):
            ax.axhline(y=i, color="k", linestyle="-")

        # Draw robot
        x, y = self.get_position(self._robot, state)
        robot_direction = self.enum_to_dir[state.get(self._robot, "dir")]
        robot_img = mpimg.imread(
            utils.get_env_asset_path(f"imgs/robot_{robot_direction}.png"))
        img_size = (0.7, 0.7)
        ax.imshow(robot_img,
                  extent=[
                      x + (1 - img_size[0]) / 2, x + (1 + img_size[0]) / 2,
                      y + (1 - img_size[1]) / 2, y + (1 + img_size[1]) / 2
                  ])
        if CFG.burger_render_set_of_marks:
            ax.text(x + 1 / 2,
                    y + (1 - img_size[1]) / 2,
                    self._robot.name,
                    fontsize=fontsize,
                    color="red",
                    ha="center",
                    va="top",
                    bbox=dict(facecolor="black",
                              alpha=0.5,
                              boxstyle="square,pad=0.0"))

        # Draw grill
        x, y = self.get_position(self._grill, state)
        grill_img = mpimg.imread(utils.get_env_asset_path("imgs/grill.png"))
        ax.imshow(grill_img, extent=[x, x + 1, y, y + 1])
        if CFG.burger_render_set_of_marks:
            ax.text(x + 1 / 2,
                    y + (1 - img_size[1]) / 2,
                    self._grill.name,
                    fontsize=fontsize,
                    color="red",
                    ha="center",
                    va="top",
                    bbox=dict(facecolor="black",
                              alpha=0.5,
                              boxstyle="square,pad=0.0"))

        # Draw cutting board
        if self._cutting_board in state:
            x, y = self.get_position(self._cutting_board, state)
            cutting_board_img = mpimg.imread(
                utils.get_env_asset_path("imgs/cutting_board.png"))
            ax.imshow(cutting_board_img, extent=[x, x + 1, y, y + 1])
            if CFG.burger_render_set_of_marks:
                ax.text(x + 1 / 2,
                        y + (1 - img_size[1]) / 2,
                        self._cutting_board.name,
                        fontsize=fontsize,
                        color="red",
                        ha="center",
                        va="top",
                        bbox=dict(facecolor="black",
                                  alpha=0.5,
                                  boxstyle="square,pad=0.0"))

        # Draw items
        type_to_img = {
            self._top_bun_type:
            mpimg.imread("predicators/envs/assets/imgs/top_bun.png"),
            self._bottom_bun_type:
            mpimg.imread("predicators/envs/assets/imgs/bottom_bun.png"),
            self._cheese_type:
            mpimg.imread("predicators/envs/assets/imgs/cheese.png"),
            self._tomato_type:
            mpimg.imread(utils.get_env_asset_path("imgs/uncut_lettuce.png")),
            self._patty_type:
            mpimg.imread(
                utils.get_env_asset_path("imgs/realistic_raw_patty_full.png"))
        }
        held_img_size = (0.3, 0.3)
        offset = held_img_size[1] * (1 / 3)
        items = [obj for obj in state if obj.is_instance(self._item_type)]
        assert state.simulator_state is not None
        assert "state" in state.simulator_state
        for item in items:
            img = type_to_img[item.type]
            if "is_cooked" in state.simulator_state["state"][
                    item] and self._IsCooked_holds(state, [item]):
                img = mpimg.imread(
                    utils.get_env_asset_path(
                        "imgs/realistic_cooked_patty_full.png"))
            elif "is_sliced" in state.simulator_state["state"][
                    item] and self._IsSliced_holds(state, [item]):
                img = mpimg.imread(
                    utils.get_env_asset_path("imgs/cut_lettuce.png"))
            zorder = state.get(item, "z")
            is_held = state.simulator_state["state"][item]["is_held"] > 0.5
            x, y = self.get_position(item, state)
            # If the item is held, make it smaller so that it does obstruct the
            # robot.
            img_size = (0.7, 0.7)
            if is_held:
                extent = [
                    x + (1 - held_img_size[0]) * (1 / 2),
                    x + (1 + held_img_size[0]) * (1 / 2), y + offset,
                    y + held_img_size[1] + offset
                ]
            # If the item is on top of something else, make it look like it by
            # moving it up a little.
            elif zorder > 0:
                offset = 0.1 * zorder
                extent = [
                    x + (1 - img_size[0]) * (1 / 2),
                    x + (1 + img_size[0]) * (1 / 2),
                    y + (1 - img_size[1]) / 2 + offset,
                    y + (1 + img_size[1]) / 2 + offset
                ]
            else:
                extent = [
                    x + (1 - img_size[0]) * (1 / 2),
                    x + (1 + img_size[0]) * (1 / 2), y + (1 - img_size[1]) / 2,
                    y + (1 + img_size[1]) / 2
                ]
            ax.imshow(img, extent=extent, zorder=zorder)
            if CFG.burger_render_set_of_marks:
                if is_held:
                    rx, _ = self.get_position(self._robot, state)
                    # If the robot is on the right edge, put text labels for
                    # held items on the left side so that they don't extend past
                    # the edge of the grid and make the image larger.
                    if rx == self.num_cols - 1:
                        horizontal_align = "right"
                        text_x = x + (1 - held_img_size[0]) * (1 / 2)
                    else:
                        horizontal_align = "left"
                        text_x = x + (1 + held_img_size[0]) * (1 / 2)
                    ax.text(text_x,
                            y + offset + held_img_size[1] / 2,
                            item.name,
                            fontsize=fontsize,
                            color="red",
                            ha=horizontal_align,
                            va="top",
                            bbox=dict(facecolor="black",
                                      alpha=0.5,
                                      boxstyle="square,pad=0.0"))
                else:
                    if zorder > 0:
                        # If the item is on the grill or cutting board, and
                        # there is not an item on top of it, then put its text
                        # label near the top of the cell.
                        if self._cutting_board in state:
                            check = (self._On_holds(state, [item, self._grill])
                                     or self._On_holds(state, [
                                         item, self._cutting_board
                                     ])) and self._Clear_holds(state, [item])
                        else:
                            check = self._On_holds(state, [
                                item, self._grill
                            ]) and self._Clear_holds(state, [item])
                        if check:
                            ax.text(x + 1 / 2,
                                    y + (1 + img_size[1]) / 2,
                                    item.name,
                                    fontsize=fontsize,
                                    color="red",
                                    ha="center",
                                    va="bottom",
                                    bbox=dict(facecolor="black",
                                              alpha=0.5,
                                              boxstyle="square,pad=0.0"))
                        else:
                            ax.text(x,
                                    y + (0.1 * zorder) + (1 - img_size[1]) / 2,
                                    item.name,
                                    fontsize=fontsize,
                                    color="red",
                                    ha="left",
                                    va="top",
                                    bbox=dict(facecolor="black",
                                              alpha=0.5,
                                              boxstyle="square,pad=0.0"))
                    else:
                        # If something is on top of this item or this item is on
                        # something else, then put the text label on the left
                        # side of the cell.
                        if not self._Clear_holds(
                                state, [item]) or not self._OnNothing_holds(
                                    state, [item]):
                            ax.text(x,
                                    y + (1 - img_size[1]) / 2,
                                    item.name,
                                    fontsize=fontsize,
                                    color="red",
                                    ha="left",
                                    va="top",
                                    bbox=dict(facecolor="black",
                                              alpha=0.5,
                                              boxstyle="square,pad=0.0"))
                        else:
                            ax.text(x + 1 / 2,
                                    y + (1 - img_size[1]) / 2,
                                    item.name,
                                    fontsize=fontsize,
                                    color="red",
                                    ha="center",
                                    va="top",
                                    bbox=dict(facecolor="black",
                                              alpha=0.5,
                                              boxstyle="square,pad=0.0"))

        # Draw background
        floor_img = mpimg.imread(
            utils.get_env_asset_path("imgs/floorwood.png"))
        for y in range(self.num_rows):
            for x in range(self.num_cols):
                ax.imshow(floor_img, extent=[x, x + 1, y, y + 1], zorder=-1)

        ax.set_xlim(0, self.num_cols)
        ax.set_ylim(0, self.num_rows)
        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()
        return fig

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        if CFG.burger_dummy_render:
            return [np.zeros((16, 16), dtype=np.uint8)]
        fig = self.render_state_plt(state, task, action, caption)
        # Create an in-memory binary stream.
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        # Rewind the stream to the beginning so that it can be read from.
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)

        # Convert the image to RGB mode to save as JPEG, because
        # JPEG format does not support images with an alpha channel
        # (transparency).
        img_rgb = img.convert("RGB")
        jpeg_buf = io.BytesIO()
        img_rgb.save(jpeg_buf, format="JPEG")
        jpeg_buf.seek(0)
        jpeg_img = Image.open(jpeg_buf)

        # If we return jpeg_img, we get this error:
        # `ValueError: I/O operation on closed file`, so we copy it before
        # closing the buffers.
        ret_img = copy.deepcopy(jpeg_img)
        ret_arr = np.array(ret_img)
        buf.close()
        jpeg_buf.close()
        return [ret_arr]

    def _copy_observation(self, obs: Observation) -> Observation:
        return copy.deepcopy(obs)

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        # Rather than have the observation be the state + the image, we just
        # package the image inside the state's simulator_state.
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = self._current_task.init_obs
        return self._copy_observation(self._current_observation)

    def step(self, action: Action) -> Observation:
        # Rather than have the observation be the state + the image, we just
        # package the image inside the state's simulator_state.
        self._current_observation = self.simulate(self._current_observation,
                                                  action)
        return self._copy_observation(self._current_observation)

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:
            del state  # unused
            logging.info(
                "Controls: arrow keys to move, wasd to change direction, " \
                "(e) to interact, (f) to pick/place, (q) to quit"
            )
            dcol, drow, turn, interact, pickplace = 0, 0, -1, 0, 0
            if event.key == "q":
                raise utils.HumanDemonstrationFailure("Human quit.")
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
            action = Action(
                np.array([dcol, drow, turn, interact, pickplace],
                         dtype=np.float32))
            return action

        return _event_to_action


class BurgerNoMoveEnv(BurgerEnv):
    """BurgerEnv but with the movement option wrapped inside each of the other
    options."""

    # Types
    _object_type = Type("object", [])
    _item_or_station_type = Type("item_or_station", [], _object_type)
    # _item_type = Type("item", [], _item_or_station_type)
    # _station_type = Type("station", [], _item_or_station_type)
    _item_type = Type("item", [], _object_type)
    _station_type = Type("station", [], _object_type)
    _robot_type = Type("robot", ["row", "col", "z", "fingers", "dir"],
                       _object_type)
    _patty_type = Type("patty", ["row", "col", "z"], _item_type)
    _tomato_type = Type("lettuce", ["row", "col", "z"], _item_type)
    _cheese_type = Type("cheese", ["row", "col", "z"], _item_type)
    _bottom_bun_type = Type("bottom_bun", ["row", "col", "z"], _item_type)
    _top_bun_type = Type("top_bun", ["row", "col", "z"], _item_type)
    _grill_type = Type("grill", ["row", "col", "z"], _station_type)
    _cutting_board_type = Type("cutting_board", ["row", "col", "z"],
                               _station_type)

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._OnGround = Predicate("OnGround", [self._item_type],
                                   self._OnGround_holds)
        self._GoalHack2 = Predicate("GoalHack2",
                                    [self._bottom_bun_type, self._patty_type],
                                    self._GoalHack2_holds)
        self._GoalHack3 = Predicate("GoalHack3",
                                    [self._bottom_bun_type, self._tomato_type],
                                    self._GoalHack3_holds)
        self._GoalHack4 = Predicate("GoalHack4",
                                    [self._patty_type, self._tomato_type],
                                    self._GoalHack3_holds)
        self._GoalHack5 = Predicate(
            "GoalHack5", [self._cutting_board_type, self._patty_type],
            self._GoalHack5_holds)
        self._GoalHack6 = Predicate("GoalHack6",
                                    [self._patty_type, self._patty_type],
                                    self._GoalHack5_holds)
        self._GoalHack7 = Predicate("GoalHack7",
                                    [self._grill_type, self._patty_type],
                                    self._GoalHack5_holds)

    def _OnGround_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        obj_z = state.get(obj, "z")
        return obj_z == 0

    def _GoalHack2_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        # The object is somewhere below the patty and the patty is cooked.
        obj, patty = objects
        obj_z = state.get(obj, "z")
        obj_x, obj_y = self.get_position(obj, state)
        p_z = state.get(patty, "z")
        p_x, p_y = self.get_position(patty, state)
        same_cell = obj_x == p_x and obj_y == p_y
        return obj_z < p_z and same_cell and self._IsCooked_holds(
            state, [patty])

    def _GoalHack3_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        # The object is somewhere below the tomato and the tomato is sliced.
        obj, tomato = objects
        obj_z = state.get(obj, "z")
        obj_x, obj_y = self.get_position(obj, state)
        t_z = state.get(tomato, "z")
        t_x, t_y = self.get_position(tomato, state)
        same_cell = obj_x == t_x and obj_y == t_y
        return obj_z < t_z and same_cell and self._IsSliced_holds(
            state, [tomato])

    def _GoalHack5_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        obj, patty = objects
        # The object is right below the patty and the patty is cooked.
        obj_z = state.get(obj, "z")
        obj_x, obj_y = self.get_position(obj, state)
        p_z = state.get(patty, "z")
        p_x, p_y = self.get_position(patty, state)
        same_cell = obj_x == p_x and obj_y == p_y
        # We check that the object is right below the patty because we don't
        # want this turning on for the top patty when there is a stack of two
        # cooked patties on a cutting board -- then the operator that covers
        # this transition will be less general.
        return obj_z == p_z - 1 and same_cell and self._IsCooked_holds(
            state, [patty])

    @classmethod
    def get_name(cls) -> str:
        return "burger_no_move"

    def get_edge_cells_for_object_placement(
            self, rng: np.random.Generator) -> List[Tuple[int, int]]:
        del rng  # unused
        n_row = self.num_rows
        n_col = self.num_cols
        top = [(n_row - 1, col) for col in range(n_col)]
        left = [(row, 0) for row in range(n_row)]
        bottom = [(0, col) for col in range(n_col)]
        right = [(row, n_col - 1) for row in range(n_row)]
        corners = [(0, 0), (0, self.num_cols - 1), (self.num_rows - 1, 0),
                   (self.num_rows - 1, self.num_cols - 1)]
        cells = (set(top) | set(left) | set(bottom)
                 | set(right)) - set(corners)
        return sorted(cells)

    def _get_tasks(self, num: int, rng: np.random.Generator,
                   train_or_test: str) -> List[EnvironmentTask]:
        spots_for_objects = self.get_edge_cells_for_object_placement(rng)

        def create_default_state() -> Tuple[dict, dict, List[Tuple[int, int]]]:
            state_dict = {}
            hidden_state = {}
            shuffled_spots = spots_for_objects.copy()
            rng.shuffle(shuffled_spots)

            # Add robot, grill, and cutting board
            state_dict[self._robot] = {
                "row": 2,  # assumes 5x5 grid
                "col": 2,  # assumes 5x5 grid
                "z": 0,
                "fingers": 0.0,
                "dir": 3
            }
            r, c = shuffled_spots[0]
            state_dict[self._grill] = {"row": r, "col": c, "z": 0}
            r, c = shuffled_spots[1]
            state_dict[self._cutting_board] = {"row": r, "col": c, "z": 0}

            # Add patty
            r, c = shuffled_spots[2]
            patty = Object("patty1", self._patty_type)
            state_dict[patty] = {"row": r, "col": c, "z": 0}
            hidden_state[patty] = {"is_cooked": 0.0, "is_held": 0.0}

            # Add tomato
            r, c = shuffled_spots[3]
            tomato = Object("lettuce1", self._tomato_type)
            state_dict[tomato] = {"row": r, "col": c, "z": 0}
            hidden_state[tomato] = {"is_sliced": 0.0, "is_held": 0.0}

            # Add cheese
            r, c = shuffled_spots[4]
            cheese = Object("cheese1", self._cheese_type)
            state_dict[cheese] = {"row": r, "col": c, "z": 0}
            hidden_state[cheese] = {"is_held": 0.0}

            # Add top bun
            r, c = shuffled_spots[5]
            top_bun = Object("top_bun1", self._top_bun_type)
            state_dict[top_bun] = {"row": r, "col": c, "z": 0}
            hidden_state[top_bun] = {"is_held": 0.0}

            # Add bottom bun
            r, c = shuffled_spots[6]
            bottom_bun = Object("bottom_bun1", self._bottom_bun_type)
            state_dict[bottom_bun] = {"row": r, "col": c, "z": 0}
            hidden_state[bottom_bun] = {"is_held": 0.0}

            return state_dict, hidden_state, shuffled_spots

        def create_task(state_dict: dict, hidden_state: dict,
                        goal: Set[GroundAtom],
                        alt_goal: Set[GroundAtom]) -> EnvironmentTask:
            state = utils.create_state_from_dict(state_dict)
            state.simulator_state = {}
            state.simulator_state["state"] = hidden_state
            # A DefaultEnvironmentTask is a dummy environment task. Our render
            # function does not use the task argument, so this is ok.
            state.simulator_state["images"] = self.render_state(
                state, DefaultEnvironmentTask)
            return EnvironmentTask(state, goal, alt_goal_desc=alt_goal)

        def name_to_obj(state_dict: dict) -> Dict[str, Object]:
            d = {}
            for obj in state_dict:
                d[obj.name] = obj
            return d

        # We'll have three kinds of train/test pairs.
        # Consider a burger as a bottom bun, a top bun, and something(s) in
        # between them.
        # (1)
        # Train:
        # - Make a burger with a cooked patty
        # - Make a burger with chopped lettuce
        # - Place chopped lettuce on a patty
        # - Place chopped lettuce on a patty, place patty on cutting board
        # Test:
        # - Make a burger with a cooked patty and chopped lettuce
        # (2)
        # Train:
        # - Make a burger with a cooked patty
        # - Cook two patties and stack them, place bottom patty on cutting
        # board
        # Test:
        # - Make a burger with two cooked patties
        # (3)
        # Train:
        # - Make a burger with a cooked patty
        # Test:
        # - Make several bottom bun + cooked patty stacks

        def create_tasks_for_type_one(
        ) -> Tuple[List[EnvironmentTask], List[EnvironmentTask]]:
            train_tasks = []
            test_tasks = []

            assert CFG.num_train_tasks % 4 == 0

            for _ in range(CFG.num_train_tasks // 4):
                # train task 1
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                bottom_bun = d["bottom_bun1"]
                patty = d["patty1"]
                top_bun = d["top_bun1"]
                train_goal = {
                    GroundAtom(self._IsCooked, [patty]),
                    GroundAtom(self._On, [patty, bottom_bun]),
                    GroundAtom(self._On, [top_bun, patty])
                }
                alt_train_goal = {
                    GroundAtom(self._GoalHack2, [bottom_bun, patty]),
                    GroundAtom(self._On, [top_bun, patty])
                }
                train_task = create_task(state_dict, hidden_state, train_goal,
                                         alt_train_goal)
                train_tasks.append(train_task)

                # train task 2
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                bottom_bun = d["bottom_bun1"]
                tomato = d["lettuce1"]
                top_bun = d["top_bun1"]
                train_goal = {
                    GroundAtom(self._IsSliced, [tomato]),
                    GroundAtom(self._On, [tomato, bottom_bun]),
                    GroundAtom(self._On, [top_bun, tomato])
                }
                alt_train_goal = {
                    GroundAtom(self._GoalHack3, [bottom_bun, tomato]),
                    GroundAtom(self._On, [top_bun, tomato])
                }
                train_task = create_task(state_dict, hidden_state, train_goal,
                                         alt_train_goal)
                train_tasks.append(train_task)

                # train task 3
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                patty = d["patty1"]
                tomato = d["lettuce1"]
                train_goal = {
                    GroundAtom(self._On, [patty, self._cutting_board]),
                    GroundAtom(self._On, [tomato, patty]),
                    GroundAtom(self._IsSliced, [tomato])
                }
                alt_train_goal = {
                    GroundAtom(self._On, [patty, self._cutting_board]),
                    GroundAtom(self._GoalHack4, [patty, tomato])
                }
                train_task = create_task(state_dict, hidden_state, train_goal,
                                         alt_train_goal)
                train_tasks.append(train_task)

                # train task 4
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                patty = d["patty1"]
                tomato = d["lettuce1"]
                # Start with the patty on the grill
                r, c = shuffled_spots[0]  # where the grill is
                state_dict[patty] = {"row": r, "col": c, "z": 1}
                hidden_state[patty] = {"is_cooked": 0.0, "is_held": 0.0}
                train_goal = {
                    GroundAtom(self._On, [patty, self._grill]),
                    GroundAtom(self._On, [tomato, patty]),
                    GroundAtom(self._IsSliced, [tomato])
                }
                alt_train_goal = {
                    GroundAtom(self._On, [patty, self._grill]),
                    GroundAtom(self._GoalHack4, [patty, tomato])
                }
                train_task = create_task(state_dict, hidden_state, train_goal,
                                         alt_train_goal)
                train_tasks.append(train_task)

            for i in range(CFG.num_test_tasks):
                # test task 1
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                bottom_bun = d["bottom_bun1"]
                patty = d["patty1"]
                tomato = d["lettuce1"]
                top_bun = d["top_bun1"]

                if i < CFG.burger_num_test_start_holding:
                    # replace the cheese with a bottom bun
                    cheese1 = [o for o in state_dict if o.name == "cheese1"][0]
                    state_dict.pop(cheese1)
                    hidden_state.pop(cheese1)
                    r, c = shuffled_spots[4]  # where the cheese was
                    bottom_bun2 = Object("bottom_bun2", self._bottom_bun_type)
                    state_dict[bottom_bun2] = {"row": r, "col": c, "z": 0}
                    hidden_state[bottom_bun2] = {"is_held": 0.0}
                    # add another patty
                    r, c = shuffled_spots[7]  # next empty cell
                    patty2 = Object("patty2", self._patty_type)
                    state_dict[patty2] = {"row": r, "col": c, "z": 0}
                    hidden_state[patty2] = {"is_cooked": 0.0, "is_held": 0.0}
                    # add another lettuce
                    r, c = shuffled_spots[8]  # next empty cell
                    lettuce2 = Object("lettuce2", self._tomato_type)
                    state_dict[lettuce2] = {"row": r, "col": c, "z": 0}
                    hidden_state[lettuce2] = {"is_sliced": 1.0, "is_held": 0.0}
                    # start out holding patty2.
                    state_dict[self._robot]["fingers"] = 1.0
                    state_dict[patty2]["row"] = state_dict[self._robot]["row"]
                    state_dict[patty2]["col"] = state_dict[self._robot]["col"]
                    hidden_state[patty2] = {"is_cooked": 0.0, "is_held": 1.0}

                test_goal = {
                    GroundAtom(self._IsCooked, [patty]),
                    GroundAtom(self._IsSliced, [tomato]),
                    GroundAtom(self._On, [patty, bottom_bun]),
                    GroundAtom(self._On, [tomato, patty]),
                    GroundAtom(self._On, [top_bun, tomato]),
                }
                if i < CFG.burger_num_test_start_holding:
                    test_goal.add(GroundAtom(self._IsCooked, [patty2]))
                    test_goal.add(GroundAtom(self._On, [patty2, bottom_bun2]))
                alt_test_goal = {
                    GroundAtom(self._GoalHack2, [bottom_bun, patty]),
                    GroundAtom(self._GoalHack4, [patty, tomato]),
                    GroundAtom(self._On, [top_bun, tomato]),
                }
                if i < CFG.burger_num_test_start_holding:
                    alt_test_goal.add(
                        GroundAtom(self._GoalHack2, [bottom_bun2, patty2]))
                test_task = create_task(state_dict, hidden_state, test_goal,
                                        alt_test_goal)
                test_tasks.append(test_task)

            return train_tasks, test_tasks

        def create_tasks_for_type_two(
        ) -> Tuple[List[EnvironmentTask], List[EnvironmentTask]]:
            train_tasks = []
            test_tasks = []

            assert CFG.num_train_tasks % 3 == 0

            for _ in range(CFG.num_train_tasks // 3):
                # train task 1
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                bottom_bun = d["bottom_bun1"]
                patty = d["patty1"]
                top_bun = d["top_bun1"]
                train_goal = {
                    GroundAtom(self._IsCooked, [patty]),
                    GroundAtom(self._On, [patty, bottom_bun]),
                    GroundAtom(self._On, [top_bun, patty])
                }
                alt_train_goal = {
                    GroundAtom(self._GoalHack2, [bottom_bun, patty]),
                    GroundAtom(self._On, [top_bun, patty])
                }
                train_task = create_task(state_dict, hidden_state, train_goal,
                                         alt_train_goal)
                train_tasks.append(train_task)

                # train task 2
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                patty = d["patty1"]
                r, c = shuffled_spots[7]  # next empty cell
                patty2 = Object("patty2", self._patty_type)
                state_dict[patty2] = {"row": r, "col": c, "z": 0}
                hidden_state[patty2] = {"is_cooked": 0.0, "is_held": 0.0}
                train_goal = {
                    GroundAtom(self._On, [patty, self._cutting_board]),
                    GroundAtom(self._IsCooked, [patty]),
                    GroundAtom(self._IsCooked, [patty2]),
                    GroundAtom(self._On, [patty2, patty])
                }
                alt_train_goal = {
                    GroundAtom(self._GoalHack5, [self._cutting_board, patty]),
                    GroundAtom(self._GoalHack6, [patty, patty2])
                }
                train_task = create_task(state_dict, hidden_state, train_goal,
                                         alt_train_goal)
                train_tasks.append(train_task)

                # train task 3
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                patty = d["patty1"]
                r, c = shuffled_spots[7]  # next empty cell
                patty2 = Object("patty2", self._patty_type)
                state_dict[patty2] = {"row": r, "col": c, "z": 0}
                hidden_state[patty2] = {"is_cooked": 0.0, "is_held": 0.0}
                train_goal = {
                    GroundAtom(self._On, [patty2, self._grill]),
                    GroundAtom(self._IsCooked, [patty]),
                    GroundAtom(self._IsCooked, [patty2]),
                    GroundAtom(self._On, [patty, patty2])
                }
                alt_train_goal = {
                    GroundAtom(self._GoalHack7, [self._grill, patty2]),
                    GroundAtom(self._GoalHack6, [patty2, patty])
                }
                train_task = create_task(state_dict, hidden_state, train_goal,
                                         alt_train_goal)
                train_tasks.append(train_task)

            for i in range(CFG.num_test_tasks):
                # test task 1
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                d = name_to_obj(state_dict)
                bottom_bun = d["bottom_bun1"]
                patty = d["patty1"]
                top_bun = d["top_bun1"]
                r, c = shuffled_spots[7]  # next empty cell
                patty2 = Object("patty2", self._patty_type)
                state_dict[patty2] = {"row": r, "col": c, "z": 0}
                hidden_state[patty2] = {"is_cooked": 0.0, "is_held": 0.0}

                if i < CFG.burger_num_test_start_holding:
                    # replace the cheese with a bottom bun
                    cheese1 = [o for o in state_dict if o.name == "cheese1"][0]
                    state_dict.pop(cheese1)
                    hidden_state.pop(cheese1)
                    r, c = shuffled_spots[4]  # where the cheese was
                    bottom_bun2 = Object("bottom_bun2", self._bottom_bun_type)
                    state_dict[bottom_bun2] = {"row": r, "col": c, "z": 0}
                    hidden_state[bottom_bun2] = {"is_held": 0.0}
                    # add another patty
                    r, c = shuffled_spots[8]  # next empty cell
                    patty3 = Object("patty3", self._patty_type)
                    state_dict[patty3] = {"row": r, "col": c, "z": 0}
                    hidden_state[patty3] = {"is_cooked": 0.0, "is_held": 0.0}
                    # start out holding patty3.
                    state_dict[self._robot]["fingers"] = 1.0
                    state_dict[patty3]["row"] = state_dict[self._robot]["row"]
                    state_dict[patty3]["col"] = state_dict[self._robot]["col"]
                    hidden_state[patty3] = {"is_cooked": 0.0, "is_held": 1.0}

                test_goal = {
                    GroundAtom(self._IsCooked, [patty]),
                    GroundAtom(self._On, [patty, bottom_bun]),
                    GroundAtom(self._On, [patty2, patty]),
                    GroundAtom(self._On, [top_bun, patty2])
                }
                if i < CFG.burger_num_test_start_holding:
                    test_goal.add(GroundAtom(self._IsCooked, [patty3]))
                    test_goal.add(GroundAtom(self._On, [patty3, bottom_bun2]))
                alt_test_goal = {
                    GroundAtom(self._GoalHack2, [bottom_bun, patty]),
                    GroundAtom(self._GoalHack6, [patty, patty2]),
                    GroundAtom(self._On, [top_bun, patty2])
                }
                if i < CFG.burger_num_test_start_holding:
                    alt_test_goal.add(
                        GroundAtom(self._GoalHack2, [bottom_bun2, patty3]))
                test_task = create_task(state_dict, hidden_state, test_goal,
                                        alt_test_goal)
                test_tasks.append(test_task)
            return train_tasks, test_tasks

        def create_tasks_for_type_three(
        ) -> Tuple[List[EnvironmentTask], List[EnvironmentTask]]:
            train_tasks = []
            test_tasks = []
            made_double_patty_train_task = False
            for _ in range(CFG.num_train_tasks):
                # train task 1
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                state_dict = {
                    self._robot: state_dict[self._robot],
                    self._grill: state_dict[self._grill]
                }
                hidden_state = {}
                # Create patty.
                r, c = shuffled_spots[1]
                patty = Object("patty1", self._patty_type)
                state_dict[patty] = {"row": r, "col": c, "z": 0}
                hidden_state[patty] = {"is_cooked": 0.0, "is_held": 0.0}
                # Create bottom bun.
                r, c = shuffled_spots[6]
                bottom_bun = Object("bottom_bun1", self._bottom_bun_type)
                state_dict[bottom_bun] = {"row": r, "col": c, "z": 0}
                hidden_state[bottom_bun] = {"is_held": 0.0}
                # Create top bun.
                r, c = shuffled_spots[8]
                top_bun = Object("top_bun1", self._top_bun_type)
                state_dict[top_bun] = {"row": r, "col": c, "z": 0}
                hidden_state[top_bun] = {"is_held": 0.0}
                if not made_double_patty_train_task:
                    # Make an extra patty, bottom bun and top bun.
                    r, c = shuffled_spots[2]
                    patty2 = Object("patty2", self._patty_type)
                    state_dict[patty2] = {"row": r, "col": c, "z": 0}
                    hidden_state[patty2] = {"is_cooked": 0.0, "is_held": 0.0}
                    r, c = shuffled_spots[7]
                    bottom_bun2 = Object("bottom_bun2", self._bottom_bun_type)
                    state_dict[bottom_bun2] = {"row": r, "col": c, "z": 0}
                    hidden_state[bottom_bun2] = {"is_held": 0.0}
                    r, c = shuffled_spots[9]
                    top_bun2 = Object("top_bun2", self._top_bun_type)
                    state_dict[top_bun2] = {"row": r, "col": c, "z": 0}
                    hidden_state[top_bun2] = {"is_held": 0.0}
                    train_goal = {
                        GroundAtom(self._IsCooked, [patty]),
                        GroundAtom(self._On, [patty, bottom_bun]),
                        GroundAtom(self._On, [top_bun, patty]),
                        GroundAtom(self._IsCooked, [patty2]),
                        GroundAtom(self._On, [patty2, bottom_bun2]),
                        GroundAtom(self._On, [top_bun2, patty2])
                    }
                    alt_train_goal = {
                        GroundAtom(self._GoalHack2, [bottom_bun, patty]),
                        GroundAtom(self._On, [top_bun, patty]),
                        GroundAtom(self._GoalHack2, [bottom_bun2, patty2]),
                        GroundAtom(self._On, [top_bun2, patty2])
                    }
                    made_double_patty_train_task = True
                else:
                    train_goal = {
                        GroundAtom(self._IsCooked, [patty]),
                        GroundAtom(self._On, [patty, bottom_bun]),
                        GroundAtom(self._On, [top_bun, patty]),
                    }
                    alt_train_goal = {
                        GroundAtom(self._GoalHack2, [bottom_bun, patty]),
                        GroundAtom(self._On, [top_bun, patty]),
                    }
                train_task = create_task(state_dict, hidden_state, train_goal,
                                         alt_train_goal)
                train_tasks.append(train_task)

            for i in range(CFG.num_test_tasks):
                # test task 1
                state_dict, hidden_state, shuffled_spots = create_default_state(
                )
                state_dict = {
                    self._robot: state_dict[self._robot],
                    self._grill: state_dict[self._grill]
                }
                hidden_state = {}
                # Create three patties.
                r, c = shuffled_spots[1]
                patty = Object("patty1", self._patty_type)
                state_dict[patty] = {"row": r, "col": c, "z": 0}
                hidden_state[patty] = {"is_cooked": 0.0, "is_held": 0.0}
                r, c = shuffled_spots[2]
                patty2 = Object("patty2", self._patty_type)
                state_dict[patty2] = {"row": r, "col": c, "z": 0}
                hidden_state[patty2] = {"is_cooked": 0.0, "is_held": 0.0}
                r, c = shuffled_spots[3]
                patty3 = Object("patty3", self._patty_type)
                state_dict[patty3] = {"row": r, "col": c, "z": 0}
                hidden_state[patty3] = {"is_cooked": 0.0, "is_held": 0.0}
                r, c = shuffled_spots[4]
                patty4 = Object("patty4", self._patty_type)
                state_dict[patty4] = {"row": r, "col": c, "z": 0}
                hidden_state[patty4] = {"is_cooked": 0.0, "is_held": 0.0}
                r, c = shuffled_spots[5]
                patty5 = Object("patty5", self._patty_type)
                state_dict[patty5] = {"row": r, "col": c, "z": 0}
                hidden_state[patty5] = {"is_cooked": 0.0, "is_held": 0.0}
                # Create three bottom buns.
                r, c = shuffled_spots[6]
                bottom_bun = Object("bottom_bun1", self._bottom_bun_type)
                state_dict[bottom_bun] = {"row": r, "col": c, "z": 0}
                hidden_state[bottom_bun] = {"is_held": 0.0}
                r, c = shuffled_spots[7]
                bottom_bun2 = Object("bottom_bun2", self._bottom_bun_type)
                state_dict[bottom_bun2] = {"row": r, "col": c, "z": 0}
                hidden_state[bottom_bun2] = {"is_held": 0.0}
                r, c = shuffled_spots[8]
                bottom_bun3 = Object("bottom_bun3", self._bottom_bun_type)
                state_dict[bottom_bun3] = {"row": r, "col": c, "z": 0}
                hidden_state[bottom_bun3] = {"is_held": 0.0}
                r, c = shuffled_spots[9]
                bottom_bun4 = Object("bottom_bun4", self._bottom_bun_type)
                state_dict[bottom_bun4] = {"row": r, "col": c, "z": 0}
                hidden_state[bottom_bun4] = {"is_held": 0.0}
                r, c = shuffled_spots[10]
                bottom_bun5 = Object("bottom_bun5", self._bottom_bun_type)
                state_dict[bottom_bun5] = {"row": r, "col": c, "z": 0}
                hidden_state[bottom_bun5] = {"is_held": 0.0}

                if i < CFG.burger_num_test_start_holding:
                    r, c = shuffled_spots[11]  # next empty cell
                    bottom_bun6 = Object("bottom_bun6", self._bottom_bun_type)
                    state_dict[bottom_bun6] = {"row": r, "col": c, "z": 0}
                    hidden_state[bottom_bun6] = {"is_held": 0.0}
                    # add another patty
                    r, c = shuffled_spots[11]  # next empty cell
                    patty6 = Object("patty6", self._patty_type)
                    state_dict[patty6] = {"row": r, "col": c, "z": 0}
                    hidden_state[patty6] = {"is_cooked": 0.0, "is_held": 0.0}
                    # start out holding patty6.
                    state_dict[self._robot]["fingers"] = 1.0
                    state_dict[patty6]["row"] = state_dict[self._robot]["row"]
                    state_dict[patty6]["col"] = state_dict[self._robot]["col"]
                    hidden_state[patty6] = {"is_cooked": 0.0, "is_held": 1.0}

                test_goal = {
                    GroundAtom(self._On, [patty, bottom_bun]),
                    GroundAtom(self._On, [patty2, bottom_bun2]),
                    GroundAtom(self._On, [patty3, bottom_bun3]),
                    GroundAtom(self._On, [patty4, bottom_bun4]),
                    GroundAtom(self._On, [patty5, bottom_bun5]),
                    GroundAtom(self._IsCooked, [patty]),
                    GroundAtom(self._IsCooked, [patty2]),
                    GroundAtom(self._IsCooked, [patty3]),
                    GroundAtom(self._IsCooked, [patty4]),
                    GroundAtom(self._IsCooked, [patty5])
                }
                if i < CFG.burger_num_test_start_holding:
                    test_goal.add(GroundAtom(self._IsCooked, [patty6]))
                    test_goal.add(GroundAtom(self._On, [patty6, bottom_bun6]))
                alt_test_goal = {
                    GroundAtom(self._GoalHack2, [bottom_bun, patty]),
                    GroundAtom(self._GoalHack2, [bottom_bun2, patty2]),
                    GroundAtom(self._GoalHack2, [bottom_bun3, patty3]),
                    GroundAtom(self._GoalHack2, [bottom_bun4, patty4]),
                    GroundAtom(self._GoalHack2, [bottom_bun5, patty5])
                }
                if i < CFG.burger_num_test_start_holding:
                    alt_test_goal.add(
                        GroundAtom(self._GoalHack2, [bottom_bun6, patty6]))
                test_task = create_task(state_dict, hidden_state, test_goal,
                                        alt_test_goal)
                test_tasks.append(test_task)
            return train_tasks, test_tasks

        if CFG.burger_no_move_task_type == "more_stacks":
            train_tasks, test_tasks = create_tasks_for_type_three()
        elif CFG.burger_no_move_task_type == "fatter_burger":
            train_tasks, test_tasks = create_tasks_for_type_two()
        elif CFG.burger_no_move_task_type == "combo_burger":
            train_tasks, test_tasks = create_tasks_for_type_one()
        else:
            raise NotImplementedError(
                f"Unrecognized task type: {CFG.burger_no_move_task_type}.")

        if train_or_test == "train":
            assert len(train_tasks) == CFG.num_train_tasks
            return train_tasks
        assert len(test_tasks) == CFG.num_test_tasks
        return test_tasks

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._IsCooked, self._IsSliced, self._HandEmpty, self._Holding,
            self._On, self._OnGround, self._Clear, self._GoalHack2,
            self._GoalHack3, self._GoalHack4, self._GoalHack5, self._GoalHack6,
            self._GoalHack7
        }

    @property
    def agent_goal_predicates(self) -> Set[Predicate]:
        preds_by_task_type = {
            "more_stacks": {
                self._On, self._OnGround, self._GoalHack2, self._Clear,
                self._Holding
            },
            "fatter_burger": {
                self._On,
                self._OnGround,
                self._GoalHack2,
                self._GoalHack5,
                self._GoalHack6,
                self._GoalHack7,
                self._Clear,
                self._Holding,
            },
            "combo_burger": {
                self._On, self._OnGround, self._GoalHack2, self._GoalHack3,
                self._GoalHack4, self._Clear, self._Holding
            }
        }
        return preds_by_task_type[CFG.burger_no_move_task_type]

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.agent_goal_predicates

    def get_vlm_debug_atom_strs(self,
                                train_tasks: List[Task]) -> List[List[str]]:
        # Convert the default value from List[List[str]] to List[str] to match
        # the other entries we'll put into the dictionary.
        default = [a[0] for a in super().get_vlm_debug_atom_strs(train_tasks)]
        atom_strs_by_task_type = {
            "more_stacks": ["Cooked(patty1)"],
            "fatter_burger": ["Cooked(patty1)"],
            "combo_burger":
            ["Cooked(patty1)", "Cut(lettuce1)", "Whole(lettuce1)"]
        }
        atom_strs_by_task_type = defaultdict(lambda: default,
                                             atom_strs_by_task_type)
        atom_strs = atom_strs_by_task_type[CFG.burger_no_move_task_type]
        return [[a] for a in atom_strs]
