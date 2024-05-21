"""A simple gridworld environment where a robot prepares a burger, inspired by
https://github.com/portal-cornell/robotouille.

This environment uses assets from robotouille that were designed by
Nicole Thean (https://github.com/nicolethean).
"""

import logging
from typing import Callable, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
# import pygame
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class GridWorldEnv(BaseEnv):
    """A simple gridworld environment where a robot prepares a burger, inspired
    by https://github.com/portal-cornell/robotouille.

    This environment is designed to showcase a predicate invention approach that
    learns geometric predicates that operate on the object-centric state and
    video-language model predicates that operate on the visual rendering of the
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
    python predicators/main.py --env gridworld
    --approach grammar_search_invention --seed 0 --num_train_tasks 10
    --option_model_terminate_on_repeat False
    --sesame_max_skeletons_optimized 1000 --timeout 80
    --sesame_max_samples_per_step 1 --make_demo_videos
    --sesame_task_planner fdopt

    Note that the default task planner is too slow -- fast downward is required.
    """

    # Types
    _object_type = Type("object", [])
    _item_type = Type("item", [], _object_type)
    _station_type = Type("station", [], _object_type)

    _robot_type = Type("robot", ["row", "col", "fingers", "dir"])

    _patty_type = Type("patty", ["row", "col", "z"], _item_type)
    _tomato_type = Type("tomato", ["row", "col", "z"], _item_type)
    _cheese_type = Type("cheese", ["row", "col", "z"], _item_type)
    _bottom_bun_type = Type("bottom_bun", ["row", "col", "z"], _item_type)
    _top_bun_type = Type("top_bun", ["row", "col", "z"], _item_type)

    _grill_type = Type("grill", ["row", "col", "z"], _station_type)
    _cutting_board_type = Type("cutting_board", ["row", "col", "z"],
                               _station_type)

    dir_to_enum = {"up": 0, "left": 1, "down": 2, "right": 3}
    enum_to_dir = {0: "up", 1: "left", 2: "down", 3: "right"}

    num_rows = CFG.gridworld_num_rows
    num_cols = CFG.gridworld_num_cols

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
        self._Clear = Predicate("Clear", [self._object_type],
                                self._Clear_holds)
        # self._GoalHack = Predicate("GoalHack", [
        #     self._bottom_bun_type, self._patty_type, self._cheese_type,
        #     self._tomato_type, self._top_bun_type
        # ], self._GoalHack_holds)

        # Static objects (exist no matter the settings)
        self._robot = Object("robby", self._robot_type)
        self._grill = Object("grill", self._grill_type)
        self._cutting_board = Object("cutting_board", self._cutting_board_type)

    @classmethod
    def get_name(cls) -> str:
        return "gridworld"

    @property
    def types(self) -> Set[Type]:
        return {
            self._object_type, self._item_type, self._station_type,
            self._robot_type, self._patty_type, self._tomato_type,
            self._cheese_type, self._bottom_bun_type, self._top_bun_type,
            self._grill_type, self._cutting_board_type
        }

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        del rng  # unused
        tasks = []

        # Add robot, grill, and cutting board
        state_dict = {}
        hidden_state = {}
        state_dict[self._robot] = {
            "row": 2,
            "col": 2,
            "fingers": 0.0,
            "dir": 3
        }
        state_dict[self._grill] = {"row": 2, "col": 3, "z": 0}
        state_dict[self._cutting_board] = {"row": 1, "col": 3, "z": 0}

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

        goal = {
            GroundAtom(self._On, [patty, bottom_bun]),
            GroundAtom(self._On, [cheese, patty]),
            GroundAtom(self._On, [tomato, cheese]),
            GroundAtom(self._On, [top_bun, tomato]),
            GroundAtom(self._IsCooked, [patty]),
            GroundAtom(self._IsSliced, [tomato]),
        }

        for _ in range(num):
            state = utils.create_state_from_dict(state_dict)
            state.simulator_state = hidden_state
            # Note: this takes in Observation, GoalDescription, whose types are
            # Any
            tasks.append(EnvironmentTask(state, goal))

        return tasks

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

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
            if obj.is_instance(self._item_type):
                if not self._Holding_holds(
                        state, [robot, obj]) and self.Adjacent_holds(
                            state, [robot, obj]):
                    return False
            elif obj.is_instance(self._station_type):
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
        return state.simulator_state[patty]["is_cooked"] > 0.5

    def _IsSliced_holds(self, state: State, objects: Sequence[Object]) -> bool:
        tomato, = objects
        return state.simulator_state[tomato]["is_sliced"] > 0.5

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        robot, = objects
        return state.get(robot, "fingers") < 0.5

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, item = objects
        return not self._HandEmpty_holds(
            state, [robot]) and state.simulator_state[item]["is_held"] > 0.5

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        a, b = objects
        ax, ay = self.get_position(a, state)
        bx, by = self.get_position(b, state)
        az = state.get(a, "z")
        bz = state.get(b, "z")
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
        for other_obj in state:
            if other_obj.is_instance(self._item_type) or other_obj.is_instance(
                    self._station_type):
                if self._On_holds(state, [other_obj, obj]):
                    return False
        return True

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

    @classmethod
    def get_position(cls, obj: Object, state: State) -> Tuple[int, int]:
        """Public for use by oracle options."""
        col = state.get(obj, "col")
        row = state.get(obj, "row")
        return col, row

    @classmethod
    def is_adjacent(cls, col_1, row_1, col_2, row_2):
        """Public for use by oracle options."""
        adjacent_vertical = col_1 == col_2 and abs(row_1 - row_2) == 1
        adjacent_horizontal = row_1 == row_2 and abs(col_1 - col_2) == 1
        return adjacent_vertical or adjacent_horizontal

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._Adjacent, self._AdjacentToNothing, self._AdjacentNotFacing,
            self._Facing, self._IsCooked, self._IsSliced, self._HandEmpty,
            self._Holding, self._On, self._OnNothing, self._Clear
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On, self._IsCooked, self._IsSliced}

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
    def _get_cell_in_direction(x, y, direction) -> Tuple[int, int]:
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
        # We assume only one of <dcol, drow>, <direction>, <interact>,
        # <pickplace> is not "null" in each action.
        # If each one was null, the action would be <0, 0, -1, 0, 0>.
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
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
                if state.simulator_state[obj]["is_held"] > 0.5:
                    continue
            ox, oy = self.get_position(obj, state)
            if abs(new_rx - ox) < 1e-3 and abs(new_ry - oy) < 1e-3:
                return next_state

        # No collision detected, so we can move the robot.
        next_state.set(self._robot, "col", new_rx)
        next_state.set(self._robot, "row", new_ry)

        # If an object was held, move it with the robot.
        for item in items:
            if state.simulator_state[item]["is_held"] > 0.5:
                next_state.set(item, "col", new_rx)
                next_state.set(item, "row", new_ry)

        # Handle interaction (cutting or cooking).
        for item in items:
            if self.Facing_holds(state,
                                 [self._robot, item]) and interact > 0.5:
                if item.is_instance(self._patty_type) and self._On_holds(
                        state, [item, self._grill]):
                    next_state.simulator_state[item]["is_cooked"] = 1.0
                elif item.is_instance(self._tomato_type) and self._On_holds(
                        state, [item, self._cutting_board]):
                    next_state.simulator_state[item]["is_sliced"] = 1.0

        # Handle picking.
        if pickplace > 0.5 and self._HandEmpty_holds(state, [self._robot]):
            facing_items = []
            for item in items:
                if self.Facing_holds(state, [self._robot, item]):
                    facing_items.append((item, state.get(item, "z")))
            if len(facing_items) > 0:
                # We'll pick up the item that is "on top".
                on_top = max(facing_items, key=lambda x: x[1])[0]
                next_state.simulator_state[on_top]["is_held"] = 1.0
                next_state.set(on_top, "col", rx)
                next_state.set(on_top, "row", ry)
                next_state.set(on_top, "z", 0)
                next_state.set(self._robot, "fingers", 1.0)

        # Handle placing.
        if pickplace > 0.5 and not self._HandEmpty_holds(state, [self._robot]):
            held_item = [
                item for item in items
                if state.simulator_state[item]["is_held"] > 0.5
            ][0]
            px, py = self._get_cell_in_direction(
                rx, ry, self.enum_to_dir[state.get(self._robot, "dir")])
            if 0 <= py <= self.num_rows and 0 <= px <= self.num_cols:
                next_state.set(self._robot, "fingers", 0.0)
                next_state.simulator_state[held_item]["is_held"] = 0.0
                next_state.set(held_item, "col", px)
                next_state.set(held_item, "row", py)
                # If any other objects are at this location, then this must go
                # on top of them.
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

        # print("state: ", state)
        # print("action: ", action)
        # print("next state: ", next_state)
        # print()
        # top_bun = [obj for obj in state if obj.name == "top_bun"][0]
        # if pickplace > 0 and state.simulator_state[top_bun]["is_held"] > 0.5:
        #     print()
        #     import pdb; pdb.set_trace()

        return next_state

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
        x, y = self.get_position(self._robot, state)
        # ax.plot(robot_col + 0.5, robot_row + 0.5, 'rs', markersize=20)
        robot_direction = self.enum_to_dir[state.get(self._robot, "dir")]
        robot_img = mpimg.imread(
            utils.get_env_asset_path(f"imgs/robot_{robot_direction}.png"))
        img_size = (0.8, 0.8)
        ax.imshow(robot_img,
                  extent=[
                      x + (1 - img_size[0]) / 2, x + (1 + img_size[0]) / 2,
                      y + (1 - img_size[1]) / 2, y + (1 + img_size[1]) / 2
                  ])

        # Draw grill
        x, y = self.get_position(self._grill, state)
        grill_img = mpimg.imread(utils.get_env_asset_path("imgs/grill.png"))
        ax.imshow(grill_img, extent=[x, x + 1, y, y + 1])

        # Draw cutting board
        x, y = self.get_position(self._cutting_board, state)
        cutting_board_img = mpimg.imread(
            utils.get_env_asset_path("imgs/cutting_board.png"))
        ax.imshow(cutting_board_img, extent=[x, x + 1, y, y + 1])

        # Draw items
        type_to_img = {
            self._top_bun_type:
            mpimg.imread("predicators/envs/assets/imgs/top_bun.png"),
            self._bottom_bun_type:
            mpimg.imread("predicators/envs/assets/imgs/bottom_bun.png"),
            self._cheese_type:
            mpimg.imread("predicators/envs/assets/imgs/cheese.png"),
            self._tomato_type:
            mpimg.imread(utils.get_env_asset_path("imgs/whole_tomato.png")),
            self._patty_type:
            mpimg.imread(utils.get_env_asset_path("imgs/raw_patty.png"))
        }
        held_img_size = (0.3, 0.3)
        offset = held_img_size[1] * (1 / 3)
        items = [obj for obj in state if obj.is_instance(self._item_type)]
        for item in items:
            img = type_to_img[item.type]
            if "is_cooked" in state.simulator_state[
                    item] and self._IsCooked_holds(state, [item]):
                img = mpimg.imread(
                    utils.get_env_asset_path("imgs/cooked_patty.png"))
            elif "is_sliced" in state.simulator_state[
                    item] and self._IsSliced_holds(state, [item]):
                img = mpimg.imread(
                    utils.get_env_asset_path("imgs/sliced_tomato.png"))
            zorder = state.get(item, "z")
            is_held = state.simulator_state[item]["is_held"] > 0.5
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

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:
            del state  # unused
            logging.info(
                "Controls: arrow keys to move, wasd to change direction, " \
                "(e) to interact, (f) to pick/place"
            )
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
            action = Action(
                np.array([dcol, drow, turn, interact, pickplace],
                         dtype=np.float32))
            return action

        return _event_to_action
