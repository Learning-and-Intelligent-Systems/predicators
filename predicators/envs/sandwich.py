"""Sandwich making domain."""

import json
import logging
from pathlib import Path
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import RGBA, Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type
from predicators.utils import _Geom2D


class SandwichEnv(BaseEnv):
    """Sandwich making domain."""
    # Parameters that aren't important enough to need to clog up settings.py
    # The table x bounds are (1.1, 1.6), but the workspace is smaller.
    x_lb: ClassVar[float] = 1.2
    x_ub: ClassVar[float] = 1.5
    # The table y bounds are (0.3, 1.2), but the workspace is smaller.
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    holder_width: ClassVar[float] = (x_ub - x_lb) * 0.8
    holder_length: ClassVar[float] = (y_ub - y_lb) * 0.4
    holder_x_lb: ClassVar[float] = x_lb + holder_width / 2
    holder_x_ub: ClassVar[float] = x_ub - holder_width / 2
    holder_y_lb: ClassVar[float] = y_lb + holder_length / 2
    holder_y_ub: ClassVar[float] = holder_y_lb + (y_ub - y_lb) * 0.2
    holder_color: ClassVar[RGBA] = (0.5, 0.5, 0.5, 1.0)
    board_width: ClassVar[float] = (x_ub - x_lb) * 0.8
    board_length: ClassVar[float] = (y_ub - y_lb) * 0.2
    board_x_lb: ClassVar[float] = x_lb + board_width / 2
    board_x_ub: ClassVar[float] = x_ub - board_width / 2
    board_y_lb: ClassVar[
        float] = holder_y_lb + holder_length + board_length / 2
    board_y_ub: ClassVar[float] = board_y_lb + (y_ub - y_lb) * 0.2
    board_color: ClassVar[RGBA] = (0.1, 0.1, 0.5, 0.8)
    held_tol: ClassVar[float] = 0.5
    on_tol: ClassVar[float] = 0.01

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._ingredient_type = Type("ingredient", [
            "pose_x", "pose_y", "pose_z", "held", "color_r", "color_g",
            "color_b", "thickness", "radius", "shape"
        ])
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "fingers"])
        self._holder_type = Type("holder",
                                 ["pose_x", "pose_y", "length", "width"])
        self._board_type = Type("board",
                                ["pose_x", "pose_y", "length", "width"])
        # Predicates
        self._InHolder = Predicate("InHolder",
                                   [self._ingredient_type, self._holder_type],
                                   self._InHolder_holds)
        self._On = Predicate("On",
                             [self._ingredient_type, self._ingredient_type],
                             self._On_holds)
        self._OnBoard = Predicate("OnBoard",
                                  [self._ingredient_type, self._board_type],
                                  self._OnBoard_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._Holding = Predicate("Holding",
                                  [self._ingredient_type, self._robot_type],
                                  self._Holding_holds)
        self._Clear = Predicate("Clear", [self._ingredient_type],
                                self._Clear_holds)
        self._IsBread = Predicate("IsBread", [self._ingredient_type],
                                  self._IsBread_holds)
        self._IsBurger = Predicate("IsBurger", [self._ingredient_type],
                                   self._IsBurger_holds)
        self._IsHam = Predicate("IsHam", [self._ingredient_type],
                                self._IsHam_holds)
        self._IsEgg = Predicate("IsEgg", [self._ingredient_type],
                                self._IsEgg_holds)
        self._IsCheese = Predicate("IsCheese", [self._ingredient_type],
                                   self._IsCheese_holds)
        self._IsLettuce = Predicate("IsLettuce", [self._ingredient_type],
                                    self._IsLettuce_holds)
        self._IsTomato = Predicate("IsTomato", [self._ingredient_type],
                                   self._IsTomato_holds)
        self._IsGreenPepper = Predicate("IsGreenPepper",
                                        [self._ingredient_type],
                                        self._IsGreenPepper_holds)
        # Options
        self._Pick: ParameterizedOption = utils.SingletonParameterizedOption(
            # variables: [robot, object to pick]
            # params: []
            "Pick",
            self._Pick_policy,
            types=[self._robot_type, self._ingredient_type])
        self._Stack: ParameterizedOption = utils.SingletonParameterizedOption(
            # variables: [robot, object on which to stack currently-held-object]
            # params: []
            "Stack",
            self._Stack_policy,
            types=[self._robot_type, self._ingredient_type])
        self._PutOnBoard: ParameterizedOption = \
            utils.SingletonParameterizedOption(
            # variables: [robot]
            "PutOnBoard",
            self._PutOnBoard_policy,
            types=[self._robot_type])
        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._holder = Object("holder", self._holder_type)
        self._board = Object("board", self._board_type)
        self._num_ingredients_train = CFG.sandwich_ingredients_train
        self._num_ingredients_test = CFG.sandwich_ingredients_test

    @classmethod
    def get_name(cls) -> str:
        return "sandwich"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        # TODO
        return state.copy()

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               num_ingredients=self._num_ingredients_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               num_ingredients=self._num_ingredients_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnBoard, self._GripperOpen, self._Holding,
            self._Clear, self._InHolder, self._IsBread, self._IsBurger,
            self._IsHam, self._IsEgg, self._IsCheese, self._IsLettuce,
            self._IsTomato, self._IsGreenPepper
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnBoard, self._IsBread, self._IsBurger,
            self._IsHam, self._IsEgg, self._IsCheese, self._IsLettuce,
            self._IsTomato, self._IsGreenPepper
        }

    @property
    def types(self) -> Set[Type]:
        return {self._ingredient_type, self._robot_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Pick, self._Stack, self._PutOnBoard}

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers]
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        figscale = 10.0
        figsize = (figscale * (self.x_ub - self.x_lb),
                   figscale * (self.y_ub - self.y_lb))
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax = plt.gca()

        # Draw workspace.
        ws_width = (self.x_ub - self.x_lb)
        ws_length = (self.y_ub - self.y_lb)
        ws_x = self.x_lb
        ws_y = self.y_lb
        workspace_rect = utils.Rectangle(ws_x, ws_y, ws_width, ws_length, 0.0)
        workspace_rect.plot(ax, facecolor="white", edgecolor="black")

        # Draw objects.
        for obj in state:
            geom = self._obj_to_geom2d(obj, state)
            color = self._obj_to_color(obj, state)
            geom.plot(ax, facecolor=color, edgecolor="black")

        title = ""
        if caption is not None:
            title += f"; {caption}"
        plt.suptitle(title, fontsize=24, wrap=True)

        ax.set_xlim(self.x_lb, self.x_ub)
        ax.set_ylim(self.y_lb, self.y_ub)
        ax.axis("off")
        plt.tight_layout()
        return fig

    def _get_tasks(self, num_tasks: int, num_ingredients: Dict[str, List[int]],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for _ in range(num_tasks):
            ing_to_num: Dict[str, int] = {}
            for ing_name, possible_ing_nums in num_ingredients.items():
                num_ing = rng.choice(possible_ing_nums)
                ing_to_num[ing_name] = num_ing
            init_state = self._sample_initial_state(ing_to_num, rng)
            goal = self._sample_goal(ing_to_num, rng)
            # goal_holds = all(goal_atom.holds(init_state) for goal_atom in goal)
            # assert not goal_holds
            tasks.append(Task(init_state, goal))
        return tasks

    def _sample_initial_state(self, ingredient_to_num: Dict[str, int],
                              rng: np.random.Generator) -> State:
        # Sample holder state.
        holder_state = {
            "pose_x": rng.uniform(self.holder_x_lb, self.holder_x_ub),
            "pose_y": rng.uniform(self.holder_y_lb, self.holder_y_ub),
            "length": self.holder_length,
            "width": self.holder_width,
        }
        # Sampler board state.
        board_state = {
            "pose_x": rng.uniform(self.board_x_lb, self.board_x_ub),
            "pose_y": rng.uniform(self.board_y_lb, self.board_y_ub),
            "length": self.board_length,
            "width": self.board_width,
        }
        # Finalize state.
        state_dict = {self._holder: holder_state, self._board: board_state}
        return utils.create_state_from_dict(state_dict)

    def _sample_goal(self, ingredient_to_num: Dict[str, int],
                     rng: np.random.Generator) -> Set[GroundAtom]:
        # TODO
        return set()

    def _On_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        if state.get(obj1, "held") >= self.held_tol or \
           state.get(obj2, "held") >= self.held_tol:
            return False
        x1 = state.get(obj1, "pose_x")
        y1 = state.get(obj1, "pose_y")
        z1 = state.get(obj1, "pose_z")
        thick1 = state.get(obj1, "thickness")
        x2 = state.get(obj2, "pose_x")
        y2 = state.get(obj2, "pose_y")
        z2 = state.get(obj2, "pose_z")
        thick2 = state.get(obj2, "thickness")
        offset = thick1 / 2 + thick2 / 2
        return np.allclose([x1, y1, z1], [x2, y2, z2 + offset],
                           atol=self.on_tol)

    def _InHolder_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _OnBoard_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    @staticmethod
    def _GripperOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        rf = state.get(robot, "fingers")
        assert rf in (0.0, 1.0)
        return rf == 1.0

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _IsBread_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _IsBurger_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _IsHam_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _IsEgg_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _IsLettuce_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _IsTomato_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _IsCheese_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _IsGreenPepper_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        # TODO
        return False

    def _Pick_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        del memory, params  # unused
        # TODO
        arr = np.add(self.action_space.low, self.action_space.high) / 2.
        return Action(arr)

    def _Stack_policy(self, state: State, memory: Dict,
                      objects: Sequence[Object], params: Array) -> Action:
        del memory, params  # unused
        # TODO
        arr = np.add(self.action_space.low, self.action_space.high) / 2.
        return Action(arr)

    def _PutOnBoard_policy(self, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
        del state, memory, objects  # unused
        # TODO
        arr = np.add(self.action_space.low, self.action_space.high) / 2.
        return Action(arr)

    def _obj_to_geom2d(self, obj: Object, state: State) -> _Geom2D:
        rect_types = [self._holder_type, self._board_type]
        if any(obj.is_instance(t) for t in rect_types):
            width = state.get(obj, "width")
            length = state.get(obj, "length")
            x = state.get(obj, "pose_x") - width / 2.
            y = state.get(obj, "pose_y") - length / 2.
            return utils.Rectangle(x, y, width, length, 0.0)
        import ipdb
        ipdb.set_trace()

    def _obj_to_color(self, obj: Object, state: State) -> RGBA:
        if obj.is_instance(self._holder_type):
            return self.holder_color
        if obj.is_instance(self._board_type):
            return self.board_color
        import ipdb
        ipdb.set_trace()
