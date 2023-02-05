"""Sandwich making domain."""

import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import ClassVar, DefaultDict, Dict, List, Optional, Sequence, \
    Set, Tuple

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
    table_height: ClassVar[float] = 0.2
    # The table x bounds are (1.1, 1.6), but the workspace is smaller.
    x_lb: ClassVar[float] = 1.2
    x_ub: ClassVar[float] = 1.5
    # The table y bounds are (0.3, 1.2), but the workspace is smaller.
    y_lb: ClassVar[float] = 0.4
    y_ub: ClassVar[float] = 1.1
    z_lb: ClassVar[float] = table_height - 0.02
    z_ub: ClassVar[float] = z_lb + 0.25
    holder_width: ClassVar[float] = (x_ub - x_lb) * 0.8
    holder_length: ClassVar[float] = (y_ub - y_lb) * 0.6
    holder_x_lb: ClassVar[float] = x_lb + holder_width / 2
    holder_x_ub: ClassVar[float] = x_ub - holder_width / 2
    holder_y_lb: ClassVar[float] = y_lb + holder_length / 2
    holder_y_ub: ClassVar[float] = holder_y_lb + (y_ub - y_lb) * 0.1
    holder_color: ClassVar[RGBA] = (0.5, 0.5, 0.5, 1.0)
    holder_thickness: ClassVar[float] = 0.01
    board_width: ClassVar[float] = (x_ub - x_lb) * 0.8
    board_length: ClassVar[float] = (y_ub - y_lb) * 0.2
    board_x_lb: ClassVar[float] = x_lb + board_width / 2
    board_x_ub: ClassVar[float] = x_ub - board_width / 2
    board_y_lb: ClassVar[
        float] = holder_y_lb + holder_length / 2 + board_length
    board_y_ub: ClassVar[float] = board_y_lb + (y_ub - y_lb) * 0.05
    board_color: ClassVar[RGBA] = (0.1, 0.1, 0.5, 0.8)
    board_thickness: ClassVar[float] = 0.01
    ingredient_thickness: ClassVar[float] = 0.02
    ingredient_colors: ClassVar[Dict[str, Tuple[float, float, float]]] = {
        "bread": (0.58, 0.29, 0.0),
        "burger": (0.32, 0.15, 0.0),
        "ham": (0.937, 0.384, 0.576),
        "egg": (0.937, 0.898, 0.384),
        "cheese": (0.937, 0.737, 0.203),
        "lettuce": (0.203, 0.937, 0.431),
        "tomato": (0.917, 0.180, 0.043),
        "green_pepper": (0.156, 0.541, 0.160),
    }
    ingredient_radii: ClassVar[Dict[str, float]] = {
        "bread": board_length / 2.5,
        "burger": board_length / 3,
        "ham": board_length / 2.75,
        "egg": board_length / 3.25,
        "cheese": board_length / 2.75,
        "lettuce": board_length / 3,
        "tomato": board_length / 3,
        "green_pepper": board_length / 3.5,
    }
    # 0 is cuboid, 1 is cylinder
    ingredient_shapes: ClassVar[Dict[str, float]] = {
        "bread": 0,
        "burger": 1,
        "ham": 0,
        "egg": 1,
        "cheese": 0,
        "lettuce": 1,
        "tomato": 1,
        "green_pepper": 1,
    }
    held_tol: ClassVar[float] = 0.5
    on_tol: ClassVar[float] = 0.01

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._ingredient_type = Type("ingredient", [
            "pose_x", "pose_y", "pose_z", "rot", "held", "color_r", "color_g",
            "color_b", "thickness", "radius", "shape"
        ])
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "pose_z", "fingers"])
        self._holder_type = Type(
            "holder", ["pose_x", "pose_y", "length", "width", "thickness"])
        self._board_type = Type(
            "board", ["pose_x", "pose_y", "length", "width", "thickness"])
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

        # Set up whole figure.
        figscale = 10.0
        x_len = (self.x_ub - self.x_lb)
        y_len = (self.y_ub - self.y_lb)
        z_len = (self.z_ub - self.z_lb)
        figsize = (figscale * (x_len + y_len), figscale * (y_len + z_len))

        width_ratio = x_len / y_len
        fig, (yz_ax, xy_ax) = plt.subplots(
            1,
            2,
            figsize=figsize,
            gridspec_kw={'width_ratios': [1, width_ratio]})

        # Draw on xy axis.

        # Draw workspace.
        ws_width = (self.x_ub - self.x_lb)
        ws_length = (self.y_ub - self.y_lb)
        ws_x = self.x_lb
        ws_y = self.y_lb
        workspace_rect = utils.Rectangle(ws_x, ws_y, ws_width, ws_length, 0.0)
        workspace_rect.plot(xy_ax, facecolor="white", edgecolor="black")

        # Draw robot base (roughly) for reference.
        rob_base_x_pad = 0.75 * ws_width
        rob_base_x = ws_x + ws_width / 2 + rob_base_x_pad
        rob_base_y = ws_y + ws_length / 2
        rob_base_radius = min(ws_width, ws_length) * 0.2
        robot_circ = utils.Circle(rob_base_x, rob_base_y, rob_base_radius)
        robot_circ.plot(xy_ax, facecolor="gray", edgecolor="black")
        xy_ax.text(rob_base_x,
                   rob_base_y,
                   "robot",
                   fontsize="x-small",
                   ha="center",
                   va="center",
                   bbox=dict(facecolor="white", edgecolor="black", alpha=0.5))

        # Draw objects, sorted in z order.
        def _get_z(obj: Object) -> float:
            if obj.is_instance(self._holder_type) or \
               obj.is_instance(self._board_type):
                return -float("inf")
            return state.get(obj, "pose_z")

        for obj in sorted(state, key=_get_z):
            geom = self._obj_to_geom2d(obj, state, "topdown")
            color = self._obj_to_color(obj, state)
            geom.plot(xy_ax, facecolor=color, edgecolor="black")
            # Add text box.
            if obj.is_instance(self._ingredient_type):
                x = state.get(obj, "pose_x")
                y = state.get(obj, "pose_y")
                s = obj.name
                xy_ax.text(x,
                           y,
                           s,
                           fontsize="x-small",
                           ha="center",
                           va="center",
                           bbox=dict(facecolor="white",
                                     edgecolor="black",
                                     alpha=0.5))
        xy_ax.set_xlim(self.x_lb, rob_base_x + rob_base_radius)
        xy_ax.set_ylim(self.y_lb, self.y_ub)
        xy_ax.set_aspect("equal", adjustable="box")
        xy_ax.axis("off")

        # Draw on yz axis.

        # Draw workspace.
        ws_length = (self.y_ub - self.y_lb)
        ws_height = self.ingredient_thickness  # just some small value
        ws_y = self.y_lb
        ws_z = self.table_height - ws_height
        workspace_rect = utils.Rectangle(ws_y, ws_z, ws_length, ws_height, 0.0)
        workspace_rect.plot(yz_ax, facecolor="white", edgecolor="black")

        # Draw objects. Order doesn't matter because there should be no
        # overlaps in these dimensions.
        for obj in state:
            geom = self._obj_to_geom2d(obj, state, "side")
            color = self._obj_to_color(obj, state)
            geom.plot(yz_ax, facecolor=color, edgecolor="black")

        yz_ax.set_xlim(self.y_lb - self.ingredient_thickness, self.y_ub)
        yz_ax.set_ylim(self.z_lb, self.z_ub)
        yz_ax.set_aspect("equal", adjustable="box")
        yz_ax.axis("off")

        # Finalize figure.
        title = ""
        if caption is not None:
            title += f"; {caption}"
        plt.suptitle(title, fontsize=24, wrap=True)
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
            goal = self._sample_goal(init_state, rng)
            goal_holds = all(goal_atom.holds(init_state) for goal_atom in goal)
            assert not goal_holds
            tasks.append(Task(init_state, goal))
        return tasks

    def _sample_initial_state(self, ingredient_to_num: Dict[str, int],
                              rng: np.random.Generator) -> State:
        # Sample holder state.
        holder_x = rng.uniform(self.holder_x_lb, self.holder_x_ub)
        holder_y = rng.uniform(self.holder_y_lb, self.holder_y_ub)
        holder_state = {
            "pose_x": holder_x,
            "pose_y": holder_y,
            "width": self.holder_width,
            "length": self.holder_length,
            "thickness": self.holder_thickness,
        }
        # Sampler board state.
        board_state = {
            "pose_x": rng.uniform(self.board_x_lb, self.board_x_ub),
            "pose_y": rng.uniform(self.board_y_lb, self.board_y_ub),
            "width": self.board_width,
            "length": self.board_length,
            "thickness": self.board_thickness,
        }
        state_dict = {self._holder: holder_state, self._board: board_state}
        # Sample ingredients.
        ing_spacing = self.ingredient_thickness
        # Add padding.
        tot_num_ings = sum(ingredient_to_num.values())
        tot_thickness = tot_num_ings * self.ingredient_thickness
        ing_spacing += (self.holder_length - tot_thickness) / (tot_num_ings -
                                                               1)
        # Randomly order the ingredients.
        order_indices = list(range(tot_num_ings))
        for ing, num in ingredient_to_num.items():
            ing_static_features = self._ingredient_to_static_features(ing)
            radius = ing_static_features["radius"]
            for ing_i in range(num):
                obj_name = f"{ing}{ing_i}"
                obj = Object(obj_name, self._ingredient_type)
                order_idx = rng.choice(order_indices)
                order_indices.remove(order_idx)
                pose_y = (holder_y - self.holder_length / 2) + \
                         order_idx * ing_spacing + \
                         self.ingredient_thickness / 2.
                pose_z = self.table_height + self.holder_thickness + radius
                state_dict[obj] = {
                    "pose_x": holder_x,
                    "pose_y": pose_y,
                    "pose_z": pose_z,
                    "rot": np.pi / 2.,
                    "held": 0.0,
                    **ing_static_features
                }
        return utils.create_state_from_dict(state_dict)

    def _sample_goal(self, state: State,
                     rng: np.random.Generator) -> Set[GroundAtom]:
        # Some possible sandwiches. Bottom to top.
        sandwiches = [
            ["bread", "cheese", "burger", "bread"],
            ["bread", "cheese", "burger", "egg", "bread"],
            ["bread", "cheese", "burger", "lettuce", "bread"],
            ["bread", "burger", "lettuce", "tomato", "bread"],
            ["bread", "cheese", "burger", "lettuce", "tomato", "bread"],
            [
                "bread", "cheese", "burger", "lettuce", "tomato",
                "green_pepper", "bread"
            ],
            ["bread", "cheese", "ham", "bread"],
            ["bread", "cheese", "ham", "tomato", "bread"],
            ["bread", "cheese", "egg", "bread"],
            ["bread", "cheese", "egg", "tomato", "bread"],
            ["bread", "cheese", "egg", "tomato", "green_pepper", "bread"],
        ]
        # For now, assume all sandwiches are feasible.
        ing_to_objs = self._state_to_ingredient_groups(state)
        # Randomize order.
        ing_to_remaining_objs = {
            n: sorted(objs, key=lambda _: rng.uniform())
            for n, objs in ing_to_objs.items()
        }
        sandwich = sandwiches[rng.choice(len(sandwiches))]
        sandwich_objs = []
        for ing in sandwich:
            obj = ing_to_remaining_objs[ing].pop()
            sandwich_objs.append(obj)
        # Create goal atoms.
        goal_atoms: Set[GroundAtom] = set()
        bottom = sandwich_objs[0]
        on_board_atom = GroundAtom(self._OnBoard, [bottom, self._board])
        goal_atoms.add(on_board_atom)
        for top, bot in zip(sandwich_objs[1:], sandwich_objs[:-1]):
            on_atom = GroundAtom(self._On, [top, bot])
            goal_atoms.add(on_atom)
        return goal_atoms

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
        obj, = objects
        return self._is_ingredient(obj, "bread", state)

    def _IsBurger_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "burger", state)

    def _IsHam_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "ham", state)

    def _IsEgg_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "egg", state)

    def _IsLettuce_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "lettuce", state)

    def _IsTomato_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "tomato", state)

    def _IsCheese_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "cheese", state)

    def _IsGreenPepper_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "green_pepper", state)

    def _Pick_policy(self, state: State, memory: Dict,
                     objects: Sequence[Object], params: Array) -> Action:
        del memory, params  # unused
        import ipdb
        ipdb.set_trace()
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

    def _obj_to_geom2d(self, obj: Object, state: State, view: str) -> _Geom2D:
        if view == "topdown":
            rect_types = [self._holder_type, self._board_type]
            if any(obj.is_instance(t) for t in rect_types):
                width = state.get(obj, "width")
                length = state.get(obj, "length")
                x = state.get(obj, "pose_x") - width / 2.
                y = state.get(obj, "pose_y") - length / 2.
                return utils.Rectangle(x, y, width, length, 0.0)
            assert obj.is_instance(self._ingredient_type)
            # Cuboid.
            if abs(state.get(obj, "shape") - 0.0) < 1e-3:
                width = 2 * state.get(obj, "radius")
                thickness = state.get(obj, "thickness")
                # Oriented facing up, i.e., in the holder.
                if abs(state.get(obj, "rot") - np.pi / 2.) < 1e-3:
                    x = state.get(obj, "pose_x") - width / 2.
                    y = state.get(obj, "pose_y") - thickness / 2.
                    return utils.Rectangle(x, y, width, thickness, 0.0)
                # Oriented facing down, i.e., on the board.
                assert abs(state.get(obj, "rot") - 0.0) < 1e-3
                import ipdb
                ipdb.set_trace()
            # Cylinder.
            assert abs(state.get(obj, "shape") - 1.0) < 1e-3
            width = 2 * state.get(obj, "radius")
            thickness = state.get(obj, "thickness")
            # Oriented facing up, i.e., in the holder.
            if abs(state.get(obj, "rot") - np.pi / 2.) < 1e-3:
                x = state.get(obj, "pose_x") - width / 2.
                y = state.get(obj, "pose_y") - thickness / 2.
                return utils.Rectangle(x, y, width, thickness, 0.0)
            # Oriented facing down, i.e., on the board.
            assert abs(state.get(obj, "rot") - 0.0) < 1e-3
            import ipdb
            ipdb.set_trace()
        assert view == "side"
        rect_types = [self._holder_type, self._board_type]
        if any(obj.is_instance(t) for t in rect_types):
            length = state.get(obj, "length")
            thickness = state.get(obj, "thickness")
            y = state.get(obj, "pose_y") - length / 2.
            z = self.table_height
            return utils.Rectangle(y, z, length, thickness, 0.0)
        assert obj.is_instance(self._ingredient_type)
        # Cuboid.
        if abs(state.get(obj, "shape") - 0.0) < 1e-3:
            thickness = state.get(obj, "thickness")
            length = 2 * state.get(obj, "radius")
            # Oriented facing up, i.e., in the holder.
            if abs(state.get(obj, "rot") - np.pi / 2.) < 1e-3:
                y = state.get(obj, "pose_y") - thickness / 2.
                z = state.get(obj, "pose_z") - length / 2.
                return utils.Rectangle(y, z, thickness, length, 0.0)
            # Oriented facing down, i.e., on the board.
            assert abs(state.get(obj, "rot") - 0.0) < 1e-3
            import ipdb
            ipdb.set_trace()
        # Cylinder.
        assert abs(state.get(obj, "shape") - 1.0) < 1e-3
        thickness = state.get(obj, "thickness")
        length = 2 * state.get(obj, "radius")
        # Oriented facing up, i.e., in the holder.
        if abs(state.get(obj, "rot") - np.pi / 2.) < 1e-3:
            y = state.get(obj, "pose_y") - thickness / 2.
            z = state.get(obj, "pose_z") - length / 2.
            return utils.Rectangle(y, z, thickness, length, 0.0)
        # Oriented facing down, i.e., on the board.
        assert abs(state.get(obj, "rot") - 0.0) < 1e-3
        import ipdb
        ipdb.set_trace()

    def _obj_to_color(self, obj: Object, state: State) -> RGBA:
        if obj.is_instance(self._holder_type):
            return self.holder_color
        if obj.is_instance(self._board_type):
            return self.board_color
        assert obj.is_instance(self._ingredient_type)
        r = state.get(obj, "color_r")
        g = state.get(obj, "color_g")
        b = state.get(obj, "color_b")
        a = 1.0
        return (r, g, b, a)

    def _ingredient_to_static_features(self,
                                       ing_name: str) -> Dict[str, float]:
        color_r, color_g, color_b = self.ingredient_colors[ing_name]
        radius = self.ingredient_radii[ing_name]
        shape = self.ingredient_shapes[ing_name]
        return {
            "color_r": color_r,
            "color_g": color_g,
            "color_b": color_b,
            "thickness": self.ingredient_thickness,
            "radius": radius,
            "shape": shape
        }

    def _is_ingredient(self, obj: Object, ing_name: str, state: State) -> bool:
        return self._obj_to_ingredient(obj, state) == ing_name

    def _obj_to_ingredient(self, obj: Object, state: State) -> str:
        obj_color = (state.get(obj, "color_r"), state.get(obj, "color_g"),
                     state.get(obj, "color_b"))
        affinities = {
            n: np.sum(np.subtract(c, obj_color)**2)
            for n, c in self.ingredient_colors.items()
        }
        closest = min(affinities, key=affinities.get)
        return closest

    def _state_to_ingredient_groups(self,
                                    state: State) -> Dict[str, Set[Object]]:
        ings = set(self.ingredient_colors)
        ing_groups: Dict[str, Set[Object]] = {n: set() for n in ings}
        for obj in state.get_objects(self._ingredient_type):
            ing_name = self._obj_to_ingredient(obj, state)
            ing_groups[ing_name].add(obj)
        return dict(ing_groups)
