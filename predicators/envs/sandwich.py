"""Sandwich making domain."""

from typing import ClassVar, Collection, Dict, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import RGBA, Action, EnvironmentTask, GroundAtom, \
    Object, Predicate, State, Type
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
    z_ub: ClassVar[float] = z_lb + 0.25  # for rendering only
    pick_z: ClassVar[float] = 0.7
    robot_init_x: ClassVar[float] = (x_lb + x_ub) / 2
    robot_init_y: ClassVar[float] = (y_lb + y_ub) / 2
    robot_init_z: ClassVar[float] = pick_z
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
        "patty": (0.32, 0.15, 0.0),
        "ham": (0.937, 0.384, 0.576),
        "egg": (0.937, 0.898, 0.384),
        "cheese": (0.937, 0.737, 0.203),
        "lettuce": (0.203, 0.937, 0.431),
        "tomato": (0.917, 0.180, 0.043),
        "green_pepper": (0.156, 0.541, 0.160),
    }
    ingredient_radii: ClassVar[Dict[str, float]] = {
        "bread": board_length / 2.5,
        "patty": board_length / 3,
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
        "patty": 1,
        "ham": 0,
        "egg": 1,
        "cheese": 0,
        "lettuce": 1,
        "tomato": 1,
        "green_pepper": 1,
    }
    held_tol: ClassVar[float] = 0.5
    on_tol: ClassVar[float] = 0.01
    pick_tol: ClassVar[float] = 0.0001

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
        self._BoardClear = Predicate("BoardClear", [self._board_type],
                                     self._BoardClear_holds)
        self._IsBread = Predicate("IsBread", [self._ingredient_type],
                                  self._IsBread_holds)
        self._IsPatty = Predicate("IsPatty", [self._ingredient_type],
                                  self._IsPatty_holds)
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
        x, y, z, fingers = action.arr
        # Infer which transition function to follow.
        # If we are not holding anything, the only possible action is pick.
        if fingers < self.held_tol:
            return self._transition_pick(state, x, y, z)
        # If the action is targetting a position that is on the surface of
        # the board, then the only possible action is putonboard.
        thickness = self.ingredient_thickness + self.board_thickness
        if z < self.table_height + thickness:
            return self._transition_putonboard(state, x, y)
        # The only remaining possible action is stack.
        return self._transition_stack(state, x, y, z)

    def _transition_pick(self, state: State, x: float, y: float,
                         z: float) -> State:
        next_state = state.copy()
        # Can only pick if fingers are open
        if not self._GripperOpen_holds(state, [self._robot]):
            return next_state
        ing = self._get_ingredient_at_xyz(state, x, y, z)
        if ing is None:  # no ingredient at this pose
            return next_state
        # Can only pick if ingredient is in the holder
        if not self._InHolder_holds(state, [ing, self._holder]):
            return next_state
        # Execute pick
        next_state.set(ing, "pose_x", x)
        next_state.set(ing, "pose_y", y)
        next_state.set(ing, "pose_z", self.pick_z)
        next_state.set(ing, "held", 1.0)
        if "clear" in self._board_type.feature_names:
            next_state.set(ing, "clear", 0.0)
        next_state.set(self._robot, "fingers", 0.0)  # close fingers
        return next_state

    def _transition_putonboard(self, state: State, x: float,
                               y: float) -> State:
        next_state = state.copy()
        # Can only putonboard if fingers are closed.
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        ing = self._get_held_object(state)
        assert ing is not None
        # Check that nothing is on the board.
        if not self._BoardClear_holds(state, [self._board]):
            return next_state
        # Execute putonboard. Rotate object.
        thickness = self.ingredient_thickness / 2 + self.board_thickness
        pose_z = self.table_height + thickness
        next_state.set(ing, "pose_x", x)
        next_state.set(ing, "pose_y", y)
        next_state.set(ing, "pose_z", pose_z)
        next_state.set(ing, "rot", 0.0)
        next_state.set(ing, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers
        if "clear" in self._board_type.feature_names:
            next_state.set(ing, "clear", 1.0)
            next_state.set(self._board, "clear", 0.0)
        return next_state

    def _transition_stack(self, state: State, x: float, y: float,
                          z: float) -> State:
        next_state = state.copy()
        # Can only stack if fingers are closed
        if self._GripperOpen_holds(state, [self._robot]):
            return next_state
        # Check that both objects exist
        ing = self._get_held_object(state)
        assert ing is not None
        other_ing = self._get_highest_object_below(state, x, y, z)
        if other_ing is None:  # no object to stack onto
            return next_state
        # Can't stack onto yourself!
        if ing == other_ing:
            return next_state
        # Need object we're stacking onto to be clear
        if not self._object_is_clear(state, other_ing):
            return next_state
        # Execute stack by snapping into place
        cur_x = state.get(other_ing, "pose_x")
        cur_y = state.get(other_ing, "pose_y")
        cur_z = state.get(other_ing, "pose_z")
        next_state.set(ing, "pose_x", cur_x)
        next_state.set(ing, "pose_y", cur_y)
        next_state.set(ing, "pose_z", cur_z + self.ingredient_thickness)
        next_state.set(ing, "rot", 0.0)
        next_state.set(ing, "held", 0.0)
        next_state.set(self._robot, "fingers", 1.0)  # open fingers
        if "clear" in self._board_type.feature_names:
            next_state.set(other_ing, "clear", 0.0)
            next_state.set(ing, "clear", 1.0)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               num_ingredients=self._num_ingredients_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               num_ingredients=self._num_ingredients_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnBoard, self._GripperOpen, self._Holding,
            self._Clear, self._InHolder, self._BoardClear, self._IsBread,
            self._IsPatty, self._IsHam, self._IsEgg, self._IsCheese,
            self._IsLettuce, self._IsTomato, self._IsGreenPepper
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._On, self._OnBoard, self._IsBread, self._IsPatty, self._IsHam,
            self._IsEgg, self._IsCheese, self._IsLettuce, self._IsTomato,
            self._IsGreenPepper
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._ingredient_type, self._robot_type, self._board_type,
            self._holder_type
        }

    @property
    def action_space(self) -> Box:
        # dimensions: [x, y, z, fingers]
        lowers = np.array([self.x_lb, self.y_lb, 0.0, 0.0], dtype=np.float32)
        uppers = np.array([self.x_ub, self.y_ub, 10.0, 1.0], dtype=np.float32)
        return Box(lowers, uppers)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
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
            if obj.is_instance(self._robot_type):
                continue
            if self._Holding_holds(state, [obj, self._robot]):
                continue
            geom = self._obj_to_geom2d(obj, state, "topdown")
            color = self._obj_to_color(obj, state)
            geom.plot(xy_ax, facecolor=color, edgecolor="black")
            # Add text box.
            if obj.is_instance(self._ingredient_type) and \
                self._InHolder_holds(state, [obj, self._holder]):
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
            if obj.is_instance(self._robot_type):
                continue
            if self._Holding_holds(state, [obj, self._robot]):
                continue
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
                   rng: np.random.Generator) -> List[EnvironmentTask]:
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
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    def _sample_initial_state(self, ingredient_to_num: Dict[str, int],
                              rng: np.random.Generator) -> State:
        # Create robot initial state.
        # Note: the robot poses are not used in this environment (they are
        # constant), but they change and get used in the PyBullet subclass.
        robot_state = {
            "pose_x": self.robot_init_x,
            "pose_y": self.robot_init_y,
            "pose_z": self.robot_init_z,
            "fingers": 1.0  # fingers start out open
        }
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
        # Sample board state.
        board_state = {
            "pose_x": rng.uniform(self.board_x_lb, self.board_x_ub),
            "pose_y": rng.uniform(self.board_y_lb, self.board_y_ub),
            "width": self.board_width,
            "length": self.board_length,
            "thickness": self.board_thickness,
        }
        if "clear" in self._board_type.feature_names:
            board_state["clear"] = 1.0
        state_dict = {
            self._holder: holder_state,
            self._board: board_state,
            self._robot: robot_state
        }
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
                if "clear" in self._board_type.feature_names:
                    # All ingredients start not clear as they are not on the
                    # board
                    state_dict[obj]["clear"] = 0.0
        return utils.create_state_from_dict(state_dict)

    def _sample_goal(self, state: State,
                     rng: np.random.Generator) -> Set[GroundAtom]:
        # Some possible sandwiches. Bottom to top.
        sandwiches = [
            ["bread", "cheese", "patty", "bread"],
            ["bread", "cheese", "patty", "egg", "bread"],
            ["bread", "cheese", "patty", "lettuce", "bread"],
            ["bread", "patty", "lettuce", "tomato", "bread"],
            ["bread", "cheese", "patty", "lettuce", "tomato", "bread"],
            [
                "bread", "cheese", "patty", "lettuce", "tomato",
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

    def _get_language_goal_prompt_prefix(self,
                                         object_names: Collection[str]) -> str:
        # pylint:disable=line-too-long
        available_predicates = ", ".join(
            [p.name for p in sorted(self.goal_predicates)])
        available_objects = ", ".join(sorted(object_names))
        # We could extract the object names, but this is simpler.
        assert {"bread0", "bread1", "patty0", "cheese0",
                "lettuce0"}.issubset(object_names)
        prompt = f"""# The available predicates are: {available_predicates}
# The available objects are: {available_objects}
# Use the available predicates and objects to convert natural language goals into JSON goals.
        
# I want a sandwich with a patty, cheese, and lettuce.
{{"OnBoard": [["bread0", "board"]], "On": [["bread1", "lettuce0"], ["lettuce0", "cheese0"], ["cheese0", "patty0"], ["patty0", "bread0"]]}}
"""
        return prompt

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
        obj, holder = objects
        ds = ["x", "y"]
        sizes = [self.holder_width, self.holder_length]
        return self._object_contained_in_object(obj, holder, state, ds, sizes)

    def _OnBoard_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, board = objects
        ds = ["x", "y"]
        sizes = [self.board_width, self.board_length]
        return self._object_contained_in_object(obj, board, state, ds, sizes)

    @staticmethod
    def _GripperOpen_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        rf = state.get(robot, "fingers")
        assert rf in (0.0, 1.0)
        return rf == 1.0

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        held_obj = self._get_held_object(state)
        if held_obj is None:
            return False
        return held_obj == obj

    def _Clear_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._object_is_clear(state, obj)

    def _BoardClear_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        board, = objects
        ings = state.get_objects(self._ingredient_type)
        return not any(self._OnBoard_holds(state, [o, board]) for o in ings)

    def _IsBread_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "bread", state)

    def _IsPatty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._is_ingredient(obj, "patty", state)

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
                length = width
                # Oriented facing up, i.e., in the holder.
                if abs(state.get(obj, "rot") - np.pi / 2.) < 1e-3:
                    x = state.get(obj, "pose_x") - width / 2.
                    y = state.get(obj, "pose_y") - thickness / 2.
                    return utils.Rectangle(x, y, width, thickness, 0.0)
                # Oriented facing down, i.e., on the board.
                assert abs(state.get(obj, "rot") - 0.0) < 1e-3
                x = state.get(obj, "pose_x") - width / 2.
                y = state.get(obj, "pose_y") - length / 2.
                return utils.Rectangle(x, y, width, length, 0.0)
            # Cylinder.
            assert abs(state.get(obj, "shape") - 1.0) < 1e-3
            radius = state.get(obj, "radius")
            width = 2 * radius
            thickness = state.get(obj, "thickness")
            # Oriented facing up, i.e., in the holder.
            if abs(state.get(obj, "rot") - np.pi / 2.) < 1e-3:
                x = state.get(obj, "pose_x") - width / 2.
                y = state.get(obj, "pose_y") - thickness / 2.
                return utils.Rectangle(x, y, width, thickness, 0.0)
            # Oriented facing down, i.e., on the board.
            assert abs(state.get(obj, "rot") - 0.0) < 1e-3
            x = state.get(obj, "pose_x")
            y = state.get(obj, "pose_y")
            return utils.Circle(x, y, radius)
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
            width = length
            # Oriented facing up, i.e., in the holder.
            if abs(state.get(obj, "rot") - np.pi / 2.) < 1e-3:
                y = state.get(obj, "pose_y") - thickness / 2.
                z = state.get(obj, "pose_z") - length / 2.
                return utils.Rectangle(y, z, thickness, length, 0.0)
            # Oriented facing down, i.e., on the board.
            assert abs(state.get(obj, "rot") - 0.0) < 1e-3
            y = state.get(obj, "pose_y") - length / 2.
            z = state.get(obj, "pose_z") - thickness / 2.
            return utils.Rectangle(y, z, length, thickness, 0.0)
        # Cylinder.
        assert abs(state.get(obj, "shape") - 1.0) < 1e-3
        thickness = state.get(obj, "thickness")
        radius = state.get(obj, "radius")
        length = 2 * radius
        # Oriented facing up, i.e., in the holder.
        if abs(state.get(obj, "rot") - np.pi / 2.) < 1e-3:
            y = state.get(obj, "pose_y") - thickness / 2.
            z = state.get(obj, "pose_z") - length / 2.
            return utils.Rectangle(y, z, thickness, length, 0.0)
        # Oriented facing down, i.e., on the board.
        assert abs(state.get(obj, "rot") - 0.0) < 1e-3
        y = state.get(obj, "pose_y") - length / 2.
        z = state.get(obj, "pose_z") - thickness / 2.
        return utils.Rectangle(y, z, length, thickness, 0.0)

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
        # We use the color of the object to determine what type of ingredient
        # it is. This is a simple version of the realistic version where the
        # type of ingredient is a function of its visual properties (color,
        # texture, detailed shape, etc). Setting it up this way means that
        # we could try to learn the predicates, i.e., do color classification.
        # Also, we don't want to use subtypes of the ingredient type because
        # our learning algorithms don't properly support hierarchical typing.
        # Finally, we wouldn't want to use the object names to determine their
        # ingredient type because all predicates must be a function of the
        # object states, and the object names are not part of the states.
        obj_color = (state.get(obj, "color_r"), state.get(obj, "color_g"),
                     state.get(obj, "color_b"))
        affinities = {
            n: np.sum(np.subtract(c, obj_color)**2)
            for n, c in self.ingredient_colors.items()
        }
        closest = min(affinities, key=lambda o: affinities[o])
        return closest

    def _state_to_ingredient_groups(self,
                                    state: State) -> Dict[str, Set[Object]]:
        ings = set(self.ingredient_colors)
        ing_groups: Dict[str, Set[Object]] = {n: set() for n in ings}
        for obj in state.get_objects(self._ingredient_type):
            ing_name = self._obj_to_ingredient(obj, state)
            ing_groups[ing_name].add(obj)
        return dict(ing_groups)

    def _get_ingredient_at_xyz(self, state: State, x: float, y: float,
                               z: float) -> Optional[Object]:
        close_ingredients = []
        for ing in state.get_objects(self._ingredient_type):
            ing_pose = np.array([
                state.get(ing, "pose_x"),
                state.get(ing, "pose_y"),
                state.get(ing, "pose_z")
            ])
            if np.allclose([x, y, z], ing_pose, atol=self.pick_tol):
                dist = np.linalg.norm(np.array([x, y, z]) - ing_pose)
                close_ingredients.append((ing, float(dist)))
        if not close_ingredients:
            return None
        return min(close_ingredients, key=lambda x: x[1])[0]  # min distance

    def _get_held_object(self, state: State) -> Optional[Object]:
        for obj in state.get_objects(self._ingredient_type):
            if state.get(obj, "held") >= self.held_tol:
                return obj
        return None

    def _get_highest_object_below(self, state: State, x: float, y: float,
                                  z: float) -> Optional[Object]:
        objs_here = []
        for obj in state.get_objects(self._ingredient_type):
            pose = np.array(
                [state.get(obj, "pose_x"),
                 state.get(obj, "pose_y")])
            obj_z = state.get(obj, "pose_z")
            if np.allclose([x, y], pose, atol=self.pick_tol) and \
               obj_z < z - self.pick_tol:
                objs_here.append((obj, obj_z))
        if not objs_here:
            return None
        return max(objs_here, key=lambda x: x[1])[0]  # highest z

    def _object_is_clear(self, state: State, obj: Object) -> bool:
        if self._Holding_holds(state, [obj, self._robot]):
            return False
        if not self._OnBoard_holds(state, [obj, self._board]):
            return False
        for other_obj in state.get_objects(self._ingredient_type):
            if self._On_holds(state, [other_obj, obj]):
                return False
        return True

    def _object_contained_in_object(self, obj: Object, container: Object,
                                    state: State, dims: List[str],
                                    sizes: List[float]) -> bool:
        assert len(dims) == len(sizes)
        for dim, size in zip(dims, sizes):
            obj_pose = state.get(obj, f"pose_{dim}")
            container_pose = state.get(container, f"pose_{dim}")
            container_lb = container_pose - size / 2.
            container_ub = container_pose + size / 2.
            if not container_lb - 1e-5 <= obj_pose <= container_ub + 1e-5:
                return False
        return True


class SandwichEnvClear(SandwichEnv):
    """Sandwich making domain where (1) each ingredient has a feature
    indicating whether it is clear, and (2) the board has a feature indicating
    whether it is clear.

    This allows us to learn all the predicates with the assumption that
    the predicates are a function of only their argument's states.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Add attribute.
        self._ingredient_type = Type("ingredient", [
            "pose_x", "pose_y", "pose_z", "rot", "held", "color_r", "color_g",
            "color_b", "thickness", "radius", "shape", "clear"
        ])
        self._board_type = Type(
            "board",
            ["pose_x", "pose_y", "length", "width", "thickness", "clear"])
        # Override predicates with board and ingredient arguments.
        self._InHolder = Predicate("InHolder",
                                   [self._ingredient_type, self._holder_type],
                                   self._InHolder_holds)
        self._On = Predicate("On",
                             [self._ingredient_type, self._ingredient_type],
                             self._On_holds)
        self._OnBoard = Predicate("OnBoard",
                                  [self._ingredient_type, self._board_type],
                                  self._OnBoard_holds)
        self._Holding = Predicate("Holding",
                                  [self._ingredient_type, self._robot_type],
                                  self._Holding_holds)
        self._Clear = Predicate("Clear", [self._ingredient_type],
                                self._Clear_holds)
        self._BoardClear = Predicate("BoardClear", [self._board_type],
                                     self._BoardClear_holds)
        self._IsBread = Predicate("IsBread", [self._ingredient_type],
                                  self._IsBread_holds)
        self._IsPatty = Predicate("IsPatty", [self._ingredient_type],
                                  self._IsPatty_holds)
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
        # Override board object.
        self._board = Object("board", self._board_type)

    @classmethod
    def get_name(cls) -> str:
        return "sandwich_clear"

    def _BoardClear_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        board, = objects
        return state.get(board, "clear") == 1

    def _object_is_clear(self, state: State, obj: Object) -> bool:
        return state.get(obj, "clear") == 1
