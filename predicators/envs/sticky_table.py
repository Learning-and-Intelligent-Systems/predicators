"""Sticky table simulated environment."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib.patches import Wedge

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class StickyTableEnv(BaseEnv):
    """An environment where a cube must be transported between tables.

    Most of the tables are flat, but one is half is smooth and half is sticky.
    When placing on the smooth side, the cube usually falls off; when placing
    on the sticky side, it usually stays. When it falls off, it falls onto the
    floor. It can be picked up from the floor.

    Note that unlike almost all of our other environments, there is real
    stochasticity in the outcomes of placing.

    The action space is 2D. When the robot is holding nothing, the only action
    that changes anything is clicking on the cube. When the robot is holding
    the cube, the action places the cube at that location.
    """
    x_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 1.0
    y_lb: ClassVar[float] = 0.0
    y_ub: ClassVar[float] = 1.0
    cube_scale: ClassVar[float] = 0.25  # as a function of table radius
    sticky_surface_mode: ClassVar[str] = "half"  # half or whole

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # For noisy simulation.
        self._noise_rng = np.random.default_rng(CFG.seed)

        # Types
        self._cube_type = Type("cube", ["x", "y", "size", "held"])
        self._table_type = Type("table", ["x", "y", "radius", "sticky"])

        # Predicates
        self._OnTable = Predicate("OnTable",
                                  [self._cube_type, self._table_type],
                                  self._OnTable_holds)
        self._OnFloor = Predicate("OnFloor", [self._cube_type],
                                  self._OnFloor_holds)
        self._Holding = Predicate("Holding", [self._cube_type],
                                  self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [], self._HandEmpty_holds)

    @classmethod
    def get_name(cls) -> str:
        return "sticky_table"

    @property
    def _pick_success_prob(self) -> float:
        return CFG.sticky_table_pick_success_prob

    @property
    def _place_sticky_fall_prob(self) -> float:
        return CFG.sticky_table_place_sticky_fall_prob

    @property
    def _place_smooth_fall_prob(self) -> float:
        return CFG.sticky_table_place_smooth_fall_prob

    def simulate(self, state: State, action: Action) -> State:
        # NOTE: noise is added here. Two calls to simulate with the same
        # inputs may produce different outputs!
        assert self.action_space.contains(action.arr)
        act_x, act_y = action.arr
        next_state = state.copy()
        hand_empty = self._HandEmpty_holds(state, [])
        cube, = state.get_objects(self._cube_type)
        # Picking logic.
        if hand_empty:
            # Fail sometimes.
            if self._noise_rng.uniform() < self._pick_success_prob:
                if self._action_grasps_object(act_x, act_y, cube, state):
                    next_state.set(cube, "held", 1.0)
        # Placing logic.
        else:
            next_state.set(cube, "held", 0.0)
            # Find the table for placing, if any.
            table: Optional[Object] = None
            cube_size = state.get(cube, "size")
            rect = utils.Rectangle(act_x, act_y, cube_size, cube_size, 0.0)
            for target in state.get_objects(self._table_type):
                circ = self._object_to_geom(target, state)
                if self._rectangle_inside_geom(rect, circ):
                    table = target
                    break
            if table is None:
                # Put on the floor here.
                next_state.set(cube, "x", act_x)
                next_state.set(cube, "y", act_y)
            else:
                # Possibly put on the table, or have it fall somewhere near.
                fall_prob = self._place_sticky_fall_prob
                if self._table_is_sticky(table, state):
                    # Check if placing on the smooth side of the sticky table.
                    table_y = state.get(table, "y")
                    if self.sticky_surface_mode == "half" and act_y < table_y:
                        fall_prob = self._place_smooth_fall_prob
                if self._noise_rng.uniform() < fall_prob:
                    fall_x, fall_y = self._sample_floor_point_around_table(
                        table, state, self._noise_rng)
                    next_state.set(cube, "x", fall_x)
                    next_state.set(cube, "y", fall_y)
                    assert self._OnFloor_holds(next_state, [cube])
                else:
                    next_state.set(cube, "x", act_x)
                    next_state.set(cube, "y", act_y)
        return next_state

    def _action_grasps_object(self, act_x: float, act_y: float, cube: Object,
                              state: State) -> bool:
        rect = self._object_to_geom(cube, state)
        return rect.contains_point(act_x, act_y)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               rng=self._test_rng,
                               sticky_table_only=True)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._OnTable, self._OnFloor, self._Holding, self._HandEmpty}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._OnTable}

    @property
    def types(self) -> Set[Type]:
        return {self._cube_type, self._table_type}

    @property
    def action_space(self) -> Box:
        return Box(np.array([self.x_lb, self.y_lb], dtype=np.float32),
                   np.array([self.x_ub, self.y_ub], dtype=np.float32))

    def _object_to_geom(self, obj: Object, state: State) -> utils._Geom2D:
        if obj.is_instance(self._cube_type):
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            size = state.get(obj, "size")
            return utils.Rectangle(x, y, size, size, 0.0)
        assert obj.is_instance(self._table_type)
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        radius = state.get(obj, "radius")
        return utils.Circle(x, y, radius)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        cube_color = "red"
        normal_table_color = "blue"
        sticky_table_color = "yellow"
        alpha = 0.75
        cube, = state.get_objects(self._cube_type)
        tables = state.get_objects(self._table_type)
        surface_mode = self.sticky_surface_mode
        for table in tables:
            is_sticky = self._table_is_sticky(table, state)
            circ = self._object_to_geom(table, state)
            color = sticky_table_color if is_sticky else normal_table_color
            hatch = "OO" if is_sticky and surface_mode == "whole" else None
            circ.plot(ax,
                      facecolor=color,
                      edgecolor="black",
                      alpha=alpha,
                      hatch=hatch)
            if is_sticky and surface_mode == "half":
                x = state.get(table, "x")
                y = state.get(table, "y")
                radius = state.get(table, "radius")
                wedge = Wedge((x, y), radius, 0, 180, fill=False, hatch="OO")
                ax.add_artist(wedge)
        cube_is_held = self._Holding_holds(state, [cube])
        edge_color = "white" if cube_is_held else "black"
        rect = self._object_to_geom(cube, state)
        rect.plot(ax, facecolor=cube_color, edgecolor=edge_color, alpha=alpha)
        if caption is not None:
            plt.suptitle(caption, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self,
                   num: int,
                   rng: np.random.Generator,
                   sticky_table_only: bool = False) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            # The goal is to move the cube to some table.
            # The table positions are static
            # The initial location of the cube and the goal are randomized.
            num_tables = CFG.sticky_table_num_tables
            assert num_tables >= 2
            state_dict: Dict[Object, Dict[str, float]] = {}
            # Generate the tables in a ring around the center of the room.
            origin_x = (self.x_ub - self.x_lb) / 2
            origin_y = (self.y_ub - self.y_lb) / 2
            d = min(self.x_ub - self.x_lb, self.y_ub - self.y_lb) / 3
            thetas = np.linspace(0, 2 * np.pi, num=num_tables, endpoint=False)
            # Select the radius to prevent any overlap. Exact would be
            # d * sin(theta / 2). Divide by 2 to be conservative.
            angle_diff = thetas[1] - thetas[0]
            radius = d * np.sin(angle_diff / 2) / 2
            for i, theta in enumerate(thetas):
                x = d * np.cos(theta) + origin_x
                y = d * np.sin(theta) + origin_y
                if i > 0:
                    prefix = "normal"
                    sticky = 0.0
                else:
                    prefix = "sticky"
                    sticky = 1.0
                obj = Object(f"{prefix}-table-{i}", self._table_type)
                state_dict[obj] = {
                    "x": x,
                    "y": y,
                    "radius": radius,
                    "sticky": sticky
                }
            tables = sorted(state_dict)
            rng.shuffle(tables)  # type: ignore
            if sticky_table_only:
                stickies = [t for t in tables if state_dict[t]["sticky"] > 0.5]
                target_table = stickies[0]
                remaining_tables = [t for t in tables if t != target_table]
                init_table = remaining_tables[0]
            else:
                init_table, target_table = tables[:2]
            # Create cube.
            size = radius * self.cube_scale
            table_x = state_dict[init_table]["x"]
            table_y = state_dict[init_table]["y"]
            while True:
                theta = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, radius)
                x = table_x + dist * np.cos(theta)
                y = table_y + dist * np.sin(theta)
                cube = Object("cube", self._cube_type)
                state_dict[cube] = {
                    "x": x,
                    "y": y,
                    "size": size,
                    "held": 0.0,
                }
                state = utils.create_state_from_dict(state_dict)
                if self._OnTable_holds(state, [cube, init_table]):
                    break
            goal = {GroundAtom(self._OnTable, [cube, target_table])}
            task = EnvironmentTask(state, goal)
            tasks.append(task)
        return tasks

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        cube, table = objects
        if self._Holding_holds(state, [cube]):
            return False
        rect = self._object_to_geom(cube, state)
        circ = self._object_to_geom(table, state)
        assert isinstance(rect, utils.Rectangle)
        return self._rectangle_inside_geom(rect, circ)

    @staticmethod
    def _rectangle_inside_geom(rect: utils.Rectangle,
                               geom: utils._Geom2D) -> bool:
        for x, y in rect.vertices:
            if not geom.contains_point(x, y):
                return False
        return True

    def _OnFloor_holds(self, state: State, objects: Sequence[Object]) -> bool:
        cube, = objects
        if self._Holding_holds(state, [cube]):
            return False
        for table in state.get_objects(self._table_type):
            if self._OnTable_holds(state, [cube, table]):
                return False
        return True

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        cube, = objects
        return state.get(cube, "held") > 0.5

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        assert not objects
        cube, = state.get_objects(self._cube_type)
        return not self._Holding_holds(state, [cube])

    def _table_is_sticky(self, table: Object, state: State) -> bool:
        return state.get(table, "sticky") > 0.5

    def _sample_floor_point_around_table(
            self, table: Object, state: State,
            rng: np.random.Generator) -> Tuple[float, float]:
        x = state.get(table, "x")
        y = state.get(table, "y")
        radius = state.get(table, "radius")
        dist = radius + rng.uniform(radius / 10, radius / 4)
        theta = rng.uniform(0, 2 * np.pi)
        return (x + dist * np.cos(theta), y + dist * np.sin(theta))


class StickyTableTrickyFloorEnv(StickyTableEnv):
    """Variation where picking from the floor is the only thing that can be
    improved through sampler learning.

    Placing on the table is still noisy, but inherently so.
    """

    sticky_surface_mode = "whole"  # the 'sticky' table is sticky everywhere

    @property
    def _place_sticky_fall_prob(self) -> float:
        return CFG.sticky_table_tricky_floor_place_sticky_fall_prob

    @classmethod
    def get_name(cls) -> str:
        return "sticky_table_tricky_floor"

    def _action_grasps_object(self, act_x: float, act_y: float, cube: Object,
                              state: State) -> bool:
        if not super()._action_grasps_object(act_x, act_y, cube, state):
            return False
        # If the cube is on the floor, make it harder to grasp.
        if not self._OnFloor_holds(state, [cube]):
            return True
        # Specifically, only succeed if grasp is in upper-right quadrant with
        # respect to the cube's center.
        size = state.get(cube, "size")
        cube_x = state.get(cube, "x") + size / 2
        cube_y = state.get(cube, "y") + size / 2
        return act_x > cube_x and act_y > cube_y
