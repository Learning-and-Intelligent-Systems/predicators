"""Sticky table simulated environment."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set

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

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

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

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        # TODO
        return state.copy()

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

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
        for table in tables:
            is_sticky = self._table_is_sticky(table, state)
            circ = self._object_to_geom(table, state)
            color = sticky_table_color if is_sticky else normal_table_color
            circ.plot(ax, facecolor=color, edgecolor="black", alpha=alpha)
            if is_sticky:
                x = state.get(table, "x")
                y = state.get(table, "y")
                radius = state.get(table, "radius")
                wedge = Wedge((x, y), radius, 0, 180, fill=False, hatch="OO")
                ax.add_artist(wedge)
        rect = self._object_to_geom(cube, state)
        rect.plot(ax, facecolor=cube_color, edgecolor="black", alpha=alpha)
        if caption is not None:
            plt.suptitle(caption, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
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
        rect = self._object_to_geom(cube, state)
        circ = self._object_to_geom(table, state)
        assert isinstance(rect, utils.Rectangle)
        for x, y in rect.vertices:
            if not circ.contains_point(x, y):
                return False
        return True

    def _OnFloor_holds(self, state: State, objects: Sequence[Object]) -> bool:
        cube, = objects
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
