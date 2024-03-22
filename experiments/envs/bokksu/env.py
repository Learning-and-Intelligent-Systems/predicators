from dataclasses import dataclass
import logging
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from experiments.envs.utils import BoxWH
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, State, Task, Type
import gym
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.affinity import translate

from matplotlib import patches, pyplot as plt
import matplotlib

class Bokksu(BaseEnv):
    """Bokksu environment"""

    # Settings
    ## Task generation settings
    num_tries: ClassVar[int] = 100000

    range_train_blocks: ClassVar[Tuple[int, int]] = (3, 3)
    range_test_blocks: ClassVar[Tuple[int, int]] = (2, 2)

    ## World shape settings
    world_range_x: ClassVar[Tuple[float, float]] = (0, 20)
    world_range_y: ClassVar[Tuple[float, float]] = (0, 5)

    bokksu_x_pos: ClassVar[float] = 0.0
    bokksu_y_pos: ClassVar[float] = 0.0
    bokksu_width: ClassVar[int] = 2
    bokksu_height: ClassVar[int] = 2

    blocks_start_x: ClassVar[float] = 4.0
    blocks_start_y: ClassVar[float] = 0.0
    sub_cell_size: ClassVar[float] = 1.0
    sub_cell_margin: ClassVar[float] = 0.1

    ## Predicate thresholds
    orientation_vertical_thresh: ClassVar[float] = 0.5
    sub_block_present_thresh: ClassVar[float] = 0.5

    # Types
    _bokksu_type = Type("bokksu", ["x", "y", "width", "height"])
    _block_type = Type("block", ["x", "y", "orientation", "sub_block_0", "sub_block_1"])

    # Predicates
    def _Inside_holds(state: State, objects: Sequence[Object]) -> bool:
        bokksu, block = objects
        return Bokksu._get_shape(state, bokksu).contains(Bokksu._get_shape(state, block))

    _Inside: ClassVar[Predicate] = Predicate("Inside", [_bokksu_type, _block_type], _Inside_holds)
    _Outside: ClassVar[Predicate] = Predicate("Outside", [_bokksu_type, _block_type], lambda s, objs: not Bokksu._Inside_holds(s, objs))


    # Common Objects
    _bokksu = Object("bokksu", _bokksu_type)

    @classmethod
    def get_name(cls) -> str:
        return "bokksu"

    def simulate(self, state: State, action: Action) -> State:
        old_x, old_y, new_x, new_y, orientation = action.arr
        next_state = state.copy()

        # Check which block was selected
        finger = Point(old_x, old_y)
        selected_blocks = [block for block in state.get_objects(self._block_type) if self._get_shape(state, block).contains_properly(finger)]
        if not selected_blocks:
            logging.info("NO BLOCK SELECTED")
            return next_state
        selected_block, = selected_blocks

        # Check new placement
        new_block_shape = self._get_block_polygon(
            new_x, new_y, state.get(selected_block, "sub_block_0"),
            state.get(selected_block, "sub_block_1"), orientation
        )
        bokksu_shape = self._get_shape(state, self._bokksu)
        if bokksu_shape.intersects(new_block_shape) and not bokksu_shape.contains(new_block_shape):
            logging.info("BLOCK INTERSECTS WITH BOUNDARY")
            return next_state

        # Check collisions
        if any(self._get_shape(state, block).intersects(new_block_shape)
               for block in state.get_objects(self._block_type) if block != selected_block):
            logging.info("BLOCK COLLIDES WITH OTHER BLOCKS")
            return next_state

        # Move the block
        next_state.set(selected_block, "x", new_x)
        next_state.set(selected_block, "y", new_y)
        next_state.set(selected_block, "orientation", orientation)
        self._set_shape(next_state, selected_block, new_block_shape)

        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        if not self._train_tasks:
            self._train_tasks = self._generate_tasks(
                rng = self._train_rng,
                num_tasks = CFG.num_train_tasks,
                range_blocks = self.range_train_blocks,
            )
        return self._train_tasks

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if not self._test_tasks:
            self._test_tasks = self._generate_tasks(
                rng = self._test_rng,
                num_tasks = CFG.num_test_tasks,
                range_blocks = self.range_train_blocks,
            )
        return self._test_tasks

    def _generate_tasks(
        self,
        rng: np.random.Generator,
        num_tasks: int,
        range_blocks: Tuple[int, int]
    ) -> List[EnvironmentTask]:
        return [self._generate_task(rng, range_blocks) for _ in range(num_tasks)]

    def _generate_task(
        self,
        rng: np.random.Generator,
        range_blocks: Tuple[int, int],
    ) -> EnvironmentTask:
        num_blocks = rng.integers(*range_blocks, endpoint=True)

        # Generating the grid of blocks
        for _ in range(self.num_tries):
            edges = {
                (x, y, True) for x in range(self.bokksu_width - 1) for y in range(self.bokksu_height) # Vertical edges
            } | {
                (x, y, False) for x in range(self.bokksu_width) for y in range(self.bokksu_height - 1) # Horizontal edges
            }
            final_edges = set()

            for _ in range(self.bokksu_width * self.bokksu_height - num_blocks):
                if not edges:
                    break
                x, y, vert = rng.choice(list(edges))
                other_x, other_y = (x + 1, y) if vert else (x, y + 1)
                edges -= {
                    (x, y, True), (x, y, False), (x - 1, y, True), (x, y - 1, False),
                    (other_x, other_y, True), (other_x, other_y, False),
                    (other_x - 1, other_y, True), (other_x, other_y - 1, False),
                }
                final_edges.add((x, y, vert))
            else:
                break
        else:
            raise ValueError("Could not generate a task with the given settings")

        # Generating blocks
        blocks, block_positions = zip(*[
            (Object(f"block_{x}_{y}", self._block_type), (x, y))
            for x in range(self.bokksu_width) for y in range(self.bokksu_height)
            if not {(x - 1, y, True), (x, y - 1, False)} & final_edges
        ])

        # Generating goal
        goal = {self._Inside([self._bokksu, block]) for block in blocks}

        # Constructing placeholder state
        state: State = State({obj: np.zeros((obj.type.dim,), dtype=np.float32) for obj in blocks + (self._bokksu,)}, {})

        # Setting bokksu params
        state.set(self._bokksu, "x", self.bokksu_x_pos)
        state.set(self._bokksu, "y", self.bokksu_y_pos)
        state.set(self._bokksu, "width", self.bokksu_width * self.sub_cell_size)
        state.set(self._bokksu, "height", self.bokksu_height * self.sub_cell_size)
        state.simulator_state[self._bokksu] = (BoxWH(
            self.bokksu_x_pos, self.bokksu_y_pos,
            self.bokksu_width * self.sub_cell_size,
            self.bokksu_height * self.sub_cell_size
        ), (0, 0, False))

        # Setting block positions
        for idx, block, (x, y) in zip(range(len(blocks)), blocks, block_positions):
            block_x, block_y = self.blocks_start_x + self.sub_cell_size * idx + self.sub_cell_margin, self.blocks_start_y
            state.set(block, "x", block_x)
            state.set(block, "y", block_y)
            state.set(block, "orientation", 1.0)
            state.set(block, "sub_block_0", 1.0)
            if (x, y, True) in final_edges:
                state.set(block, "sub_block_1", 1.0)
                state.simulator_state[block] = (
                    self._get_block_polygon(block_x, block_y, 1.0, 1.0, 1.0),
                    (x, y, False)
                )
            elif (x, y, False) in final_edges:
                state.set(block, "sub_block_1", 1.0)
                state.simulator_state[block] = (
                    self._get_block_polygon(block_x, block_y, 1.0, 1.0, 1.0),
                    (x, y, True)
                )
            else:
                state.set(block, "sub_block_1", 0.0)
                state.simulator_state[block] = (
                    self._get_block_polygon(block_x, block_y, 1.0, 1.0, 1.0),
                    (x, y, True)
                )

        return EnvironmentTask(state, goal)

    @classmethod
    def _get_block_polygon(cls, x: float, y: float, sub_block_0: float, sub_block_1: float, orientation: float) -> Polygon:
        return BoxWH(*cls._get_block_shape(x, y, sub_block_0, sub_block_1, orientation))

    @classmethod
    def _get_block_shape(cls, x: float, y: float, sub_block_0: float, sub_block_1: float, orientation: float) -> Tuple[float, float, float, float]:
        sub_block_0 = sub_block_0 > Bokksu.sub_block_present_thresh
        sub_block_1 = sub_block_1 > Bokksu.sub_block_present_thresh
        vertical = orientation > Bokksu.orientation_vertical_thresh

        shape = MultiPolygon()
        x_min = y_min = np.inf
        x_max = y_max = -np.inf
        sub_blocks_present = False
        sub_block_size = cls.sub_cell_size - cls.sub_cell_margin * 2
        if sub_block_0:
            sub_blocks_present = True
            x_min, x_max = min(x_min, x), max(x_max, x + sub_block_size)
            y_min, y_max = min(y_min, y), max(y_max, y + sub_block_size)
        if sub_block_1:
            if vertical:
                y += cls.sub_cell_size
            else:
                x += cls.sub_cell_size
            sub_blocks_present = True
            x_min, x_max = min(x_min, x), max(x_max, x + sub_block_size)
            y_min, y_max = min(y_min, y), max(y_max, y + sub_block_size)
        if sub_blocks_present:
            return x_min, y_min, x_max - x_min, y_max - y_min
        else:
            return (x, y, 0, 0)

    @classmethod
    def _get_shape(cls, state: State, obj: Object) -> Polygon:
        poly, _ = cast(Dict[Object, Tuple[Polygon, Tuple[int, int, bool]]], state.simulator_state)[obj]
        return poly

    @classmethod
    def _set_shape(cls, state: State, obj: Object, poly: Polygon) -> None:
        helper_dict = cast(Dict[Object, Tuple[Polygon, Tuple[int, int, bool]]], state.simulator_state)
        _, desired_pos = helper_dict[obj]
        helper_dict[obj] = (poly, desired_pos)

    @classmethod
    def _get_desired_pos(cls, state: State, obj: Object) -> Tuple[int, int, bool]:
        _, desired_pos = cast(Dict[Object, Tuple[Polygon, Tuple[int, int, bool]]], state.simulator_state)[obj]
        return desired_pos

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Inside, self._Outside}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Inside}

    @property
    def types(self) -> Set[Type]:
        return {self._bokksu_type, self._block_type}

    @property
    def action_space(self) -> gym.spaces.Box:
        """(current_block_x, current_block_y, new_block_x, new_block_y, new_block_orientation)"""
        lower_bound = np.array([self.world_range_x[0], self.world_range_y[0]] * 2 + [0], dtype=np.float32)
        upper_bound = np.array([self.world_range_x[1], self.world_range_y[1]] * 2 + [1], dtype=np.float32)
        return gym.spaces.Box(lower_bound, upper_bound)

    def render_state_plt(
        self,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None
    ) -> matplotlib.figure.Figure:
        fig = plt.figure()
        ax = fig.add_subplot()
        fig.suptitle(caption)

        # Drawing the box
        ax.add_patch(patches.Rectangle(
            (state.get(self._bokksu, "x"), state.get(self._bokksu, "y")),
            state.get(self._bokksu, "width"), state.get(self._bokksu, "height"),
            color = 'pink'
        ))

        # Drawing the blocks
        for block in state.get_objects(self._block_type):
            sub_block_size = self.sub_cell_size - self.sub_cell_margin * 2
            if state.get(block, "sub_block_1") > self.sub_block_present_thresh:
                if state.get(block, "orientation") > self.orientation_vertical_thresh:
                    width = sub_block_size
                    height = sub_block_size * 2
                else:
                    width = sub_block_size * 2
                    height = sub_block_size
            else:
                width = sub_block_size
                height = sub_block_size
            ax.add_patch(patches.Rectangle(
                (state.get(self._bokksu, "x"), state.get(self._bokksu, "y")),
                width, height, color = 'black'
            ))
        return fig