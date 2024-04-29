import dataclasses
import itertools
import logging
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple, cast
from copy import deepcopy

import numpy as np
import numpy.typing as npt
from experiments.envs.utils import BoxWH, construct_subcell_box, plot_geometry
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, Predicate, State, Type
import gym
from shapely import Geometry
from shapely.geometry import Polygon, Point, MultiPolygon, box
from shapely.affinity import translate

from matplotlib import patches, pyplot as plt
import matplotlib

from predicators.utils import abstract
matplotlib.use("tkagg")


@dataclasses.dataclass(eq=False)
class SimulatorState():
    polys: Dict[Object, Polygon] = dataclasses.field(default_factory=dict)
    desired_poses: Dict[Object, Tuple[float, float]
                        ] = dataclasses.field(default_factory=dict)
    held_block: Optional[Object] = None
    dx: float = 0.0
    dy: float = 0.0

    def copy(self) -> 'SimulatorState':
        return dataclasses.replace(self)


class WBox(BaseEnv):
    """WBox environment"""

    # Settings
    # Task generation settings
    num_tries: ClassVar[int] = 100000

    range_train_containers: ClassVar[Tuple[int, int]] = (2, 2)
    max_num_containers = max(*range_train_containers,
                             CFG.wbox_test_num_containers)
    object_placement_margin = 1.2

    # World shape settings
    world_range_x: ClassVar[Tuple[float, float]] = (0, 25)
    world_range_y: ClassVar[Tuple[float, float]] = (0, 25)
    visualization_margin: ClassVar[float] = 3.0
    robot_point = Point(0, 0)

    container_x_pos: ClassVar[float] = 0.0
    container_y_pos: ClassVar[float] = 0.0

    blocks_start_x: ClassVar[float] = 3.0
    blocks_start_y: ClassVar[float] = 0.0
    sub_cell_size: ClassVar[float] = 1.0
    sub_cell_margin: ClassVar[float] = 0.2

    # Predicate thresholds
    sub_block_present_thresh: ClassVar[float] = 0.5
    next_to_thresh: ClassVar[float] = 0.5
    holding_thresh: ClassVar[float] = 0.5

    # Types
    _robot_type = Type("robot", ["x", "y", "holding"])
    _object_type = Type("object", ["x", "y"])
    _container_type = Type("container", ["x", "y"], _object_type)
    _block_type = Type("block", ["x", "y", "held", "type"], _object_type)

    # Predicates
    # Inside predicate
    def _Inside_holds(state: State, objects: Sequence[Object]) -> bool:
        container, block = objects
        return WBox._get_held_block(state) != block and \
            WBox._get_shape(state, container).contains(
                WBox._get_shape(state, block))

    _Inside: ClassVar[Predicate] = Predicate(
        "Inside", [_container_type, _block_type], _Inside_holds)

    # Outside predicate
    def _Outside_holds(state: State, objects: Sequence[Object]) -> bool:
        return not WBox._Inside_holds(state, objects)
    _Outside: ClassVar[Predicate] = Predicate(
        "Outside", [_container_type, _block_type], _Outside_holds)

    # NextTo predicate
    def _NextTo_holds(state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        return WBox._get_shape(state, obj1).distance(WBox._get_shape(state, obj2)) <= WBox.next_to_thresh

    _NextToRobot: ClassVar[Predicate] = Predicate(
        "NextToRobot", [_robot_type, _object_type], _NextTo_holds)

    # Held predicate
    def _Held_holds(state: State, objects: Sequence[Object]) -> bool:
        _, block, = objects
        return WBox._get_held_block(state) == block

    _Held: ClassVar[Predicate] = Predicate(
        "Held", [_robot_type, _block_type], _Held_holds)

    # NotHeld predicate
    def _NotHeld_holds(state: State, objects: Sequence[Object]) -> bool:
        return WBox._get_held_block(state) is None

    _NotHeld: ClassVar[Predicate] = Predicate(
        "NotHeld", [_robot_type], _NotHeld_holds)

    # Common Objects
    _robot = Object("robot", _robot_type)

    # Common Geometries
    container_poly = construct_subcell_box(np.array([
        [False, False, True, True],
        [False, True, True, True],
        [True, True, True, True],
        [True, True, True, True],
    ]), sub_cell_size)

    d_block_poly = construct_subcell_box(np.array([
        [True, True],
        [True, True],
    ]), sub_cell_size).buffer(-sub_cell_margin, join_style='mitre')

    i_block_poly = construct_subcell_box(np.array([
        [True], [True], [True],
    ]), sub_cell_size).buffer(-sub_cell_margin, join_style='mitre')

    # Miscellaneous Data
    num_d_blocks_offsets = np.array([
        [[1, 0], [2, 0], [3, 0]],
        [[2, 0], [3, 0], [0, 0]],
        [[1, 0], [2, 0], [2, 2]],
        [[0, 0], [2, 0], [2, 2]]
    ]) * sub_cell_size

    @classmethod
    def get_name(cls) -> str:
        return "wbox"

    def simulate(self, state: State, action: Action) -> State:
        grab, move, place, _, _ = action.arr

        affinities = [
            (grab, self._transition_grab),
            (move, self._transition_move),
            (place, self._transition_place),
        ]
        _, transition_fn = max(affinities, key=lambda t: t[0])
        next_state = transition_fn(state, action)
        return next_state

    def _transition_grab(self, state: State, action: Action) -> State:
        logging.info("GRAB TRANSITION")
        _, _, _, x, y = action.arr
        finger = Point(x, y)
        next_state = state.copy()

        # Check if the robot is holding anything
        if self._get_held_block(state) is not None:
            logging.info("ALREADY HOLDING A BLOCK")
            return next_state

        # Check which block was selected
        selected_blocks = [block for block in state.get_objects(
            self._block_type) if self._get_shape(state, block).contains_properly(finger)]
        if not selected_blocks:
            logging.info("NO BLOCK SELECTED")
            return next_state
        selected_block, = selected_blocks

        # Check if the block is not too far away
        if not WBox._NextTo_holds(state, [self._robot, selected_block]):
            logging.info("ROBOT NOT NEXT TO THE BLOCK")
            return next_state

        # Changing the state
        center = self._get_shape(state, selected_block).boundary.centroid
        next_state.set(selected_block, "x", next_state.get(
            selected_block, "x") - center.x)
        next_state.set(selected_block, "y", next_state.get(
            selected_block, "y") - center.y)
        self._set_held_block(next_state, selected_block)
        return next_state

    def _transition_move(self, state: State, action: Action) -> State:
        logging.info("MOVE TRANSITION")
        _, _, _, x, y = action.arr
        next_state = state.copy()

        # Move the robot
        next_state.set(self._robot, "x", x)
        next_state.set(self._robot, "y", y)

        # Check if the robot is holding anything
        held_block = self._get_held_block(state)
        if held_block is not None:
            center = self._get_shape(state, held_block).boundary.centroid
            next_state.set(held_block, "x", x +
                           next_state.get(held_block, "x") - center.x)
            next_state.set(held_block, "y", y +
                           next_state.get(held_block, "y") - center.y)

        # Renormalize objects
        self._renormalize_state(next_state)

        return next_state

    def _transition_place(self, state: State, action: Action) -> State:
        logging.info("PLACE TRANSITION")
        _, _, _, x, y = action.arr
        next_state = state.copy()

        # Check if a block is held
        held_block = self._get_held_block(state)
        if held_block is None:
            logging.info("NO BLOCK HELD")
            return next_state

        # Check new placement
        new_block_shape = self._get_shape(
            state, held_block, x, y
        )
        container_shapes = [poly for c in state.get_objects(self._container_type) for poly in [
            self._get_shape(state, c)] if poly.intersects(new_block_shape)]
        if container_shapes and container_shapes[0].boundary.intersects(new_block_shape):
            logging.info("BLOCK INTERSECTS WITH A CONTAINER BOUNDARY")
            return next_state

        # Check collisions
        if any(self._get_shape(state, block).intersects(new_block_shape)
               for block in state.get_objects(self._block_type) if block != held_block):
            logging.info("BLOCK COLLIDES WITH OTHER BLOCKS")
            return next_state

        # Set the new state
        next_state.set(held_block, "x", x)
        next_state.set(held_block, "y", y)
        self._set_held_block(next_state, None)

        # Check if the new placement isn't too far away
        if not WBox._NextTo_holds(next_state, [self._robot, held_block]):
            logging.info("ROBOT NOT NEXT TO THE BLOCK")
            return state.copy()

        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        if not self._train_tasks:
            self._train_tasks = self._generate_tasks(
                rng=self._train_rng,
                num_tasks=CFG.num_train_tasks,
                range_blocks=self.range_train_containers,
            )
        return self._train_tasks

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if not self._test_tasks:
            self._test_tasks = self._generate_tasks(
                rng=self._test_rng,
                num_tasks=CFG.num_test_tasks,
                range_blocks=(CFG.wbox_test_num_containers,
                              CFG.wbox_test_num_containers),
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
        range_containers: Tuple[int, int],
    ) -> EnvironmentTask:
        # Constructing helper objects
        goal: Set[GroundAtom] = set()
        simulator_state = SimulatorState()
        state = State({}, simulator_state)

        # Constructing subtasks
        num_containers = rng.integers(*range_containers, endpoint=True)
        # so that the blocks cannot be sorted
        block_hashes = rng.permutation(
            num_containers * 3).reshape(num_containers, -1)
        allowed_area = box(
            self.world_range_x[0], self.world_range_y[0], self.world_range_x[1], self.world_range_y[1])
        for idx in range(num_containers):
            allowed_area = self._generate_subtask(
                rng, idx, block_hashes[idx], goal,
                simulator_state, state, allowed_area
            )

        # Inserting the robot
        simulator_state.polys[self._robot] = self.robot_point
        state.set(self._robot, "holding", 0.0)
        for _ in range(self.num_tries):
            state.set(self._robot, "x", rng.uniform(*self.world_range_x))
            state.set(self._robot, "y", rng.uniform(*self.world_range_y))
            if any(WBox._NextTo_holds(state, [self._robot, container]) for container in state.get_objects(self._container_type)):
                break
        else:
            raise ValueError(
                "Could not generate a task with the given settings")

        # Normalizing the state
        self._renormalize_state(state)
        return EnvironmentTask(state, goal)

    def _generate_subtask(
        self,
        rng: np.random.Generator,
        idx: int,
        block_hashes: npt.NDArray[np.int64],
        goal: Set[GroundAtom],
        simulator_state: SimulatorState,
        state: State,
        allowed_area: Geometry
    ) -> Geometry:
        # Finding the container position
        container = Object(f"{idx}_container", self._container_type)
        simulator_state.polys[container] = self.container_poly
        for _ in range(self.num_tries):
            state.set(container, "x", rng.uniform(*self.world_range_x))
            state.set(container, "y", rng.uniform(*self.world_range_y))
            if not allowed_area.contains(self._get_shape(state, container)):
                continue
            break
        else:
            raise ValueError(
                "Could not generate a task with the given settings")
        allowed_area = allowed_area.difference(self._get_shape(
            state, container).buffer(self.object_placement_margin))

        # Finding the block positions
        num_d_blocks = rng.integers(4)
        blocks_data = [
            (Object(f"{block_hashes[block_idx + num_d_blocks]}_{idx}_i_block_{block_idx}",
             self._block_type), self.i_block_poly, False)
            for block_idx in range(3 - num_d_blocks)
        ] + [
            (Object(f"{block_hashes[block_idx]}_{idx}_d_block_{block_idx}",
             self._block_type), self.d_block_poly, True)
            for block_idx in range(num_d_blocks)
        ]
        for block, poly, is_d_block in blocks_data:
            simulator_state.polys[block] = poly
            state.set(block, "held", 0.0)
            state.set(block, "type", is_d_block)
            for _ in range(self.num_tries):
                state.set(block, "x", rng.uniform(*self.world_range_x))
                state.set(block, "y", rng.uniform(*self.world_range_y))
                if not allowed_area.contains(self._get_shape(state, block)):
                    continue
                break
            else:
                raise ValueError(
                    "Could not generate a task with the given settings")
            allowed_area = allowed_area.difference(self._get_shape(
                state, block).buffer(self.object_placement_margin))

        # Inserting the goal
        goal.update(self._Inside([container, block])
                    for block, _, _ in blocks_data)

        # Inserting the desired poses
        blocks, _, is_d_block = zip(*sorted(blocks_data, key=lambda d: d[2]))
        for block, (dx, dy) in zip(
            blocks, self.num_d_blocks_offsets[sum(
                is_d_block)] + [state.get(container, "x"), state.get(container, "y")]
        ):
            simulator_state.desired_poses[block] = (dx, dy)

        return allowed_area

    @classmethod
    def _get_shape(
        cls, state: State, obj: Object, x: Optional[float] = None,
        y: Optional[float] = None
    ) -> Polygon:
        if x is None:
            x = state.get(obj, "x")
        if y is None:
            y = state.get(obj, "y")

        poly = cast(SimulatorState, state.simulator_state).polys[obj]
        return translate(poly, x, y)

    @classmethod
    def _get_desired_pos(cls, state: State, obj: Object) -> Tuple[float, float]:
        simulator_state = cast(SimulatorState, state.simulator_state)
        dx, dy = simulator_state.dx, simulator_state.dy
        desired_x, desired_y = simulator_state.desired_poses[obj]
        return desired_x + dx, desired_y + dy

    @classmethod
    def _renormalize_state(cls, state: State) -> None:
        dx, dy = -state.get(cls._robot, "x"), -state.get(cls._robot, "y")
        for obj in state:
            state.set(obj, "x", state.get(obj, "x") + dx)
            state.set(obj, "y", state.get(obj, "y") + dy)
        simulator_state = cast(SimulatorState, state.simulator_state)
        simulator_state.dx += dx
        simulator_state.dy += dy

    @classmethod
    def _set_held_block(cls, state: State, held_block: Optional[Object]) -> None:
        simulator_state = cast(SimulatorState, state.simulator_state)
        old_held_block = simulator_state.held_block
        simulator_state.held_block = held_block
        if old_held_block is not None:
            state.set(old_held_block, "held", 0.0)
        if held_block is not None:
            state.set(held_block, "held", 1.0)
            state.set(cls._robot, "holding", 1.0)
        else:
            state.set(cls._robot, "holding", 0.0)

    @classmethod
    def _get_held_block(cls, state: State) -> Optional[Object]:
        held_block = cast(SimulatorState, state.simulator_state).held_block
        assert (state.get(cls._robot, "holding") >=
                cls.holding_thresh) ^ (held_block is None)
        assert held_block is None or state.get(
            held_block, "held") >= cls.holding_thresh
        return held_block

    @classmethod
    def _get_unnormalized_coordinates(cls, state: State, obj: Object) -> Tuple[float, float]:
        x, y = state.get(obj, "x"), state.get(obj, "y")
        simulator_state = cast(SimulatorState, state.simulator_state)
        dx, dy = simulator_state.dx, simulator_state.dy
        return x - dx, y - dy

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._Inside, self._Outside, self._NextToRobot, self._Held, self._NotHeld}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._Inside}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._object_type, self._container_type, self._block_type}

    @property
    def action_space(self) -> gym.spaces.Box:
        """(grab_action, move_action, place_action, x, y)"""
        move_range_x = self.world_range_x[1] - self.world_range_x[0]
        move_range_y = self.world_range_y[1] - self.world_range_y[0]
        lower_bound = np.array(
            [0.0, 0.0, 0.0, -move_range_x, -move_range_y], dtype=np.float32)
        upper_bound = np.array(
            [1.0, 1.0, 1.0, move_range_x, move_range_y], dtype=np.float32)
        return gym.spaces.Box(lower_bound, upper_bound)

    @classmethod
    def _get_obj_patch(cls, state: State, obj: Object, **kwargs) -> patches.Patch:
        x, y = cls._get_unnormalized_coordinates(state, obj)
        return plot_geometry(cls._get_shape(state, obj, x, y), **kwargs)

    @classmethod
    def render_state_plt(
        cls,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None
    ) -> matplotlib.figure.Figure:  # type: ignore
        fig = plt.figure()
        ax = fig.add_subplot()
        if caption:
            fig.suptitle(caption)

        # Drawing the containers
        for container in state.get_objects(cls._container_type):
            ax.add_patch(cls._get_obj_patch(state, container,
                         color='pink', linestyle='--', fill=False))

        # Drawing the blocks
        held_block = cls._get_held_block(state)
        for block in state.get_objects(cls._block_type):
            if block == held_block:
                continue
            x, y = cls._get_unnormalized_coordinates(state, block)
            ax.add_patch(cls._get_obj_patch(
                state, block, facecolor='green', edgecolor='darkgreen'))

        # Drawing the held block
        if held_block is not None:
            ax.add_patch(cls._get_obj_patch(state, held_block,
                         edgecolor='darkgreen', linestyle=':', fill=False))

        # Drawing the robot
        ax.add_patch(cls._get_obj_patch(state, cls._robot, color='red'))

        ax.set_xlim(cls.world_range_x[0] - cls.visualization_margin,
                    cls.world_range_x[1] + cls.visualization_margin)
        ax.set_ylim(cls.world_range_y[0] - cls.visualization_margin,
                    cls.world_range_y[1] + cls.visualization_margin)
        return fig
