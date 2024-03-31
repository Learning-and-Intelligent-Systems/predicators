from dataclasses import dataclass, field
import itertools
import gym
from typing import Callable, ClassVar, List, Optional, Sequence, Set, Tuple, Dict, Union, cast

from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, Predicate, State, Type

from experiments.envs.utils import BoxWH

import numpy as np
import numpy.typing as npt

from shapely import point_on_surface
from shapely.geometry import box as Box, Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.affinity import translate

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import matplotlib

__all__ = ['Shelves2DEnv']

class Shelves2DEnv(BaseEnv):
    '''This environment consists of several subtasks. Each subtask consists of boxes, shelves to put those boxes in,
    a dummy bundle object representing all the shelves as well as a cover the width of the row of the shelves. Actions
    work by specifying a point that we are gripping and how much to move the thing we grip. The goal of each subtask is
    to put all the boxes in the right shelves and cover the shelves from the bottom or from the top.

    Important points to note:
        - Only boxes and covers are movable
        - Shelves have 0-thickness walls on their left and right
        - Shelves are arranged in a row and they all have the same heigh
        - We cannot have intersections between objects
        - The cover must be within some distance from the shelves and it must be aligned with them sideways
        - A cover cannot be moved before moving all the boxes from the subtask

    Important geometrical constraints:
        - A box is only in a shelf if it has some intersection with the shelf
        - Boxes are always taller than their respective shelves
        - Boxes are always thinner than their respective shelves


    Object types:
        - box(x, y, width, height) -> movable
        - shelf(x, y, width, height)
        - bundle(x, y, width, height)
        - cover(x, y, width, height) -> movable

    Predicates:
        - In(box, shelf)
        - Bundles(bundle, shelf)
        - CoversTop(cover, bundle)
        - CoversBottom(cover, bundle)

    '''

    # Simulator settings
    cover_max_distance: ClassVar[float] = 1#0.2
    cover_sideways_tolerance: ClassVar[float] = 1#0.2

    range_world_x: ClassVar[Tuple[(float, float)]] = (-60, 60)
    range_world_y: ClassVar[Tuple[(float, float)]] = (-60, 60)

    num_tries: ClassVar[int] = 100000

    # Task generation settings
    range_subtasks_train: ClassVar[Tuple[(int, int)]] = (1, 1)#(5, 6)
    range_subtasks_test: ClassVar[Tuple[(int, int)]] = (1, 1)#(10, 12)

    range_shelves_train: ClassVar[Tuple[(int, int)]] = (2, 5)#(2, 3)
    # range_shelves_test: ClassVar[Tuple[(int, int)]] = (10, 10)#(12, 15)

    range_box_width: ClassVar[Tuple[(float, float)]] = (1, 5)
    range_box_height: ClassVar[Tuple[(float, float)]] = (7, 15)

    range_shelf_width: ClassVar[Tuple[(float, float)]] = (2, 6)
    range_shelf_height: ClassVar[Tuple[(float, float)]] = (3.5, 12)
    range_shelf_sep: ClassVar[Tuple[(float, float)]] = (0.4, 0.7)

    min_box_width_clearance: ClassVar[float] = 0.4
    min_box_height_clearance: ClassVar[float] = 3

    cover_thickness: ClassVar[float] = 1
    bundle_margin: ClassVar[float] = 0.1

    # Types
    _box_type = Type('box', [
        'pose_x', 'pose_y',
        'size_x', 'size_y'
    ])
    _shelf_type = Type('shelf', [
        'pose_x', 'pose_y',
        'size_x', 'size_y'
    ])
    _bundle_type = Type('bundle', [
        'pose_x', 'pose_y',
        'size_x', 'size_y'
    ])
    _cover_type = Type('cover',[
        'pose_x', 'pose_y',
        'size_x', 'size_y',
        'start_x', 'start_y'
    ])

    def __init__(self, use_gui: bool):
        super().__init__(use_gui)

        self._GoesInto = Predicate('GoesInto', [
            self._box_type,
            self._shelf_type], self._GoesInto_holds)
        self._In = Predicate('In', [
            self._box_type,
            self._shelf_type], self._In_holds)
        self._CoverAtStart = Predicate('CoverAtStart', [
            self._cover_type], self._CoverAtStart_holds)
        self._Bundles = Predicate('Bundles', [
            self._bundle_type,
            self._shelf_type], self._Bundles_holds)
        self._CoverFor = Predicate('CoverFor', [
            self._cover_type,
            self._bundle_type], self._CoverFor_holds)
        self._CoversTop = Predicate('CoversTop', [
            self._cover_type,
            self._bundle_type], self._CoversTop_holds)
        self._CoversBottom = Predicate('CoversBottom', [
            self._cover_type,
            self._bundle_type], self._CoversBottom_holds)

        self._train_tasks = None
        self._test_tasks = None

    @classmethod
    def get_name(cls):
        return 'shelves2d'

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        if self._train_tasks is None:
            self._train_tasks = self._get_tasks(
                num_tasks=CFG.num_train_tasks,
                range_subtasks=self.range_subtasks_train,
                range_shelves=self.range_shelves_train,
                rng=self._train_rng
            )
        # self._generate_tasks()
        return self._train_tasks

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if self._test_tasks is None:
            self._test_tasks = self._get_tasks(
                num_tasks=CFG.num_test_tasks,
                range_subtasks=self.range_subtasks_test,
                range_shelves=(CFG.shelves2d_test_num_boxes, CFG.shelves2d_test_num_boxes),
                rng=self._test_rng
            )
        # self._generate_tasks()
        return self._test_tasks

    def _generate_tasks(self):
        if self._tasks is None:
            real_tasks = self._get_tasks(
                num_tasks=CFG.num_test_tasks,
                range_subtasks=self.range_subtasks_train,
                range_shelves=self.range_shelves_train,
                rng=self._train_rng
            )
            self._tasks = [real_task for _, real_task in zip(range(CFG.num_train_tasks), itertools.cycle(real_tasks))]


    @property
    def predicates(self) -> Set[Predicate]:
        '''Get the set of predicates that are given with this environment.'''
        return {self._GoesInto, self._In, self._CoverAtStart, self._Bundles,
                self._CoverFor, self._CoversTop, self._CoversBottom}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        '''Get the subset of self.predicates that are used in goals.'''
        return {self._In, self._CoversTop, self._CoversBottom}

    @property
    def types(self):
        '''Get the set of types that are given with this environment.'''
        return {self._box_type, self._shelf_type, self._bundle_type, self._cover_type}

    @classmethod
    def action_space_bounds(cls) -> gym.spaces.Box:
        """The action space is as follows:
         0, 1 - gripper coords (will grab blocks that are within those coords and the cover if it's within a margin)
         2, 3 - object movement delta (will move the gripped objects by the given amount)
        """
        lower = np.array([
            cls.range_world_x[0], cls.range_world_y[0],
            cls.range_world_x[0] - cls.range_world_x[1],
            cls.range_world_y[0] - cls.range_world_y[1]
        ], dtype=np.float32)
        upper = np.array([
            cls.range_world_x[1], cls.range_world_y[1],
            cls.range_world_x[1] - cls.range_world_x[0],
            cls.range_world_y[1] - cls.range_world_y[0]
        ], dtype=np.float32)
        return gym.spaces.Box(lower, upper)

    @property
    def action_space(self) -> gym.spaces.Box:
        return self.action_space_bounds()

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)

        aux_data = cast(Shelves2DState, state.simulator_state)
        next_state = state.copy()

        # Extract objects
        movables = state.get_objects(self._box_type) + state.get_objects(self._cover_type)
        shelves = state.get_objects(self._shelf_type)

        # Check which object is grasped
        gripper_pose = Point(*self._get_action_gripper_pose(action))
        movable_polys = [(movable, self._get_obj_polygon(state, movable)) for movable in movables]
        gripped_obj_polys = [(movable, movable_poly) for movable, movable_poly in movable_polys if movable_poly.contains(gripper_pose)]

        if not gripped_obj_polys:
            return next_state
        (gripped_obj, gripped_poly), = gripped_obj_polys

        # Move the grasped object
        if not gripped_obj.is_instance(self._box_type) or self._is_cover_at_start(state, aux_data.box2cover(gripped_obj)):
            (dx, dy) = self._get_action_translation(action)
            gripped_poly = translate(gripped_poly, dx, dy)

            # Check if within bounds of the world
            if not self._world_poly.contains(gripped_poly):
                return next_state

            # Check if intersects movable objects
            if MultiPolygon(obj_poly for obj, obj_poly in movable_polys if obj is not gripped_obj).intersects(gripped_poly):
                return next_state

            # Check if intersects shelf sides
            if MultiLineString([side for shelf in shelves for side in self._get_shelf_sides(state, shelf)]).intersects(gripped_poly):
                return next_state

            next_state.set(gripped_obj, 'pose_x', state.get(gripped_obj, 'pose_x') + dx)
            next_state.set(gripped_obj, 'pose_y', state.get(gripped_obj, 'pose_y') + dy)
        return next_state

    @classmethod
    def render_state_plt(
            cls,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        '''Renders the enviornment state. For now only for debugging purposes.'''
        fig = plt.figure()
        ax = fig.add_subplot()

        # Draw shelves
        for shelf in state.get_objects(cls._shelf_type):
            (x, y, w, h) = cls.get_shape_data(state, shelf)
            ax.add_patch(patches.Rectangle((x, y), w, h, color='#00ff00'))
            ax.add_patch(patches.PathPatch(Path(
                [(x, y), (x, y + h), (x + w, y), (x + w, y + h)],
                [Path.MOVETO, Path.LINETO] * 2
            ), color='darkgreen'))

        # Draw boxes
        for box in state.get_objects(cls._box_type):
            (x, y, w, h) = cls.get_shape_data(state, box)
            ax.add_patch(patches.Rectangle((x, y), w, h, color='#0090ff'))

        # Draw covers
        for cover in state.get_objects(cls._cover_type):
            (x, y, w, h) = cls.get_shape_data(state, cover)
            ax.add_patch(patches.Rectangle((x, y), w, h, color='#ff2020'))

        # # Draw bundles
        # for bundle in state.get_objects(cls._bundle_type):
        #     (x, y, w, h) = cls.get_shape_data(state, bundle)
        #     ax.add_patch(patches.Rectangle((x, y), w, h, color='#ff20ff70'))

        # ax.set_xlim(*cls.range_world_x)
        # ax.set_ylim(*cls.range_world_y)
        # ax.axis('off')
        ax.autoscale_view()
        return fig

    def _get_tasks(self, num_tasks: int, range_subtasks: Tuple[int, int], range_shelves, rng: np.random.Generator) -> List[EnvironmentTask]:
        return [self._get_task(range_subtasks, range_shelves, rng) for _ in range(num_tasks)]

    def _get_task(self, range_subtasks: Tuple[int, int], range_shelves: Tuple[int, int], rng: np.random.Generator) -> EnvironmentTask:
        num_subtasks = rng.integers(range_subtasks[0], range_subtasks[1] + 1)

        overall_state = State({}, Shelves2DState())
        overall_goal = set()
        objs_multipoly = MultiPolygon()

        for subtask_id in range(num_subtasks):
            (subtask_goal, objs_multipoly) = self._get_subtask(overall_state, objs_multipoly, subtask_id, range_shelves, rng)
            overall_goal |= subtask_goal

        return EnvironmentTask(overall_state, overall_goal)

    def _get_subtask(self, state: State, objs_multipoly: MultiPolygon, subtask_id: int,
                     range_shelves: Tuple[int, int], rng: np.random.Generator) -> Tuple[Set[GroundAtom], MultiPolygon]:
        num_shelves = rng.integers(range_shelves[0], range_shelves[1] + 1)

        # Create objects belonging to the subtasks
        boxes = [Object(f's{subtask_id}b{id}', self._box_type) for id in range(num_shelves)]
        shelves = [Object(f's{subtask_id}s{id}', self._shelf_type) for id in range(num_shelves)]
        bundle = Object(f's{subtask_id}d', self._bundle_type)
        cover = Object(f's{subtask_id}c', self._cover_type)

        # Create the goal
        cover_predicate = rng.choice([
            self._CoversBottom,
            self._CoversTop,
        ])
        goal = {cover_predicate([cover, bundle])} | {self._In([box, shelf]) for box, shelf in zip(boxes, shelves)}

        # Update auxiliary info
        aux_data = cast(Shelves2DState, state.simulator_state)
        aux_data.bundle2cover[bundle] = cover
        for box, shelf in zip(boxes, shelves):
            aux_data.box2shelf[box] = shelf
            aux_data.shelf2bundle[shelf] = bundle

        # Generate boxes
        box_widths = np.empty((0,))
        box_heights = np.empty((0,))
        for box in boxes:
            for _ in range(self.num_tries):
                box_w = self._get_uniform(rng, self.range_box_width)
                box_h = self._get_uniform(rng, self.range_box_height)
                box_x = self._get_uniform(rng, (self.range_world_x[0], self.range_world_x[1] - box_w))
                box_y = self._get_uniform(rng, (self.range_world_y[0], self.range_world_y[1] - box_h))
                box_poly = BoxWH(box_x, box_y, box_w, box_h)
                if self._world_poly.contains(box_poly) and not objs_multipoly.intersects(box_poly):
                    break
            else:
                raise ValueError('Could not generate a task with given settings')

            state.data[box] = np.array([box_x, box_y, box_w, box_h], dtype=np.float32)
            objs_multipoly = objs_multipoly.union(box_poly)

            box_widths = np.r_[(box_widths, box_w)]
            box_heights = np.r_[(box_heights, box_h)]

            min_box_height = box_heights.min()
            max_box_height = box_heights.max()

        # Generate the bundle
        for _ in range(self.num_tries):
            shelf_widths = self._get_uniform(rng, (
                np.maximum(box_widths + self.min_box_width_clearance, self.range_shelf_width[0]),
                self.range_shelf_width[1]
            ))
            shelf_seps = self._get_uniform(rng, self.range_shelf_sep, num_shelves - 1)

            bundle_width = shelf_widths.sum() + shelf_seps.sum()
            bundle_height = self._get_uniform(rng, (
                self.range_shelf_height[0],
                min(self.range_shelf_height[1], min_box_height - self.min_box_height_clearance)
            ))

            bundle_x = self._get_uniform(rng, (self.range_world_x[0], self.range_world_x[1] - bundle_width))
            bundle_y = self._get_uniform(rng, (self.range_world_y[0], self.range_world_y[1] - bundle_height))

            bundle_margin_x = self.cover_sideways_tolerance + self.bundle_margin
            bundle_margin_y = max(max_box_height, self.cover_max_distance + self.cover_thickness) + self.bundle_margin

            bundle_margin = Box(
                bundle_x - bundle_margin_x,
                bundle_y - bundle_margin_y,
                bundle_x + bundle_width + bundle_margin_x,
                bundle_y + bundle_height + bundle_margin_y
            )
            if self._world_poly.contains(bundle_margin) and not objs_multipoly.intersects(bundle_margin):
                break
        else:
            raise ValueError('Could not generate a task with given settings')

        objs_multipoly = objs_multipoly.union(bundle_margin)
        state.data[bundle] = np.array([bundle_x, bundle_y, bundle_width, bundle_height], dtype=np.float32)

        # Generate shelves
        shelves_x = bundle_x
        shelves_y = bundle_y
        for shelf, shelf_width, shelf_sep in zip(shelves, shelf_widths, list(shelf_seps) + [0]):
            state.data[shelf] = np.array([
                shelves_x,
                shelves_y,
                shelf_width,
                bundle_height], dtype=np.float32)
            shelf_poly = BoxWH(shelves_x, shelves_y, shelf_width, bundle_height)
            shelves_x += shelf_sep + shelf_width

        # Generate the cover
        for _ in range(self.num_tries):
            cover_w = bundle_width
            cover_h = self.cover_thickness
            cover_x = rng.uniform(self.range_world_x[0], self.range_world_x[1] - cover_w)
            cover_y = rng.uniform(self.range_world_y[0], self.range_world_y[1] - cover_h)
            cover_poly = BoxWH(cover_x, cover_y, cover_w, cover_h)
            if self._world_poly.contains(cover_poly) and not objs_multipoly.intersects(cover_poly):
                break
        else:
            raise ValueError('Could not generate a task with given settings')
        state.data[cover] = np.array([
            cover_x,
            cover_y,
            cover_w,
            cover_h,
            cover_x + cover_w / 2,
            cover_y + cover_h / 2], dtype=np.float32)
        objs_multipoly = objs_multipoly.union(cover_poly)

        return (goal, objs_multipoly)

    def _In_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (box, shelf) = objects
        box_poly = self._get_obj_polygon(state, box)
        shelf_poly = self._get_obj_polygon(state, shelf)
        return shelf_poly.intersects(box_poly)

    def _GoesInto_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (box, shelf) = objects
        box2shelf = cast(Shelves2DState, state.simulator_state).box2shelf
        return box2shelf[box] == shelf

    def _Bundles_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (bundle, shelf) = objects
        shelf2bundle = cast(Shelves2DState, state.simulator_state).shelf2bundle
        return shelf2bundle[shelf] == bundle

    def _CoversTop_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return self._Covers_helper(state, objects, True)

    def _CoversBottom_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return self._Covers_helper(state, objects, False)

    def _Covers_helper(self, state: State, objects: Sequence[Object], cover_from_top: bool) -> bool:
        (cover, bundle) = objects

        cover_poly = self._get_obj_polygon(state, cover)
        bundle_poly = self._get_obj_polygon(state, bundle)

        cover_center = cover_poly.boundary.centroid
        bundle_center = bundle_poly.boundary.centroid
        if np.abs(cover_center.x - bundle_center.x) > self.cover_sideways_tolerance:
            return False

        bundle_distance = bundle_poly.distance(cover_poly)
        if bundle_distance > self.cover_max_distance:
            return False

        if cover_from_top:
            return cover_center.y > bundle_center.y
        return cover_center.y < bundle_center.y

    def _CoverAtStart_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (cover,) = objects
        return self._is_cover_at_start(state, cover)

    def _CoverFor_holds(self, state: State, objects: Sequence[Object]) -> bool:
        (cover, bundle) = objects
        bundle2cover = cast(Shelves2DState, state.simulator_state).bundle2cover
        return bundle2cover[bundle] == cover

    @property
    def _world_poly(self) -> Polygon:
        return Box(self.range_world_x[0], self.range_world_y[0], self.range_world_x[1], self.range_world_y[1])

    @classmethod
    def _is_cover_at_start(cls, state: State, cover: Object) -> bool:
        assert cover.is_instance(cls._cover_type)
        cover_poly = cls._get_obj_polygon(state, cover)
        return cover_poly.contains(cls._get_cover_start_pose(state, cover))

    @classmethod
    def _get_obj_polygon(cls, state: State, obj: Object) -> Polygon:
        x, y, w, h = cls.get_shape_data(state, obj)
        return BoxWH(x, y, w, h)

    @classmethod
    def _get_uniform(cls, rng: np.random.Generator, range: Tuple[Union[float, npt.NDArray], Union[float, npt.NDArray]],
                     size: Optional[Tuple[int, ...]] = None) -> npt.NDArray[np.float32]:
        assert type(range[0]) == type(range[1]) == float or size is None

        if not (np.array(range[0]) <= np.array(range[1])).all():
            raise ValueError('Could not generate a task with given settings')
        return rng.uniform(*range, size=size)

    @classmethod
    def get_shape_data(cls, state: State, obj: Object) -> Tuple[np.float32, np.float32, np.float32, np.float32]:
        x = state.get(obj, 'pose_x')
        y = state.get(obj, 'pose_y')
        w = state.get(obj, 'size_x')
        h = state.get(obj, 'size_y')
        assert type(x) == type(y) == type(w) == type(h) == np.float32
        return (x, y, w, h)

    @classmethod
    def _get_cover_start_pose(cls, state: State, obj: Object):
        assert obj.is_instance(cls._cover_type)

        x = state.get(obj, 'start_x')
        y = state.get(obj, 'start_y')
        assert type(x) == type(y) == np.float32

        return Point((x, y))

    @classmethod
    def _get_shelf_sides(cls, state: State, obj: Object):
        assert obj.is_instance(cls._shelf_type)

        x = state.get(obj, 'pose_x')
        y = state.get(obj, 'pose_y')
        w = state.get(obj, 'size_x')
        h = state.get(obj, 'size_y')

        return (LineString([[x, y], [x,y + h]]), LineString([[x + w, y], [x + w, y + h]]))

    @classmethod
    def _get_action_gripper_pose(cls, action: Action) -> Tuple[np.float32, np.float32]:
        return (action.arr[0], action.arr[1])

    @classmethod
    def _get_action_translation(cls, action: Action) -> Tuple[np.float32, np.float32]:
        return (action.arr[2], action.arr[3])

@dataclass
class Shelves2DState:
     box2shelf: Dict[Object, Object] = field(default_factory=dict)
     shelf2bundle: Dict[Object, Object] = field(default_factory=dict)
     bundle2cover: Dict[Object, Object] = field(default_factory=dict)

     def box2cover(self, box: Object) -> Object:
         shelf = self.box2shelf[box]
         bundle = self.shelf2bundle[shelf]
         cover = self.bundle2cover[bundle]
         return cover