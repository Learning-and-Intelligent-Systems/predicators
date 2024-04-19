from dataclasses import dataclass
import logging
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
from predicators.envs.base_env import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Object, Predicate, State, Type
from experiments.envs.utils import BoxWH
from shapely.geometry import box as Box, Point, Polygon, MultiPolygon, LineString, MultiLineString
from shapely.affinity import translate
import gym

import matplotlib.patches as patches
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("tkagg")

from predicators.utils import abstract

__all__ = ['Donuts']

class Donuts(BaseEnv):
    """Base environment."""

    # Settings
    ## Task generation settings
    range_train_toppings: ClassVar[Tuple[int, int]] = (1, 3)
    range_train_donuts: ClassVar[Tuple[int, int]] = (1, 1)

    num_tries: ClassVar[int] = 10000
    world_intersection_margin = 1000

    ## World shape
    range_world_x: ClassVar[Tuple[float, float]] = (-20.0, 20.0)
    range_world_y: ClassVar[Tuple[float, float]] = (-2.5, 10.0)
    topper_size: ClassVar[Tuple[float, float]] = (1.0, 1.0)
    donut_radius: ClassVar[float] = 0.2
    robot_size: ClassVar[Tuple[float, float]] = (1.0, 1.0)
    container_size: ClassVar[Tuple[float, float]] = (1.0, 1.0)
    robot_pos: ClassVar[Tuple[float, float]] = (0.0, 0.0)
    range_table_y: ClassVar[Tuple[float, float]] = (-2.5, -1.0)

    ## Predicate thresholds
    coveredin_thresh: ClassVar[float] = 0.5
    held_thresh: ClassVar[float] = 0.5
    nextto_dist_thresh: ClassVar[float] = 2.0
    fingers_closed_thresh: ClassVar[float] = 0.5
    side_grasp_thresh: ClassVar[float] = 0.1
    top_grasp_thresh: ClassVar[float] = 0.9
    fresh_thresh = 0.5

    # Variables for parametric types and predicates
    toppings: ClassVar[List[str]] = [
        "Sprinkles", "Frosting", "Sugar", "ChocolateChips", "Strawberries",
        "Blueberries", "Nuts", "Honey", "Cinnamon", "Coconut",
        # "MapleSyrup", "Caramel", "PeanutButter", "ChocolateGlaze", "VanillaGlaze",
        # "LemonZest", "Nutella", "Bacon", "Marshmallows", "OreoCrumbs",
    ]
    amount_format: ClassVar[str] = "amount{}"
    topper_format: ClassVar[str] = "topperFor{}"
    covered_in_format: ClassVar[str] = "CoveredIn{}"

    # Types
    _object_type: ClassVar[Type] = Type("object", ["x", "y"])
    _robot_type: ClassVar[Type] = Type("robot", ["x", "y", "fingers"], _object_type)
    _donut_type: ClassVar[Type] = Type("donut", ["x", "y", "grasp", "held", "fresh"] + list(map(amount_format.format, toppings)), _object_type)
    _position_type: ClassVar[Type] = Type("position", ["x", "y"], _object_type)
    _container_type: ClassVar[Type] = Type("container", ["x", "y"], _position_type)
    _shelf_type: ClassVar[Type] = Type("shelf", ["x", "y"], _container_type)
    _box_type: ClassVar[Type] = Type("box", ["x", "y"], _container_type)
    _topper_type: ClassVar[Type] = Type("topper", ["x", "y"], _position_type)
    _topper_types: ClassVar[Dict[str, Type]] = {}
    for topping in toppings:
        _topper_types[topping] = Type(topper_format.format(topping), ["x", "y"], _topper_type)

    # Predicates
    ## CoveredIn Predicates
    class CoveredIn_holds:
        def __init__(self, amount: str):
            self._amount = amount

        def __call__(self, state: State, objects: Sequence[Object]) -> bool:
            donut, = objects
            return state.get(donut, self._amount) >= Donuts.coveredin_thresh

    _CoveredInPreds: ClassVar[Dict[str, Predicate]] = {}
    for topping in toppings: # Comprehension does not see class scope
        _CoveredInPreds[topping] = Predicate(
            covered_in_format.format(topping), [_donut_type], CoveredIn_holds(amount_format.format(topping))
        )

    ## Fresh Predicate
    def _Fresh_holds(state: State, objects: Sequence[Object]) -> bool:
        donut, = objects
        return state.get(donut, "fresh") >= Donuts.fresh_thresh

    _Fresh: ClassVar[Predicate] = Predicate("Fresh", [_donut_type], _Fresh_holds)

    ## NextTo Predicate
    def _objects_distance(state: State, obj1: Object, obj2: Object) -> float:
        shapes = Donuts._get_shapes(state)
        return shapes[obj1].distance(shapes[obj2])

    def _NextTo_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, obj = objects
        return robot != obj and Donuts._objects_distance(state, robot, obj) <= Donuts.nextto_dist_thresh

    _NextTo: ClassVar[Predicate] = Predicate("NextTo", [_robot_type, _object_type], _NextTo_holds)

    ## NotHeld Predicate
    def _NotHeld_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, = objects
        fingers = state.get(robot, "fingers") <= Donuts.fingers_closed_thresh
        return not fingers

    _NotHeld: ClassVar[Predicate] = Predicate("NotHeld", [_robot_type], _NotHeld_holds)

    ## Held Predicate
    def _Held_holds(state: State, objects: Sequence[Object]) -> bool:
        robot, donut = objects
        fingers = state.get(robot, "fingers") <= Donuts.fingers_closed_thresh
        held = state.get(donut, "held") >= Donuts.held_thresh
        assert not held or Donuts._objects_distance(state, robot, donut) < 1e-2 and fingers
        return held

    _Held: ClassVar[Predicate] = Predicate("Held", [_robot_type, _donut_type], _Held_holds)

    ## In Predicate
    def _In_holds(state: State, objects: Sequence[Object]) -> bool:
        donut, container = objects
        shapes = Donuts._get_shapes(state)
        return shapes[container].contains(shapes[donut])

    _In: ClassVar[Predicate] = Predicate("In", [_donut_type, _container_type], _In_holds)

    # Common objects
    _robot: ClassVar[Object] = Object("robot", _robot_type)
    _toppers: ClassVar[Dict[str, Object]] = {}
    for topping in toppings:
        _toppers[topping] = Object(topper_format.format(topping), _topper_types[topping])

    @classmethod
    def get_name(cls) -> str:
        return "donuts"

    @classmethod
    def simulate(cls, state: State, action: Action) -> State:
        move, grab_place = action.arr[3:5]
        topping_affinities = action.arr[5:]
        affinities = [
            (move, cls._transition_move),
            (1 - grab_place, cls._transition_grab),
            (grab_place + 1, cls._transition_place),
        ] + [
            (affinity, cls._transition_toppings_gen(topping))
            for topping, affinity in zip(cls.toppings, topping_affinities)#, strict=True)
        ]
        _, transition_fn = min(affinities, key = lambda t: t[0])
        return transition_fn(state, action)

    @classmethod
    def _transition_move(cls, state: State, action: Action) -> State:
        logging.info("TRANSITION MOVE")
        next_state = state.copy()
        dx, dy = action.arr[:2]

        # Check if the new robot position does not collide with anything
        shapes = Donuts._get_shapes(next_state)
        new_robot_polygon = translate(shapes[cls._robot], dx, dy)
        mb_donut = cls._get_held_donut(state)
        non_active_objects = [
            obj for obj in state.get_objects(cls._object_type)
            if obj not in [cls._robot, mb_donut]
        ]
        if cls._collides(state, non_active_objects, new_robot_polygon):
            logging.info("ROBOT COLLIDES WITH OBJECTS")
            return next_state

        # Check if the new robot position is within the bounds of the world
        world = Box(
            cls.range_world_x[0], cls.range_world_y[0],
            cls.range_world_x[1], cls.range_world_y[1],
        )
        if not world.contains(new_robot_polygon):
            logging.info("ROBOT OOB")
            return next_state

        # Move the robot
        shapes[cls._robot] = new_robot_polygon

        # Move the held donut
        if mb_donut is not None:
            shapes[mb_donut] = cls._move_donut_polygon(state, mb_donut, dx, dy)

        # Move the relative positions of the rest of the objects
        for obj in non_active_objects:
            next_state.set(obj, "x", next_state.get(obj, "x") - dx)
            next_state.set(obj, "y", next_state.get(obj, "y") - dy)
        return next_state

    @classmethod
    def _transition_grab(cls, state: State, action: Action) -> State:
        logging.info("TRANSITION GRAB")
        next_state = state.copy()
        grasp = action.arr[2]

        # Check if a donut isn't already grabbed
        if cls._get_held_donut(state) is not None:
            logging.info("DONUT ALREADY HELD")
            return next_state

        # Get closest donut
        donut, _ = min((
            (donut, cls._objects_distance(state, cls._robot, donut))
            for donut in state.get_objects(cls._donut_type)
        ), key = lambda t: t[1])

        # Check if the donut is next to the robot
        if not cls._NextTo_holds(state, [cls._robot, donut]):
            logging.info("DONUT NOT NEXT TO THE ROBOT")
            return next_state

        # Move the donut
        cls._move_donut(next_state, donut, state.get(cls._robot, "x"), state.get(cls._robot, "y"))
        next_state.set(donut, "grasp", grasp)
        next_state.set(donut, "held", 1.0)
        next_state.set(cls._robot, "fingers", 0.0)
        return next_state

    @classmethod
    def _transition_place(cls, state: State, action: Action) -> State:
        logging.info("TRANSITION PLACE")
        next_state = state.copy()
        x, y = action.arr[:2]

        # Check what donut is held
        mb_donut = cls._get_held_donut(state)
        if mb_donut is None:
            logging.info("DONUT NOT HELD")
            return next_state

        # Check if the placement spot doesn't collide with others
        new_donut_polygon = cls._move_donut_polygon(state, mb_donut, x, y)
        if Donuts._collides(state, list(cls._toppers.values()) + [
            donut for donut in state.get_objects(cls._donut_type)
            if donut != mb_donut
        ], new_donut_polygon):
            logging.info("DONUT COLLIDES WITH OBJECTS")
            return next_state

        # Check what the donut is placed into
        mb_container = cls._get_container(state, new_donut_polygon)
        if mb_container is None:
            logging.info("DONUT NOT PLACED INTO CONTAINER")
            return next_state

        # Check if we are next to the container
        if not cls._NextTo_holds(state, [cls._robot, mb_container]):
            logging.info("ROBOT NOT NEXT TO A CONTAINER")
            return next_state

        # Check if the donut would be placed into the container
        shapes = Donuts._get_shapes(state)
        if not shapes[mb_container].contains(new_donut_polygon):
            logging.info("DONUT NOT PLACED INTO THE CONTAINER")
            return next_state

        # Check for correct grasp
        grasp = state.get(mb_donut, "grasp")
        if mb_container.is_instance(cls._box_type):
            if grasp < cls.top_grasp_thresh:
                logging.info("SIDE GRASP CANNOT BE PLACED INTO A BOX")
                return next_state
        else:
            if grasp > cls.side_grasp_thresh:
                logging.info("TOP GRASP CANNOT BE PLACED ON A SHELF")
                return next_state

        # Move the donut
        cls._move_donut(next_state, mb_donut, x, y)
        next_state.set(mb_donut, "held", 0.0)
        next_state.set(mb_donut, "fresh", 0.0)
        next_state.set(cls._robot, "fingers", 1.0)
        return next_state

    @classmethod
    def _transition_toppings_gen(cls, topping: str) -> Callable[[State, Action], State]:
        def _transition_toppings(state: State, action: Action) -> State:
            logging.info(f"TRANSITION TOPPING {topping}")
            next_state = state.copy()

            # Robot not next to the topper
            if not Donuts._NextTo_holds(state, [cls._robot, cls._toppers[topping]]):
                logging.info("ROBOT NOT NEXT TO THE TOPER")
                return next_state

            # Robot not holding anything
            mb_donut = cls._get_held_donut(state)
            if mb_donut is None:
                logging.info("ROBOT NOT HOLDING ANYTHING")
                return next_state

            # Donut is not fresh
            if not Donuts._Fresh_holds(state, [mb_donut]):
                logging.info("DONUT NOT FRESH")
                return next_state

            # Adding topping to the donut
            next_state.set(mb_donut, cls.amount_format.format(topping), 1.0)
            return next_state
        return _transition_toppings

    @classmethod
    def _get_held_donut(cls, state: State) -> Optional[Object]:
        if state.get(cls._robot, "fingers") >= cls.fingers_closed_thresh:
            return None
        return next(filter(
            lambda donut: cls._Held_holds(state, [cls._robot, donut]),
            state.get_objects(cls._donut_type)
        ), None)

    @classmethod
    def _get_container(cls, state: State, donut_polygon: Polygon) -> Optional[Object]:
        boxes = state.get_objects(cls._box_type)
        shelves = state.get_objects(cls._shelf_type)
        shapes = Donuts._get_shapes(state)
        return next(filter(lambda obj: shapes[obj].contains(donut_polygon), boxes + shelves), None)

    @classmethod
    def _move_donut(cls, state: State, donut: Object, new_x: float, new_y: float) -> None:
        assert donut.is_instance(cls._donut_type)
        Donuts._get_shapes(state)[donut] = \
            cls._move_donut_polygon(state, donut, new_x, new_y)

        state.set(donut, "x", new_x)
        state.set(donut, "y", new_y)

    @classmethod
    def _move_donut_polygon(cls, state: State, donut: Object, new_x: float, new_y: float) -> Polygon:
        assert donut.is_instance(cls._donut_type)
        old_x, old_y = state.get(donut, "x"), state.get(donut, "y")

        return translate(
            Donuts._get_shapes(state)[donut],
            new_x - old_x, new_y - old_y
        )

    @staticmethod
    def _collides(state: State, objects: Sequence[Object], polygon: Polygon) -> bool:
        return MultiPolygon([
            Donuts._get_shapes(state)[obj]
            for obj in objects
        ]).intersects(polygon)

    @staticmethod
    def _get_shapes(state: State) -> Dict[Object, Polygon]:
        return cast(Dict[Object, Polygon], state.simulator_state)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        if not self._train_tasks:
            self._train_tasks = self._generate_tasks(
                rng = self._train_rng,
                num_tasks = CFG.num_train_tasks,
                range_toppings = self.range_train_toppings,
                range_donuts = self.range_train_donuts,
            )
        return self._train_tasks

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        if not self._test_tasks:
            assert CFG.donuts_test_num_toppings <= len(self.toppings)
            self._test_tasks = self._generate_tasks(
                rng = self._test_rng,
                num_tasks = CFG.num_test_tasks,
                range_toppings = (CFG.donuts_test_num_toppings, CFG.donuts_test_num_toppings),
                range_donuts = (CFG.donuts_test_num_donuts, CFG.donuts_test_num_donuts),
            )
        return self._test_tasks

    def _generate_tasks(
        self,
        rng: np.random.Generator,
        num_tasks: int,
        range_toppings: Tuple[int, int],
        range_donuts: Tuple[int, int],
    ) -> List[EnvironmentTask]:
        return [
            self._generate_task(
                rng,
                range_toppings,
                range_donuts,
            ) for _ in range(num_tasks)
        ]

    def _generate_task(
        self,
        rng: np.random.Generator,
        range_toppings: Tuple[int, int],
        range_donuts: Tuple[int, int],
    ) -> EnvironmentTask:
        num_donuts = rng.integers(*range_donuts, endpoint=True)

        # Generating objects
        donuts = [
            Object(f"donut{i}", self._donut_type)
            for i in range(num_donuts)
        ]
        containers = [
            rng.choice([Object(f"box{i}", self._box_type), Object(f"shelf{i}", self._shelf_type)])
            for i in range(num_donuts)
        ]

        # Generating Goal
        goal = {
            Donuts._In([donut, container])
            for donut, container in zip(donuts, containers)
        } | {
            Donuts._CoveredInPreds[topping]([donut])
            for donut in donuts
            for topping in rng.choice(
                self.toppings, rng.integers(*range_toppings, endpoint=True), replace=False
            )
        }

        # Constructing placeholder state and total geometry (to check for collision)
        shapes: Dict[Object, Polygon] = {}
        state: State = State({
            obj: np.zeros((obj.type.dim,), dtype=np.float32)
            for obj in donuts + containers + [self._robot] + list(self._toppers.values())
        }, shapes)
        world = Box(
            self.range_world_x[0] - self.world_intersection_margin,
            self.range_world_y[0] - self.world_intersection_margin,
            self.range_world_x[1] + self.world_intersection_margin,
            self.range_world_y[1] + self.world_intersection_margin
        ).difference(Box(
            self.range_world_x[0], self.range_world_y[0],
            self.range_world_x[1], self.range_world_y[1]
        ))
        table = Box(
            self.range_world_x[0], self.range_table_y[0],
            self.range_world_x[1], self.range_table_y[1],
        )

        # Adding robot
        state.set(self._robot, "x", self.robot_pos[0])
        state.set(self._robot, "y", self.robot_pos[1])
        state.set(self._robot, "fingers", 1.0)
        robot_polygon = BoxWH(
            self.robot_pos[0] - self.robot_size[0]/2,
            self.robot_pos[1] - self.robot_size[1]/2,
            *self.robot_size
        )
        if world.intersects(robot_polygon):
            raise ValueError("Could not generate a task with the given settings")
        shapes[self._robot] = robot_polygon
        world = world.union(robot_polygon)

        # Adding toppers
        for topping in self.toppings:
            for _ in range(self.num_tries):
                # Generating the topper's position
                topper_x, topper_y = rng.uniform(
                    [self.range_world_x[0], self.range_table_y[0]],
                    [self.range_world_x[1], self.range_table_y[1]]
                )

                # Checking the new position
                topper_polygon = BoxWH(topper_x - self.topper_size[0]/2, topper_y - self.topper_size[1] / 2, *self.topper_size)
                if world.intersects(topper_polygon) or not table.contains(topper_polygon):
                    continue
                break
            else:
                raise ValueError("Could not generate a task with the given settings")
            state.set(self._toppers[topping], "x", topper_x)
            state.set(self._toppers[topping], "y", topper_y)
            shapes[self._toppers[topping]] = topper_polygon
            world = world.union(topper_polygon)

        # Adding containers
        for container in containers:
            for _ in range(self.num_tries):
                # Generating the container's position
                container_x, container_y = rng.uniform(
                    [self.range_world_x[0], self.range_table_y[0]],
                    [self.range_world_x[1], self.range_table_y[1]]
                )

                # Checking the new position
                container_polygon = BoxWH(
                    container_x - self.container_size[0] / 2,
                    container_y - self.container_size[1] / 2,
                    *self.container_size
                )
                if world.intersects(container_polygon) or not table.contains(container_polygon):
                    continue
                break
            else:
                raise ValueError("Could not generate a task with the given settings")
            state.set(container, "x", container_x)
            state.set(container, "y", container_y)
            shapes[container] = container_polygon
            world = world.union(container_polygon)

        # Adding donuts
        for donut in donuts:
            for _ in range(self.num_tries):
                # Generating the donut's position
                donut_x, donut_y = rng.uniform(
                    [self.range_world_x[0], self.range_table_y[0]],
                    [self.range_world_x[1], self.range_table_y[1]]
                )

                # Checking the new position
                donut_polygon = Point(donut_x, donut_y).buffer(self.donut_radius, quad_segs=64)
                if world.intersects(donut_polygon) or not table.contains(donut_polygon):
                    continue
                break
            else:
                raise ValueError("Could not generate a task with the given settings")
            state.set(donut, "x", donut_x)
            state.set(donut, "y", donut_y)
            state.set(donut, "fresh", 1.0)
            state.set(donut, "held", 0.0)
            shapes[donut] = donut_polygon
            world = world.union(donut_polygon)

        return EnvironmentTask(state, goal)

    @property
    def predicates(self) -> Set[Predicate]:
        return {Donuts._In, Donuts._Held, Donuts._NotHeld, Donuts._NextTo, Donuts._Fresh} | set(Donuts._CoveredInPreds.values())

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {Donuts._In} | set(Donuts._CoveredInPreds.values())

    @property
    def types(self) -> Set[Type]:
        return {
            self._object_type, self._robot_type, self._donut_type, self._topper_type,
            self._position_type, self._container_type, self._box_type, self._shelf_type
        } | set(self._topper_types.values())

    @property
    def action_space(self) -> gym.spaces.Box:
        "(x, y, grasp, move_robot, grab/place donut, add toppings...)"
        lower_bound = np.array(
            [self.range_world_x[0] - self.range_world_x[1], self.range_world_x[0] - self.range_world_x[1], 0, 0, -1] +
            [0 for _ in self.toppings],
            dtype=np.float32
        )
        upper_bound = np.array(
            [self.range_world_x[1] - self.range_world_x[0], self.range_world_x[1] - self.range_world_x[0], 1, 1, 1] +
            [1 for _ in self.toppings],
            dtype=np.float32
        )
        return gym.spaces.Box(lower_bound, upper_bound)

    @classmethod
    def render_state_plt(
        cls,
        state: State,
        task: EnvironmentTask,
        action: Optional[Action] = None,
        caption: Optional[str] = None
    ) -> matplotlib.figure.Figure:
        fig = plt.figure()
        fig.suptitle(caption)
        ax = fig.add_subplot()

        # Drawing the robot
        robot_x, robot_y = state.get(cls._robot, "x"), state.get(cls._robot, "y")
        ax.add_patch(patches.Rectangle((
            robot_x - cls.robot_size[0] / 2, robot_y - cls.robot_size[1] / 2, *cls.robot_size
        ), *cls.robot_size, color='#000000'))

        # Drawing the table
        ax.add_patch(patches.Rectangle(
            (cls.range_world_x[0], cls.range_table_y[0]),
            cls.range_world_x[1] - cls.range_world_x[0],
            cls.range_table_y[1] - cls.range_table_y[0],
            color='#8800FF'
        ))

        # Drawing the toppers
        for topping in cls.toppings:
            topper_x, topper_y = state.get(cls._toppers[topping], "x"), state.get(cls._toppers[topping], "y")
            ax.add_patch(patches.Rectangle((
                topper_x - cls.topper_size[0] / 2, topper_y - cls.topper_size[1] / 2, *cls.topper_size
            ), *cls.topper_size, color='#1144EE'))

        # Drawing the boxes
        for box in state.get_objects(cls._box_type):
            box_x, box_y = state.get(box, "x"), state.get(box, "y")
            ax.add_patch(patches.Rectangle((
                box_x - cls.container_size[0] / 2, box_y - cls.container_size[1] / 2
            ), *cls.container_size, color='#55CC33'))

        # Drawing the shelves
        for shelf in state.get_objects(cls._shelf_type):
            shelf_x, shelf_y = state.get(shelf, "x"), state.get(shelf, "y")
            ax.add_patch(patches.Rectangle((
                shelf_x - cls.container_size[0] / 2, shelf_y - cls.container_size[1] / 2
            ), *cls.container_size, color='#DDDD66'))

        # Drawing the donuts
        for donut in state.get_objects(cls._donut_type):
            donut_x, donut_y = state.get(donut, "x"), state.get(donut, "y")
            ax.add_patch(patches.Circle((donut_x, donut_y), cls.donut_radius, color='#DC7737'))

        ax.set_xlim(*cls.range_world_x)
        ax.set_ylim(*cls.range_world_y)
        return fig