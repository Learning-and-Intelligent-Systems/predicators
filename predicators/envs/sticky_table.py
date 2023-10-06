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
    """An environment where a cube and a ball must be transported between
    tables.

    Most of the tables are flat, but one is half is smooth and half is sticky.
    When placing on the smooth side, the cube usually falls off; when placing
    on the sticky side, it usually stays. When it falls off, it falls onto the
    floor. It can be picked up from the floor.

    The ball falls off with some probability when placed on even normal tables,
    and it falls off certainly when placed on the smooth side of a sticky table.
    When placed on the sticky side, it only sometimes falls off. However, if
    the ball is placed inside a cup first, then the ball + cup system stays
    on table surfaces as well as a cube does.

    Note that unlike almost all of our other environments, there is real
    stochasticity in the outcomes of placing.

    The action space is 4D. TODO: explain what's going on with the action space.
    """
    x_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 1.0
    y_lb: ClassVar[float] = 0.0
    y_ub: ClassVar[float] = 1.0
    reachable_thresh: ClassVar[float] = 0.25
    objs_scale: ClassVar[float] = 0.25  # as a function of table radius
    sticky_surface_mode: ClassVar[str] = "half"  # half or whole
    # Types
    _cube_type: ClassVar[Type] = Type("cube", ["x", "y", "size", "held"])
    _table_type: ClassVar[Type] = Type("table", ["x", "y", "radius", "sticky"])
    _robot_type: ClassVar[Type] = Type("robot", ["x", "y"])
    _ball_type: ClassVar[Type] = Type("ball", ["x", "y", "radius", "held"])
    _cup_type: ClassVar[Type] = Type("cup", ["x", "y", "radius", "held"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # For noisy simulation.
        self._noise_rng = np.random.default_rng(CFG.seed)

        # Predicates
        self._CubeOnTable = Predicate("CubeOnTable",
                                      [self._cube_type, self._table_type],
                                      self._OnTable_holds)
        self._CubeOnFloor = Predicate("CubeOnFloor", [self._cube_type],
                                      self._OnFloor_holds)
        self._BallOnTable = Predicate("BallOnTable",
                                      [self._ball_type, self._table_type],
                                      self._OnTable_holds)
        self._BallOnFloor = Predicate("BallOnFloor", [self._ball_type],
                                      self._OnFloor_holds)
        self._CupOnTable = Predicate("CupOnTable",
                                     [self._cup_type, self._table_type],
                                     self._OnTable_holds)
        self._CupOnFloor = Predicate("CupOnFloor", [self._cup_type],
                                     self._OnFloor_holds)
        self._HoldingCube = Predicate("HoldingCube", [self._cube_type],
                                      self._Holding_holds)
        self._HoldingBall = Predicate("HoldingBall", [self._ball_type],
                                      self._Holding_holds)
        self._HoldingCup = Predicate("HoldingCup", [self._cup_type],
                                     self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [], self._HandEmpty_holds)
        self._IsReachableSurface = Predicate(
            "IsReachableSurface", [self._robot_type, self._table_type],
            self._IsReachable_holds)
        self._IsReachableCube = Predicate("IsReachableCube",
                                          [self._robot_type, self._cube_type],
                                          self._IsReachable_holds)
        self._IsReachableBall = Predicate("IsReachableBall",
                                          [self._robot_type, self._ball_type],
                                          self._IsReachable_holds)
        self._IsReachableCup = Predicate("IsReachableCup",
                                         [self._robot_type, self._cup_type],
                                         self._IsReachable_holds)
        self._BallInCup = Predicate("BallInCup",
                                    [self._ball_type, self._cup_type],
                                    self._BallInCup_holds)
        self._BallNotInCup = Predicate("BallNotInCup",
                                       [self._ball_type, self._cup_type],
                                       self._BallNotInCup_holds)

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
    def _place_ball_fall_prob(self) -> float:
        return CFG.sticky_table_place_ball_fall_prob

    @property
    def _place_smooth_fall_prob(self) -> float:
        return CFG.sticky_table_place_smooth_fall_prob

    def simulate(self, state: State, action: Action) -> State:
        # NOTE: noise is added here. Two calls to simulate with the same
        # inputs may produce different outputs!
        assert self.action_space.contains(action.arr)
        move_or_pickplace, obj_type_id, act_x, act_y = action.arr
        next_state = state.copy()
        hand_empty = self._HandEmpty_holds(state, [])
        cube, = state.get_objects(self._cube_type)
        ball, = state.get_objects(self._ball_type)
        cup, = state.get_objects(self._cup_type)
        robot, = state.get_objects(self._robot_type)
        cube_held = self._Holding_holds(state, [cube])
        ball_held = self._Holding_holds(state, [ball])
        cup_held = self._Holding_holds(state, [cup])
        assert (not (cube_held and cup_held)) and (not (cube_held
                                                        and ball_held))
        ball_in_cup = self._BallInCup_holds(state, [ball, cup])
        obj_being_held: Optional[Object] = None
        if cube_held:
            obj_being_held = cube
        if ball_held and not cup_held:
            obj_being_held = ball
        elif cup_held:
            obj_being_held = cup

        if move_or_pickplace == 1.0:
            # Picking logic.
            if hand_empty:
                # Fail sometimes.
                if self._noise_rng.uniform() < self._pick_success_prob:
                    if obj_type_id == 0.0:
                        if self._action_grasps_object(act_x, act_y, cube,
                                                      state):
                            next_state.set(cube, "held", 1.0)
                    elif obj_type_id == 1.0:
                        if self._action_grasps_object(act_x, act_y, ball,
                                                      state):
                            next_state.set(ball, "held", 1.0)
                    else:
                        assert obj_type_id == 2.0
                        if self._action_grasps_object(act_x, act_y, cup,
                                                      state):
                            next_state.set(cup, "held", 1.0)
                            if ball_in_cup:
                                next_state.set(ball, "held", 1.0)
            # Placing logic.
            else:
                if obj_being_held is not None:
                    next_state.set(obj_being_held, "held", 0.0)
                # Find the table for placing, if any.
                table: Optional[Object] = None
                for target in state.get_objects(self._table_type):
                    rect = self._object_to_geom(target, state)
                    if rect.contains_point(act_x, act_y):
                        table = target
                        break
                if table is None:
                    # Put on the floor here.
                    next_state = self._handle_placing_object(
                        act_x, act_y, next_state, obj_being_held, ball, cup,
                        ball_in_cup)
                else:
                    # TODO: we're currently not checking that the robot is reachable to
                    # where it's trying to pick or place at; we probably want to do this!
                    if obj_type_id == 3.0:
                        # Possibly put on the table, or have it fall somewhere near.
                        fall_prob = self._place_sticky_fall_prob
                        if obj_being_held == ball:
                            fall_prob = self._place_ball_fall_prob
                        if self._table_is_sticky(table, state):
                            # Check if placing on the smooth side of the sticky table.
                            table_y = state.get(table, "y")
                            if self.sticky_surface_mode == "half" and act_y < table_y + 0.3 * (state.get(table, "radius") - (state.get(cube, "size") / 2)):
                                if obj_being_held in [cube, cup]:
                                    fall_prob = self._place_smooth_fall_prob
                                else:
                                    assert obj_being_held == ball
                                    fall_prob = 1.0

                        if obj_being_held == cup and fall_prob != 1.0:
                            import ipdb; ipdb.set_trace()

                        if self._noise_rng.uniform() < fall_prob:
                            fall_x, fall_y = self._sample_floor_point_around_table(
                                table, state, self._noise_rng)
                            next_state = self._handle_placing_object(
                                fall_x, fall_y, next_state, obj_being_held, ball,
                                cup, ball_in_cup)
                            assert self._OnFloor_holds(next_state,
                                                    [obj_being_held])
                        else:
                            next_state = self._handle_placing_object(
                                act_x, act_y, next_state, obj_being_held, ball,
                                cup, ball_in_cup)
                    else:
                        assert obj_type_id == 2.0 # corresponding to placing in cup
                        assert obj_being_held == ball
                        next_state.set(ball, "x", act_x)
                        next_state.set(ball, "y", act_y)
                        next_state.set(ball, "held", 0.0)
                        assert self._BallInCup_holds(next_state, [ball, cup])
                        assert self._HandEmpty_holds(next_state, [])
        else:
            # Navigation logic.
            pseudo_next_state = state.copy()
            pseudo_next_state.set(robot, "x", act_x)
            pseudo_next_state.set(robot, "y", act_y)
            if self.exists_robot_collision(pseudo_next_state):
                return next_state
            next_state.set(robot, "x", act_x)
            next_state.set(robot, "y", act_y)
        return next_state

    def _action_grasps_object(self, act_x: float, act_y: float, obj: Object,
                              state: State) -> bool:
        obj_geom = self._object_to_geom(obj, state)
        return obj_geom.contains_point(act_x, act_y)

    def _handle_placing_object(self, act_x: float, act_y: float, state: State,
                               obj_being_held: Object, ball: Object,
                               cup: Object, ball_in_cup: bool) -> State:
        assert ball.type == self._ball_type
        assert cup.type == self._cup_type
        next_state = state.copy()
        next_state.set(obj_being_held, "x", act_x)
        next_state.set(obj_being_held, "y", act_y)
        if ball_in_cup and obj_being_held == cup:
            next_state.set(ball, "x", act_x)
            next_state.set(ball, "y", act_y)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._CubeOnTable, self._CubeOnFloor, self._BallOnTable,
            self._BallOnFloor, self._CupOnTable, self._CupOnFloor,
            self._HoldingCube, self._HoldingBall, self._HoldingCup,
            self._HandEmpty, self._IsReachableSurface, self._IsReachableCube,
            self._IsReachableBall, self._IsReachableCup, self._BallInCup,
            self._BallNotInCup
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._cube_type, self._table_type, self._robot_type,
            self._ball_type, self._cup_type
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._CubeOnTable, self._BallOnTable}

    @property
    def action_space(self) -> Box:
        # Action space is [move_or_pickplace, obj_type_id, x, y].
        # If move_or_pickplace is 0, robot will move to the indicated
        # x, y location.
        # Otherwise, if move_or_pickplace is 1, it will either pick or place
        # the object with obj_type_id the x, y location.
        # obj_type_id 0.0 = cube, 1.0 = ball, 2.0 = cup, 3.0 table
        return Box(
            np.array([0.0, 0.0, self.x_lb, self.y_lb], dtype=np.float32),
            np.array([1.0, 3.0, self.x_ub, self.y_ub], dtype=np.float32))

    @classmethod
    def _object_to_geom(self, obj: Object, state: State) -> utils._Geom2D:
        if obj.is_instance(self._cube_type):
            x = state.get(obj, "x")
            y = state.get(obj, "y")
            size = state.get(obj, "size")
            return utils.Rectangle(x, y, size, size, 0.0)
        assert obj.is_instance(self._table_type) or obj.is_instance(
            self._cup_type) or obj.is_instance(self._ball_type)
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
        # TODO: this is completely broken; need to fix!
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

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            # The goal is to move the cube to some table.
            # The table positions are static
            # The initial location of the cube, the goal
            # and the robot are randomized.
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
                if i >= CFG.sticky_table_num_sticky_tables:
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
            # rng.shuffle(tables)  # type: ignore
            target_table, cube_table, cup_table  = tables[:3]
            # Create cube.
            size = radius * self.objs_scale
            table_x = state_dict[cube_table]["x"]
            table_y = state_dict[cube_table]["y"]
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
                if self._OnTable_holds(state, [cube, cube_table]):
                    break
            # Create cup.
            table_x = state_dict[cup_table]["x"]
            table_y = state_dict[cup_table]["y"]
            while True:
                theta = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, radius)
                x = table_x + dist * np.cos(theta)
                y = table_y + dist * np.sin(theta)
                cup = Object("cup", self._cup_type)
                state_dict[cup] = {
                    "x": x,
                    "y": y,
                    "radius": size +
                    0.05 * size,  # need to make sure cup is bigger than ball
                    "held": 0.0,
                }
                state = utils.create_state_from_dict(state_dict)
                if self._OnTable_holds(state, [cup, cup_table]):
                    break
            # Create ball.
            while True:
                x = rng.uniform(self.x_lb, self.x_ub)
                y = rng.uniform(self.y_lb, self.y_ub)
                ball = Object("ball", self._ball_type)
                state_dict[ball] = {
                    "x": x,
                    "y": y,
                    "radius": size -
                    0.05 * size,  # need to make sure cup is bigger than ball
                    "held": 0.0
                }
                state = utils.create_state_from_dict(state_dict)
                if self._OnFloor_holds(state, [ball]):
                    break
            # Create robot.
            while True:
                x = rng.uniform(self.x_lb, self.x_ub)
                y = rng.uniform(self.y_lb, self.y_ub)
                robot = Object("robot", self._robot_type)
                state_dict[robot] = {
                    "x": x,
                    "y": y,
                }
                state = utils.create_state_from_dict(state_dict)
                if not self.exists_robot_collision(state):
                    break

            goal = {
                GroundAtom(self._CubeOnTable, [cube, target_table]),
                GroundAtom(self._BallOnTable, [ball, target_table])
            }
            task = EnvironmentTask(state, goal)
            tasks.append(task)
        return tasks

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, table = objects
        if self._Holding_holds(state, [obj]):
            return False
        obj_geom = self._object_to_geom(obj, state)
        circ = self._object_to_geom(table, state)
        assert isinstance(circ, utils.Circle)
        if isinstance(obj_geom, utils.Rectangle):
            for x, y in obj_geom.vertices:
                if not circ.contains_point(x, y):
                    return False
            return True
        assert isinstance(obj_geom, utils.Circle)
        return circ.contains_circle(obj_geom)

    def _OnFloor_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        if self._Holding_holds(state, [obj]):
            return False
        for table in state.get_objects(self._table_type):
            if self._OnTable_holds(state, [obj, table]):
                return False
        return True

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "held") > 0.5

    def _HandEmpty_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        assert not objects
        cube, = state.get_objects(self._cube_type)
        ball, = state.get_objects(self._ball_type)
        cup, = state.get_objects(self._cup_type)
        return not (self._Holding_holds(state, [cube]) or self._Holding_holds(
            state, [ball]) or self._Holding_holds(state, [cup]))

    def _IsReachable_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robot, other_obj = objects
        x_squared_dist = (state.get(robot, "x") - state.get(other_obj, "x"))**2
        y_squared_dist = (state.get(robot, "y") - state.get(other_obj, "y"))**2
        curr_dist = np.sqrt((x_squared_dist + y_squared_dist))
        return curr_dist <= self.reachable_thresh

    def _BallInCup_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        ball, cup = objects
        ball_geom = self._object_to_geom(ball, state)
        cup_geom = self._object_to_geom(cup, state)
        assert isinstance(ball_geom, utils.Circle)
        assert isinstance(cup_geom, utils.Circle)
        return cup_geom.contains_circle(ball_geom)

    def _BallNotInCup_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        return not self._BallInCup_holds(state, objects)

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

    @classmethod
    def exists_robot_collision(self, state: State) -> bool:
        """Return true if there is a collision between the robot and any other
        object in the environment."""
        robot, = state.get_objects(self._robot_type)
        all_possible_collision_objs = state.get_objects(
            self._cube_type) + state.get_objects(self._table_type)
        for obj in all_possible_collision_objs:
            obj_geom = self._object_to_geom(obj, state)
            if obj_geom.contains_point(state.get(robot, "x"),
                                       state.get(robot, "y")):
                return True
        return False


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
        if not self._CubeOnFloor_holds(state, [cube]):
            return True
        # Specifically, only succeed if grasp is in upper-right quadrant with
        # respect to the cube's center.
        size = state.get(cube, "size")
        cube_x = state.get(cube, "x") + size / 2
        cube_y = state.get(cube, "y") + size / 2
        return act_x > cube_x and act_y > cube_y
