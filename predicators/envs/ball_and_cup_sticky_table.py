"""Ball and cup with sticky table simulated environment."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class BallAndCupStickyTableEnv(BaseEnv):
    """An environment where a ball must be transported between different
    tables. This environment is a more-complex (but significantly different)
    version of the sticky-table environment.

    Most of the tables are completely flat, but one is half is mostly smooth
    sticky in a particular circular region on the table. If the agent tries
    to place the ball directly on any table, it will roll off with high
    probability. If it tries to place it on a special table,
    the ball will *certainly* roll off. However, if the ball is placed inside
    a cup first, then the ball + cup system stays on the sticky and normal
    table surfaces with high probability.

    Note that unlike almost all of our other environments, there is real
    stochasticity in the outcomes of placing.

    The action space is 5D and slightly complicated: please see the comment
    under the action_space class property below.
    """
    x_lb: ClassVar[float] = 0.0
    x_ub: ClassVar[float] = 1.0
    y_lb: ClassVar[float] = 0.0
    y_ub: ClassVar[float] = 1.0
    reachable_thresh: ClassVar[float] = 0.1
    objs_scale: ClassVar[float] = 0.25  # as a function of table radius
    sticky_region_radius_scale: ClassVar[float] = 0.35
    # Types
    _table_type: ClassVar[Type] = Type("table", [
        "x", "y", "radius", "sticky", "sticky_region_x_offset",
        "sticky_region_y_offset", "sticky_region_radius"
    ])
    _robot_type: ClassVar[Type] = Type("robot", ["x", "y"])
    _ball_type: ClassVar[Type] = Type("ball", ["x", "y", "radius", "held"])
    _cup_type: ClassVar[Type] = Type("cup", ["x", "y", "radius", "held"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # For noisy simulation.
        self._noise_rng = np.random.default_rng(CFG.seed)

        # Predicates
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
        self._HoldingBall = Predicate("HoldingBall", [self._ball_type],
                                      self._Holding_holds)
        self._HoldingCup = Predicate("HoldingCup", [self._cup_type],
                                     self._Holding_holds)
        self._HandEmpty = Predicate("HandEmpty", [], self._HandEmpty_holds)
        self._IsReachableSurface = Predicate(
            "IsReachableSurface", [self._robot_type, self._table_type],
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

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("Rendering not implemented yet!")

    @classmethod
    def get_name(cls) -> str:
        return "ball_and_cup_sticky_table"

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._BallOnTable, self._BallOnFloor, self._CupOnTable,
            self._CupOnFloor, self._HoldingBall, self._HoldingCup,
            self._HandEmpty, self._IsReachableSurface, self._IsReachableBall,
            self._IsReachableCup, self._BallInCup, self._BallNotInCup
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._table_type, self._robot_type, self._ball_type, self._cup_type
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._BallOnTable}

    @property
    def action_space(self) -> Box:
        # Action space is [move_or_pickplace, obj_type_id, ball_only, x, y].
        # If move_or_pickplace is 0, robot will move to the indicated
        # x, y location.
        # Otherwise, if move_or_pickplace is 1, it will either pick or place
        # the object with obj_type_id at the x, y location.
        # obj_type_id 1.0 = ball, 2.0 = cup, 3.0 table
        # The ball_only var is used to handle the case where we're holding the
        # ball and cup and want to only place the ball somewhere.
        return Box(
            np.array([0.0, 0.0, 0.0, self.x_lb, self.y_lb], dtype=np.float32),
            np.array([1.0, 3.0, 1.0, self.x_ub, self.y_ub], dtype=np.float32))

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

    @classmethod
    def _object_to_geom(cls, obj: Object, state: State) -> utils._Geom2D:
        x = state.get(obj, "x")
        y = state.get(obj, "y")
        radius = state.get(obj, "radius")
        return utils.Circle(x, y, radius)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            # The initial location of the the robot is randomized.
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
            size = radius * self.objs_scale
            # Add a random spin to offset the circle. This is to ensure
            # the tables are in different positions along the circle every
            # time.
            sticky_region_radius = radius * self.sticky_region_radius_scale
            # Now, actually instantiate the tables.
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
                sticky_region_dist_from_center = rng.uniform(
                    0.0, radius - sticky_region_radius)
                sticky_region_theta_from_center = rng.uniform(0.0, 2 * np.pi)
                state_dict[obj] = {
                    "x":
                    x,
                    "y":
                    y,
                    "radius":
                    radius,
                    "sticky":
                    sticky,
                    "sticky_region_x_offset":
                    sticky_region_dist_from_center *
                    np.cos(sticky_region_theta_from_center),
                    "sticky_region_y_offset":
                    sticky_region_dist_from_center *
                    np.sin(sticky_region_theta_from_center),
                    "sticky_region_radius":
                    sticky_region_radius
                }
            tables = sorted(state_dict)
            target_table = tables[-1]
            ball_table = tables[0]
            # Create cup and initialize it to be somewhere
            # on the floor.
            while True:
                x = rng.uniform(self.x_lb, self.x_ub)
                y = rng.uniform(self.y_lb, self.y_ub)
                cup = Object("cup", self._cup_type)
                state_dict[cup] = {
                    "x": x,
                    "y": y,
                    "radius": size +
                    0.05 * size,  # need to make sure cup is bigger than ball
                    "held": 0.0,
                }
                state = utils.create_state_from_dict(state_dict)
                if self._OnFloor_holds(state, [cup]):
                    break
            # Create ball and place it delicately balanced atop
            # a table initially. This is intentional: we want the agent
            # to really struggle/be unable to recreate the initial
            # set of atoms.
            table_x = state_dict[ball_table]["x"]
            table_y = state_dict[ball_table]["y"]
            while True:
                theta = rng.uniform(0, 2 * np.pi)
                dist = rng.uniform(0, radius)
                x = table_x + dist * np.cos(theta)
                y = table_y + dist * np.sin(theta)
                ball = Object("ball", self._ball_type)
                state_dict[ball] = {
                    "x": x,
                    "y": y,
                    "radius": size -
                    0.05 * size,  # need to make sure cup is bigger than ball
                    "held": 0.0
                }
                state = utils.create_state_from_dict(state_dict)
                if self._OnTable_holds(state, [ball, ball_table]):
                    break
            # Create robot. Set the robot's pose by randomly sampling
            # valid poses in the room, but ensure that the pose is
            # such that the robot is initially only reachable to 1
            # object (otherwise, the domain is not necessarily
            # reversible given our defined NSRTs).
            while True:
                x = rng.uniform(self.x_lb, self.x_ub)
                y = rng.uniform(self.y_lb, self.y_ub)
                robot = Object("robot", self._robot_type)
                state_dict[robot] = {
                    "x": x,
                    "y": y,
                }
                state = utils.create_state_from_dict(state_dict)
                if not self._invalid_robot_init_pos(state):
                    break

            goal = {GroundAtom(self._BallOnTable, [ball, target_table])}
            task = EnvironmentTask(state, goal)
            tasks.append(task)
        return tasks

    @classmethod
    def exists_robot_collision(cls, state: State) -> bool:
        """Return true if there is a collision between the robot and any other
        object in the environment."""
        robot, = state.get_objects(cls._robot_type)
        all_possible_collision_objs = state.get_objects(
            cls._table_type) + state.get_objects(
                cls._cup_type) + state.get_objects(cls._ball_type)
        for obj in all_possible_collision_objs:
            obj_geom = cls._object_to_geom(obj, state)
            if obj_geom.contains_point(state.get(robot, "x"),
                                       state.get(robot, "y")):
                return True
        return False

    def _invalid_robot_init_pos(self, state: State) -> bool:
        """Return true if the robot position either (1) is in collision or (2)
        is reachable to either no objects, or more than one object (important
        for reversibility of domain)."""
        robot, = state.get_objects(self._robot_type)
        all_possible_collision_objs = state.get_objects(
            self._table_type) + state.get_objects(
                self._cup_type) + state.get_objects(self._ball_type)
        num_objs_reachable = 0
        for obj in all_possible_collision_objs:
            if self._IsReachable_holds(state, [robot, obj]):
                num_objs_reachable += 1
            if num_objs_reachable > 1:
                return True
        if num_objs_reachable != 1:
            return True
        return self.exists_robot_collision(state)

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, table = objects
        if self._Holding_holds(state, [obj]):
            return False
        obj_geom = self._object_to_geom(obj, state)
        circ = self._object_to_geom(table, state)
        assert isinstance(circ, utils.Circle)
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
        ball, = state.get_objects(self._ball_type)
        cup, = state.get_objects(self._cup_type)
        return not (self._Holding_holds(state, [ball])
                    or self._Holding_holds(state, [cup]))

    def _euclidean_reachability_check(self, x1: float, y1: float, x2: float,
                                      y2: float) -> bool:
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) <= self.reachable_thresh

    def _IsReachable_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robot, other_obj = objects
        return self._euclidean_reachability_check(state.get(robot, "x"),
                                                  state.get(robot, "y"),
                                                  state.get(other_obj, "x"),
                                                  state.get(other_obj, "y"))

    def _BallInCup_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        ball, cup = objects
        ball_geom = self._object_to_geom(ball, state)
        cup_geom = self._object_to_geom(cup, state)
        assert isinstance(ball_geom, utils.Circle)
        assert isinstance(cup_geom, utils.Circle)
        ball_and_cup_at_same_pos = cup_geom.contains_circle(ball_geom)
        holding_ball = self._Holding_holds(state, [ball])
        holding_cup = self._Holding_holds(state, [cup])
        return ball_and_cup_at_same_pos and (
            (holding_ball and holding_cup) or
            (not holding_ball and not holding_cup))

    def _BallNotInCup_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        return not self._BallInCup_holds(state, objects)

    def _table_is_sticky(self, table: Object, state: State) -> bool:
        return state.get(table, "sticky") > 0.5

    def simulate(self, state: State, action: Action) -> State:
        # NOTE: noise is added here. Two calls to simulate with the same
        # inputs may produce different outputs!
        assert self.action_space.contains(action.arr)
        move_or_pickplace, obj_type_id, ball_only, act_x, act_y = action.arr
        next_state = state.copy()
        hand_empty = self._HandEmpty_holds(state, [])
        ball, = state.get_objects(self._ball_type)
        cup, = state.get_objects(self._cup_type)
        robot, = state.get_objects(self._robot_type)
        ball_held = self._Holding_holds(state, [ball])
        cup_held = self._Holding_holds(state, [cup])
        ball_in_cup = self._BallInCup_holds(state, [ball, cup])
        obj_being_held: Optional[Object] = None
        if (ball_held and not cup_held) or ball_only > 0.5:
            obj_being_held = ball
        elif cup_held:
            obj_being_held = cup
        # In this case, handle picking/placing.
        if move_or_pickplace == 1.0:
            # Picking logic.
            if hand_empty:
                # Fail sometimes.
                if self._noise_rng.uniform() < self._pick_success_prob:
                    if obj_type_id == 1.0:
                        # Pick ball.
                        if self._action_grasps_object(act_x, act_y, ball,
                                                      state):
                            next_state.set(ball, "held", 1.0)
                        assert self._Holding_holds(next_state, [ball])
                    else:
                        assert obj_type_id == 2.0
                        if self._action_grasps_object(act_x, act_y, cup,
                                                      state):
                            # Pick cup.
                            next_state.set(cup, "held", 1.0)
                            if ball_in_cup:
                                # Pick both ball and cup simultaneously.
                                next_state.set(ball, "held", 1.0)
                                assert self._Holding_holds(next_state, [ball])
                            assert self._Holding_holds(next_state, [cup])
            # Placing logic.
            else:
                if not hand_empty:
                    assert obj_being_held is not None
                    # Find the table for placing, if any.
                    table: Optional[Object] = None
                    for target in state.get_objects(self._table_type):
                        circ = self._object_to_geom(target, state)
                        if circ.contains_point(act_x, act_y):
                            table = target
                            break
                    if table is None:
                        # Put on the floor at the commanded position.
                        next_state = self._handle_placing_object(
                            act_x, act_y, next_state, obj_being_held, ball,
                            cup, ball_in_cup, ball_only)
                        # Release object being held.
                        if obj_being_held is not None:
                            next_state.set(obj_being_held, "held", 0.0)
                        assert self._OnFloor_holds(next_state,
                                                   [obj_being_held])
                    else:
                        # Check that we are only attempting to place
                        # within our reachable radius. Note that we don't
                        # check this for placing on the floor, because the
                        # robot is allowed to 'throw' things onto the floor.
                        table_x = state.get(table, "x")
                        table_y = state.get(table, "y")
                        if self._euclidean_reachability_check(
                                state.get(robot, "x"), state.get(robot, "y"),
                                table_x, table_y):
                            # Release object being held.
                            if obj_being_held is not None:
                                next_state.set(obj_being_held, "held", 0.0)
                            if obj_type_id == 3.0:
                                # Possibly put on the table, or have it fall
                                # somewhere near.
                                fall_prob = self._place_sticky_fall_prob
                                if obj_being_held == ball:
                                    fall_prob = self._place_ball_fall_prob
                                if self._table_is_sticky(table, state):
                                    # Check if placing on the smooth part of
                                    # the sticky table, and set fall prob
                                    # accordingly.
                                    sticky_region_x = state.get(
                                        table,
                                        "sticky_region_x_offset") + table_x
                                    sticky_region_y = state.get(
                                        table,
                                        "sticky_region_y_offset") + table_y
                                    sticky_region = utils.Circle(
                                        sticky_region_x, sticky_region_y,
                                        state.get(table,
                                                  "sticky_region_radius"))
                                    if not sticky_region.contains_point(
                                            act_x, act_y):
                                        if obj_being_held == cup:
                                            fall_prob = \
                                                self._place_smooth_fall_prob
                                        else:
                                            assert obj_being_held == ball
                                            fall_prob = 1.0
                                # Handle object falling or placing on table
                                # surface.
                                if self._noise_rng.uniform() < fall_prob:
                                    fall_x, fall_y = \
                                        self._sample_floor_point_around_table(
                                        table, state, self._noise_rng)
                                    next_state = self._handle_placing_object(
                                        fall_x, fall_y, next_state,
                                        obj_being_held, ball, cup, ball_in_cup,
                                        ball_only)
                                    assert self._OnFloor_holds(
                                        next_state, [obj_being_held])
                                else:
                                    next_state = self._handle_placing_object(
                                        act_x, act_y, next_state,
                                        obj_being_held, ball, cup, ball_in_cup,
                                        ball_only)
                                    assert self._OnTable_holds(
                                        next_state, [obj_being_held, table])
                            else:
                                # corresponding to placing in cup
                                assert obj_type_id == 2.0
                                assert obj_being_held == ball
                                next_state.set(ball, "x", act_x)
                                next_state.set(ball, "y", act_y)
                                next_state.set(ball, "held", 0.0)
                                assert self._BallInCup_holds(
                                    next_state, [ball, cup])
                            if ball_only < 0.5:
                                assert self._HandEmpty_holds(next_state, [])
        else:
            # Navigation logic.
            pseudo_next_state = state.copy()
            pseudo_next_state.set(robot, "x", act_x)
            pseudo_next_state.set(robot, "y", act_y)
            if self.exists_robot_collision(pseudo_next_state):
                return next_state  # pragma: no cover
            next_state.set(robot, "x", act_x)
            next_state.set(robot, "y", act_y)
        return next_state

    def _action_grasps_object(self, act_x: float, act_y: float, obj: Object,
                              state: State) -> bool:
        obj_geom = self._object_to_geom(obj, state)
        return obj_geom.contains_point(act_x, act_y)

    def _handle_placing_object(self, act_x: float, act_y: float, state: State,
                               obj_being_held: Object, ball: Object,
                               cup: Object, ball_in_cup: bool,
                               ball_only: float) -> State:
        """Logic for correctly setting the location of the held object after
        executing the place skill."""
        assert ball.type == self._ball_type
        assert cup.type == self._cup_type
        next_state = state.copy()
        next_state.set(obj_being_held, "x", act_x)
        next_state.set(obj_being_held, "y", act_y)
        next_state.set(obj_being_held, "held", 0.0)
        if ball_in_cup and obj_being_held == cup and ball_only < 0.5:
            next_state.set(ball, "x", act_x)
            next_state.set(ball, "y", act_y)
            next_state.set(ball, "held", 0.0)
        return next_state

    def _sample_floor_point_around_table(
            self, table: Object, state: State,
            rng: np.random.Generator) -> Tuple[float, float]:
        x = state.get(table, "x")
        y = state.get(table, "y")
        radius = state.get(table, "radius")
        dist_from_table = self.objs_scale * radius
        dist = radius + rng.uniform(radius + dist_from_table, radius +
                                    (1.15 * dist_from_table))
        theta = rng.uniform(0, 2 * np.pi)
        sampled_x = x + dist * np.cos(theta)
        sampled_y = y + dist * np.sin(theta)
        while sampled_x < self.x_lb or sampled_x > self.x_ub or \
                sampled_y < self.y_lb or sampled_y > self.y_ub:
            dist = radius + rng.uniform(radius + dist_from_table, radius +
                                        (1.15 * dist_from_table))
            theta = rng.uniform(0, 2 * np.pi)
            sampled_x = x + dist * np.cos(theta)
            sampled_y = y + dist * np.sin(theta)
        return (sampled_x, sampled_y)
