from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, \
    Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class SamplerVizEnv(BaseEnv):
    """SamplerViz domain."""

    env_x_lb: ClassVar[float] = 0
    env_y_lb: ClassVar[float] = 0
    env_x_ub: ClassVar[float] = 20#12#
    env_y_ub: ClassVar[float] = 20#12#
    robot_radius: ClassVar[float] = 2
    gripper_length: ClassVar[float] = 2
    shelf_w: ClassVar[float] = 7
    shelf_h: ClassVar[float] = 3
    book_w_lb: ClassVar[float] = 0.5
    book_w_ub: ClassVar[float] = 1
    book_h_lb: ClassVar[float] = 1
    book_h_ub: ClassVar[float] = 1.5
    _shelf_color: ClassVar[List[float]] = [0.89, 0.82, 0.68]
    _robot_color: ClassVar[List[float]] = [0.5, 0.5, 0.5]

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._book_type = Type(
            "book", ["pose_x", "pose_y", "width", "height", "yaw", "held"])
        self._shelf_type = Type("shelf",
                                ["pose_x", "pose_y", "width", "height", "yaw", "held"])
        self._goal_type = Type("goal", ["x", "y"])
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "yaw", "gripper_free"])
        # Predicates
        self._CanReach = Predicate("CanReach",
                                  [self._shelf_type, self._robot_type],
                                  self._CanReach_holds)
        self._Holding = Predicate("Holding", [self._shelf_type],
                                  self._Holding_holds)
        self._GripperFree = Predicate("GripperFree", [self._robot_type],
                                      self._GripperFree_holds)
        self._OnGoal = Predicate("OnGoal", [self._shelf_type], self._OnGoal_holds)

        # Options
        lo = [-8, -4]
        hi = [9, 5]
        self._NavigateTo = utils.SingletonParameterizedOption(
            # variables: [robot, object to navigate to]
            # params: [offset_x, offset_y]
            "NavigateTo",
            self._NavigateTo_policy,
            types=[self._robot_type, self._shelf_type],
            params_space=Box(np.array(lo, dtype=np.float32),
                             np.array(hi, dtype=np.float32)))
        lo = []
        hi = []
        self._PickShelf = utils.SingletonParameterizedOption(
            # variables: [robot, book to pick]
            # params: [offset_gripper, book_yaw]
            "PickShelf",
            self._PickShelf_policy,
            types=[self._robot_type, self._shelf_type],
            params_space=Box(np.array(lo, dtype=np.float32),
                             np.array(hi, dtype=np.float32)))
        lo = [0]
        hi = [20]
        self._PushShelf = utils.SingletonParameterizedOption(
            # variables: [robot, book, shelf]
            # params: [offset_gripper]
            "PushShelf",
            self._PushShelf_policy,
            types=[self._robot_type, self._shelf_type],
            params_space=Box(np.array(lo, dtype=np.float32),
                             np.array(hi, dtype=np.float32)))
        # Static objects (always exist no matter the settings)
        self._shelf = Object("shelf", self._shelf_type)
        self._robot = Object("robby", self._robot_type)
        self._goal = Object("goal", self._goal_type)

    @classmethod
    def get_name(cls) -> str:
        return "sampler_viz"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        arr = action.arr
        if arr[-1] < -0.33:
            transition_fn = self._transition_pick_shelf
        elif -0.33 <= arr[-1] < 0.33:
            transition_fn = self._transition_push_shelf
        elif 0.33 <= arr[-1]:
            transition_fn = self._transition_navigate_to
        return transition_fn(state, action)

    def _transition_pick_shelf(self, state: State, action: Action) -> State:
        offset_gripper = action.arr[3]
        next_state = state.copy()

        shelf = self._shelf
        held_shelf = state.get(shelf, "held")
        if held_shelf:
            return next_state

        robby = self._robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 1.0:
            return next_state

        tip_x = robby_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)

        shelf_is_pickable, shelf_rect = self._get_pickable_shelf(state, tip_x, tip_y)

        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        ignore_objects = {shelf, robby}
        # Check that there is a graspable book and that there are no gripper collisions
        if not shelf_is_pickable or self.check_collision(state, gripper_line,
                                                     ignore_objects):
            return next_state
        # Execute pick
        rel_x = shelf_rect.x - tip_x
        rel_y = shelf_rect.y - tip_y
        next_state.set(shelf, "held", 1.0)
        next_state.set(shelf, "pose_x", rel_x)
        next_state.set(shelf, "pose_y", rel_y)
        next_state.set(shelf, "yaw", -np.pi / 2)
        next_state.set(robby, "gripper_free", 0.0)
        return next_state

    def _transition_push_shelf(self, state: State, action: Action) -> State:
        displacement = action.arr[0]
        offset_gripper = 1
        next_state = state.copy()

        shelf = self._shelf
        held = state.get(shelf, "held")
        if not held:
            return next_state
        shelf_relative_x = state.get(shelf, "pose_x")
        shelf_relative_y = state.get(shelf, "pose_y")
        shelf_relative_yaw = state.get(shelf, "yaw")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")

        robby = self._robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        new_x = robby_x + displacement * np.cos(robby_yaw)
        new_y = robby_y + displacement * np.sin(robby_yaw)

        tip_x = new_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = new_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)

        # TODO: this needs to be wrt the shelf and the shelf yaw needs to be properly set (to 0? pi/2?)
        place_x = tip_x + shelf_relative_x * np.sin(
            robby_yaw) + shelf_relative_y * np.cos(robby_yaw)
        place_y = tip_y + shelf_relative_y * np.sin(
            robby_yaw) - shelf_relative_x * np.cos(robby_yaw)

        place_yaw = shelf_relative_yaw + robby_yaw
        while place_yaw > np.pi:
            place_yaw -= (2 * np.pi)
        while place_yaw < -np.pi:
            place_yaw += (2 * np.pi)

        # Some collision checking
        next_state.set(shelf, "held", 0.0)
        next_state.set(shelf, "pose_x", place_x)
        next_state.set(shelf, "pose_y", place_y)
        next_state.set(shelf, "yaw", place_yaw)
        next_state.set(robby, "gripper_free", 1.0)
        next_state.set(robby, "pose_x", new_x)
        next_state.set(robby, "pose_y", new_y)
        return next_state

    def _transition_navigate_to(self, state: State, action: Action) -> State:
        pos_x, pos_y, yaw = action.arr[:3]
        next_state = state.copy()

        robby = self._robot
        robby_geom = utils.Circle(pos_x, pos_y, self.robot_radius)
        ignore_objects = {robby}
        if self.check_collision(state, robby_geom, ignore_objects):
            return next_state

        next_state.set(robby, "pose_x", pos_x)
        next_state.set(robby, "pose_y", pos_y)
        next_state.set(robby, "yaw", yaw)
        return next_state

    @property
    def _num_obstacles_train(self) -> List[int]:
        return CFG.sampler_viz_num_obstacles_train

    @property
    def _num_obstacles_test(self) -> List[int]:
        return CFG.sampler_viz_num_obstacles_test

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.bookshelf_train_tasks_overwrite if CFG.bookshelf_train_tasks_overwrite is not None else CFG.num_train_tasks,
                               possible_num_obstacles=self._num_obstacles_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               possible_num_obstacles=self._num_obstacles_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._OnGoal, self._CanReach, self._Holding, self._GripperFree}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._OnGoal}

    @property
    def types(self) -> Set[Type]:
        return {
            self._book_type, self._shelf_type, self._robot_type, self._goal_type
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._NavigateTo, self._PickShelf, self._PushShelf}

    @property
    def action_space(self) -> Box:
        # Actions are 5-dimensional vectors:
        # [x, y, yaw, offset_gripper, pick_place_navigate]
        # yaw is only used for pick actions
        # pick_place_navigate is -1 for pick, 0 for push, 1 for navigate
        lowers = np.array([self.env_x_lb, self.env_y_lb, -np.pi, 0.0, -1.0],
                          dtype=np.float32)
        uppers = np.array([self.env_x_ub, self.env_y_ub, np.pi, 1.0, 1.0],
                          dtype=np.float32)
        return Box(lowers, uppers)

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        books = [b for b in state if b.is_instance(self._book_type)]
        
        shelf = self._shelf
        robby = self._robot
        goal = self._goal
        
        # Draw robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free = state.get(robby, "gripper_free") == 1.0
        circ = utils.Circle(robby_x, robby_y, self.robot_radius)
        circ.plot(ax, facecolor=self._robot_color)
        offset_gripper = 0  # if gripper_free else 1
        tip_x = robby_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        gripper_line.plot(ax, color="white")

        # Draw shelf
        shelf_x = state.get(shelf, "pose_x")
        shelf_y = state.get(shelf, "pose_y")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")
        shelf_yaw = state.get(shelf, "yaw")
        holding = state.get(shelf, "held") == 1.0
        if holding:
            aux_x = tip_x + shelf_x * np.sin(robby_yaw) + shelf_y * np.cos(robby_yaw)
            aux_y = tip_y + shelf_y * np.sin(robby_yaw) - shelf_x * np.cos(robby_yaw)
            shelf_x = aux_x
            shelf_y = aux_y
            shelf_yaw += robby_yaw
        while shelf_yaw > np.pi:
            shelf_yaw -= (2 * np.pi)
        while shelf_yaw < -np.pi:
            shelf_yaw += (2 * np.pi)
        rect = utils.Rectangle(shelf_x, shelf_y, shelf_w, shelf_h, shelf_yaw)
        rect.plot(ax, facecolor=self._shelf_color)

        # Draw books
        for b in sorted(books):
            x = state.get(b, "pose_x")
            y = state.get(b, "pose_y")
            w = state.get(b, "width")
            h = state.get(b, "height")
            yaw = state.get(b, "yaw")
            holding = state.get(b, "held") == 1.0
            fc = "gray"
            ec = "gray"
            while yaw > np.pi:
                yaw -= (2 * np.pi)
            while yaw < -np.pi:
                yaw += (2 * np.pi)
            rect = utils.Rectangle(x, y, w, h, yaw)
            rect.plot(ax, facecolor=fc, edgecolor=ec)

        # Draw goal
        goal_x = state.get(goal, "x")
        goal_y = state.get(goal, "y")
        circle = utils.Circle(goal_x, goal_y, 0.1)
        circle.plot(ax, facecolor="red")

        ax.set_xlim(self.env_x_lb, self.env_x_ub)
        ax.set_ylim(self.env_y_lb, self.env_y_ub)
        plt.suptitle(caption, fontsize=12, wrap=True)
        plt.tight_layout()
        plt.axis("off")
        return fig

    def _get_tasks(self, num_tasks: int, possible_num_obstacles: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for i in range(num_tasks):
            num_obstacles = rng.choice(possible_num_obstacles)
            data = {}

            # Fixed shelf state
            shelf_w = self.shelf_w
            shelf_h = self.shelf_h
            shelf_x = 6.5
            shelf_y = 3
            shelf_yaw = 0
            data[self._shelf] = np.array(
                [shelf_x, shelf_y, shelf_w, shelf_h, shelf_yaw, 0.0])

            tmp_state = State(data)
            # Initialize robot pos
            robot_collision = True
            while robot_collision:
                robot_init_x = rng.uniform(self.env_x_lb, self.env_x_ub)
                robot_init_y = rng.uniform(self.env_y_lb, self.env_y_ub)
                robot_init_yaw = rng.uniform(-np.pi, np.pi)
                robot_circ = utils.Circle(robot_init_x, robot_init_y,
                                          self.robot_radius)
                robot_collision = self.check_collision(tmp_state, robot_circ)

            gripper_free = 1.0
            data[self._robot] = np.array(
                [robot_init_x, robot_init_y, robot_init_yaw, gripper_free])

            # Sample goal location
            goal_x = rng.uniform(shelf_x, shelf_x + shelf_w)
            goal_y = rng.uniform(shelf_y + shelf_y, self.env_y_ub)
            goal_obj = Object("goal", self._goal_type)
            data[goal_obj] = np.array([goal_x, goal_y], dtype=np.float32)
            if CFG.sampler_viz_singlestep_goal:
                goal = {GroundAtom(self._CanReach, [self._shelf, self._robot])}
            else:
                goal = {GroundAtom(self._OnGoal, [self._shelf])}

            for j in range(num_obstacles):
                obstacle_collision = True
                tmp_state = State(data)
                ignore_objects = {self._shelf}   # allow obstacles to be placed on shelf
                while obstacle_collision:
                    obstacle_init_x = rng.uniform(self.env_x_lb, self.env_x_ub)
                    obstacle_init_y = rng.uniform(self.env_y_lb, self.env_y_ub)
                    obstacle_init_yaw = rng.uniform(-np.pi, np.pi)
                    obstacle_width = rng.uniform(self.book_w_lb, self.book_w_ub)
                    obstacle_height = rng.uniform(self.book_h_lb, self.book_h_ub)
                    obstacle_rect = utils.Rectangle(obstacle_init_x, obstacle_init_y,
                                                obstacle_width, obstacle_height,
                                                obstacle_init_yaw)
                    obstacle_collision = self.check_collision(tmp_state, obstacle_rect,
                                                              ignore_objects)
                obstacle = Object(f"book{j}", self._book_type)
                held = 0.0
                data[obstacle] = np.array([
                    obstacle_init_x, obstacle_init_y, obstacle_width, obstacle_height,
                    obstacle_init_yaw, held
                ],
                                      dtype=np.float32)

            state = State(data)
            tasks.append(Task(state, goal))
        return tasks

    def check_collision(self,
                        state: State,
                        geom: utils._Geom2D,
                        ignore_objects: Optional[Set[Object]] = None) -> bool:
        if ignore_objects is None:
            ignore_objects = set()
        for obj in state.data:
            if obj in ignore_objects:
                continue
            if obj.is_instance(self._book_type) or obj.is_instance(
                    self._shelf_type):
                x = state.get(obj, "pose_x")
                y = state.get(obj, "pose_y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "yaw")
                while theta > np.pi:
                    theta -= (2 * np.pi)
                while theta < -np.pi:
                    theta += (2 * np.pi)
                obj_geom = utils.Rectangle(x, y, width, height, theta)
            elif obj.is_instance(self._robot_type):
                x = state.get(obj, "pose_x")
                y = state.get(obj, "pose_y")
                obj_geom = utils.Circle(x, y, self.robot_radius)
            else:
                assert obj.type.name in ['dummy', "goal"]
                continue
            if utils.geom2ds_intersect(geom, obj_geom):
                return True
        return False

    '''
    Note: the pick policy takes a parameter delta, representing how far to stretch the
    gripper (between 0, 1 relative to the gripper size), and a parameter yaw, representing
    the relative orientation of the book once gripped. 
    '''

    def _PickShelf_policy(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> Action:
        del memory  # unused
        robby, shelf = objects
        shelf_x = state.get(shelf, "pose_x")
        shelf_y = state.get(shelf, "pose_y")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")
        shelf_yaw = state.get(shelf, "yaw")

        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")

        arr = np.array([robby_x, robby_y, shelf_yaw, 1.0, -1],
                       dtype=np.float32)
        return Action(arr)

    '''
    Note: the place policy takes a single parameter delta, representing how far to stretch
    the gripper, between 0 and 1 relative to the gripper size. 
    '''

    def _PushShelf_policy(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        del memory  # unused
        robby, shelf = objects

        shelf_x = state.get(shelf, "pose_x")
        shelf_y = state.get(shelf, "pose_y")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")
        shelf_yaw = state.get(shelf, "yaw")

        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")

        displacement, = params

        arr = np.array([displacement, 0.0, 0.0, 0.0, 0],
                       dtype=np.float32)
        return Action(arr)

    '''
    Note: the navigation policy takes xy parameters between -.5 and 1.5 representing
    the relative placement wrt the object, normalized by the object's dimensions.
    '''

    def _NavigateTo_policy(self, state: State, memory: Dict,
                           objects: Sequence[Object], params: Array) -> Action:
        del memory  # unused
        robby, obj = objects
        obj_x = state.get(obj, "pose_x")
        obj_y = state.get(obj, "pose_y")
        obj_w = state.get(obj, "width")
        obj_h = state.get(obj, "height")
        obj_yaw = state.get(obj, "yaw")
        if CFG.bookshelf_add_sampler_idx_to_params:
            _, offset_x, offset_y = params
        else:
            offset_x, offset_y = params

        pos_x = obj_x + obj_w * offset_x * np.cos(obj_yaw) - \
                obj_h * offset_y * np.sin(obj_yaw)
        pos_y = obj_y + obj_w * offset_x * np.sin(obj_yaw) + \
                obj_h * offset_y * np.cos(obj_yaw)

        if offset_x < 0 and 0 <= offset_y <= 1:
            yaw = 0
        elif offset_x > 1 and 0 <= offset_y <= 1:
            yaw = -np.pi
        elif 0 <= offset_x <= 1 and offset_y < 0:
            yaw = np.pi / 2
        elif 0 <= offset_x <= 1 and offset_y > 1:
            yaw = -np.pi / 2
        elif 0 <= offset_x <= 1 and 0 <= offset_y <= 1:
            # Collision with object; will fail, so set any value
            yaw = 0
        else:
            x = offset_x - 1 if offset_x > 1 else offset_x
            y = offset_y - 1 if offset_y > 1 else offset_y
            yaw = np.arctan2(-y * obj_h, -x * obj_w)
        yaw += obj_yaw
        while yaw > np.pi:
            yaw -= (2 * np.pi)
        while yaw < -np.pi:
            yaw += (2 * np.pi)
        arr = np.array([pos_x, pos_y, yaw, 0.0, 1.0], dtype=np.float32)
        # pos_x and pos_y might take the robot out of the env bounds, so we clip
        # the action back into its bounds for safety.
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        shelf, = objects
        return state.get(shelf, "held") == 1.0

    def _OnGoal_holds(self, state: State, objects: Sequence[Object]) -> bool:
        shelf, = objects

        shelf_x = state.get(shelf, "pose_x")
        shelf_y = state.get(shelf, "pose_y")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")
        shelf_yaw = state.get(shelf, "yaw")

        shelf_rect = utils.Rectangle(shelf_x, shelf_y, shelf_w, shelf_h,
                                     shelf_yaw)

        goal = self._goal
        goal_x = state.get(self._goal, "x")
        goal_y = state.get(self._goal, "y")

        shelf_rect = utils.Rectangle(shelf_x, shelf_y, shelf_w, shelf_h,
                                     shelf_yaw)
        return shelf_rect.contains_point(goal_x, goal_y)

    def _GripperFree_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robby, = objects
        return state.get(robby, "gripper_free") == 1.0

    def _CanReach_holds(self, state: State,
                        objects: Sequence[Object]) -> bool:
        obj, robby = objects
        if obj.is_instance(self._shelf_type) and self._Holding_holds(state, [obj]):
            return False
        if obj.is_instance(self._book_type) or obj.is_instance(self._shelf_type):
            x = state.get(obj, "pose_x")
            y = state.get(obj, "pose_y")
            width = state.get(obj, "width")
            height = state.get(obj, "height")
            theta = state.get(obj, "yaw")
            while theta > np.pi:
                theta -= (2 * np.pi)
            while theta < -np.pi:
                theta += (2 * np.pi)
            obj_geom = utils.Rectangle(x, y, width, height, theta)
        else:
            raise ValueError("Can only compute reachable for books and shelves")

        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        tip_x = robby_x + (self.robot_radius + self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + self.gripper_length) * np.sin(robby_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        return utils.geom2ds_intersect(gripper_line, obj_geom)


    def _get_pickable_shelf(self, state: State, tip_x: float, tip_y: float):
        shelf = self._shelf

        x = state.get(shelf, "pose_x")
        y = state.get(shelf, "pose_y")
        width = state.get(shelf, "width")
        height = state.get(shelf, "height")
        theta = state.get(shelf, "yaw")
        while theta > np.pi:
            theta -= (2 * np.pi)
        while theta < -np.pi:
            theta += (2 * np.pi)
        shelf_rect = utils.Rectangle(x, y, width, height, theta)
        if shelf_rect.contains_point(tip_x, tip_y):
            return True, shelf_rect
        return False, shelf_rect
