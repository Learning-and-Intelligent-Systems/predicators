from typing import Any, ClassVar, Dict, List, Optional, Sequence, Set, Tuple, \
    Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators import utils
from predicators.envs import BaseEnv
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class BookShelf(BaseEnv):
    """Bookshelf domain."""

    env_x_lb: ClassVar[float] = 0
    env_y_lb: ClassVar[float] = 0
    env_x_ub: ClassVar[float] = 20
    env_y_ub: ClassVar[float] = 20
    robot_radius: ClassVar[float] = 2
    gripper_length: ClassVar[float] = 2
    shelf_w_lb: ClassVar[float] = 5
    shelf_w_ub: ClassVar[float] = 10
    shelf_h_lb: ClassVar[float] = 2
    shelf_h_ub: ClassVar[float] = 5
    book_w_lb: ClassVar[float] = 0.5
    book_w_ub: ClassVar[float] = 1
    book_h_lb: ClassVar[float] = 1
    book_h_ub: ClassVar[float] = 1.5
    _shelf_color: ClassVar[List[float]] = [0.89, 0.82, 0.68]
    _robot_color: ClassVar[List[float]] = [0.5, 0.5, 0.5]

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._object_type = Type("object", [])
        # Book: when held, pose becomes relative to gripper
        self._book_type = Type(
            "book", ["pose_x", "pose_y", "width", "height", "yaw", "held"],
            self._object_type)
        self._shelf_type = Type("shelf",
                                ["pose_x", "pose_y", "width", "height", "yaw"],
                                self._object_type)
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "yaw", "gripper_free"])
        # Predicates
        self._OnShelf = Predicate("OnShelf",
                                  [self._book_type, self._shelf_type],
                                  self._OnShelf_holds)
        self._Holding = Predicate("Holding", [self._book_type],
                                  self._Holding_holds)
        self._GripperFree = Predicate("GripperFree", [self._robot_type],
                                      self._GripperFree_holds)
        # Options
        self._NavigateTo = utils.SingletonParameterizedOption(
            # variables: [robot, object to navigate to]
            # params: [offset_x, offset_y]
            "NavigateTo",
            self._NavigateTo_policy,
            types=[self._robot_type, self._object_type],
            params_space=Box(np.array([-4, -4], dtype=np.float32),
                             np.array([5, 5], dtype=np.float32)))
        self._PickBook = utils.SingletonParameterizedOption(
            # variables: [robot, book to pick]
            # params: [offset_gripper, book_yaw]
            "PickBook",
            self._PickBook_policy,
            types=[self._robot_type, self._book_type],
            params_space=Box(np.array([0.0, -np.pi], dtype=np.float32),
                             np.array([1.0, np.pi], dtype=np.float32)))
        self._PlaceBookOnShelf = utils.SingletonParameterizedOption(
            # variables: [robot, book, shelf]
            # params: [offset_gripper]
            "PlaceBookOnShelf",
            self._PlaceBookOnShelf_policy,
            types=[self._robot_type, self._book_type, self._shelf_type],
            params_space=Box(np.array([0.0], dtype=np.float32),
                             np.array([1.0], dtype=np.float32)))
        # Static objects (always exist no matter the settings)
        self._shelf = Object("shelf", self._shelf_type)
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "bookshelf"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        arr = action.arr
        if arr[-1] < -0.33:
            transition_fn = self._transition_pick_book
        elif -0.33 <= arr[-1] < 0.33:
            transition_fn = self._transition_place_book_on_shelf
        elif 0.33 <= arr[-1]:
            transition_fn = self._transition_navigate_to
        return transition_fn(state, action)

    def _transition_pick_book(self, state: State, action: Action) -> State:
        yaw, offset_gripper = action.arr[2:4]
        next_state = state.copy()
        held_book = self._get_held_book(state)
        if held_book is not None:
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

        pick_book, book_rect = self._get_pickable_book(state, tip_x, tip_y)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        ignore_objects = {pick_book, robby}
        # Check that there is a graspable book and that there are no gripper collisions
        if pick_book is None or self.check_collision(state, gripper_line,
                                                     ignore_objects):
            return next_state
        # Execute pick
        book_rect = book_rect.rotate_about_point(tip_x, tip_y, yaw)
        yaw = book_rect.theta - robby_yaw
        if yaw > np.pi:
            yaw -= (2 * np.pi)
        elif yaw < -np.pi:
            yaw += (2 * np.pi)
        rel_x = book_rect.x - tip_x
        rel_y = book_rect.y - tip_y
        rot_rel_x = rel_x * np.sin(robby_yaw) - rel_y * np.cos(robby_yaw)
        rot_rel_y = rel_x * np.cos(robby_yaw) + rel_y * np.sin(robby_yaw)
        next_state.set(pick_book, "held", 1.0)
        next_state.set(pick_book, "pose_x", rot_rel_x)
        next_state.set(pick_book, "pose_y", rot_rel_y)
        next_state.set(pick_book, "yaw", yaw)
        next_state.set(robby, "gripper_free", 0.0)
        return next_state

    def _transition_place_book_on_shelf(self, state: State,
                                        action: Action) -> State:
        offset_gripper = action.arr[3]
        next_state = state.copy()

        book = self._get_held_book(state)
        if book is None:
            return next_state
        book_relative_x = state.get(book, "pose_x")
        book_relative_y = state.get(book, "pose_y")
        book_relative_yaw = state.get(book, "yaw")
        book_w = state.get(book, "width")
        book_h = state.get(book, "height")

        robby = self._robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        shelf = self._shelf
        shelf_x = state.get(shelf, "pose_x")
        shelf_y = state.get(shelf, "pose_y")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")
        shelf_yaw = state.get(shelf, "yaw")

        tip_x = robby_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)

        place_x = tip_x + book_relative_x * np.sin(
            robby_yaw) + book_relative_y * np.cos(robby_yaw)
        place_y = tip_y + book_relative_y * np.sin(
            robby_yaw) - book_relative_x * np.cos(robby_yaw)
        place_yaw = book_relative_yaw + robby_yaw
        if place_yaw > np.pi:
            place_yaw -= (2 * np.pi)
        elif place_yaw < -np.pi:
            place_yaw += (2 * np.pi)

        # Check whether the books center-of-mass is within the shelf bounds
        # and that there are no other collisions with the book or gripper
        book_rect = utils.Rectangle(place_x, place_y, book_w, book_h,
                                    place_yaw)
        shelf_rect = utils.Rectangle(shelf_x, shelf_y, shelf_w, shelf_h,
                                     shelf_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        ignore_objects = {book, robby, shelf}
        if not shelf_rect.contains_point(*(book_rect.center)) or \
            self.check_collision(state, book_rect, ignore_objects) or \
            self.check_collision(state, gripper_line, ignore_objects):
            return next_state

        next_state.set(book, "held", 0.0)
        next_state.set(book, "pose_x", place_x)
        next_state.set(book, "pose_y", place_y)
        next_state.set(book, "yaw", place_yaw)
        next_state.set(robby, "gripper_free", 1.0)
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
    def _num_books_train(self) -> List[int]:
        return CFG.bokshelf_num_books_train

    @property
    def _num_books_test(self) -> List[int]:
        return CFG.num_books_test

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               num_books_lst=self._num_books_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               num_books_lst=self._num_books_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._OnShelf, self._Holding}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._OnShelf}

    @property
    def types(self) -> Set[Type]:
        return {
            self._object_type, self._book_type, self._shelf_type,
            self._robot_type
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._NavigateTo, self._PickBook, self._PlaceBookOnShelf}

    @property
    def action_space(self) -> Box:
        # Actions are 5-dimensional vectors:
        # [x, y, yaw, offset_gripper, pick_place_navigate]
        # yaw is only used for pick actions
        # pick_place_navigate is -1 for pick, 0 for place, 1 for navigate
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
        # Draw shelf
        shelf_x = state.get(shelf, "pose_x")
        shelf_y = state.get(shelf, "pose_y")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")
        shelf_yaw = state.get(shelf, "yaw")
        rect = utils.Rectangle(shelf_x, shelf_y, shelf_w, shelf_h, shelf_yaw)
        rect.plot(ax, facecolor=self._shelf_color)

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
        gripper_line.plot(ax)

        # Draw books
        for b in sorted(books):
            x = state.get(b, "pose_x")
            y = state.get(b, "pose_y")
            w = state.get(b, "width")
            h = state.get(b, "height")
            yaw = state.get(b, "yaw")
            holding = state.get(b, "held") == 1.0
            fc = "blue"
            ec = "red" if holding else "black"
            if holding:
                aux_x = tip_x + x * np.sin(robby_yaw) + y * np.cos(robby_yaw)
                aux_y = tip_y + y * np.sin(robby_yaw) - x * np.cos(robby_yaw)
                x = aux_x
                y = aux_y
                yaw += robby_yaw
                if yaw > np.pi:
                    yaw -= (2 * np.pi)
                elif yaw < -np.pi:
                    yaw += (2 * np.pi)
            rect = utils.Rectangle(x, y, w, h, yaw)
            rect.plot(ax, facecolor=fc, edgecolor=ec)

        ax.set_xlim(self.env_x_lb, self.env_x_ub)
        ax.set_ylim(self.env_y_lb, self.env_y_ub)
        plt.suptitle(caption, fontsize=12, wrap=True)
        plt.tight_layout()
        return fig

    def _get_tasks(self, num_tasks: int, num_books_lst: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for i in range(num_tasks):
            num_books = num_books_lst[i % len(num_books_lst)]
            data = {}

            # Sample shelf variables
            shelf_out_of_bounds = True
            while shelf_out_of_bounds:
                # Size
                shelf_w = rng.uniform(self.shelf_w_lb, self.shelf_w_ub)
                shelf_h = rng.uniform(self.shelf_h_lb, self.shelf_h_ub)
                # Pose
                shelf_x = rng.uniform(self.env_x_lb, self.env_x_ub - shelf_w)
                shelf_y = rng.uniform(self.env_y_lb, self.env_y_ub - shelf_h)
                shelf_yaw = rng.uniform(-np.pi, np.pi)

                if shelf_yaw >= 0:
                    min_x = shelf_x - shelf_h * np.sin(shelf_yaw)
                    max_x = shelf_x + shelf_w * np.cos(shelf_yaw)
                    min_y = shelf_y
                    max_y = shelf_y + shelf_w * np.sin(
                        shelf_yaw) + shelf_h * np.cos(shelf_yaw)
                else:
                    min_x = shelf_x
                    max_x = shelf_x + shelf_h * np.sin(
                        -shelf_yaw) + shelf_w * np.cos(-shelf_yaw)
                    min_y = shelf_y - shelf_w * np.sin(-shelf_yaw)
                    max_y = shelf_y + shelf_h * np.cos(-shelf_yaw)

                if min_x >= self.env_x_lb and max_x <= self.env_x_ub \
                    and min_y >= self.env_y_lb and max_y <= self.env_y_ub:
                    shelf_out_of_bounds = False
            shelf_rect = utils.Rectangle(shelf_x, shelf_y, shelf_w, shelf_h,
                                         shelf_yaw)
            data[self._shelf] = np.array(
                [shelf_x, shelf_y, shelf_w, shelf_h, shelf_yaw])

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

            # Sample book poses
            goal = set()
            for j in range(num_books):
                book_collision = True
                tmp_state = State(data)
                while book_collision:
                    book_init_x = rng.uniform(self.env_x_lb, self.env_x_ub)
                    book_init_y = rng.uniform(self.env_y_lb, self.env_y_ub)
                    book_init_yaw = rng.uniform(-np.pi, np.pi)
                    book_width = rng.uniform(self.book_w_lb, self.book_w_ub)
                    book_height = rng.uniform(self.book_h_lb, self.book_h_ub)
                    book_rect = utils.Rectangle(book_init_x, book_init_y,
                                                book_width, book_height,
                                                book_init_yaw)
                    book_collision = self.check_collision(tmp_state, book_rect)
                book = Object(f"book{j}", self._book_type)
                held = 0.0
                data[book] = np.array([
                    book_init_x, book_init_y, book_width, book_height,
                    book_init_yaw, held
                ],
                                      dtype=np.float32)
                goal.add(GroundAtom(self._OnShelf, [book, self._shelf]))

            state = State(data)
            tasks.append(Task(state, goal))
        return tasks

    # From https://stackoverflow.com/questions/37101001/pythonic-way-to-generate-random-uniformly-distributed-points-within-hollow-squarl
    def sample_outside_bbox(self,
                            a,
                            b,
                            inner_width,
                            inner_height,
                            rng,
                            n_samples=1):
        """(a, b) is the lower-left corner of the "hollow"."""
        outer_width = self.env_x_ub - self.env_x_lb
        outer_height = self.env_y_ub - self.env_y_lb
        llcorners = np.array([[0, 0], [a, 0], [a + inner_width, 0], [0, b],
                              [a + inner_width, b], [0, b + inner_height],
                              [a, b + inner_height],
                              [a + inner_width, b + inner_height]])
        top_height = outer_height - (b + inner_height)
        right_width = outer_width - (a + inner_width)
        widths = np.array([
            a, inner_width, right_width, a, right_width, a, inner_width,
            right_width
        ])
        heights = np.array([
            b, b, b, inner_height, inner_height, top_height, top_height,
            top_height
        ])
        areas = widths * heights
        shapes = np.column_stack((widths, heights))

        regions = rng.multinomial(n_samples, areas / areas.sum())
        indices = np.repeat(range(8), regions)
        unit_coords = rng.random(size=(n_samples, 2))
        pts = unit_coords * shapes[indices] + llcorners[indices]

        return pts + np.array([[self.env_x_lb, self.env_y_lb]])

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
                obj_geom = utils.Rectangle(x, y, width, height, theta)
            elif obj.is_instance(self._robot_type):
                x = state.get(obj, "pose_x")
                y = state.get(obj, "pose_y")
                obj_geom = utils.Circle(x, y, self.robot_radius)
            if utils.geom2ds_intersect(geom, obj_geom):
                return True
        return False

    '''
    Note: the pick policy takes a parameter delta, representing how far to stretch the
    gripper (between 0, 1 relative to the gripper size), and a parameter yaw, representing
    the relative orientation of the book once gripped. 
    '''

    def _PickBook_policy(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> Action:
        del memory  # unused
        robby, book = objects
        book_x = state.get(book, "pose_x")
        book_y = state.get(book, "pose_y")
        book_w = state.get(book, "width")
        book_h = state.get(book, "height")
        book_yaw = state.get(book, "yaw")

        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")

        offset_gripper, book_yaw = params

        arr = np.array([robby_x, robby_y, book_yaw, offset_gripper, -1],
                       dtype=np.float32)
        return Action(arr)

    '''
    Note: the place policy takes a single parameter delta, representing how far to stretch
    the gripper, between 0 and 1 relative to the gripper size. 
    '''

    def _PlaceBookOnShelf_policy(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        del memory  # unused
        robby, book, shelf = objects
        book_x = state.get(book, "pose_x")
        book_y = state.get(book, "pose_y")
        book_w = state.get(book, "width")
        book_h = state.get(book, "height")
        book_yaw = state.get(book, "yaw")

        shelf_x = state.get(shelf, "pose_x")
        shelf_y = state.get(shelf, "pose_y")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")
        shelf_yaw = state.get(shelf, "yaw")

        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")

        offset_gripper, = params

        arr = np.array([robby_x, robby_y, robby_yaw, offset_gripper, 0],
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
        if yaw > np.pi:
            yaw -= (2 * np.pi)
        elif yaw < -np.pi:
            yaw += (2 * np.pi)
        arr = np.array([pos_x, pos_y, yaw, 0.0, 1.0], dtype=np.float32)
        # pos_x and pos_y might take the robot out of the env bounds, so we clip
        # the action back into its bounds for safety.
        arr = np.clip(arr, self.action_space.low, self.action_space.high)
        return Action(arr)

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        book, = objects
        return state.get(book, "held") == 1.0

    def _OnShelf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        book, shelf = objects

        book_x = state.get(book, "pose_x")
        book_y = state.get(book, "pose_y")
        book_w = state.get(book, "width")
        book_h = state.get(book, "height")
        book_yaw = state.get(book, "yaw")

        shelf_x = state.get(shelf, "pose_x")
        shelf_y = state.get(shelf, "pose_y")
        shelf_w = state.get(shelf, "width")
        shelf_h = state.get(shelf, "height")
        shelf_yaw = state.get(shelf, "yaw")

        book_rect = utils.Rectangle(book_x, book_y, book_w, book_h, book_yaw)
        shelf_rect = utils.Rectangle(shelf_x, shelf_y, shelf_w, shelf_h,
                                     shelf_yaw)
        return shelf_rect.contains_point(*(book_rect.center))

    def _GripperFree_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robby, = objects
        return state.get(robby, "gripper_free") == 1.0

    def _get_pickable_book(self, state: State, tip_x: float, tip_y: float):
        pick_book = None
        book_rect = None
        for obj in state.data:
            if obj.is_instance(self._book_type):
                x = state.get(obj, "pose_x")
                y = state.get(obj, "pose_y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "yaw")
                book_rect = utils.Rectangle(x, y, width, height, theta)
                if book_rect.contains_point(tip_x, tip_y):
                    pick_book = obj
                    break
        return pick_book, book_rect

    def _get_held_book(self, state):
        for obj in state:
            if obj.is_instance(self._book_type) and state.get(obj,
                                                              "held") == 1.0:
                return obj
        return None
