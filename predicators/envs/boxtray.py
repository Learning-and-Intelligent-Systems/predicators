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


class BoxtrayEnv(BaseEnv):
    """Boxtray domain."""

    env_x_lb: ClassVar[float] = 0
    env_y_lb: ClassVar[float] = 0
    env_x_ub: ClassVar[float] = 20#12#
    env_y_ub: ClassVar[float] = 20#12#
    robot_radius: ClassVar[float] = 2
    gripper_length: ClassVar[float] = 2
    tray_w_lb: ClassVar[float] = 11
    tray_w_ub: ClassVar[float] = 13
    tray_h_lb: ClassVar[float] = 3
    tray_h_ub: ClassVar[float] = 5
    box_w_lb: ClassVar[float] = 0.5
    box_w_ub: ClassVar[float] = 1
    box_h_lb: ClassVar[float] = 1
    box_h_ub: ClassVar[float] = 1.5
    _tray_color: ClassVar[List[float]] = [0.89, 0.82, 0.68]
    _robot_color: ClassVar[List[float]] = [0.5, 0.5, 0.5]

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._object_type = Type("object", ["pose_x", "pose_y", "width", "height", "yaw", "held"])
        self._pickable_type = Type("pickable", ["pose_x", "pose_y", "width", "height", "yaw", "held"], self._object_type)
        self._placeable_type = Type("placeable", ["pose_x", "pose_y", "width", "height", "yaw", "held"], self._object_type)
        # Box: when held, pose becomes relative to gripper
        self._box_type = Type(
            "box", ["pose_x", "pose_y", "width", "height", "yaw", "held"],
            self._pickable_type)
        self._tray_type = Type("tray",
                                ["pose_x", "pose_y", "width", "height", "yaw", "held"],
                                self._placeable_type)
        self._robot_type = Type("robot",
                                ["pose_x", "pose_y", "yaw", "gripper_free"])
        # Predicates
        self._OnTray = Predicate("OnTray",
                                  [self._box_type, self._tray_type],
                                  self._OnTray_holds)
        self._CanReach = Predicate("CanReach",
                                  [self._object_type, self._robot_type],
                                  self._CanReach_holds)
        self._Holding = Predicate("Holding", [self._box_type],
                                  self._Holding_holds)
        self._GripperFree = Predicate("GripperFree", [self._robot_type],
                                      self._GripperFree_holds)
        # Options
        if CFG.bookshelf_add_sampler_idx_to_params:
            if CFG.use_fully_uniform_actions:
                lo = [0, self.env_x_lb, self.env_y_lb]
                hi = [2, self.env_x_ub, self.env_y_ub]
            else:
                lo = [0, -8, -4]
                hi = [2, 9, 5]
        else:
            if CFG.use_fully_uniform_actions:
                lo = [self.env_x_lb, self.env_y_lb]
                hi = [self.env_x_ub, self.env_y_ub]
            else:
                lo = [-8, -4]
                hi = [9, 5]
        self._NavigateTo = utils.SingletonParameterizedOption(
            # variables: [robot, object to navigate to]
            # params: [offset_x, offset_y]
            "NavigateTo",
            self._NavigateTo_policy,
            types=[self._robot_type, self._object_type],
            params_space=Box(np.array(lo, dtype=np.float32),
                             np.array(hi, dtype=np.float32)))
        if CFG.bookshelf_add_sampler_idx_to_params:
            lo = [0, 0, -np.pi]
            hi = [2, 1, np.pi]
        else:
            lo = [0, -np.pi]
            hi = [1, np.pi]
        self._PickBox = utils.SingletonParameterizedOption(
            # variables: [robot, box to pick]
            # params: [offset_gripper, box_yaw]
            "PickBox",
            self._PickBox_policy,
            types=[self._robot_type, self._box_type],
            params_space=Box(np.array(lo, dtype=np.float32),
                             np.array(hi, dtype=np.float32)))
        if CFG.bookshelf_add_sampler_idx_to_params:
            lo = [0, 0]
            hi = [2, 1]
        else:
            lo = [0]
            hi = [1]
        self._PlaceBoxOnTray = utils.SingletonParameterizedOption(
            # variables: [robot, box, tray]
            # params: [offset_gripper]
            "PlaceBoxOnTray",
            self._PlaceBoxOnTray_policy,
            types=[self._robot_type, self._box_type, self._tray_type],
            params_space=Box(np.array(lo, dtype=np.float32),
                             np.array(hi, dtype=np.float32)))
        # Static objects (always exist no matter the settings)
        self._tray = Object("tray", self._tray_type)
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "boxtray"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        arr = action.arr
        if arr[-1] < -0.33:
            transition_fn = self._transition_pick_box
        elif -0.33 <= arr[-1] < 0.33:
            transition_fn = self._transition_place_box_on_tray
        elif 0.33 <= arr[-1]:
            transition_fn = self._transition_navigate_to
        return transition_fn(state, action)

    def _transition_pick_box(self, state: State, action: Action) -> State:
        yaw, offset_gripper = action.arr[2:4]
        next_state = state.copy()
        held_box = self._get_held_box(state)
        if held_box is not None:
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

        pick_box, box_rect = self._get_pickable_box(state, tip_x, tip_y)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        ignore_objects = {pick_box, robby}
        # Check that there is a graspable box and that there are no gripper collisions
        if pick_box is None or self.check_collision(state, gripper_line,
                                                     ignore_objects):
            return next_state
        # Execute pick
        box_rect = box_rect.rotate_about_point(tip_x, tip_y, yaw)
        yaw = box_rect.theta - robby_yaw
        while yaw > np.pi:
            yaw -= (2 * np.pi)
        while yaw < -np.pi:
            yaw += (2 * np.pi)
        rel_x = box_rect.x - tip_x
        rel_y = box_rect.y - tip_y
        rot_rel_x = rel_x * np.sin(robby_yaw) - rel_y * np.cos(robby_yaw)
        rot_rel_y = rel_x * np.cos(robby_yaw) + rel_y * np.sin(robby_yaw)
        next_state.set(pick_box, "held", 1.0)
        next_state.set(pick_box, "pose_x", rot_rel_x)
        next_state.set(pick_box, "pose_y", rot_rel_y)
        next_state.set(pick_box, "yaw", yaw)
        next_state.set(robby, "gripper_free", 0.0)
        return next_state

    def _transition_place_box_on_tray(self, state: State,
                                        action: Action) -> State:
        offset_gripper = action.arr[3]
        next_state = state.copy()

        box = self._get_held_box(state)
        if box is None:
            return next_state
        box_relative_x = state.get(box, "pose_x")
        box_relative_y = state.get(box, "pose_y")
        box_relative_yaw = state.get(box, "yaw")
        box_w = state.get(box, "width")
        box_h = state.get(box, "height")

        robby = self._robot
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        gripper_free = state.get(robby, "gripper_free")
        if gripper_free != 0.0:
            return next_state

        tray = self._tray
        tray_x = state.get(tray, "pose_x")
        tray_y = state.get(tray, "pose_y")
        tray_w = state.get(tray, "width")
        tray_h = state.get(tray, "height")
        tray_yaw = state.get(tray, "yaw")
        tray_hole1_x = tray_x
        tray_hole1_y = tray_y
        tray_hole2_x = tray_x + (tray_w - tray_h) * np.cos(tray_yaw)
        tray_hole2_y = tray_y + (tray_w - tray_h) * np.sin(tray_yaw)

        tip_x = robby_x + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + offset_gripper *
                           self.gripper_length) * np.sin(robby_yaw)

        place_x = tip_x + box_relative_x * np.sin(
            robby_yaw) + box_relative_y * np.cos(robby_yaw)
        place_y = tip_y + box_relative_y * np.sin(
            robby_yaw) - box_relative_x * np.cos(robby_yaw)
        place_yaw = box_relative_yaw + robby_yaw
        while place_yaw > np.pi:
            place_yaw -= (2 * np.pi)
        while place_yaw < -np.pi:
            place_yaw += (2 * np.pi)

        # Check whether the boxs center-of-mass is within the tray bounds
        # and that there are no other collisions with the box or gripper
        box_rect = utils.Rectangle(place_x, place_y, box_w, box_h,
                                    place_yaw)
        tray_hole1_rect = utils.Rectangle(tray_hole1_x, tray_hole1_y, tray_h, tray_h, tray_yaw)
        tray_hole2_rect = utils.Rectangle(tray_hole2_x, tray_hole2_y, tray_h, tray_h, tray_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        ignore_objects = {box, robby, tray}
        if not (tray_hole1_rect.contains_point(*(box_rect.center)) or \
                tray_hole2_rect.contains_point(*(box_rect.center))) or \
            self.check_collision(state, box_rect, ignore_objects) or \
            self.check_collision(state, gripper_line, ignore_objects):
            return next_state

        next_state.set(box, "held", 0.0)
        next_state.set(box, "pose_x", place_x)
        next_state.set(box, "pose_y", place_y)
        next_state.set(box, "yaw", place_yaw)
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
    def _num_boxs_train(self) -> List[int]:
        return CFG.bookshelf_num_books_train

    @property
    def _num_boxs_test(self) -> List[int]:
        return CFG.bookshelf_num_books_test

    @property
    def _num_obstacles_train(self) -> List[int]:
        return CFG.bookshelf_num_obstacles_train if not CFG.bookshelf_no_obstacles else [0]

    @property
    def _num_obstacles_test(self) -> List[int]:
        return CFG.bookshelf_num_obstacles_test if not CFG.bookshelf_no_obstacles else [0]

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.bookshelf_train_tasks_overwrite if CFG.bookshelf_train_tasks_overwrite is not None else CFG.num_train_tasks,
                               possible_num_boxs=self._num_boxs_train,
                               possible_num_obstacles=self._num_obstacles_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               possible_num_boxs=self._num_boxs_test,
                               possible_num_obstacles=self._num_obstacles_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._OnTray, self._CanReach, self._Holding, self._GripperFree}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._OnTray}

    @property
    def types(self) -> Set[Type]:
        return {
            self._object_type, self._pickable_type, self._placeable_type,
            self._box_type, self._tray_type, self._robot_type
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._NavigateTo, self._PickBox, self._PlaceBoxOnTray}

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
        boxs = [b for b in state if b.is_instance(self._box_type)]
        if task is not None:
            goal_boxs = {obj for atom in task.goal for obj in atom.objects if obj.is_instance(self._box_type) }
        else:
            goal_boxs = boxs
        
        tray = self._tray
        robby = self._robot
        # Draw tray
        tray_x = state.get(tray, "pose_x")
        tray_y = state.get(tray, "pose_y")
        tray_w = state.get(tray, "width")
        tray_h = state.get(tray, "height")
        tray_yaw = state.get(tray, "yaw")
        rect = utils.Rectangle(tray_x, tray_y, tray_w, tray_h, tray_yaw)
        rect.plot(ax, facecolor=self._tray_color)
        tray_hole1_x = tray_x
        tray_hole1_y = tray_y
        rect = utils.Rectangle(tray_hole1_x, tray_hole1_y, tray_h, tray_h, tray_yaw)
        rect.plot(ax, facecolor='sienna')
        tray_hole2_x = tray_x + (tray_w - tray_h) * np.cos(tray_yaw)
        tray_hole2_y = tray_y + (tray_w - tray_h) * np.sin(tray_yaw)
        rect = utils.Rectangle(tray_hole2_x, tray_hole2_y, tray_h, tray_h, tray_yaw)
        rect.plot(ax, facecolor='sienna')


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

        # Draw boxs
        for b in sorted(boxs):
            x = state.get(b, "pose_x")
            y = state.get(b, "pose_y")
            w = state.get(b, "width")
            h = state.get(b, "height")
            yaw = state.get(b, "yaw")
            holding = state.get(b, "held") == 1.0
            fc = "seagreen" if b in goal_boxs else "gray"
            ec = "red" if holding else "gray"
            if holding:
                aux_x = tip_x + x * np.sin(robby_yaw) + y * np.cos(robby_yaw)
                aux_y = tip_y + y * np.sin(robby_yaw) - x * np.cos(robby_yaw)
                x = aux_x
                y = aux_y
                yaw += robby_yaw
            while yaw > np.pi:
                yaw -= (2 * np.pi)
            while yaw < -np.pi:
                yaw += (2 * np.pi)
            rect = utils.Rectangle(x, y, w, h, yaw)
            rect.plot(ax, facecolor=fc, edgecolor=ec)

        ax.set_xlim(self.env_x_lb, self.env_x_ub)
        ax.set_ylim(self.env_y_lb, self.env_y_ub)
        plt.suptitle(caption, fontsize=12, wrap=True)
        plt.tight_layout()
        plt.axis("off")
        return fig

    def _get_tasks(self, num_tasks: int, possible_num_boxs: List[int],
                   possible_num_obstacles: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for i in range(num_tasks):
            num_boxs = rng.choice(possible_num_boxs)
            num_obstacles = rng.choice(possible_num_obstacles)
            data = {}

            # Sample tray variables
            if CFG.boxtray_against_wall:
                raise ValueError('This env is not supposed to have the tray fixed against a wall')
                # # Size
                # tray_w = rng.uniform(self.tray_w_lb, self.tray_w_ub)
                # tray_h = rng.uniform(self.tray_h_lb, self.tray_h_ub)
                # # Pick wall
                # wall_idx = rng.integers(4)  # l, b, r, t
                # # Pose
                # if wall_idx == 0 or wall_idx == 2:
                #     tray_y = rng.uniform(self.env_y_lb + tray_w, self.env_y_ub)
                #     if wall_idx == 0:
                #         tray_x = self.env_x_lb
                #     else:
                #         tray_x = self.env_x_ub - tray_h
                #     tray_yaw = - np.pi / 2
                # else:
                #     tray_x = rng.uniform(self.env_x_lb, self.env_x_ub - tray_w)
                #     if wall_idx == 1:
                #         tray_y = self.env_y_lb
                #     else:
                #         tray_y = self.env_y_ub - tray_h
                #     tray_yaw = 0
            else:
                tray_out_of_bounds = True
                env_w = self.env_x_ub - self.env_x_lb
                env_h = self.env_y_ub - self.env_y_lb
                env_x = self.env_x_lb
                env_y = self.env_y_lb
                env_rect = utils.Rectangle(env_x, env_y, env_w, env_h, 0)
                while tray_out_of_bounds:
                    # Size
                    tray_w = rng.uniform(self.tray_w_lb, self.tray_w_ub)
                    tray_h = rng.uniform(self.tray_h_lb, self.tray_h_ub)
                    # Pose
                    tray_x = rng.uniform(self.env_x_lb, self.env_x_ub - tray_w)
                    tray_y = rng.uniform(self.env_y_lb, self.env_y_ub - tray_h)
                    tray_yaw = rng.uniform(-np.pi, np.pi)

                    tray_rect = utils.Rectangle(tray_x, tray_y, tray_w, tray_h,
                                                 tray_yaw)
                    tray_out_of_bounds = not(
                        env_rect.contains_point(*(tray_rect.vertices[0])) and
                        env_rect.contains_point(*(tray_rect.vertices[1])) and
                        env_rect.contains_point(*(tray_rect.vertices[2])) and
                        env_rect.contains_point(*(tray_rect.vertices[3]))
                    )

            data[self._tray] = np.array(
                [tray_x, tray_y, tray_w, tray_h, tray_yaw, 0.0])

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

            # Sample box poses
            goal = set()
            for j in range(num_boxs):
                box_collision = True
                tmp_state = State(data)
                while box_collision:
                    box_init_x = rng.uniform(self.env_x_lb, self.env_x_ub)
                    box_init_y = rng.uniform(self.env_y_lb, self.env_y_ub)
                    box_init_yaw = rng.uniform(-np.pi, np.pi)
                    box_width = rng.uniform(self.box_w_lb, self.box_w_ub)
                    box_height = rng.uniform(self.box_h_lb, self.box_h_ub)
                    box_rect = utils.Rectangle(box_init_x, box_init_y,
                                                box_width, box_height,
                                                box_init_yaw)
                    box_collision = self.check_collision(tmp_state, box_rect)
                box = Object(f"box{j}", self._box_type)
                held = 0.0
                data[box] = np.array([
                    box_init_x, box_init_y, box_width, box_height,
                    box_init_yaw, held
                ],
                                      dtype=np.float32)
                if not CFG.boxtray_singlestep_goal:
                    goal.add(GroundAtom(self._OnTray, [box, self._tray]))
                elif j == 0:
                    goal.add(GroundAtom(self._CanReach, [box, self._robot]))

            for j in range(num_boxs, num_boxs + num_obstacles):
                obstacle_collision = True
                tmp_state = State(data)
                ignore_objects = {self._tray}   # allow obstacles to be placed on tray
                while obstacle_collision:
                    obstacle_init_x = rng.uniform(self.env_x_lb, self.env_x_ub)
                    obstacle_init_y = rng.uniform(self.env_y_lb, self.env_y_ub)
                    obstacle_init_yaw = rng.uniform(-np.pi, np.pi)
                    obstacle_width = rng.uniform(self.box_w_lb, self.box_w_ub)
                    obstacle_height = rng.uniform(self.box_h_lb, self.box_h_ub)
                    obstacle_rect = utils.Rectangle(obstacle_init_x, obstacle_init_y,
                                                obstacle_width, obstacle_height,
                                                obstacle_init_yaw)
                    obstacle_collision = self.check_collision(tmp_state, obstacle_rect,
                                                              ignore_objects)
                obstacle = Object(f"box{j}", self._box_type)
                held = 0.0
                data[obstacle] = np.array([
                    obstacle_init_x, obstacle_init_y, obstacle_width, obstacle_height,
                    obstacle_init_yaw, held
                ],
                                      dtype=np.float32)

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
            if obj.is_instance(self._box_type) or obj.is_instance(
                    self._tray_type):
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
                assert obj.type.name == 'dummy'
                continue
            if utils.geom2ds_intersect(geom, obj_geom):
                return True
        return False

    '''
    Note: the pick policy takes a parameter delta, representing how far to stretch the
    gripper (between 0, 1 relative to the gripper size), and a parameter yaw, representing
    the relative orientation of the box once gripped. 
    '''

    def _PickBox_policy(self, state: State, memory: Dict,
                         objects: Sequence[Object], params: Array) -> Action:
        del memory  # unused
        robby, box = objects
        box_x = state.get(box, "pose_x")
        box_y = state.get(box, "pose_y")
        box_w = state.get(box, "width")
        box_h = state.get(box, "height")
        box_yaw = state.get(box, "yaw")

        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")

        if CFG.bookshelf_add_sampler_idx_to_params:
            _, offset_gripper, box_yaw = params
        else:
            offset_gripper, box_yaw = params

        arr = np.array([robby_x, robby_y, box_yaw, offset_gripper, -1],
                       dtype=np.float32)
        return Action(arr)

    '''
    Note: the place policy takes a single parameter delta, representing how far to stretch
    the gripper, between 0 and 1 relative to the gripper size. 
    '''

    def _PlaceBoxOnTray_policy(self, state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> Action:
        del memory  # unused
        robby, box, tray = objects
        box_x = state.get(box, "pose_x")
        box_y = state.get(box, "pose_y")
        box_w = state.get(box, "width")
        box_h = state.get(box, "height")
        box_yaw = state.get(box, "yaw")

        tray_x = state.get(tray, "pose_x")
        tray_y = state.get(tray, "pose_y")
        tray_w = state.get(tray, "width")
        tray_h = state.get(tray, "height")
        tray_yaw = state.get(tray, "yaw")

        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")

        if CFG.bookshelf_add_sampler_idx_to_params:
            _, offset_gripper = params
        else:
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
        if CFG.use_fully_uniform_actions:
            if CFG.bookshelf_add_sampler_idx_to_params:
                _, param_pos_x, param_pos_y = params
            else:
                param_pos_x, param_pos_y = params

            # pos_x cos = obj_x cos + obj_w offset_x cos2 - obj_h offset_y sincos
            # pos_y sin = obj_y sin + obj_w offset_x sin2 + obj_h offset_y sincos
            # pos_x cos + pos_y sin = obj_x cos + obj_y sin + obj_w offset_x
            # offset_x = ((pos_x - obj_x) cos + (pos_y - obj_y) sin) / obj_w

            # pos_x sin = obj_x sin + obj_w offset_x sincos - obj_h offset_y sin2
            # pos_y cos = obj_y cos + obj_w offset_x sincos + obj_h offset_y cos2
            # pos_y cos - pos_x sin = obj_y cos - obj_x sin + obj_h offset_y
            # offset_y = ((pos_y - obj_y) cos - (pos_x - obj_x) sin) / obj_h

            offset_x = ((param_pos_x - obj_x) * np.cos(obj_yaw) + (param_pos_y - obj_y) * np.sin(obj_yaw)) / obj_w
            offset_y = ((param_pos_y - obj_y) * np.cos(obj_yaw) - (param_pos_x - obj_x) * np.sin(obj_yaw)) / obj_h
        else:
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
        box, = objects
        return state.get(box, "held") == 1.0

    def _OnTray_holds(self, state: State, objects: Sequence[Object]) -> bool:
        box, tray = objects

        box_x = state.get(box, "pose_x")
        box_y = state.get(box, "pose_y")
        box_w = state.get(box, "width")
        box_h = state.get(box, "height")
        box_yaw = state.get(box, "yaw")

        tray_x = state.get(tray, "pose_x")
        tray_y = state.get(tray, "pose_y")
        tray_w = state.get(tray, "width")
        tray_h = state.get(tray, "height")
        tray_yaw = state.get(tray, "yaw")
        tray_hole1_x = tray_x
        tray_hole1_y = tray_y
        tray_hole2_x = tray_x + (tray_w - tray_h) * np.cos(tray_yaw)
        tray_hole2_y = tray_y + (tray_w - tray_h) * np.sin(tray_yaw)

        while box_yaw > np.pi:
            box_yaw -= (2 * np.pi)
        while box_yaw < -np.pi:
            box_yaw += (2 * np.pi)
        box_rect = utils.Rectangle(box_x, box_y, box_w, box_h, box_yaw)
        tray_hole1_rect = utils.Rectangle(tray_hole1_x, tray_hole1_y, tray_h, tray_h, tray_yaw)
        tray_hole2_rect = utils.Rectangle(tray_hole2_x, tray_hole2_y, tray_h, tray_h, tray_yaw)
        return tray_hole1_rect.contains_point(*(box_rect.center)) or \
               tray_hole2_rect.contains_point(*(box_rect.center))

    def _GripperFree_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robby, = objects
        return state.get(robby, "gripper_free") == 1.0

    def _CanReach_holds(self, state: State,
                        objects: Sequence[Object]) -> bool:
        obj, robby = objects
        robby_x = state.get(robby, "pose_x")
        robby_y = state.get(robby, "pose_y")
        robby_yaw = state.get(robby, "yaw")
        tip_x = robby_x + (self.robot_radius + self.gripper_length) * np.cos(robby_yaw)
        tip_y = robby_y + (self.robot_radius + self.gripper_length) * np.sin(robby_yaw)
        gripper_line = utils.LineSegment(robby_x, robby_y, tip_x, tip_y)
        if obj.is_instance(self._box_type) and self._Holding_holds(state, [obj]):
            return False
        if obj.is_instance(self._box_type):
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
            return utils.geom2ds_intersect(gripper_line, obj_geom)
        if obj.is_instance(self._tray_type):
            x = state.get(obj, "pose_x")
            y = state.get(obj, "pose_y")
            width = state.get(obj, "width")
            height = state.get(obj, "height")
            theta = state.get(obj, "yaw")
            while theta > np.pi:
                theta -= (2 * np.pi)
            while theta < -np.pi:
                theta += (2 * np.pi)
            x1 = x
            y1 = y
            x2 = x + (width - height) * np.cos(theta)
            y2 = y + (width - height) * np.sin(theta)
            obj_geom1 = utils.Rectangle(x1, y1, height, height, theta)
            obj_geom2 = utils.Rectangle(x2, y2, height, height, theta)
            return utils.geom2ds_intersect(gripper_line, obj_geom1) or \
                   utils.geom2ds_intersect(gripper_line, obj_geom2)

        raise ValueError("Can only compute reachable for boxs and bins")



    def _get_pickable_box(self, state: State, tip_x: float, tip_y: float):
        pick_box = None
        box_rect = None
        for obj in state.data:
            if obj.is_instance(self._box_type):
                x = state.get(obj, "pose_x")
                y = state.get(obj, "pose_y")
                width = state.get(obj, "width")
                height = state.get(obj, "height")
                theta = state.get(obj, "yaw")
                while theta > np.pi:
                    theta -= (2 * np.pi)
                while theta < -np.pi:
                    theta += (2 * np.pi)
                box_rect = utils.Rectangle(x, y, width, height, theta)
                if box_rect.contains_point(tip_x, tip_y):
                    pick_box = obj
                    break
        return pick_box, box_rect

    def _get_held_box(self, state):
        for obj in state:
            if obj.is_instance(self._box_type) and state.get(obj,
                                                              "held") == 1.0:
                return obj
        return None
