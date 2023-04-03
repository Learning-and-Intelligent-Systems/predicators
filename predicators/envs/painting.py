"""Painting domain, which allows for two different grasps on an object (side or
top).

Side grasping allows for placing into the shelf, and top grasping allows
for placing into the box. The box has a lid which may need to be opened;
this lid is NOT modeled by any of the given predicates.
"""

import logging
from typing import Any, Callable, ClassVar, List, Optional, Sequence, Set, \
    Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from matplotlib import patches

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type
from predicators.utils import EnvironmentFailure, HumanDemonstrationFailure


class PaintingEnv(BaseEnv):
    """Painting domain."""
    # Parameters that aren't important enough to need to clog up settings.py
    table_lb: ClassVar[float] = -10.1
    table_ub: ClassVar[float] = -1.0
    table_height: ClassVar[float] = 0.2
    table_x: ClassVar[float] = 1.65
    shelf_l: ClassVar[float] = 2.0  # shelf length
    shelf_lb: ClassVar[float] = 1.
    shelf_ub: ClassVar[float] = shelf_lb + shelf_l - 0.05
    shelf_x: ClassVar[float] = 1.65
    shelf_y: ClassVar[float] = (shelf_lb + shelf_ub) / 2.0
    box_s: ClassVar[float] = 0.8  # side length
    box_y: ClassVar[float] = 0.5  # y coordinate
    box_lb: ClassVar[float] = box_y - box_s / 10
    box_ub: ClassVar[float] = box_y + box_s / 10
    box_x: ClassVar[float] = 1.65
    env_lb: ClassVar[float] = min(table_lb, shelf_lb, box_lb)
    env_ub: ClassVar[float] = max(table_ub, shelf_ub, box_ub)
    obj_height: ClassVar[float] = 0.13
    obj_radius: ClassVar[float] = 0.03
    obj_x: ClassVar[float] = 1.65
    obj_z: ClassVar[float] = table_height + obj_height / 2
    pick_tol: ClassVar[float] = 1e-2
    color_tol: ClassVar[float] = 1e-2
    wetness_tol: ClassVar[float] = 0.5
    dirtiness_tol: ClassVar[float] = 0.5
    open_fingers: ClassVar[float] = 0.8
    top_grasp_thresh: ClassVar[float] = 0.5 + 1e-2
    side_grasp_thresh: ClassVar[float] = 0.5 - 1e-2
    robot_x: ClassVar[float] = table_x - 0.5
    nextto_thresh: ClassVar[float] = 1.0
    on_table_height_tol: ClassVar[float] = 5e-02

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._obj_type = Type("obj", [
            "pose_x", "pose_y", "pose_z", "dirtiness", "wetness", "color",
            "grasp", "held"
        ])
        self._box_type = Type("box", ["pose_x", "pose_y", "color"])
        self._lid_type = Type("lid", ["is_open"])
        self._shelf_type = Type("shelf", ["pose_x", "pose_y", "color"])
        self._robot_type = Type("robot", ["pose_x", "pose_y", "fingers"])
        # Predicates
        self._InBox = Predicate("InBox", [self._obj_type, self._box_type],
                                self._InBox_holds)
        self._InShelf = Predicate("InShelf",
                                  [self._obj_type, self._shelf_type],
                                  self._InShelf_holds)
        self._IsBoxColor = Predicate("IsBoxColor",
                                     [self._obj_type, self._box_type],
                                     self._IsBoxColor_holds)
        self._IsShelfColor = Predicate("IsShelfColor",
                                       [self._obj_type, self._shelf_type],
                                       self._IsShelfColor_holds)
        self._GripperOpen = Predicate("GripperOpen", [self._robot_type],
                                      self._GripperOpen_holds)
        self._OnTable = Predicate("OnTable", [self._obj_type],
                                  self._OnTable_holds)
        self._NotOnTable = Predicate("NotOnTable", [self._obj_type],
                                     self._NotOnTable_holds)
        self._HoldingTop = Predicate("HoldingTop", [self._obj_type],
                                     self._HoldingTop_holds)
        self._HoldingSide = Predicate("HoldingSide", [self._obj_type],
                                      self._HoldingSide_holds)
        self._Holding = Predicate("Holding", [self._obj_type],
                                  self._Holding_holds)
        self._IsWet = Predicate("IsWet", [self._obj_type], self._IsWet_holds)
        self._IsDry = Predicate("IsDry", [self._obj_type], self._IsDry_holds)
        self._IsDirty = Predicate("IsDirty", [self._obj_type],
                                  self._IsDirty_holds)
        self._IsClean = Predicate("IsClean", [self._obj_type],
                                  self._IsClean_holds)
        self._IsOpen = Predicate("IsOpen", [self._lid_type],
                                 self._IsOpen_holds)
        # Static objects (always exist no matter the settings).
        self._box = Object("receptacle_box", self._box_type)
        self._lid = Object("box_lid", self._lid_type)
        self._shelf = Object("receptacle_shelf", self._shelf_type)
        self._robot = Object("robby", self._robot_type)

    @classmethod
    def get_name(cls) -> str:
        return "painting"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        arr = action.arr
        # Infer which transition function to follow
        wash_affinity = 0 if arr[5] > 0.5 else abs(arr[5] - 0.5)
        dry_affinity = 0 if arr[6] > 0.5 else abs(arr[6] - 0.5)
        paint_affinity = min(abs(arr[7] - state.get(self._box, "color")),
                             abs(arr[7] - state.get(self._shelf, "color")))
        affinities = [
            (abs(1 - arr[4]), self._transition_pick_or_openlid),
            (wash_affinity, self._transition_wash),
            (dry_affinity, self._transition_dry),
            (paint_affinity, self._transition_paint),
            (abs(-1 - arr[4]), self._transition_place),
        ]
        _, transition_fn = min(affinities, key=lambda item: item[0])
        return transition_fn(state, action)

    def _transition_pick_or_openlid(self, state: State,
                                    action: Action) -> State:
        x, y, z, grasp = action.arr[:4]
        next_state = state.copy()
        # Open lid
        if self.box_lb < y < self.box_ub:
            next_state.set(self._lid, "is_open", 1.0)
            return next_state
        held_obj = self._get_held_object(state)
        # Cannot pick if already holding something
        if held_obj is not None:
            return next_state
        # Cannot pick if object pose not on table
        if not self.table_lb < y < self.table_ub:
            return next_state
        # Cannot pick if grasp is invalid
        if self.side_grasp_thresh < grasp < self.top_grasp_thresh:
            return next_state
        # Check if some object is close enough to (x, y, z)
        target_obj = self._get_object_at_xyz(state, x, y, z, self.pick_tol)
        if target_obj is None:
            return next_state
        # Execute pick
        next_state.set(self._robot, "fingers", 0.0)
        next_state.set(target_obj, "grasp", grasp)
        next_state.set(target_obj, "held", 1.0)
        return next_state

    def _transition_wash(self, state: State, action: Action) -> State:
        target_wetness = action.arr[5]
        next_state = state.copy()
        held_obj = self._get_held_object(state)
        # Can only wash if holding obj
        if held_obj is None:
            return next_state
        # Execute wash
        cur_dirtiness = state.get(held_obj, "dirtiness")
        next_dirtiness = max(cur_dirtiness - target_wetness, 0.0)
        next_state.set(held_obj, "wetness", target_wetness)
        next_state.set(held_obj, "dirtiness", next_dirtiness)
        return next_state

    def _transition_dry(self, state: State, action: Action) -> State:
        target_wetness = max(1.0 - action.arr[6], 0.0)
        next_state = state.copy()
        held_obj = self._get_held_object(state)
        # Can only dry if holding obj
        if held_obj is None:
            return next_state
        # Execute dry
        next_state.set(held_obj, "wetness", target_wetness)
        return next_state

    def _transition_paint(self, state: State, action: Action) -> State:
        color = action.arr[7]
        next_state = state.copy()
        # Can only paint if holding obj
        held_obj = self._get_held_object(state)
        if held_obj is None:
            return next_state
        # Can only paint if dry and clean
        if state.get(held_obj, "dirtiness") > self.dirtiness_tol or \
           state.get(held_obj, "wetness") > self.wetness_tol:
            return next_state
        # Execute paint
        next_state.set(held_obj, "color", color)
        return next_state

    def _transition_place(self, state: State, action: Action) -> State:
        # Action args are target pose for held obj
        x, y, z = action.arr[:3]
        next_state = state.copy()
        # Can only place if holding obj
        held_obj = self._get_held_object(state)
        if held_obj is None:
            return next_state
        # Detect table vs shelf vs box place
        if self.table_lb < y < self.table_ub:
            receptacle = "table"
        elif self.shelf_lb < y < self.shelf_ub:
            receptacle = "shelf"
        elif self.box_lb < y < self.box_ub:
            receptacle = "box"
        else:
            # Cannot place outside of table, shelf, or box
            return next_state
        if receptacle == "box" and state.get(self._lid, "is_open") < 0.5:
            # Cannot place in box if lid is not open
            if CFG.painting_raise_environment_failure:
                raise EnvironmentFailure("Box lid is closed.",
                                         {"offending_objects": {self._lid}})
            return next_state
        # Detect top grasp vs side grasp
        grasp = state.get(held_obj, "grasp")
        if grasp > self.top_grasp_thresh:
            top_or_side = "top"
        elif grasp < self.side_grasp_thresh:
            top_or_side = "side"
        # Can only place in shelf if side grasping, box if top grasping. If the
        # receptacle is table, we don't care what kind of grasp it is.
        if receptacle == "shelf" and top_or_side != "side":
            return next_state
        if receptacle == "box" and top_or_side != "top":
            return next_state
        # Detect collisions
        collider = self._get_object_at_xyz(state, x, y, z, self.pick_tol)
        if receptacle == "table" and \
           collider is not None and \
           collider != held_obj:
            return next_state
        # Execute place
        next_state.set(self._robot, "fingers", 1.0)
        next_state.set(held_obj, "pose_x", x)
        next_state.set(held_obj, "pose_y", y)
        if self._update_z_poses:
            if receptacle == "table" and np.allclose(
                    z,
                    self.table_height + self.obj_height / 2,
                    rtol=self.on_table_height_tol):
                # If placing on table, snap the object to the correct z
                # position as long as the place location is close enough
                # (measured by rtol) to the correct table height. This
                # is necessary for learned samplers to have any hope of
                # placing objects on the table.
                next_state.set(held_obj, "pose_z",
                               self.table_height + self.obj_height / 2)
            else:
                next_state.set(held_obj, "pose_z", z)
        next_state.set(held_obj, "grasp", 0.5)
        next_state.set(held_obj, "held", 0.0)
        return next_state

    @property
    def _num_objects_train(self) -> List[int]:
        return CFG.painting_num_objs_train

    @property
    def _num_objects_test(self) -> List[int]:
        return CFG.painting_num_objs_test

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_train_tasks,
                               num_objs_lst=self._num_objects_train,
                               rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num_tasks=CFG.num_test_tasks,
                               num_objs_lst=self._num_objects_test,
                               rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._InBox, self._InShelf, self._IsBoxColor, self._IsShelfColor,
            self._GripperOpen, self._OnTable, self._NotOnTable,
            self._HoldingTop, self._HoldingSide, self._Holding, self._IsWet,
            self._IsDry, self._IsDirty, self._IsClean, self._IsOpen
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {
            self._InBox, self._InShelf, self._IsBoxColor, self._IsShelfColor
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._obj_type, self._box_type, self._lid_type, self._shelf_type,
            self._robot_type
        }

    @property
    def action_space(self) -> Box:
        # Actions are 8-dimensional vectors:
        # [x, y, z, grasp, pickplace, water level, heat level, color]
        # Note that pickplace is 1 for pick, -1 for place, and 0 otherwise,
        # while grasp, water level, heat level, and color are in [0, 1].
        # We set the lower bound for z to 0.0, rather than self.obj_z - 1e-2,
        # because in RepeatedNextToPainting, we use this dimension to check
        # affinity of the move action
        lowers = np.array(
            [self.obj_x - 1e-2, self.env_lb, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            dtype=np.float32)
        uppers = np.array([
            self.obj_x + 1e-2, self.env_ub, self.obj_z + 1e-2, 1.0, 1.0, 1.0,
            1.0, 1.0
        ],
                          dtype=np.float32)
        return Box(lowers, uppers)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1)
        objs = [o for o in state if o.is_instance(self._obj_type)]
        denom = (self.env_ub - self.env_lb)
        # The factor of "2" here should actually be 0.5, but this
        # makes the objects too small, so we'll let it be bigger.
        # Don't be alarmed if objects seem to be intersecting in
        # the resulting videos.
        r = 2 * self.obj_radius / denom
        h = 2 * self.obj_height / denom
        z = (self.obj_z - self.env_lb) / denom
        # Draw box
        box_color = state.get(self._box, "color")
        box_lower = (self.box_lb - self.obj_radius - self.env_lb) / denom
        box_upper = (self.box_ub + self.obj_radius - self.env_lb) / denom
        rect = plt.Rectangle((box_lower, z - h),
                             box_upper - box_lower,
                             2 * h,
                             facecolor=[box_color, 0, 0],
                             alpha=0.25)
        ax.add_patch(rect)
        # Draw box lid
        if state.get(self._lid, "is_open") < 0.5:
            plt.plot([box_lower, box_upper], [z + h, z + h],
                     color=[box_color, 0, 0])
        # Draw shelf
        shelf_color = state.get(self._shelf, "color")
        shelf_lower = (self.shelf_lb - self.obj_radius - self.env_lb) / denom
        shelf_upper = (self.shelf_ub + self.obj_radius - self.env_lb) / denom
        rect = plt.Rectangle((shelf_lower, z - h),
                             shelf_upper - shelf_lower,
                             2 * h,
                             facecolor=[shelf_color, 0, 0],
                             alpha=0.25)
        ax.add_patch(rect)
        # Draw objects
        held_obj = self._get_held_object(state)
        for obj in sorted(objs):
            x = state.get(obj, "pose_x")
            y = state.get(obj, "pose_y")
            z = state.get(obj, "pose_z")
            facecolor: Union[None, str, List[Any]] = None
            if state.get(obj, "wetness") > self.wetness_tol and \
               state.get(obj, "dirtiness") < self.dirtiness_tol:
                # wet and clean
                facecolor = "blue"
            elif state.get(obj, "wetness") < self.wetness_tol and \
                 state.get(obj, "dirtiness") > self.dirtiness_tol:
                # dry and dirty
                facecolor = "green"
            elif state.get(obj, "wetness") < self.wetness_tol and \
                 state.get(obj, "dirtiness") < self.dirtiness_tol:
                # dry and clean
                facecolor = "cyan"
            obj_color = state.get(obj, "color")
            if obj_color > 0:
                facecolor = [obj_color, 0, 0]
            if held_obj == obj:
                assert state.get(self._robot, "fingers") < self.open_fingers
                grasp = state.get(held_obj, "grasp")
                assert grasp < self.side_grasp_thresh or \
                    grasp > self.top_grasp_thresh
                edgecolor = ("yellow"
                             if grasp < self.side_grasp_thresh else "orange")
            else:
                edgecolor = "gray"
            # Normalize poses to [0, 1]
            x = (x - self.env_lb) / denom
            y = (y - self.env_lb) / denom
            z = (z - self.env_lb) / denom
            # Plot as rectangle
            rect = patches.Rectangle((y - r, z - h),
                                     2 * r,
                                     2 * h,
                                     zorder=-x,
                                     linewidth=1,
                                     edgecolor=edgecolor,
                                     facecolor=facecolor)
            ax.add_patch(rect)
            # Annotate the object with its name.
            ax.text(y - r / 2,
                    z + 2 * h,
                    obj.name,
                    fontsize="x-small",
                    ha="center",
                    va="center",
                    bbox=dict(facecolor="white", edgecolor="black", alpha=0.5))
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0.6, 1.0)
        title = ("blue = wet+clean, green = dry+dirty, cyan = dry+clean;\n"
                 "yellow border = side grasp, orange border = top grasp")
        if caption is not None:
            title += f";\n{caption}"
        plt.suptitle(title, fontsize=8, wrap=True)
        plt.tight_layout()
        return fig

    @property
    def _max_objs_in_goal(self) -> int:
        return CFG.painting_max_objs_in_goal

    @property
    def _update_z_poses(self) -> bool:
        return False

    def _get_tasks(self, num_tasks: int, num_objs_lst: List[int],
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        tasks = []
        for i in range(num_tasks):
            num_objs = num_objs_lst[i % len(num_objs_lst)]
            data = {}
            # Initialize robot pos with open fingers
            robot_init_y = rng.uniform(self.table_lb, self.table_ub)
            data[self._robot] = np.array([self.robot_x, robot_init_y, 1.0],
                                         dtype=np.float32)
            # Sample distinct colors for shelf and box
            color1 = rng.uniform(0.2, 0.4)
            color2 = rng.uniform(0.6, 1.0)
            if rng.choice(2):
                box_color, shelf_color = color1, color2
            else:
                shelf_color, box_color = color1, color2
            # Create box, lid, and shelf objects
            lid_is_open = int(rng.uniform() < CFG.painting_lid_open_prob)
            data[self._box] = np.array([self.box_x, self.box_y, box_color],
                                       dtype=np.float32)
            data[self._lid] = np.array([lid_is_open], dtype=np.float32)
            data[self._shelf] = np.array(
                [self.shelf_x, self.shelf_y, shelf_color], dtype=np.float32)
            # Create moveable objects and goal
            objs = []
            obj_poses: List[Tuple[float, float, float]] = []
            goal = set()
            assert CFG.painting_goal_receptacles in ("box_and_shelf", "box",
                                                     "shelf")
            if CFG.painting_goal_receptacles == "shelf":
                # No box; all max_objs_in_goal objects must go in the shelf
                num_objs_in_shelf = self._max_objs_in_goal
            else:
                # The last object is destined for the box, so the remaining
                # (max_objs_in_goal - 1) objects must go in the shelf
                num_objs_in_shelf = self._max_objs_in_goal - 1
            for j in range(num_objs):
                obj = Object(f"obj{j}", self._obj_type)
                objs.append(obj)
                pose = self._sample_initial_object_pose(obj_poses, rng)
                obj_poses.append(pose)
                # Start out wet and clean, dry and dirty, or dry and clean
                choice = rng.choice(3)
                if choice == 0:
                    wetness = 0.0
                    dirtiness = rng.uniform(0.5, 1.)
                elif choice == 1:
                    wetness = rng.uniform(0.5, 1.)
                    dirtiness = 0.0
                else:
                    wetness = 0.0
                    dirtiness = 0.0
                color = 0.0
                grasp = 0.5
                held = 0.0
                data[obj] = np.array([
                    pose[0], pose[1], pose[2], dirtiness, wetness, color,
                    grasp, held
                ],
                                     dtype=np.float32)
                if CFG.painting_goal_receptacles in (
                        "box_and_shelf", "box") and j == num_objs - 1:
                    # This object must go in the box
                    # NOTE: the box can only fit one object
                    goal.add(GroundAtom(self._InBox, [obj, self._box]))
                    goal.add(GroundAtom(self._IsBoxColor, [obj, self._box]))
                elif CFG.painting_goal_receptacles in (
                        "box_and_shelf", "shelf") and j < num_objs_in_shelf:
                    # This object must go in the shelf
                    # NOTE: any number of objects can fit in the shelf
                    goal.add(GroundAtom(self._InShelf, [obj, self._shelf]))
                    goal.add(GroundAtom(self._IsShelfColor,
                                        [obj, self._shelf]))
            assert len(goal) <= 2 * self._max_objs_in_goal
            state = State(data)
            # Sometimes start out holding an object, possibly with the wrong
            # grip, so that we'll have to put it on the table and regrasp
            if rng.uniform() < CFG.painting_initial_holding_prob:
                grasp = rng.choice([0.0, 1.0])
                target_obj = objs[rng.choice(len(objs))]
                state.set(self._robot, "fingers", 0.0)
                state.set(target_obj, "grasp", grasp)
                state.set(target_obj, "held", 1.0)
                state.set(target_obj, "pose_y",
                          state.get(self._robot, "pose_y"))
                if self._update_z_poses:
                    state.set(target_obj, "pose_z",
                              state.get(target_obj, "pose_z") + 1.0)
            tasks.append(EnvironmentTask(state, goal))
        return tasks

    def _sample_initial_object_pose(
            self, existing_poses: List[Tuple[float, float, float]],
            rng: np.random.Generator) -> Tuple[float, float, float]:
        existing_ys = [p[1] for p in existing_poses]
        while True:
            this_y = rng.uniform(self.table_lb, self.table_ub)
            if all(
                    abs(this_y - other_y) > 3.5 * self.obj_radius
                    for other_y in existing_ys):
                return (self.obj_x, this_y,
                        self.table_height + self.obj_height / 2)

    def _InBox_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        # If the object is held, not yet in box
        if self._obj_is_held(state, obj):
            return False
        # Check pose of object
        obj_y = state.get(obj, "pose_y")
        return self.box_lb < obj_y < self.box_ub

    def _InShelf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        # If the object is held, not yet in shelf
        if self._obj_is_held(state, obj):
            return False
        # Check pose of object
        obj_y = state.get(obj, "pose_y")
        return self.shelf_lb < obj_y < self.shelf_ub

    def _IsBoxColor_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        obj, box = objects
        return abs(state.get(obj, "color") -
                   state.get(box, "color")) < self.color_tol

    def _IsShelfColor_holds(self, state: State,
                            objects: Sequence[Object]) -> bool:
        obj, shelf = objects
        return abs(state.get(obj, "color") -
                   state.get(shelf, "color")) < self.color_tol

    def _GripperOpen_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robot, = objects
        fingers = state.get(robot, "fingers")
        return fingers >= self.open_fingers

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        obj_y = state.get(obj, "pose_y")
        if not self.table_lb < obj_y < self.table_ub:
            return False
        # Note that obj_z is not updated in this class, but it may be updated
        # by subclasses by overriding self._update_z_poses.
        obj_z = state.get(obj, "pose_z")
        if not np.allclose(obj_z, self.table_height + self.obj_height / 2):
            assert self._update_z_poses
            return False
        return True

    def _NotOnTable_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        return not self._OnTable_holds(state, objects)

    def _HoldingTop_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        obj, = objects
        grasp = state.get(obj, "grasp")
        return grasp > self.top_grasp_thresh

    def _HoldingSide_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        obj, = objects
        grasp = state.get(obj, "grasp")
        return grasp < self.side_grasp_thresh

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._obj_is_held(state, obj)

    def _IsWet_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "wetness") > self.wetness_tol

    def _IsDry_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return not self._IsWet_holds(state, [obj])

    def _IsDirty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "dirtiness") > self.dirtiness_tol

    def _IsClean_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return not self._IsDirty_holds(state, [obj])

    def _IsOpen_holds(self, state: State, objects: Sequence[Object]) -> bool:
        lid, = objects
        return state.get(lid, "is_open") > 0.5

    def _get_held_object(self, state: State) -> Optional[Object]:
        for obj in state:
            if obj.type != self._obj_type:
                continue
            if self._obj_is_held(state, obj):
                return obj
        return None

    def _obj_is_held(self, state: State, obj: Object) -> bool:
        # These two pieces of information are redundant. We include
        # the "held" feature only because it allows the Holding
        # predicate to be expressed with a single inequality.
        # Either feature can be used to implement this method.
        grasp = state.get(obj, "grasp")
        held_feat = state.get(obj, "held")
        is_held = (grasp > self.top_grasp_thresh
                   or grasp < self.side_grasp_thresh)
        assert is_held == (held_feat > 0.5)  # ensure redundancy
        return is_held

    def _get_object_at_xyz(self, state: State, x: float, y: float, z: float,
                           tol: float) -> Optional[Object]:
        target_obj = None
        for obj in state:
            if obj.type != self._obj_type:
                continue
            if np.allclose([x, y, z], [
                    state.get(obj, "pose_x"),
                    state.get(obj, "pose_y"),
                    state.get(obj, "pose_z")
            ],
                           atol=tol):
                target_obj = obj
        return target_obj

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:

        instructions = [
            "Click to place a held object.",
            "Click ON an object to pick it with a side grasp.",
            "Click ABOVE an object to pick it with a top grasp.",
            "Press (w) to wash the held object.",
            "Press (d) to dry the held object.",
            "Press (b) to open the box lid.",
            "Press (s) to paint the held object the shelf color.",
            "Press (p) to paint the held object the box color.",
            "Press (q) to quit.",
        ]
        instruction_str = "Controls: " + "\n - ".join(instructions)
        logging.info(instruction_str)

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:

            if event.key == "q":
                raise HumanDemonstrationFailure("Human quit.")

            # Wash held object.
            if event.key == "w":
                arr = np.array([
                    self.obj_x, self.table_lb, self.obj_z, 0.0, 0.0, 1.0, 0.0,
                    0.0
                ],
                               dtype=np.float32)
                return Action(arr)

            # Dry held object.
            if event.key == "d":
                arr = np.array([
                    self.obj_x, self.table_lb, self.obj_z, 0.0, 0.0, 0.0, 1.0,
                    0.0
                ],
                               dtype=np.float32)
                return Action(arr)

            # Open the box lid.
            if event.key == "b":
                arr = np.array([
                    self.obj_x, (self.box_lb + self.box_ub) / 2, self.obj_z,
                    0.0, 1.0, 0.0, 0.0, 0.0
                ],
                               dtype=np.float32)
                return Action(arr)

            # Paint shelf color.
            if event.key == "s":
                shelf_color = state.get(self._shelf, "color")
                arr = np.array([
                    self.obj_x,
                    self.table_lb,
                    self.obj_z,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    shelf_color,
                ],
                               dtype=np.float32)
                return Action(arr)

            # Paint box color.
            if event.key == "p":
                box_color = state.get(self._box, "color")
                arr = np.array([
                    self.obj_x,
                    self.table_lb,
                    self.obj_z,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    box_color,
                ],
                               dtype=np.float32)
                return Action(arr)

            # Only remaining actions are ones involving a click.
            if event.xdata is None or event.ydata is None:
                raise NotImplementedError("No valid action found.")

            held_obj = self._get_held_object(state)
            y = event.xdata * (self.env_ub - self.env_lb) + self.env_lb
            # Clicked z used to decide top or side grasp.
            clicked_z = event.ydata * (self.env_ub - self.env_lb) + self.env_lb
            x = self.obj_x
            z = self.obj_z
            # Use a generous tolerance.
            clicked_obj = self._get_object_at_xyz(state, x, y, z, tol=1e-1)

            # Place held object.
            if event.key is None and held_obj is not None:
                arr = np.array([x, y, z, 0.0, -1.0, 0.0, 0.0, 0.0])
                return Action(np.array(arr, dtype=np.float32))

            # Pick.
            if held_obj is None and clicked_obj is not None:
                # Set the y position to the object y position so that the
                # pick is successfully executed.
                y = state.get(clicked_obj, "pose_y")
                # Side grasp.
                if clicked_z <= self.obj_z + 2 * self.obj_height:
                    grasp = self.side_grasp_thresh - 1e-3
                # Top grasp.
                else:
                    grasp = self.top_grasp_thresh + 1e-3
                arr = np.array([x, y, z, grasp, 1.0, 0.0, 0.0, 0.0],
                               dtype=np.float32)
                return Action(arr)

            # Something went wrong.
            raise NotImplementedError("No valid action found.")

        return _event_to_action
