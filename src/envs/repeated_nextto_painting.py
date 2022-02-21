"""RepeatedNextToPainting domain, which is a merge of our Painting and
RepeatedNextTo environments.

It is exactly the same as the Painting domain, but requires movement to
navigate between objects in order to pick or place them. Also, the move
option can turn on any number of NextTo predicates.
"""

from typing import List, Set, Sequence, Dict, Tuple, Optional, Union, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from gym.spaces import Box
from predicators.src.envs.painting import PaintingEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


class RepeatedNextToPaintingEnv(PaintingEnv):
    """RepeatedNextToPainting domain."""

    def __init__(self) -> None:
        super().__init__()
        # Additional class attributes
        self.shelf_x = 1.65
        self.shelf_y = (self.shelf_lb + self.shelf_ub) / 2.0
        self.box_x = 1.65
        self.table_x = 1.65
        self.robot_x = self.table_x - 0.5
        self.nextto_thresh = 1.0
        # Overriden Types
        self._box_type = Type("box", ["pose_x", "pose_y", "color"])
        self._shelf_type = Type("shelf", ["pose_x", "pose_y", "color"])
        self._robot_type = Type("robot", ["x", "y", "fingers"])
        # Overriden Predicates. This contains some redundancy but is necessary
        # because we're overriding some of the types above.
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
        self._NextTo = Predicate("NextTo", [self._robot_type, self._obj_type],
                                 self._NextTo_holds)
        self._NextToBox = Predicate("NextToBox",
                                    [self._robot_type, self._box_type],
                                    self._NextTo_holds)
        self._NextToShelf = Predicate("NextToShelf",
                                      [self._robot_type, self._shelf_type],
                                      self._NextTo_holds)
        self._NextToNothing = Predicate("NextToNothing", [self._robot_type],
                                        self._NextToNothing_holds)
        # Overriden Options. This contains some redundancy but is necessary
        # because we're overriding some of the predicates above.
        self._Pick = ParameterizedOption(
            # variables: [robot, object to pick]
            # params: [delta x, delta y, delta z, grasp]
            "Pick",
            types=[self._robot_type, self._obj_type],
            params_space=Box(
                np.array([-1.0, -1.0, -1.0, -0.01], dtype=np.float32),
                np.array([1.0, 1.0, 1.0, 1.01], dtype=np.float32)),
            _policy=self._Pick_policy,
            _initiable=self._handempty_initiable,
            _terminal=utils.onestep_terminal)
        self._Wash = ParameterizedOption(
            # variables: [robot]
            # params: [water level]
            "Wash",
            types=[self._robot_type],
            params_space=Box(-0.01, 1.01, (1, )),
            _policy=self._Wash_policy,
            _initiable=self._holding_initiable,
            _terminal=utils.onestep_terminal)
        self._Dry = ParameterizedOption(
            # variables: [robot]
            # params: [heat level]
            "Dry",
            types=[self._robot_type],
            params_space=Box(-0.01, 1.01, (1, )),
            _policy=self._Dry_policy,
            _initiable=self._holding_initiable,
            _terminal=utils.onestep_terminal)
        self._Paint = ParameterizedOption(
            # variables: [robot]
            # params: [new color]
            "Paint",
            types=[self._robot_type],
            params_space=Box(-0.01, 1.01, (1, )),
            _policy=self._Paint_policy,
            _initiable=self._holding_initiable,
            _terminal=utils.onestep_terminal)
        self._Place = ParameterizedOption(
            # variables: [robot]
            # params: [absolute x, absolute y, absolute z]
            "Place",
            types=[self._robot_type],
            params_space=Box(
                np.array([self.obj_x - 1e-2, self.env_lb, self.obj_z - 1e-2],
                         dtype=np.float32),
                np.array([self.obj_x + 1e-2, self.env_ub, self.obj_z + 1e-2],
                         dtype=np.float32)),
            _policy=self._Place_policy,
            _initiable=self._holding_initiable,
            _terminal=utils.onestep_terminal)
        self._OpenLid = ParameterizedOption(
            # variables: [robot, lid]
            # params: []
            "OpenLid",
            types=[self._robot_type, self._lid_type],
            params_space=Box(-0.01, 1.01, (0, )),  # no parameters
            _policy=self._OpenLid_policy,
            _initiable=self._handempty_initiable,
            _terminal=utils.onestep_terminal)
        self._MoveToObj = ParameterizedOption(
            "MoveToObj",
            types=[self._robot_type, self._obj_type],
            params_space=Box(self.env_lb, self.env_ub, (1, )),
            _policy=self._Move_policy,
            _initiable=utils.always_initiable,
            _terminal=utils.onestep_terminal)
        self._MoveToBox = ParameterizedOption(
            "MoveToBox",
            types=[self._robot_type, self._box_type],
            params_space=Box(self.env_lb, self.env_ub, (1, )),
            _policy=self._Move_policy,
            _initiable=utils.always_initiable,
            _terminal=utils.onestep_terminal)
        self._MoveToShelf = ParameterizedOption(
            "MoveToShelf",
            types=[self._robot_type, self._shelf_type],
            params_space=Box(self.env_lb, self.env_ub, (1, )),
            _policy=self._Move_policy,
            _initiable=utils.always_initiable,
            _terminal=utils.onestep_terminal)
        # Overriden objects. This contains some redundancy but is necessary
        # because we're overriding some of the types above.
        self._box = Object("receptacle_box", self._box_type)
        self._lid = Object("box_lid", self._lid_type)
        self._shelf = Object("receptacle_shelf", self._shelf_type)
        self._robot = Object("robby", self._robot_type)

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        arr = action.arr
        # Infer which transition function to follow
        wash_affinity = 0 if arr[5] > 0.5 else abs(arr[5] - 0.5)
        dry_affinity = 0 if arr[6] > 0.5 else abs(arr[6] - 0.5)
        paint_affinity = min(abs(arr[7] - state.get(self._box, "color")),
                             abs(arr[7] - state.get(self._shelf, "color")))
        # NOTE: added move_affinity
        move_affinity = sum(abs(val) for val in arr[2:])
        affinities = [
            (abs(1 - arr[4]), self._transition_pick_or_openlid),
            (wash_affinity, self._transition_wash),
            (dry_affinity, self._transition_dry),
            (paint_affinity, self._transition_paint),
            (abs(-1 - arr[4]), self._transition_place),
            (move_affinity, self._transition_move),
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
        target_obj = self._get_object_at_xyz(state, x, y, z)
        if target_obj is None:
            return next_state

        # NOTE: added cannot pick if obj is too far
        if abs(state.get(self._robot, "y") - state.get(target_obj, "pose_y")) \
            >= self.nextto_thresh:
            return next_state
        # Execute pick
        next_state.set(self._robot, "fingers", 0.0)
        next_state.set(target_obj, "grasp", grasp)
        next_state.set(target_obj, "held", 1.0)
        # NOTE: added this to assign picked obj position to
        # robot y and z
        next_state.set(target_obj, "pose_y", state.get(self._robot, "y"))
        next_state.set(target_obj, "pose_z",
                       state.get(target_obj, "pose_z") + 1.0)
        #
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
            raise utils.EnvironmentFailure("Box lid is closed.",
                                           {"offending_objects": {self._lid}})
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
        collider = self._get_object_at_xyz(state, x, y, z)
        if receptacle == "table" and \
           collider is not None and \
           collider != held_obj:
            raise utils.EnvironmentFailure("Collision during place on table.",
                                           {"offending_objects": {collider}})
        #  NOTE: added restriction on place if not close enough to location
        if abs(state.get(self._robot, "y") - y) \
            >= self.nextto_thresh:
            return next_state
        #
        # Execute place
        next_state.set(self._robot, "fingers", 1.0)
        next_state.set(held_obj, "pose_x", x)
        next_state.set(held_obj, "pose_y", y)
        next_state.set(held_obj, "pose_z", z)
        next_state.set(held_obj, "grasp", 0.5)
        next_state.set(held_obj, "held", 0.0)
        return next_state

    def _transition_move(self, state: State, action: Action) -> State:
        # Action args are target y for robot
        y = action.arr[1]
        next_state = state.copy()
        # Execute move
        next_state.set(self._robot, "y", y)
        held_obj = self._get_held_object(state)
        if held_obj is not None:
            next_state.set(held_obj, "pose_y", y)
        return next_state

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._InBox, self._InShelf, self._IsBoxColor, self._IsShelfColor,
            self._GripperOpen, self._OnTable, self._HoldingTop,
            self._HoldingSide, self._Holding, self._IsWet, self._IsDry,
            self._IsDirty, self._IsClean, self._NextTo, self._NextToBox,
            self._NextToShelf, self._NextToNothing
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {
            self._Pick, self._Wash, self._Dry, self._Paint, self._Place,
            self._OpenLid, self._MoveToObj, self._MoveToBox, self._MoveToShelf
        }

    @property
    def action_space(self) -> Box:
        # Actions are 8-dimensional vectors:
        # [x, y, z, grasp, pickplace, water level, heat level, color]
        # Note that pickplace is 1 for pick, -1 for place, and 0 otherwise,
        # while grasp, water level, heat level, and color are in [0, 1].
        lowers = np.array(
            [self.obj_x - 1e-2, self.env_lb, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            dtype=np.float32)
        uppers = np.array([
            self.obj_x + 1e-2, self.env_ub, self.obj_z + 1e-2, 1.0, 1.0, 1.0,
            1.0, 1.0
        ],
                          dtype=np.float32)
        return Box(lowers, uppers)

    def render(self,
               state: State,
               task: Task,
               action: Optional[Action] = None) -> List[Image]:
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

        # List of NextTo objects to render
        # NOTE: added this to display what objects we are nextto
        # during video rendering
        nextto_objs = []
        for obj in state:
            if obj.is_instance(self._obj_type) or \
                obj.is_instance(self._box_type) or \
                obj.is_instance(self._shelf_type):
                if abs(state.get(self._robot, "y") -
                       state.get(obj, "pose_y")) < self.nextto_thresh:
                    nextto_objs.append(obj)
        #

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
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(0.6, 1.0)
        # NOTE: added this to display what objects we are nextto
        # during video rendering
        plt.suptitle("blue = wet+clean, green = dry+dirty, cyan = dry+clean;\n"
                     "yellow border = side grasp, orange border = top grasp\n"
                     "NextTo: " + str(nextto_objs),
                     fontsize=12)
        #
        img = utils.fig2data(fig)
        plt.close()
        return [img]

    def _get_tasks(self, num_tasks: int, num_objs_lst: List[int],
                   rng: np.random.Generator) -> List[Task]:
        tasks = []
        for i in range(num_tasks):
            num_objs = num_objs_lst[i % len(num_objs_lst)]
            data = {}
            # NOTE: added this to assign robot pos at initialization
            # Initialize robot pos with open fingers
            robot_init_y = rng.uniform(self.table_lb, self.table_ub)
            data[self._robot] = np.array([self.robot_x, robot_init_y, 1.0],
                                         dtype=np.float32)
            #
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
            # Create moveable objects
            objs = []
            obj_poses: List[Tuple[float, float, float]] = []
            goal = set()
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
                # Last object should go in box
                if j == num_objs - 1:
                    goal.add(GroundAtom(self._InBox, [obj, self._box]))
                    goal.add(GroundAtom(self._IsBoxColor, [obj, self._box]))
                else:
                    goal.add(GroundAtom(self._InShelf, [obj, self._shelf]))
                    goal.add(GroundAtom(self._IsShelfColor,
                                        [obj, self._shelf]))
            state = State(data)
            # Sometimes start out holding an object, possibly with the wrong
            # grip, so that we'll have to put it on the table and regrasp
            if rng.uniform() < CFG.painting_initial_holding_prob:
                grasp = rng.choice([0.0, 1.0])
                target_obj = objs[rng.choice(len(objs))]
                state.set(self._robot, "fingers", 0.0)
                state.set(target_obj, "grasp", grasp)
                state.set(target_obj, "held", 1.0)
                # NOTE:  I added is to assign held_obj initial position to
                # robot y and z
                state.set(target_obj, "pose_y", state.get(self._robot, "y"))
                state.set(target_obj, "pose_z",
                          state.get(target_obj, "pose_z") + 1.0)
                #
            tasks.append(Task(state, goal))
        return tasks

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        obj_y = state.get(obj, "pose_y")
        return self.table_lb < obj_y < self.table_ub and \
            np.allclose(state.get(obj, "pose_z"), self.table_height\
                + self.obj_height / 2)

    @staticmethod
    def _Move_policy(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> Action:
        del memory  # unused
        _, obj = objects
        next_x = state.get(obj, "pose_x")
        next_y = params[0]
        return Action(
            np.array([next_x, next_y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                     dtype=np.float32))

    def _NextTo_holds(self, state: State, objects: Sequence[Object]) -> bool:
        robot, obj = objects
        return abs(state.get(robot, "y") -
                   state.get(obj, "pose_y")) < self.nextto_thresh

    def _NextToNothing_holds(self, state: State,
                             objects: Sequence[Object]) -> bool:
        robot, = objects
        for typed_obj in state:
            if typed_obj.type in \
                [self._obj_type, self._box_type, self._shelf_type] and \
                self._NextTo_holds(state, [robot, typed_obj]) and \
                typed_obj is not self._get_held_object(state):
                return False
        return True
