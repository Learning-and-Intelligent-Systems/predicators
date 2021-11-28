"""Painting domain, which allows for two different grasps on an object
(side or top). Side grasping allows for placing into the shelf, and top
grasping allows for placing into the box. The box has a lid which
may need to be opened; this lid is NOT modeled by any of the given
predicates, but could be modeled by a learned predicate.
"""

from typing import List, Set, Sequence, Dict, Tuple, Optional
import numpy as np
from gym.spaces import Box
from matplotlib import pyplot as plt
from matplotlib import patches
from predicators.src.envs import BaseEnv
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG
from predicators.src import utils


class PaintingEnv(BaseEnv):
    """Painting domain.
    """
    # Parameters that aren't important enough to need to clog up settings.py
    table_lb = -10.1
    table_ub = -0.2
    table_height = 0.2
    shelf_l = 2.0 # shelf length
    shelf_lb = 1.
    shelf_ub = shelf_lb + shelf_l - 0.05
    box_s = 0.8  # side length
    box_y = 0.5  # y coordinate
    box_lb = box_y - box_s/10.
    box_ub = box_y + box_s/10.
    obj_height = 0.13
    obj_radius = 0.03
    obj_x = 1.65
    pick_tol = 1e-1
    color_tol = 1e-2
    wetness_tol = 0.5
    dirtiness_tol = 0.5
    open_fingers = 0.8
    top_grasp_thresh = 0.5 + 1e-5
    side_grasp_thresh = 0.5 - 1e-5
    held_tol = 0.5

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._obj_type = Type("obj", ["pose_x", "pose_y", "pose_z", "color",
                                      "wetness", "dirtiness", "held"])
        self._box_type = Type("box", ["color"])
        self._lid_type = Type("lid", ["open"])
        self._shelf_type = Type("shelf", ["color"])
        self._robot_type = Type("robot", ["gripper_rot", "fingers"])
        # Predicates
        self._InBox = Predicate(
            "InBox", [self._obj_type, self._box_type], self._InBox_holds)
        self._InShelf = Predicate(
            "InShelf", [self._obj_type, self._shelf_type], self._InShelf_holds)
        self._IsBoxColor = Predicate(
            "IsBoxColor", [self._obj_type, self._box_type],
            self._IsBoxColor_holds)
        self._IsShelfColor = Predicate(
            "IsShelfColor", [self._obj_type, self._shelf_type],
            self._IsShelfColor_holds)
        self._GripperOpen = Predicate(
            "GripperOpen", [self._robot_type], self._GripperOpen_holds)
        self._OnTable = Predicate(
            "OnTable", [self._obj_type], self._OnTable_holds)
        self._HoldingTop = Predicate(
            "HoldingTop", [self._obj_type, self._robot_type],
            self._HoldingTop_holds)
        self._HoldingSide = Predicate(
            "HoldingSide", [self._obj_type, self._robot_type],
            self._HoldingSide_holds)
        self._Holding = Predicate(
            "Holding", [self._obj_type], self._Holding_holds)
        self._IsWet = Predicate(
            "IsWet", [self._obj_type], self._IsWet_holds)
        self._IsDry = Predicate(
            "IsDry", [self._obj_type], self._IsDry_holds)
        self._IsDirty = Predicate(
            "IsDirty", [self._obj_type], self._IsDirty_holds)
        self._IsClean = Predicate(
            "IsClean", [self._obj_type], self._IsClean_holds)
        # Options
        # Objects
        self._box = Object("receptacle_box", self._box_type)
        self._lid = Object("box_lid", self._lid_type)
        self._shelf = Object("receptacle_shelf", self._shelf_type)
        self._robot = Object("robby", self._robot_type)

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError

    def get_train_tasks(self) -> List[Task]:
        raise NotImplementedError

    def get_test_tasks(self) -> List[Task]:
        raise NotImplementedError

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._InBox, self._InShelf, self._IsBoxColor,
                self._IsShelfColor, self._GripperOpen, self._OnTable,
                self._HoldingTop, self._HoldingSide, self._Holding,
                self._IsWet, self._IsDry, self._IsDirty, self._IsClean}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._InBox, self._InShelf, self._IsBoxColor,
                self._IsShelfColor}

    @property
    def types(self) -> Set[Type]:
        raise NotImplementedError

    @property
    def options(self) -> Set[ParameterizedOption]:
        raise NotImplementedError

    @property
    def action_space(self) -> Box:
        raise NotImplementedError

    def render(self, state: State, task: Task,
               action: Optional[Action] = None) -> List[Image]:
        raise NotImplementedError  # TODO

    def _InBox_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        # If the object is held, not yet in box
        if state.get(obj, "held") > 0.5:
            return False
        # Check pose of object
        obj_y = state.get(obj, "pose_y")
        return self.box_lb < obj_y < self.box_ub

    def _InShelf_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, _ = objects
        # If the object is held, not yet in shelf
        if state.get(obj, "held") > 0.5:
            return False
        # Check pose of object
        obj_y = state.get(obj, "pose_y")
        return self.shelf_lb < obj_y < self.shelf_ub

    def _IsBoxColor_holds(self, state: State, objects: Sequence[Object]
                          ) -> bool:
        obj, box = objects
        return abs(state.get(obj, "color") -
                   state.get(box, "color")) < self.color_tol

    def _IsShelfColor_holds(self, state: State, objects: Sequence[Object]
                            ) -> bool:
        obj, shelf = objects
        return abs(state.get(obj, "color") -
                   state.get(shelf, "color")) < self.color_tol

    def _GripperOpen_holds(self, state: State, objects: Sequence[Object]
                           ) -> bool:
        robot, = objects
        fingers = state.get(robot, "fingers")
        return fingers >= self.open_fingers

    def _OnTable_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        obj_y = state.get(obj, "pose_y")
        return self.table_lb < obj_y < self.table_ub

    def _HoldingTop_holds(self, state: State, objects: Sequence[Object]
                          ) -> bool:
        obj, robot = objects
        rot = state.get(robot, "gripper_rot")
        if rot < self.top_grasp_thresh:
            return False
        return self._Holding_holds(state, obj)

    def _HoldingSide_holds(self, state: State, objects: Sequence[Object]
                           ) -> bool:
        obj, robot = objects
        rot = state.get(robot, "gripper_rot")
        if rot > self.side_grasp_thresh:
            return False
        return self._Holding_holds(state, obj)

    def _Holding_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return self._get_held_object(state) == obj

    def _IsWet_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "wetness") > self.wetness_tol

    def _IsDry_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return not self._IsWet_holds(state, obj)

    def _IsDirty_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "dirtiness") > self.dirtiness_tol

    def _IsClean_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return not self._IsDirty_holds(state, obj)

    def _get_held_object(self, state):
        for obj in state:
            if obj.var_type != self._obj_type:
                continue
            if state.get(obj, "held") >= self.held_tol:
                return obj
        return None
