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
        # Additional Predicates
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
        # Additional Options
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
            (abs(1 - arr[4]), self._transition_nextto_pick_or_openlid),
            (wash_affinity, self._transition_wash),
            (dry_affinity, self._transition_dry),
            (paint_affinity, self._transition_paint),
            (abs(-1 - arr[4]), self._transition_nextto_place),
            (move_affinity, self._transition_move),
        ]
        _, transition_fn = min(affinities, key=lambda item: item[0])
        return transition_fn(state, action)

    def _transition_nextto_pick_or_openlid(self, state: State,
                                           action: Action) -> State:
        x, y, z, grasp = action.arr[:4]
        next_state = self._transition_pick_or_openlid(state, action)
        target_obj = self._get_object_at_xyz(state, x, y, z)
        # if PaintingEnv pick was sucessful check if the object was next to
        if target_obj is not None and state.get(
                target_obj, "held") == 0.0 and next_state.get(
                    target_obj, "held") == 1.0:
            # Added cannot pick if obj is too far
            if abs(state.get(self._robot, "y") - state.get(target_obj, "pose_y")) \
                >= self.nextto_thresh:
                return state
        return next_state

    def _transition_nextto_place(self, state: State, action: Action) -> State:
        # Action args are target pose for held obj
        x, y, z = action.arr[:3]
        next_state = self._transition_place(state, action)
        #  Added restriction on place if not close enough to location
        if abs(state.get(self._robot, "y") - y) \
            >= self.nextto_thresh:
            return state
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
        # Changed lower bound for z to 0.0 for RepeatedNextToPainting
        # This is needed to check affinity of the move action
        lowers = np.array(
            [self.obj_x - 1e-2, self.env_lb, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            dtype=np.float32)
        uppers = np.array([
            self.obj_x + 1e-2, self.env_ub, self.obj_z + 1e-2, 1.0, 1.0, 1.0,
            1.0, 1.0
        ],
                          dtype=np.float32)
        return Box(lowers, uppers)

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
