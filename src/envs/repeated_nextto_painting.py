"""RepeatedNextToPainting domain, which is a merge of our Painting and
RepeatedNextTo environments.

It is exactly the same as the Painting domain, but requires movement to
navigate between objects in order to pick or place them. Also, the move
option can turn on any number of NextTo predicates.
"""

from typing import Dict, List, Optional, Sequence, Set

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs.painting import PaintingEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, Object, \
    ParameterizedOption, Predicate, State, Task


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
        self._NextToTable = Predicate("NextToTable", [self._robot_type],
                                      self._NextToTable_holds)
        # Additional Options
        self._MoveToObj = utils.SingletonParameterizedOption(
            "MoveToObj",
            self._Move_policy,
            types=[self._robot_type, self._obj_type],
            params_space=Box(self.env_lb, self.env_ub, (1, )))
        self._MoveToBox = utils.SingletonParameterizedOption(
            "MoveToBox",
            self._Move_policy,
            types=[self._robot_type, self._box_type],
            params_space=Box(self.env_lb, self.env_ub, (1, )))
        self._MoveToShelf = utils.SingletonParameterizedOption(
            "MoveToShelf",
            self._Move_policy,
            types=[self._robot_type, self._shelf_type],
            params_space=Box(self.env_lb, self.env_ub, (1, )))

    @classmethod
    def get_name(cls) -> str:
        return "repeated_nextto_painting"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        arr = action.arr
        move_affinity = sum(abs(val) for val in arr[2:])
        if move_affinity <= 1e-4:
            return self._transition_move(state, action)
        return super().simulate(state, action)

    def _transition_pick_or_openlid(self, state: State,
                                    action: Action) -> State:
        x, y, z, _ = action.arr[:4]
        next_state = super()._transition_pick_or_openlid(state, action)
        target_obj = self._get_object_at_xyz(state, x, y, z)
        # In this environment, we disallow picking an object if the robot
        # is not currently next to it. To implement this, whenever the
        # parent class's pick is successful, we check the NextTo constraint,
        # and just return the current state if it fails.
        if target_obj is not None and \
           state.get(target_obj, "held") < 0.5 \
           < next_state.get(target_obj, "held"):
            abs_state_diff = abs(state.get(self._robot, "pose_y") \
                - state.get(target_obj, "pose_y"))
            if abs_state_diff >= self.nextto_thresh:
                return state
        return next_state

    def _transition_place(self, state: State, action: Action) -> State:
        # Action args are target pose for held obj
        y = action.arr[1]
        next_state = super()._transition_place(state, action)
        # In this environment, we disallow placing an object if the robot
        # is not currently next to the target place pose. To implement this,
        # whenever the parent class's place is successful, we check the
        # NextTo constraint, and just return the current state if it fails.
        if abs(state.get(self._robot, "pose_y") - y) >= self.nextto_thresh:
            return state
        return next_state

    def _transition_move(self, state: State, action: Action) -> State:
        # Action args are target y for robot
        y = action.arr[1]
        next_state = state.copy()
        # Execute move
        next_state.set(self._robot, "pose_y", y)
        held_obj = self._get_held_object(state)
        if held_obj is not None:
            next_state.set(held_obj, "pose_y", y)
        return next_state

    @property
    def predicates(self) -> Set[Predicate]:
        return super().predicates | {
            self._NextTo, self._NextToBox, self._NextToShelf, self._NextToTable
        }

    @property
    def options(self) -> Set[ParameterizedOption]:
        return super().options | {
            self._MoveToObj, self._MoveToBox, self._MoveToShelf
        }

    @property
    def _num_objects_train(self) -> List[int]:
        return CFG.rnt_painting_num_objs_train

    @property
    def _num_objects_test(self) -> List[int]:
        return CFG.rnt_painting_num_objs_test

    @property
    def _max_objs_in_goal(self) -> int:
        return CFG.rnt_painting_max_objs_in_goal

    @property
    def _update_z_poses(self) -> bool:
        return True

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        # List of NextTo objects to render
        nextto_objs = []
        for obj in state:
            if obj.is_instance(self._obj_type) or \
                obj.is_instance(self._box_type) or \
                obj.is_instance(self._shelf_type):
                if abs(
                        state.get(self._robot, "pose_y") -
                        state.get(obj, "pose_y")) < self.nextto_thresh:
                    nextto_objs.append(obj)
        # Call the parent's renderer, but include information about what
        # objects we are NextTo as a caption
        assert caption is None
        return super().render_state_plt(state, task, caption="NextTo: " + \
            str(nextto_objs))

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
        return abs(state.get(robot, "pose_y") -
                   state.get(obj, "pose_y")) < self.nextto_thresh

    def _NextToTable_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        robot, = objects
        return self.table_lb < state.get(robot, "pose_y") < self.table_ub
