"""A Sokoban environment wrapping https://github.com/mpSchrader/gym-sokoban."""
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import gym
import gym_sokoban
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box
from numpy.typing import NDArray

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, Video

# Free, goals, boxes, player masks.
_Observation = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8],
                     NDArray[np.uint8]]

# TODO: Big bug here: After we get a number of tasks, the simulator gets into the
# configuration of the last task. However, when we start planning, we need to
# set it to the current task we're trying to solve!


class SokobanEnv(BaseEnv):
    """Sokoban environment wrapping gym-sokoban."""

    _type_to_enum: ClassVar[Dict[str, int]] = {
        "free": 0,
        "goal": 1,
        "box": 2,
        "player": 3
    }

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._object_type = Type("obj", ["row", "column", "type"])

        # Predicates
        self._At = Predicate("At", [self._object_type, self._object_type],
                             self._At_holds)
        self._IsLoc = Predicate("Loc", [self._object_type], self._IsLoc_holds)
        self._NoBoxAtLoc = Predicate("NoBoxAtLoc", [self._object_type],
                                     self._NoBoxAtLoc_holds)
        self._Above = Predicate("Above",
                                [self._object_type, self._object_type],
                                self._Above_holds)
        self._Below = Predicate("Below",
                                [self._object_type, self._object_type],
                                self._Below_holds)
        self._RightOf = Predicate("RightOf",
                                  [self._object_type, self._object_type],
                                  self._RightOf_holds)
        self._LeftOf = Predicate("LeftOf",
                                 [self._object_type, self._object_type],
                                 self._LeftOf_holds)
        self._IsBox = Predicate("IsBox", [self._object_type],
                                self._IsBox_holds)
        self._IsPlayer = Predicate("IsPlayer", [self._object_type],
                                   self._IsPlayer_holds)
        self._IsGoal = Predicate("IsGoal", [self._object_type],
                                 self._IsGoal_holds)
        self._NotIsGoal = Predicate("NotIsGoal", [self._object_type],
                                    self._NotIsGoal_holds)
        self._GoalCovered = Predicate("GoalCovered", [self._object_type],
                                      self._GoalCovered_holds)

        # TODO: change to a different level?
        self._gym_env = gym.make("Sokoban-v0")

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, seed_offset=0)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               seed_offset=CFG.test_env_seed_offset)

    @classmethod
    def get_name(cls) -> str:
        return "sokoban"

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("A gym environment cannot render "
                                  "arbitrary states.")

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        assert caption is None
        arr = self._gym_env.render('rgb_array')
        return [arr]

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._At, self._GoalCovered, self._IsLoc, self._Above, self._Below,
            self._RightOf, self._LeftOf, self._IsBox, self._IsPlayer,
            self._NoBoxAtLoc, self._IsGoal, self._NotIsGoal
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._GoalCovered}

    @property
    def types(self) -> Set[Type]:
        return {self._object_type}

    @property
    def action_space(self) -> Box:
        # One-hot encoding of discrete action space.
        assert self._gym_env.action_space.n == 9
        lowers = np.zeros(9, dtype=np.float32)
        uppers = np.ones(9, dtype=np.float32)
        return Box(lowers, uppers)

    def get_state(self) -> State:
        obs = self._gym_env.render(mode='raw')
        return self._observation_to_state(obs)

    # def _fix_object_tracking_errors(self, state: State, next_state: State) -> State:
    #     # First, get a list of all the boxes that have changed their position.
    #     changed_boxes = []
    #     for obj in state:
    #         if (state.data[obj] != next_state.data[obj]).any() and state.get(obj, 'type') == self._type_to_enum['box']:
    #             changed_boxes.append(obj)
    #     # If this list has length greater than 1, there is an object-tracking issue
    #     # (since two boxes cannot change position simultaneously).
    #     if len(changed_boxes) <= 1:
    #         return next_state
    #     # This object-tracking issue can only happen when two boxes are confused
    #     # for one another.
    #     assert len(changed_boxes) == 2
    #     box0 = changed_boxes[0]
    #     box1 = changed_boxes[1]
    #     box0_final = next_state.data[box0]
    #     box1_final = next_state.data[box1]
    #     # We simply need to swap the data of these two boxes to make the
    #     # correction to the state!
    #     next_state.data[box0] = box1_final
    #     next_state.data[box1] = box0_final
    #     # We can now explicitly assert that this swapping works.
    #     new_changed_boxes = []
    #     for box in changed_boxes:
    #         if (state.data[box] != next_state.data[box]).any() and state.get(box, 'type') == self._type_to_enum['box']:
    #             new_changed_boxes.append(box)
    #     assert len(new_changed_boxes) == 1
    #     return next_state

    def simulate(self, state: State, action: Action) -> State:
        orig_simulator_state = state.simulator_state
        state.simulator_state = None
        if not state.allclose(self.get_state()):
            # If this check fails, then we've switched tasks
            # and need to reset our simulator to its original state.
            try:
                assert orig_simulator_state is not None
            except AssertionError:
                # TODO: need to fix object tracking between current state
                # and get_state()!
                import ipdb
                ipdb.set_trace()
            self._reset_initial_state_from_seed(orig_simulator_state)
            assert state.allclose(self.get_state())
        state.simulator_state = orig_simulator_state
        next_state = self.step(action)
        return next_state

    # def simulate(self, state: State, action: Action) -> State:
    #     next_state = self.step(action)
    #     return next_state

    def reset(self, train_or_test: str, task_idx: int) -> State:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_state = self._current_task.init
        # We now need to reset the underlying gym environment to the correct
        # state.
        seed_offset = CFG.seed
        if train_or_test == "test":
            seed_offset += CFG.test_env_seed_offset
        self._reset_initial_state_from_seed(seed_offset + task_idx)
        return self._current_state.copy()

    def step(self, action: Action) -> State:
        # Convert our actions to their discrete action space.
        discrete_action = np.argmax(action.arr)
        self._gym_env.step(discrete_action)
        obs = self._gym_env.render(mode='raw')
        self._current_state = self._observation_to_state(obs)
        return self._current_state.copy()

    @property
    def options(self) -> Set[ParameterizedOption]:  # pragma: no cover
        raise NotImplementedError(
            "This base class method will be deprecated soon!")

    def _get_tasks(self, num: int, seed_offset: int) -> List[Task]:
        tasks = []
        for i in range(num):
            seed = i + seed_offset + CFG.seed
            obs = self._reset_initial_state_from_seed(seed)
            init_state = self._observation_to_state(obs, seed)
            # The goal is always for all goal objects to be covered.
            goal_objs = [
                o for o in init_state
                if init_state.get(o, "type") == self._type_to_enum["goal"]
            ]
            goal = {GroundAtom(self._GoalCovered, [o]) for o in goal_objs}
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> _Observation:
        self._gym_env.seed(seed)
        self._gym_env.reset()
        return self._gym_env.render(mode='raw')

    def _observation_to_state(self,
                              obs: _Observation,
                              seed: Optional[int] = None) -> State:
        """Extract a State from a self._gym_env observation."""
        state_dict = {}

        walls, goals, boxes, player = obs
        type_to_mask = {
            "free": np.logical_not(walls | goals),
            "goal": goals,
            "box": boxes,
            "player": player
        }
        assert set(type_to_mask) == set(self._type_to_enum)

        for type_name, mask in type_to_mask.items():
            enum = self._type_to_enum[type_name]
            i = 0
            for r, c in np.argwhere(mask):
                # Put the location of the free spaces in the name for easier
                # debugging.
                if type_name == "free":
                    obj = Object(f"{type_name}_{r}_{c}", self._object_type)
                else:
                    obj = Object(f"{type_name}_{i}", self._object_type)
                state_dict[obj] = {
                    "row": r,
                    "column": c,
                    "type": enum,
                }
                i += 1

        state = utils.create_state_from_dict(state_dict)
        state.simulator_state = seed
        return state

    def _IsLoc_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # A location is 'free' if it has type 'free'.
        loc, = objects
        loc_type = state.get(loc, "type")
        return loc_type in {
            self._type_to_enum["free"], self._type_to_enum["goal"]
        }

    def _NoBoxAtLoc_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        # Only holds if the the object has a 'loc' type.
        if not (self._IsLoc_holds(state, objects)
                or self._IsGoal_holds(state, objects)):
            return False
        loc, = objects
        loc_r = state.get(loc, "row")
        loc_c = state.get(loc, "column")
        boxes = [
            o for o in state
            if state.get(o, "type") == self._type_to_enum["box"]
        ]
        # If any box is at this location, return False.
        for box in boxes:
            r = state.get(box, "row")
            c = state.get(box, "column")
            if r == loc_r and c == loc_c:
                return False
        return True

    def _IsBox_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        obj_type = state.get(obj, "type")
        return obj_type == self._type_to_enum["box"]

    def _IsGoal_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        obj_type = state.get(obj, "type")
        return obj_type == self._type_to_enum["goal"]

    def _NotIsGoal_holds(self, state: State,
                         objects: Sequence[Object]) -> bool:
        return self._IsLoc_holds(state, objects) and not \
            self._IsGoal_holds(state, objects)

    def _IsPlayer_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        obj_type = state.get(obj, "type")
        return obj_type == self._type_to_enum["player"]

    @staticmethod
    def _At_holds(state: State, objects: Sequence[Object]) -> bool:
        obj, loc, = objects
        obj_r = state.get(obj, "row")
        loc_r = state.get(loc, "row")
        obj_c = state.get(obj, "column")
        loc_c = state.get(loc, "column")
        return obj_r == loc_r and obj_c == loc_c

    @staticmethod
    def _Above_holds(state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2, = objects
        obj1_r = state.get(obj1, "row")
        obj1_c = state.get(obj1, "column")
        obj2_r = state.get(obj2, "row")
        obj2_c = state.get(obj2, "column")
        return ((obj1_r + 1) == obj2_r) and (obj1_c == obj2_c)

    @staticmethod
    def _Below_holds(state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2, = objects
        obj1_r = state.get(obj1, "row")
        obj1_c = state.get(obj1, "column")
        obj2_r = state.get(obj2, "row")
        obj2_c = state.get(obj2, "column")
        return ((obj1_r - 1) == obj2_r) and (obj1_c == obj2_c)

    @staticmethod
    def _RightOf_holds(state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2, = objects
        obj1_r = state.get(obj1, "row")
        obj1_c = state.get(obj1, "column")
        obj2_r = state.get(obj2, "row")
        obj2_c = state.get(obj2, "column")
        return (obj1_r == obj2_r) and ((obj1_c - 1) == obj2_c)

    @staticmethod
    def _LeftOf_holds(state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2, = objects
        obj1_r = state.get(obj1, "row")
        obj1_c = state.get(obj1, "column")
        obj2_r = state.get(obj2, "row")
        obj2_c = state.get(obj2, "column")
        return (obj1_r == obj2_r) and ((obj1_c + 1) == obj2_c)

    def _GoalCovered_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        goal, = objects
        if not self._IsGoal_holds(state, objects):
            return False
        goal_r = state.get(goal, "row")
        goal_c = state.get(goal, "column")
        boxes = [
            o for o in state
            if state.get(o, "type") == self._type_to_enum["box"]
        ]
        for box in boxes:
            r = state.get(box, "row")
            c = state.get(box, "column")
            if r == goal_r and c == goal_c:
                return True
        return False
