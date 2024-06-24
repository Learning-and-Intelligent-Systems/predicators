"""A Sokoban environment wrapping https://github.com/mpSchrader/gym-sokoban."""
from typing import ClassVar, Dict, List, Optional, Sequence, Set

import gym
import matplotlib
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Image, Object, \
    Observation, Predicate, State, Type, Video


class SokobanEnv(BaseEnv):
    """Sokoban environment wrapping gym-sokoban."""

    name_to_enum: ClassVar[Dict[str, int]] = {
        "free": 0,
        "goal": 1,
        "box": 2,
        "player": 3
    }

    object_type = Type("obj", ["row", "column", "type"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._At = Predicate("At", [self.object_type, self.object_type],
                             self._At_holds)
        self._IsLoc = Predicate("IsLoc", [self.object_type], self._IsLoc_holds)
        self._NoBoxAtLoc = Predicate("NoBoxAtLoc", [self.object_type],
                                     self._NoBoxAtLoc_holds)
        self._Above = Predicate("Above", [self.object_type, self.object_type],
                                self._Above_holds)
        self._Below = Predicate("Below", [self.object_type, self.object_type],
                                self._Below_holds)
        self._RightOf = Predicate("RightOf",
                                  [self.object_type, self.object_type],
                                  self._RightOf_holds)
        self._LeftOf = Predicate("LeftOf",
                                 [self.object_type, self.object_type],
                                 self._LeftOf_holds)
        self._IsBox = Predicate("IsBox", [self.object_type], self._IsBox_holds)
        self._IsPlayer = Predicate("IsPlayer", [self.object_type],
                                   self._IsPlayer_holds)
        self._IsGoal = Predicate("IsGoal", [self.object_type],
                                 self._IsGoal_holds)
        self._IsNonGoalLoc = Predicate("IsNonGoalLoc", [self.object_type],
                                       self._IsNonGoalLoc_holds)
        self._GoalCovered = self.get_goal_covered_predicate()

        # NOTE: we can change the level by modifying what we pass
        # into gym.make here.
        self._gym_env = gym.make(CFG.sokoban_gym_name)

    @classmethod
    def get_goal_covered_predicate(cls) -> Predicate:
        """Defined public so that the perceiver can use it."""
        return Predicate("GoalCovered", [cls.object_type],
                         cls._GoalCovered_holds)

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, train_or_test="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, train_or_test="test")

    @classmethod
    def get_name(cls) -> str:
        return "sokoban"

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("A gym environment cannot render "
                                  "arbitrary states.")

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        assert caption is None
        arr: Image = self._gym_env.render('rgb_array')  # type: ignore
        return [arr]

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._At, self._GoalCovered, self._IsLoc, self._Above, self._Below,
            self._RightOf, self._LeftOf, self._IsBox, self._IsPlayer,
            self._NoBoxAtLoc, self._IsGoal, self._IsNonGoalLoc
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._GoalCovered}

    @property
    def types(self) -> Set[Type]:
        return {self.object_type}

    @property
    def action_space(self) -> Box:
        # One-hot encoding of discrete action space.
        assert self._gym_env.action_space.n == 9  # type: ignore
        lowers = np.zeros(9, dtype=np.float32)
        uppers = np.ones(9, dtype=np.float32)
        return Box(lowers, uppers)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = self._current_task.init_obs
        # We now need to reset the underlying gym environment to the correct
        # state.
        seed = utils.get_task_seed(train_or_test, task_idx)
        self._reset_initial_state_from_seed(seed)
        return self._copy_observation(self._current_observation)

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for gym envs. " +
                                  "Try using --bilevel_plan_without_sim True")

    def step(self, action: Action) -> Observation:
        # Convert our actions to their discrete action space.
        discrete_action = np.argmax(action.arr)
        self._gym_env.step(discrete_action)
        self._current_observation = self._gym_env.render(
            mode='raw')  # type: ignore
        return self._copy_observation(self._current_observation)

    def goal_reached(self) -> bool:
        _, goals, boxes, _ = self._current_observation
        return not np.any(boxes & np.logical_not(goals))

    def _get_tasks(self, num: int,
                   train_or_test: str) -> List[EnvironmentTask]:
        tasks = []
        for task_idx in range(num):
            seed = utils.get_task_seed(train_or_test, task_idx)
            init_obs = self._reset_initial_state_from_seed(seed)
            goal_description = "Cover all the goals with boxes"
            task = EnvironmentTask(init_obs, goal_description)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> Observation:
        self._gym_env.seed(seed)  # type: ignore
        self._gym_env.reset()
        return self._gym_env.render(mode='raw')  # type: ignore

    @classmethod
    def _IsLoc_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        # Free spaces and goals are locations.
        loc, = objects
        obj_type = state.get(loc, "type")
        return obj_type in {cls.name_to_enum["free"], cls.name_to_enum["goal"]}

    def _NoBoxAtLoc_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        # Only holds if the the object has a 'loc' type.
        if not self._IsLoc_holds(state, objects):
            return False
        loc, = objects
        loc_r = state.get(loc, "row")
        loc_c = state.get(loc, "column")
        boxes = self.get_objects_of_enum(state, "box")
        # If any box is at this location, return False.
        for box in boxes:
            r = state.get(box, "row")
            c = state.get(box, "column")
            if r == loc_r and c == loc_c:
                return False
        return True

    @classmethod
    def _IsBox_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "box")

    @classmethod
    def _IsGoal_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "goal")

    @classmethod
    def _IsPlayer_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "player")

    @classmethod
    def _IsNonGoalLoc_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "free")

    @classmethod
    def _At_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        if not cls._is_dynamic(obj1, state):
            return False
        if not cls._is_static(obj2, state):
            return False
        return cls._check_spatial_relation(state, objects, 0, 0)

    @classmethod
    def _Above_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        if not (cls._is_static(obj1, state) and cls._is_static(obj2, state)):
            return False
        return cls._check_spatial_relation(state, objects, 1, 0)

    @classmethod
    def _Below_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        if not (cls._is_static(obj1, state) and cls._is_static(obj2, state)):
            return False
        return cls._check_spatial_relation(state, objects, -1, 0)

    @classmethod
    def _RightOf_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        if not (cls._is_static(obj1, state) and cls._is_static(obj2, state)):
            return False
        return cls._check_spatial_relation(state, objects, 0, -1)

    @classmethod
    def _LeftOf_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        if not (cls._is_static(obj1, state) and cls._is_static(obj2, state)):
            return False
        return cls._check_spatial_relation(state, objects, 0, 1)

    @classmethod
    def _GoalCovered_holds(cls, state: State,
                           objects: Sequence[Object]) -> bool:
        goal, = objects
        if not cls._IsGoal_holds(state, objects):
            return False
        goal_r = state.get(goal, "row")
        goal_c = state.get(goal, "column")
        boxes = cls.get_objects_of_enum(state, "box")
        for box in boxes:
            r = state.get(box, "row")
            c = state.get(box, "column")
            if r == goal_r and c == goal_c:
                return True
        return False

    @classmethod
    def get_objects_of_enum(cls, state: State, enum_name: str) -> Set[Object]:
        """Made public for use by perceiver."""
        return {
            o
            for o in state
            if state.get(o, "type") == cls.name_to_enum[enum_name]
        }

    @classmethod
    def _check_spatial_relation(cls, state: State, objects: Sequence[Object],
                                dr: int, dc: int) -> bool:
        obj1, obj2 = objects
        obj1_r = state.get(obj1, "row")
        obj1_c = state.get(obj1, "column")
        obj2_r = state.get(obj2, "row")
        obj2_c = state.get(obj2, "column")
        return ((obj1_r + dr) == obj2_r) and ((obj1_c + dc) == obj2_c)

    @classmethod
    def _check_enum(cls, state: State, objects: Sequence[Object],
                    enum_name: str) -> bool:
        obj, = objects
        obj_type = state.get(obj, "type")
        return obj_type == cls.name_to_enum[enum_name]

    @classmethod
    def _is_static(cls, obj: Object, state: State) -> bool:
        return cls._IsGoal_holds(state, [obj]) or \
               cls._IsNonGoalLoc_holds(state, [obj])

    @classmethod
    def _is_dynamic(cls, obj: Object, state: State) -> bool:
        return not cls._is_static(obj, state)

    def _copy_observation(self, obs: Observation) -> Observation:
        return tuple(m.copy() for m in obs)
