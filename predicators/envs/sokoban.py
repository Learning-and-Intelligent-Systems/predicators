"""A Sokoban environment wrapping https://github.com/mpSchrader/gym-sokoban."""
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import gym
import gym_sokoban  # pylint:disable=unused-import
import matplotlib
import numpy as np
from gym.spaces import Box
from numpy.typing import NDArray

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Object, Predicate, State, \
    EnvironmentTask, Type, Video

# Free, goals, boxes, player masks.
_Observation = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8],
                     NDArray[np.uint8]]


class SokobanEnv(BaseEnv):
    """Sokoban environment wrapping gym-sokoban."""

    _name_to_enum: ClassVar[Dict[str, int]] = {
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
        self._IsLoc = Predicate("IsLoc", [self._object_type],
                                self._IsLoc_holds)
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
        self._IsNonGoalLoc = Predicate("IsNonGoalLoc", [self._object_type],
                                       self._IsNonGoalLoc_holds)
        self._GoalCovered = Predicate("GoalCovered", [self._object_type],
                                      self._GoalCovered_holds)

        # NOTE: we can change the level by modifying what we pass
        # into gym.make here.
        self._gym_env = gym.make(CFG.sokoban_gym_name)

        # Used for object tracking of the boxes, which are the only objects
        # with ambiguity. The keys are the object names.
        self._box_loc_to_name: Dict[Tuple[int, int], str] = {}

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, seed_offset=0)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               seed_offset=CFG.test_env_seed_offset)

    @classmethod
    def get_name(cls) -> str:
        return "sokoban"

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
        arr = self._gym_env.render('rgb_array')
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

    def reset(self, train_or_test: str, task_idx: int) -> State:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        # NOTE: current_state will be deprecated soon in favor of current_obs.
        self._current_state = self._current_task.init
        # We now need to reset the underlying gym environment to the correct
        # state.
        seed_offset = CFG.seed
        if train_or_test == "test":
            seed_offset += CFG.test_env_seed_offset
        self._reset_initial_state_from_seed(seed_offset + task_idx)
        assert self.get_state().allclose(self._current_state)
        return self._current_state.copy()

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for gym envs.")

    def step(self, action: Action) -> State:
        # Convert our actions to their discrete action space.
        discrete_action = np.argmax(action.arr)
        self._gym_env.step(discrete_action)
        obs = self._gym_env.render(mode='raw')
        self._current_state = self._observation_to_state(obs)
        return self._current_state.copy()

    def _get_tasks(self, num: int, seed_offset: int) -> List[EnvironmentTask]:
        tasks = []
        for i in range(num):
            seed = i + seed_offset + CFG.seed
            obs = self._reset_initial_state_from_seed(seed)
            init_state = self._observation_to_state(obs)
            # The goal is always for all goal objects to be covered.
            goal_objs = self._get_objects_of_enum(init_state, "goal")
            goal = {GroundAtom(self._GoalCovered, [o]) for o in goal_objs}
            task = EnvironmentTask(init_state, goal)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> _Observation:
        self._gym_env.seed(seed)
        self._gym_env.reset()
        self._box_loc_to_name.clear()  # reset the object tracking dictionary
        return self._gym_env.render(mode='raw')

    def _observation_to_state(self, obs: _Observation) -> State:
        """Extract a State from an _Observation."""
        state_dict = {}

        walls, goals, boxes, player = obs
        type_to_mask = {
            "free": np.logical_not(walls | goals),
            "goal": goals,
            "box": boxes,
            "player": player
        }
        assert set(type_to_mask) == set(self._name_to_enum)

        # Handle moving boxes.
        new_locs = set((r, c) for r, c in np.argwhere(boxes))
        if not self._box_loc_to_name:
            # First time, so name the boxes arbitrarily.
            for i, (r, c) in enumerate(sorted(new_locs)):
                self._box_loc_to_name[(r, c)] = f"box_{i}"
        else:
            # Assume that at most one box has changed.
            old_locs = set(self._box_loc_to_name)
            changed_new_locs = new_locs - old_locs
            changed_old_locs = old_locs - new_locs
            if changed_new_locs:
                assert len(changed_new_locs) == 1
                assert len(changed_old_locs) == 1
                new_loc, = changed_new_locs
                old_loc, = changed_old_locs
                moved_box_name = self._box_loc_to_name.pop(old_loc)
                self._box_loc_to_name[new_loc] = moved_box_name

        def _get_object_name(r: int, c: int, type_name: str) -> str:
            # Put the location of the static objects in their names for easier
            # debugging.
            if type_name in {"free", "goal"}:
                return f"{type_name}_{r}_{c}"
            if type_name == "player":
                return "player"
            assert type_name == "box"
            return self._box_loc_to_name[(r, c)]

        for type_name, mask in type_to_mask.items():
            enum = self._name_to_enum[type_name]
            i = 0
            for r, c in np.argwhere(mask):
                object_name = _get_object_name(r, c, type_name)
                obj = Object(object_name, self._object_type)
                state_dict[obj] = {
                    "row": r,
                    "column": c,
                    "type": enum,
                }
                i += 1

        state = utils.create_state_from_dict(state_dict)
        return state

    @classmethod
    def _IsLoc_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        # Free spaces and goals are locations.
        loc, = objects
        obj_type = state.get(loc, "type")
        return obj_type in {
            cls._name_to_enum["free"], cls._name_to_enum["goal"]
        }

    def _NoBoxAtLoc_holds(self, state: State,
                          objects: Sequence[Object]) -> bool:
        # Only holds if the the object has a 'loc' type.
        if not self._IsLoc_holds(state, objects):
            return False
        loc, = objects
        loc_r = state.get(loc, "row")
        loc_c = state.get(loc, "column")
        boxes = self._get_objects_of_enum(state, "box")
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

    def _GoalCovered_holds(self, state: State,
                           objects: Sequence[Object]) -> bool:
        goal, = objects
        if not self._IsGoal_holds(state, objects):
            return False
        goal_r = state.get(goal, "row")
        goal_c = state.get(goal, "column")
        boxes = self._get_objects_of_enum(state, "box")
        for box in boxes:
            r = state.get(box, "row")
            c = state.get(box, "column")
            if r == goal_r and c == goal_c:
                return True
        return False

    def _get_objects_of_enum(self, state: State,
                             enum_name: str) -> Set[Object]:
        return {
            o
            for o in state
            if state.get(o, "type") == self._name_to_enum[enum_name]
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
        return obj_type == cls._name_to_enum[enum_name]

    @classmethod
    def _is_static(cls, obj: Object, state: State) -> bool:
        return cls._IsGoal_holds(state, [obj]) or \
               cls._IsNonGoalLoc_holds(state, [obj])

    @classmethod
    def _is_dynamic(cls, obj: Object, state: State) -> bool:
        return not cls._is_static(obj, state)
