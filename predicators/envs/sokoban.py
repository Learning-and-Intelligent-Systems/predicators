"""A Sokoban environment wrapping https://github.com/mpSchrader/gym-sokoban."""
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from gym.spaces import Box
import gym
import gym_sokoban

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, Predicate, State, Task, Type, Video


# Free, goals, boxes, player masks
_Observation = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]


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
        self._At = Predicate("At", [self._object_type, self._object_type], self._At_holds)
        self._GoalCovered = Predicate("GoalCovered", [self._object_type], self._GoalCovered_holds)

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
        # TODO
        return {self._At, self._GoalCovered}

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

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for gym envs")

    def step(self, action: Action) -> State:
        # Convert our actions to their discrete action space.
        discrete_action = np.argmax(action.arr)
        self._gym_env.step(discrete_action)
        obs = self._gym_env.render(mode='raw')
        return self._observation_to_state(obs)

    @property
    def options(self) -> Set[ParameterizedOption]:  # pragma: no cover
        raise NotImplementedError(
            "This base class method will be deprecated soon!")

    def _get_tasks(self, num: int, seed_offset: int) -> List[Task]:
        tasks = []
        for i in range(num):
            seed = i + seed_offset
            obs = self._reset_initial_state_from_seed(seed)
            init_state = self._observation_to_state(obs)
            # The goal is always for all goal objects to be covered.
            goal_objs = [o for o in init_state if init_state.get(o, "type") == self._type_to_enum["goal"]]
            goal = {GroundAtom(self._GoalCovered, [o]) for o in goal_objs}
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> _Observation:
        # TODO: seeding doesn't seem to be working...
        self._gym_env.seed(seed)
        self._gym_env.reset()
        return self._gym_env.render(mode='raw')

    def _observation_to_state(self, obs: _Observation) -> State:
        """Extract a State from a self._gym_env observation."""
        state_dict = {}

        walls, goals, boxes, player = obs
        type_to_mask = {
            "free": np.logical_not(walls),
            "goal": goals,
            "box": boxes,
            "player": player
        }
        assert set(type_to_mask) == set(self._type_to_enum)

        for type_name, mask in  type_to_mask.items():
            enum = self._type_to_enum[type_name]
            for r, c in np.argwhere(mask):
                obj = Object(f"{type_name}_{r}_{c}", self._object_type)
                state_dict[obj] = {
                    "row": r,
                    "column": c,
                    "type": enum,
                }

        state = utils.create_state_from_dict(state_dict)
        return state

    @staticmethod
    def _At_holds(state: State, objects: Sequence[Object]) -> bool:
        # TODO
        return False
    
    def _GoalCovered_holds(self, state: State, objects: Sequence[Object]) -> bool:
        goal, = objects
        goal_r = state.get(goal, "row")
        goal_c = state.get(goal, "column")
        boxes = [o for o in state if state.get(o, "type") == self._type_to_enum["box"]]
        for box in boxes:
            r = state.get(box, "row")
            c = state.get(box, "column")
            if r == goal_r and c == goal_c:
                return True
        return False
