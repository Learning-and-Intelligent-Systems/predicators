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


# Walls, goals, boxes, player masks
_Observation = Tuple[NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8], NDArray[np.uint8]]


class SokobanEnv(BaseEnv):
    """Sokoban environment wrapping gym-sokoban."""

    _type_to_enum: ClassVar[Dict[str, int]] = {
        "wall": 0,
        "goal": 1,
        "box": 2,
        "player": 3
    }

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        self._object_type = Type("obj", ["row", "column", "type"])

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
        return self._gym_env.render('rgb_array')

    @property
    def predicates(self) -> Set[Predicate]:
        # TODO
        return set()

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # TODO
        return set()

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
        import ipdb; ipdb.set_trace()

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
            # TODO
            goal = set()
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> _Observation:
        self._gym_env.seed(seed)
        self._gym_env.reset()
        walls, goals, boxes, player = self._gym_env.render(mode='raw')
        assert walls.shape == goals.shape == boxes.shape == player.shape
        return walls, goals, boxes, player

    def _observation_to_state(self, obs: _Observation) -> State:
        """Extract a State from a self._gym_env observation."""
        state_dict = {}

        walls, goals, boxes, player = obs
        type_to_mask = {
            "wall": walls,
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
