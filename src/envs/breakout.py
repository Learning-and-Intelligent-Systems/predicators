"""An Atari Breakout environment."""

from typing import ClassVar, Dict, List, Optional, Sequence, Set

import gym
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from gym.spaces import Box

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, Object, \
    ParameterizedOption, Predicate, State, Task, Type


class BreakoutEnv(BaseEnv):
    """An Atari Breakout environment."""

    def __init__(self) -> None:
        super().__init__()
        # Types
        self._paddle_type = Type("paddle", ["x"])
        self._ball_type = Type("ball", ["x", "y", "dx", "dy"])
        self._brick_type = Type("brick", ["x", "y", "alive"])
        # Predicates
        self._BrickAlive = Predicate("BrickAlive",
                                     [self._brick_type],
                                     self._BrickAlive_holds)
        self._BrickDead = Predicate("BrickDead",
                                     [self._brick_type],
                                     self._BrickDead_holds)
        # Options
        # TODO
        # Static objects (always exist no matter the settings).
        self._paddle = Object("paddle", self._paddle_type)
        # Gym environment.
        self._gym_env = gym.make("Breakout-v0")

    @classmethod
    def get_name(cls) -> str:
        return "breakout"

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not supported for Gym envs.")

    def reset(self, train_or_test: str, task_idx: int) -> State:
        if train_or_test == "train":
            seed_offset = 0
        else:
            assert train_or_test == "test"
            seed_offset = CFG.test_env_seed_offset
        seed = task_idx + seed_offset
        return self._reset_initial_state_from_seed(seed)

    def step(self, action: Action) -> State:
        # Actions are [0, 1, 2, 3] = ['NOOP', 'FIRE', 'RIGHT', 'LEFT'].
        continuous_action, = action.arr
        if continuous_action <= -0.5:
            gym_action = 3  # left
        elif continuous_action >= 0.5:
            gym_action = 2  # right
        else:
            assert -0.5 < continuous_action < 0.5
            gym_action = 0  # noop
        obs, _, _, _ = self._gym_env.step(gym_action)
        return self._observation_to_state(obs)

    def _generate_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks, seed_offset=0)

    def _generate_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks, seed_offset=CFG.test_env_seed_offset)

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._BrickAlive, self._BrickDead}

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._BrickDead}

    @property
    def types(self) -> Set[Type]:
        return {self._paddle_type, self._ball_type, self._brick_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return set()

    @property
    def action_space(self) -> Box:
        # Move the paddle left or right. Magnitudes don't matter.
        return Box(-1, 1, (1, ), dtype=np.float32)

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        raise NotImplementedError("Render state not supported for Gym envs.")

    def render(
        self,
        action: Optional[Action] = None,
        caption: Optional[str] = None) -> List[Image]:
        assert caption is None
        del action  # unused
        rgb_arr = self._gym_env.render(mode="rgb_array")
        return [rgb_arr]

    def _get_tasks(self, num: int, seed_offset: int) -> List[Task]:
        tasks = []
        for i in range(num):
            seed = i + seed_offset
            init_state = self._reset_initial_state_from_seed(seed)
            bricks = init_state.get_objects(self._brick_type)
            goal = {GroundAtom(self._BrickDead, [b]) for b in bricks}
            task = Task(init_state, goal)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> State:
        self._gym_env.seed(seed)
        self._gym_env.reset()
        # Firing starts the game.
        obs, _, _, _ = self._gym_env.step(1)
        init_state = self._observation_to_state(obs)
        return init_state

    def _observation_to_state(self, obs: NDArray[np.uint8]) -> State:
        """Extract a State from a self._gym_env observation."""
        # TODO
        bricks = [Object("brick0", self._brick_type)]
        return utils.create_state_from_dict({
            self._paddle: {
                "x": 10.0,
            },
            bricks[0]: {
                "x": 10.0,
                "y": 10.0,
                "alive": 1.0,
            }
        })

    @staticmethod
    def _BrickAlive_holds(state: State, objects: Sequence[Object]) -> bool:
        brick, = objects
        return state.get(brick, "alive") > 0.5

    def _BrickDead_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return not self._BrickAlive_holds(state, objects)
