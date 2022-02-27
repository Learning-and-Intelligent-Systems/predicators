"""Base class for an environment."""

import abc
from typing import List, Set, Optional
import numpy as np
from gym.spaces import Box
from predicators.src.structs import State, Task, Predicate, \
    ParameterizedOption, Type, Action, Image
from predicators.src.settings import CFG


class BaseEnv(abc.ABC):
    """Base environment."""

    def __init__(self) -> None:
        self.seed(CFG.seed)

    @abc.abstractmethod
    def simulate(self, state: State, action: Action) -> State:
        """Get the next state, given a state and an action.

        Note that this action is a low-level action (i.e., its array
        representation is a member of self.action_space), NOT an option.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_train_tasks(self) -> List[Task]:
        """Get an ordered list of tasks for training."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_test_tasks(self) -> List[Task]:
        """Get an ordered list of tasks for testing / evaluation."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def predicates(self) -> Set[Predicate]:
        """Get the set of predicates that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def goal_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def types(self) -> Set[Type]:
        """Get the set of types that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def options(self) -> Set[ParameterizedOption]:
        """Get the set of parameterized options that are given with this
        environment."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def action_space(self) -> Box:
        """Get the action space of this environment."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def render(self,
               state: State,
               task: Task,
               action: Optional[Action] = None) -> List[Image]:
        """Render a state and action into a list of images."""
        raise NotImplementedError("Override me!")

    def seed(self, seed: int) -> None:
        """Reset seed and rngs."""
        self._seed = seed
        # The train/test rng should be used when generating
        # train/test tasks respectively.
        self._train_rng = np.random.default_rng(self._seed)
        self._test_rng = np.random.default_rng(self._seed +
                                               CFG.test_env_seed_offset)
