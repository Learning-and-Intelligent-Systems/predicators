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
        self._current_state = State({})  # set in reset
        self.seed(CFG.seed)
        self._train_tasks = self._generate_train_tasks()
        self._test_tasks = self._generate_test_tasks()

    @abc.abstractmethod
    def simulate(self, state: State, action: Action) -> State:
        """Get the next state, given a state and an action.

        Note that this action is a low-level action (i.e., its array
        representation is a member of self.action_space), NOT an option.

        This function is primarily used in the default option model, and
        for implementing the default self.step(action). It is not meant to
        be part of the "final system", where the environment is the real world.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _generate_train_tasks(self) -> List[Task]:
        """Create an ordered list of tasks for training."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _generate_test_tasks(self) -> List[Task]:
        """Create an ordered list of tasks for testing / evaluation."""
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

    def get_train_tasks(self) -> List[Task]:
        """Return the ordered list of tasks for training."""
        return self._train_tasks

    def get_test_tasks(self) -> List[Task]:
        """Return the ordered list of tasks for testing / evaluation."""
        return self._test_tasks

    def seed(self, seed: int) -> None:
        """Reset seed and rngs."""
        self._seed = seed
        # The train/test rng should be used when generating
        # train/test tasks respectively.
        self._train_rng = np.random.default_rng(self._seed)
        self._test_rng = np.random.default_rng(self._seed +
                                               CFG.test_env_seed_offset)

    def reset(self, train_or_test: str, test_task_idx: int) -> None:
        """Resets the current state to the train or test task initial state."""
        if train_or_test == "train":
            tasks = self._train_tasks
        elif train_or_test == "test":
            tasks = self._test_tasks
        else:
            raise ValueError(f"Reset called with invalid train_or_test:"
                             f"{train_or_test}.")
        self._current_state = tasks[test_task_idx].init
        # Copy to prevent external changes to the environment's state.
        return self._current_state.copy()

    def step(self, action: Action) -> State:
        """Apply the action, and update and return the current state.

        Note that this action is a low-level action (i.e., its array
        representation is a member of self.action_space), NOT an option.

        By default, this funciton just calls self.simulate. However,
        environments that maintain a more complicated internal state may
        override this method.
        """
        self._current_state = self.simulate(self._current_state, action)
        # Copy to prevent external changes to the environment's state.
        return self._current_state.copy()
