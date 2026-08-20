"""Base class for an explorer."""

import abc
import itertools
import logging
from typing import List, Set

import numpy as np
from gym.spaces import Box

from predicators.settings import CFG
from predicators.structs import ExplorationStrategy, ParameterizedOption, \
    Predicate, State, Task, Type

_RNG_COUNT = itertools.count()  # make sure RNG changes per instantiation


class BaseExplorer(abc.ABC):
    """Creates a policy and termination function for exploring in a task.

    The explorer is created at the beginning of every interaction cycle
    with the latest predicates and options.
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int) -> None:
        self._predicates = predicates
        self._options = options
        self._types = types
        self._action_space = action_space
        self._train_tasks = train_tasks
        self._max_steps_before_termination = max_steps_before_termination
        self._set_seed(CFG.seed)

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this explorer."""
        raise NotImplementedError("Override me!")

    def get_exploration_strategy(
        self,
        train_task_idx: int,
        timeout: int,
        log_info: bool = False,
    ) -> ExplorationStrategy:
        """Wrap the base exploration strategy."""

        policy, termination_fn = self._get_exploration_strategy(
            train_task_idx, timeout)

        # Terminate after the given number of steps.
        remaining_steps = self._max_steps_before_termination

        def wrapped_termination_fn(state: State) -> bool:
            nonlocal remaining_steps
            if termination_fn(state):
                logging.info("[Base Explorer] terminating due to term fn")
                return True
            steps_taken = self._max_steps_before_termination - remaining_steps
            actual_remaining_steps = min(
                remaining_steps,
                CFG.max_num_steps_interaction_request - steps_taken)
            if actual_remaining_steps <= 0:
                logging.info("[Base Explorer] terminating due to max steps")
                return True
            if log_info:
                logging.info(
                    "[Base Explorer] not yet terminating (remaining steps: "
                    f"{actual_remaining_steps})")
            remaining_steps -= 1
            return False

        return policy, wrapped_termination_fn

    @abc.abstractmethod
    def _get_exploration_strategy(
        self,
        train_task_idx: int,
        timeout: int,
    ) -> ExplorationStrategy:
        """Given a train task idx, create an ExplorationStrategy, which is a
        tuple of a policy and a termination function."""
        raise NotImplementedError("Override me!")

    def _set_seed(self, seed: int) -> None:
        """Reset seed and rng."""
        self._seed = seed
        self._rng = np.random.default_rng(self._seed + next(_RNG_COUNT))
