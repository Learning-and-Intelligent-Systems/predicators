"""An approach that wraps a base approach for a spot environment. Detects when
objects have been "lost" and executes a specific controller to "find" them.
Passes control back to the main approach once all objects are not lost.

Assumes that some objects in the environment have a feature called "lost" that
is 1.0 if the object is lost and 0.0 otherwise. This feature should be tracked
by a perceiver.

For now, the "find" policy is represented with a single action that is
extracted from the environment.
"""

import logging
from functools import cached_property
from typing import Callable, Optional, Set

import numpy as np

from predicators.approaches import BaseApproachWrapper
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv
from predicators.settings import CFG
from predicators.structs import Action, Object, State, Task


class SpotWrapperApproach(BaseApproachWrapper):
    """Always "find" if some object is lost."""

    @classmethod
    def get_name(cls) -> str:
        return "spot_wrapper"

    @property
    def is_learning_based(self) -> bool:
        return self._base_approach.is_learning_based

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        # Maintain policy from the base approach.
        base_approach_policy: Optional[Callable[[State], Action]] = None

        def _policy(state: State) -> Action:
            nonlocal base_approach_policy
            # If some objects are lost, find them.
            lost_objects: Set[Object] = set()
            for obj in state:
                if "lost" in obj.type.feature_names and \
                    state.get(obj, "lost") > 0.5:
                    lost_objects.add(obj)
            # Need to find the objects.
            if lost_objects:
                logging.info(f"Looking for lost objects: {lost_objects}")
                # Reset the base approach policy.
                base_approach_policy = None
                return self._find_action
            # Check if we need to re-solve.
            if base_approach_policy is None:
                cur_task = Task(state, task.goal)
                base_approach_policy = self._base_approach.solve(
                    cur_task, timeout)
            # Use the base policy.
            return base_approach_policy(state)

        return _policy

    @cached_property
    def _find_action(self) -> Action:
        env = get_or_create_env(CFG.env)
        assert isinstance(env, SpotEnv)
        # In the future, may want to make this object-specific.
        return env.get_find_action()
