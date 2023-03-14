"""Cognitive manager (CogMan).

A wrapper around an approach that manages interaction with the environment.

Implements a perception module, which produces a State at each time step based
on the history of observations, and an execution monitor, which determines
whether to re-query the approach at each time step based on the states.

The name "CogMan" is due to Leslie Kaelbling.
"""
from typing import Callable, Optional, Set

from predicators.approaches import BaseApproach
from predicators.execution_monitoring import create_execution_monitor
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, \
    Observation, State, Task


class CogMan:
    """Cognitive manager."""

    def __init__(self, approach: BaseApproach) -> None:
        self._approach = approach
        self._perceiver = create_perceiver(CFG.perceiver)
        self._exec_monitor = create_execution_monitor(CFG.execution_monitor)
        self._current_policy: Optional[Callable[[State], Action]] = None
        self._current_goal: Optional[Set[GroundAtom]] = None

    def reset(self, env_task: EnvironmentTask) -> None:
        """Start a new episode of environment interaction."""
        task = self._perceiver.reset(env_task)
        self._current_goal = task.goal
        self._current_policy = self._approach.solve(task, timeout=CFG.timeout)
        self._exec_monitor.reset(task)

    def step(self, observation: Observation) -> Action:
        """Receive an observation and produce an action."""
        state = self._perceiver.step(observation)
        # Check if we should replan.
        if self._exec_monitor.step(state):
            assert self._current_goal is not None
            task = Task(state, self._current_goal)
            new_policy = self._approach.solve(task, timeout=CFG.timeout)
            self._current_policy = new_policy
            self._exec_monitor.reset(task)
        assert self._current_policy is not None
        act = self._current_policy(state)
        return act
