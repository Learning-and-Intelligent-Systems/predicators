"""Cognitive manager (CogMan).

A wrapper around an approach that manages interaction with the environment.

Implements a perception module, which produces a State at each time step based
on the history of observations, and an execution monitor, which determines
whether to re-query the approach at each time step based on the states.

The name "CogMan" is due to Leslie Kaelbling.
"""
from typing import Callable, List, Optional, Sequence, Set

from predicators.approaches import BaseApproach
from predicators.execution_monitoring import BaseExecutionMonitor
from predicators.perception import BasePerceiver
from predicators.settings import CFG
from predicators.structs import Action, Dataset, EnvironmentTask, GroundAtom, \
    InteractionRequest, InteractionResult, Metrics, Observation, State, Task


class CogMan:
    """Cognitive manager."""

    def __init__(self, approach: BaseApproach, perceiver: BasePerceiver,
                 execution_monitor: BaseExecutionMonitor) -> None:
        self._approach = approach
        self._perceiver = perceiver
        self._exec_monitor = execution_monitor
        self._current_policy: Optional[Callable[[State], Action]] = None
        self._current_goal: Optional[Set[GroundAtom]] = None

    def reset(self, env_task: EnvironmentTask) -> None:
        """Start a new episode of environment interaction."""
        task = self._perceiver.reset(env_task)
        self._current_goal = task.goal
        self._current_policy = self._approach.solve(task, timeout=CFG.timeout)
        self._exec_monitor.reset(task)
        self._exec_monitor.update_approach_info(
            self._approach.get_execution_monitoring_info())

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
            self._exec_monitor.update_approach_info(
                self._approach.get_execution_monitoring_info())
            assert not self._exec_monitor.step(state)
        assert self._current_policy is not None
        act = self._current_policy(state)
        return act

    # The methods below provide an interface to the approach. In the future,
    # we may want to move some of these methods into cogman properly, e.g.,
    # if we want the perceiver or execution monitor to learn from data.

    @property
    def is_learning_based(self) -> bool:
        """See BaseApproach docstring."""
        return self._approach.is_learning_based

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """See BaseApproach docstring."""
        return self._approach.learn_from_offline_dataset(dataset)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        """See BaseApproach docstring."""
        return self._approach.load(online_learning_cycle)

    def get_interaction_requests(self) -> List[InteractionRequest]:
        """See BaseApproach docstring."""
        return self._approach.get_interaction_requests()

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        """See BaseApproach docstring."""
        return self._approach.learn_from_interaction_results(results)

    @property
    def metrics(self) -> Metrics:
        """See BaseApproach docstring."""
        return self._approach.metrics

    def reset_metrics(self) -> None:
        """See BaseApproach docstring."""
        return self._approach.reset_metrics()
