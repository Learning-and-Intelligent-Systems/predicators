"""Cognitive manager (CogMan).

A wrapper around an approach that manages interaction with the environment.

Implements a perception module, which produces a State at each time step based
on the history of observations, and an execution monitor, which determines
whether to re-query the approach at each time step based on the states.

The name "CogMan" is due to Leslie Kaelbling.
"""
import logging
from typing import Callable, List, Optional, Sequence, Set

from predicators import utils
from predicators.approaches import BaseApproach
from predicators.execution_monitoring import BaseExecutionMonitor
from predicators.perception import BasePerceiver
from predicators.settings import CFG
from predicators.structs import Action, Dataset, EnvironmentTask, GroundAtom, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, Metrics, \
    Observation, State, Task, Video


class CogMan:
    """Cognitive manager."""

    def __init__(self, approach: BaseApproach, perceiver: BasePerceiver,
                 execution_monitor: BaseExecutionMonitor) -> None:
        self._approach = approach
        self._perceiver = perceiver
        self._exec_monitor = execution_monitor
        self._current_policy: Optional[Callable[[State], Action]] = None
        self._current_goal: Optional[Set[GroundAtom]] = None
        self._override_policy: Optional[Callable[[State], Action]] = None
        self._termination_fn: Optional[Callable[[State], bool]] = None
        self._current_env_task: Optional[EnvironmentTask] = None
        self._episode_state_history: List[State] = []
        self._episode_action_history: List[Action] = []
        self._episode_images: Video = []
        self._episode_num = -1

    def reset(self, env_task: EnvironmentTask) -> None:
        """Start a new episode of environment interaction."""
        logging.info("[CogMan] Reset called.")
        self._episode_num += 1
        task = self._perceiver.reset(env_task)
        self._current_env_task = env_task
        self._current_goal = task.goal
        self._reset_policy(task)
        self._exec_monitor.reset(task)
        self._exec_monitor.update_approach_info(
            self._approach.get_execution_monitoring_info())
        self._episode_state_history = [task.init]
        self._episode_action_history = []
        self._episode_images = []
        if CFG.make_cogman_videos:
            imgs = self._perceiver.render_mental_images(task.init, env_task)
            self._episode_images.extend(imgs)

    def step(self, observation: Observation) -> Optional[Action]:
        """Receive an observation and produce an action, or None for done."""
        state = self._perceiver.step(observation)
        if CFG.make_cogman_videos:
            assert self._current_env_task is not None
            imgs = self._perceiver.render_mental_images(
                state, self._current_env_task)
            self._episode_images.extend(imgs)
        # Replace the first step because the state was already added in reset().
        if not self._episode_action_history:
            self._episode_state_history[0] = state
        else:
            self._episode_state_history.append(state)
        if self._termination_fn is not None and self._termination_fn(state):
            logging.info("[CogMan] Termination triggered.")
            return None
        # Check if we should replan.
        if self._exec_monitor.step(state):
            logging.info("[CogMan] Replanning triggered.")
            assert self._current_goal is not None
            task = Task(state, self._current_goal)
            self._reset_policy(task)
            self._exec_monitor.reset(task)
            self._exec_monitor.update_approach_info(
                self._approach.get_execution_monitoring_info())
            # We only reset the approach if the override policy is
            # None, so this below assertion only works in this
            # case.
            if self._override_policy is None:
                assert not self._exec_monitor.step(state)
        assert self._current_policy is not None
        act = self._current_policy(state)
        self._exec_monitor.update_approach_info(
            self._approach.get_execution_monitoring_info())
        self._episode_action_history.append(act)
        return act

    def finish_episode(self, observation: Observation) -> None:
        """Called at the end of an episode."""
        logging.info("[CogMan] Finishing episode.")
        if len(self._episode_state_history) == len(
                self._episode_action_history):
            state = self._perceiver.step(observation)
            self._episode_state_history.append(state)
        if CFG.make_cogman_videos:
            save_prefix = utils.get_config_path_str()
            outfile = f"{save_prefix}__cogman__episode{self._episode_num}.mp4"
            utils.save_video(outfile, self._episode_images)

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

    def set_override_policy(self, policy: Callable[[State], Action]) -> None:
        """Used during online interaction."""
        self._override_policy = policy

    def unset_override_policy(self) -> None:
        """Give control back to the approach."""
        self._override_policy = None

    def set_termination_function(
            self, termination_fn: Callable[[State], bool]) -> None:
        """Used during online interaction."""
        self._termination_fn = termination_fn

    def unset_termination_function(self) -> None:
        """Reset to never willfully terminating."""
        self._termination_fn = None

    def get_current_history(self) -> LowLevelTrajectory:
        """Expose the most recent state, action history for learning."""
        return LowLevelTrajectory(self._episode_state_history,
                                  self._episode_action_history)

    def _reset_policy(self, task: Task) -> None:
        """Call the approach or use the override policy."""
        if self._override_policy is not None:
            self._current_policy = self._override_policy
        else:
            self._current_policy = self._approach.solve(task,
                                                        timeout=CFG.timeout)
