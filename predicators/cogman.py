"""Cognitive manager (CogMan).

A wrapper around an approach that manages interaction with the environment.

Implements a perception module, which produces a State at each time step based
on the history of observations, and an execution monitor, which determines
whether to re-query the approach at each time step based on the states.

The name "CogMan" is due to Leslie Kaelbling.
"""
import dataclasses
import logging
import time
import traceback
from collections import defaultdict
from typing import Callable, List, Optional, Sequence, Set, Tuple
from typing import Type as TypingType

from predicators import utils
from predicators.approaches import BaseApproach
from predicators.envs import BaseEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.execution_monitoring import BaseExecutionMonitor
from predicators.perception import BasePerceiver
from predicators.settings import CFG
from predicators.structs import Action, Dataset, EnvironmentTask, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, Metrics, \
    Observation, State, Task, Video, _Option


class CogMan:
    """Cognitive manager."""

    def __init__(self, approach: BaseApproach, perceiver: BasePerceiver,
                 execution_monitor: BaseExecutionMonitor) -> None:
        self._approach = approach
        self._perceiver = perceiver
        self._exec_monitor = execution_monitor
        self._current_policy: Optional[Callable[[State], Action]] = None
        self._current_solve_task: Optional[Task] = None
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
        self._approach.reset_for_new_episode()
        task = self._perceiver.reset(env_task)
        self._current_env_task = env_task
        self._current_solve_task = task
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
            logging.debug("[CogMan] step returning None: termination_fn fired")
            return None
        # Check if we should replan.
        if self._exec_monitor.step(state):
            logging.info("[CogMan] Replanning triggered.")
            assert self._current_solve_task is not None
            # Re-solve the episode's task from the current state; every
            # non-init Task field (goal, goal_nl, evaluator) must survive
            # a mid-episode replan.
            task = dataclasses.replace(self._current_solve_task, init=state)
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

    @property
    def get_approach_name(self) -> str:
        """See BaseApproach docstring."""
        return self._approach.get_name()

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


def run_episode_and_get_observations(
    cogman: CogMan,
    env: BaseEnv,
    train_or_test: str,
    task_idx: int,
    max_num_steps: int,
    do_env_reset: bool = True,
    terminate_on_goal_reached: bool = True,
    exceptions_to_break_on: Optional[Set[TypingType[Exception]]] = None,
    monitor: Optional[utils.LoggingMonitor] = None
) -> Tuple[Tuple[List[Observation], List[Action]], bool, Metrics]:
    """Execute cogman starting from the initial state of a train or test task
    in the environment.

    Note that the environment and cogman internal states are updated.
    Terminates when any of these conditions hold: (1) cogman.step
    returns None, indicating termination (2) max_num_steps is reached
    (3) cogman or env raise an exception of type in
    exceptions_to_break_on (4) terminate_on_goal_reached is True and the
    env goal is reached. Note that in the case where the exception is
    raised in step, we exclude the last action from the returned
    trajectory to maintain the invariant that the trajectory states are
    of length one greater than the actions. Ideally, this method would
    live in utils.py, but that results in import errors with this file.
    So we keep it here for now. It might be moved in the future.
    """
    if do_env_reset:
        env.reset(train_or_test, task_idx)
        if monitor is not None:
            monitor.reset(train_or_test, task_idx)
    render_obs = True if (
        cogman.get_approach_name == "oracle" and CFG.offline_data_method
        == "geo_and_demo_with_vlm_imgs") else CFG.rgb_observation
    if isinstance(env, PyBulletEnv):
        obs = env.get_observation(render=render_obs)
    else:
        obs = env.get_observation()
    observations = [obs]
    actions: List[Action] = []

    def _certified_solved() -> bool:
        """Goal atoms/reward hold AND the env accepts the trajectory."""
        if not env.goal_reached():
            return False
        ok, reason = env.check_episode_trajectory(observations, actions)
        if not ok:
            logging.info(
                "[CogMan] Goal atoms hold but the trajectory was "
                "REJECTED: %s", reason)
        return ok

    curr_option: Optional[_Option] = None
    metrics: Metrics = defaultdict(float)
    metrics["policy_call_time"] = 0.0
    metrics["num_options_executed"] = 0.0
    exception_raised_in_step = False
    # Whether this episode ran to a clean end, as opposed to being cut short
    # by an exception. An env driving something outside itself needs the
    # difference: a prefix of a plan is not a plan.
    episode_completed = True
    step_num = -1
    if not (terminate_on_goal_reached and env.goal_reached()):
        for step_num in range(max_num_steps):
            monitor_observed = False
            exception_raised_in_step = False
            try:
                start_time = time.perf_counter()
                act = cogman.step(obs)
                metrics["policy_call_time"] += time.perf_counter() - start_time
                if act is None:
                    logging.debug("[CogMan] loop break: act is None")
                    break
                if act.has_option() and act.get_option() != curr_option:
                    curr_option = act.get_option()
                    metrics["num_options_executed"] += 1
                # Note: it's important to call monitor.observe() before
                # env.step(), because the monitor may, for example, call
                # env.render(), which outputs images of the current env
                # state. If we instead called env.step() first, we would
                # mistakenly record images of the next time step instead of
                # the current one.
                if monitor is not None:
                    monitor.observe(obs, act)
                    monitor_observed = True
                if isinstance(env, PyBulletEnv):
                    obs = env.step(act, render_obs=render_obs)
                else:
                    obs = env.step(act)
                actions.append(act)
                observations.append(obs)
            except Exception as e:
                logging.debug(f"[CogMan] State at the exception {e}: "
                              f"{utils.abstract(obs, env.predicates)}")
                logging.debug(
                    f"[CogMan] Full traceback:\n{traceback.format_exc()}")
                # pylint: disable-next=import-outside-toplevel
                from predicators.agent_sdk.session_base import \
                    AgentSessionFatalError
                if isinstance(e, AgentSessionFatalError):
                    # The agent session backend is unusable; neither
                    # break_on handling nor keep_failed_demos may absorb
                    # this into a failed episode - the run terminates.
                    env.finish_execution(False)
                    raise
                if exceptions_to_break_on is not None and \
                   any(issubclass(type(e), c) for c in exceptions_to_break_on):
                    if monitor_observed:
                        exception_raised_in_step = True
                    # Running out of options is how a plan ENDS. It arrives as
                    # the same OptionExecutionFailure an actual failure does,
                    # so without the flag every completed plan would be
                    # reported as an aborted episode -- and an executor that
                    # defers its work to the end of a COMPLETED episode would
                    # then discard every one.
                    if not getattr(e, "info", {}).get("plan_exhausted"):
                        episode_completed = False
                    logging.debug(
                        f"[CogMan] loop break: exception in break_on set: {e}")
                    break
                if CFG.terminate_on_goal_reached_and_option_terminated and \
                    env.goal_reached():
                    episode_completed = False
                    logging.debug(
                        f"[CogMan] loop break: goal_reached+option_terminated "
                        f"(exception: {e})")
                    break
                if monitor is not None and not monitor_observed:
                    monitor.observe(obs, None)
                if CFG.keep_failed_demos:
                    cogman.finish_episode(obs)
                    env.finish_execution(False)
                    traj = (observations, actions)
                    solved = _certified_solved()
                    return traj, solved, metrics
                env.finish_execution(False)
                raise e
            if terminate_on_goal_reached and env.goal_reached():
                logging.debug("[CogMan] loop break: terminate_on_goal_reached")
                break
        else:
            option_str = (None
                          if curr_option is None else curr_option.simple_str())
            logging.info(
                "[CogMan] Reached max_num_steps=%d while executing "
                "option %s.", max_num_steps, option_str)
            logging.debug("[CogMan] Final loop step index before horizon: %d",
                          step_num)
            logging.debug("[CogMan] Atoms at horizon: %s",
                          sorted(utils.abstract(obs, env.predicates)))
    if monitor is not None and not exception_raised_in_step:
        monitor.observe(obs, None)
    cogman.finish_episode(obs)
    env.finish_execution(episode_completed)
    traj = (observations, actions)
    solved = _certified_solved()
    return traj, solved, metrics


def run_episode_and_get_states(
    cogman: CogMan,
    env: BaseEnv,
    train_or_test: str,
    task_idx: int,
    max_num_steps: int,
    do_env_reset: bool = True,
    terminate_on_goal_reached: bool = True,
    exceptions_to_break_on: Optional[Set[TypingType[Exception]]] = None,
    monitor: Optional[utils.LoggingMonitor] = None
) -> Tuple[LowLevelTrajectory, bool, Metrics]:
    """Execute cogman starting from the initial state of a train or test task
    in the environment.

    Return a trajectory involving States (which come from running a
    perceiver on observations). Having states instead of observations is
    useful for downstream learning (e.g. predicates, operators,
    samplers, etc.) Note that the only difference between this and the
    above run_episode_and_get_observations is that this method returns a
    trajectory of states instead of one of observations.
    """
    _, solved, metrics = run_episode_and_get_observations(
        cogman, env, train_or_test, task_idx, max_num_steps, do_env_reset,
        terminate_on_goal_reached, exceptions_to_break_on, monitor)
    ll_traj = cogman.get_current_history()
    return ll_traj, solved, metrics
