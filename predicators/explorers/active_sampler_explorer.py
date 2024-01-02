"""An explorer for active sampler learning."""

import logging
import os
from collections import deque
from typing import Callable, Deque, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.competence_models import SkillCompetenceModel, \
    create_competence_model
from predicators.explorers.base_explorer import BaseExplorer
from predicators.planning import PlanningFailure, PlanningTimeout, \
    run_task_plan_once
from predicators.settings import CFG
from predicators.structs import NSRT, Action, ExplorationStrategy, \
    GroundAtom, NSRTSamplerWithEpsilonIndicator, ParameterizedOption, \
    Predicate, State, Task, Type, _GroundNSRT, _GroundSTRIPSOperator, \
    _Option

# Helper type to distinguish training tasks from replanning tasks.
_TaskID = Tuple[str, int]


class ActiveSamplerExplorer(BaseExplorer):
    """Uses past ground operator successes and failures to choose a ground
    operator to "practice". Makes a plan to visit the ground operator and try
    out the respective sampler. Like the ActiveSamplerLearningApproach, we
    assume that operators and NSRTs are static except for the samplers. Updates
    ground_op_hist in-place.

    Starts by attempting to solve the given task, repeatedly trying
    until the goal is reached or time expires. With any remaining time,
    starts planning to practice.
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, nsrts: Set[NSRT],
                 ground_op_hist: Dict[_GroundSTRIPSOperator, List[bool]],
                 competence_models: Dict[_GroundSTRIPSOperator,
                                         SkillCompetenceModel],
                 nsrt_to_explorer_sampler: Dict[
                     NSRT, NSRTSamplerWithEpsilonIndicator],
                 seen_train_task_idxs: Set[int],
                 pursue_task_goal_first: bool) -> None:

        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"

        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._nsrts = nsrts
        self._ground_op_hist = ground_op_hist
        self._competence_models = competence_models
        self._last_executed_nsrt: Optional[_GroundNSRT] = None
        # Indicator that tells us whether the last executed NSRT used
        # a random sample for the purposes of exploration, or whether
        # it used something from the current learned sampler distribution
        # (exploitation).
        self._last_executed_nsrt_was_exploration: Optional[bool] = None
        self._nsrt_to_explorer_sampler = nsrt_to_explorer_sampler
        self._seen_train_task_idxs = seen_train_task_idxs
        self._pursue_task_goal_first = pursue_task_goal_first
        # If the plan is None, that means none can be found, e.g., due to
        # timeouts or dead-ends.
        self._task_plan_cache: Dict[
            _TaskID, Optional[List[_GroundSTRIPSOperator]]] = {}
        self._task_plan_calls_since_replan: Dict[_TaskID, int] = {}
        self._sorted_options = sorted(options, key=lambda o: o.name)

        # Set the default cost for skills.
        alpha, beta = CFG.skill_competence_default_alpha_beta
        c = utils.beta_bernoulli_posterior([], alpha=alpha, beta=beta).mean()
        self._default_cost = -np.log(c)

        # Tasks created through re-planning.
        n = CFG.active_sampler_explorer_planning_progress_max_replan_tasks
        self._replanning_tasks: Deque[Task] = deque([], maxlen=n)

        # To more easily see what's going on, create a log file for just
        # the explorer.
        logger_name = f"{utils.get_config_path_str()}-explore"
        self._logger = logging.getLogger(logger_name)
        # Only create the logger once.
        if not self._logger.handlers:
            self._logger.setLevel(logging.DEBUG)
            os.makedirs(CFG.log_dir, exist_ok=True)
            fh = logging.FileHandler(f"{CFG.log_dir}/{logger_name}.log")
            fh.setLevel(logging.DEBUG)
            self._logger.addHandler(fh)

    @classmethod
    def get_name(cls) -> str:
        return "active_sampler"

    def _log(self, msg: str) -> None:
        self._logger.info(msg)

    def get_exploration_strategy(self, train_task_idx: int,
                                 timeout: int) -> ExplorationStrategy:
        """Wrap the parent termination function so that we can log the final
        outcome in ground_op_hist."""
        policy, termination_fn = super().get_exploration_strategy(
            train_task_idx, timeout)

        self._log("***** New Exploration Strategy Created *****")

        def wrapped_termination_fn(state: State) -> bool:
            terminate = termination_fn(state)
            if terminate:
                self._update_ground_op_hist(state)
            return terminate

        return policy, wrapped_termination_fn

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:

        assigned_task = self._train_tasks[train_task_idx]
        assigned_task_finished = not self._pursue_task_goal_first
        assigned_task_horizon = CFG.horizon
        current_policy: Optional[Callable[[State], _Option]] = None
        next_practice_nsrt: Optional[_GroundNSRT] = None
        current_task_repeat_goal: Optional[Set[GroundAtom]] = None
        using_random = False

        def _option_policy(state: State) -> Tuple[_Option, bool]:
            nonlocal assigned_task_finished, current_policy, \
                next_practice_nsrt, using_random, assigned_task_horizon

            # Need to wait for policy to get called to "see" the train task.
            self._seen_train_task_idxs.add(train_task_idx)

            if using_random:
                self._log("[Explorer] Using random option policy.")
                return (self._get_random_option(state), False)

            # Record if we've reached the assigned goal; can now practice.
            if not assigned_task_finished and \
                assigned_task.goal_holds(state):
                self._log(
                    f"[Explorer] Reached assigned goal: {assigned_task.goal}")
                assigned_task_finished = True
                current_policy = None

            # Record if we've exhausted the time limit for the assigned task.
            elif not assigned_task_finished and assigned_task_horizon <= 0:
                self._log("[Explorer] Exhausted horizon for assigned task.")
                assigned_task_finished = True
                current_policy = None

            else:
                assigned_task_horizon -= 1

            # If we've just reached the preconditions for next_practice_nsrt,
            # then immediately execute it.
            if next_practice_nsrt is not None and all(
                    a.holds(state) for a in next_practice_nsrt.preconditions):
                g: Set[GroundAtom] = set()  # goal assumed unused
                nsrt_str = next_practice_nsrt.short_str
                self._log(f"[Explorer] Practicing: {nsrt_str}")
                exploration_sampler = self._nsrt_to_explorer_sampler[
                    next_practice_nsrt.parent]
                # We want to generate a sample to use to ground the option,
                # and also save the epsilon indicator.
                params, indicator = exploration_sampler(
                    state, g, self._rng, next_practice_nsrt.option_objs)
                # Clip the params into the params_space of self.option,
                # for safety.
                low = next_practice_nsrt.option.params_space.low
                high = next_practice_nsrt.option.params_space.high
                params = np.clip(params, low, high)
                option = next_practice_nsrt.option.ground(
                    next_practice_nsrt.option_objs, params)
                next_practice_nsrt = None
                current_policy = None
                return option, indicator

            # Check if it's time to select a new goal and re-plan.
            if current_policy is None:
                # If the assigned goal hasn't yet been reached, try for it.
                if not assigned_task_finished:
                    self._log("[Explorer] Pursuing assigned task goal")

                    def generate_goals() -> Iterator[Set[GroundAtom]]:
                        # Just a single goal.
                        yield assigned_task.goal

                # Baseline where we try the assigned task over and over,
                # going back to the initial (abstract) state after reaching
                # the goal.
                elif CFG.active_sampler_explore_task_strategy == "task_repeat":
                    self._log("[Explorer] Pursuing repeat task")

                    def generate_goals() -> Iterator[Set[GroundAtom]]:
                        nonlocal current_task_repeat_goal
                        # Loop through seen tasks in random order. Propose
                        # their initial abstract states and their goals until
                        # one is found that is not already achieved.
                        train_task_idxs = sorted(self._seen_train_task_idxs)
                        self._rng.shuffle(train_task_idxs)
                        for train_task_idx in train_task_idxs:
                            task = self._train_tasks[train_task_idx]
                            # Can only practice the task if the objects match.
                            if set(task.init) == set(state):
                                # If we've already been trying to achieve a
                                # particular goal, then keep trying to achieve
                                # it.
                                if current_task_repeat_goal is not None:
                                    current_pursuit_goal_achieved = all(
                                        a.holds(state)
                                        for a in current_task_repeat_goal)
                                    if not current_pursuit_goal_achieved:
                                        yield current_task_repeat_goal
                                # Else, figure out the next goal to plan to!
                                possible_goals = [
                                    task.goal,
                                    utils.abstract(task.init, self._predicates)
                                ]
                                for goal in possible_goals:
                                    if any(not a.holds(state) for a in goal):
                                        current_task_repeat_goal = goal
                                        yield goal

                # Otherwise, practice.
                else:
                    self._log("[Explorer] Pursuing NSRT preconditions")

                    def generate_goals() -> Iterator[Set[GroundAtom]]:
                        nonlocal next_practice_nsrt
                        # Generate goals sorted by their descending score.
                        for op in sorted(self._ground_op_hist,
                                         key=self._score_ground_op,
                                         reverse=True):
                            nsrt = [
                                n for n in self._nsrts if n.op == op.parent
                            ][0]
                            # NOTE: setting nonlocal variable.
                            next_practice_nsrt = nsrt.ground(op.objects)
                            nsrt_name = next_practice_nsrt.short_str
                            self._log(f"[Explorer] Considering: {nsrt_name}")
                            yield next_practice_nsrt.preconditions

                # Try to plan to each goal until a task plan is found.
                for goal in generate_goals():
                    task = Task(state, goal)
                    self._log(f"[Explorer] Replanning to {task.goal}")
                    # If the goal is empty, then we can just recursively
                    # call the policy, since we don't need to execute
                    # anything.
                    if len(goal) == 0:
                        return _option_policy(state)  # pragma: no cover
                    # Add this task to the re-planning task queue.
                    self._replanning_tasks.append(task)

                    try:
                        current_policy = self._get_option_policy_for_task(task)
                    # Not covering this case because the intention of this
                    # explorer is to be used in environments where any goal can
                    # be reached from anywhere, but we still don't want to
                    # crash in case that assumption is not met.
                    except (PlanningFailure,
                            PlanningTimeout):  # pragma: no cover
                        self._log(
                            "WARNING: Planning graph is not "
                            "fully-connected! This violates a key "
                            "assumption of our active sampler learning "
                            "framework; ensure you DO NOT see this message "
                            "if you're running experiments comparing"
                            "different active sampler learning approaches.")
                        continue
                    self._log("[Explorer] Plan found.")
                    break
                # Terminate early if no goal could be found.
                else:
                    self._log("[Explorer] No reachable goal found. "
                              "Switching to random exploration.")
                    using_random = True
                    return (self._get_random_option(state), False)
            # Query the current policy.
            assert current_policy is not None
            try:
                act = current_policy(state)
                return (act, False)
            except utils.OptionExecutionFailure:
                self._log("[Explorer] Option execution failure!")
                current_policy = None
            # Call recursively to trigger re-planning.
            return _option_policy(state)

        # Wrap the option policy to keep track of the executed NSRTs and if
        # they succeeded, to update the ground_op_hist.
        initialized = False

        def _wrapped_option_policy(state: State) -> _Option:
            nonlocal initialized
            if not initialized:
                self._last_executed_nsrt = None
                initialized = True
            # Update ground_op_hist.
            self._update_ground_op_hist(state)
            # Record last executed NSRT.
            option, exploration_indicator = _option_policy(state)
            ground_nsrt = utils.option_to_ground_nsrt(option, self._nsrts)
            self._log(f"[Explorer] Starting NSRT: {ground_nsrt.short_str}")
            self._last_executed_nsrt = ground_nsrt
            self._last_executed_nsrt_was_exploration = exploration_indicator
            return option

        # Finalize policy.
        policy = utils.option_policy_to_policy(
            _wrapped_option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout)

        # Catch exceptions and update the ground op history.
        def _wrapped_policy(state: State) -> Action:
            try:
                return policy(state)
            except utils.OptionTimeoutFailure as e:
                # If the option was cut off due to max_option_steps, then
                # we consider the option to be terminated.
                self._update_ground_op_hist(state)
                raise e

        # Never terminate.
        termination_fn = lambda _: False

        return _wrapped_policy, termination_fn

    def _update_ground_op_hist(self, state: State) -> None:
        """Should be called when an NSRT has just terminated."""
        nsrt = self._last_executed_nsrt
        exploration_indicator = self._last_executed_nsrt_was_exploration
        if nsrt is None:
            return
        assert exploration_indicator is not None
        success = all(a.holds(state) for a in nsrt.add_effects)
        logging.info(f"[Explorer] Last NSRT: {nsrt.short_str}")
        logging.info(f"[Explorer]   outcome: {success}")
        last_executed_op = nsrt.op
        if last_executed_op not in self._ground_op_hist:
            self._ground_op_hist[last_executed_op] = []
        self._ground_op_hist[last_executed_op].append(success)
        # Update the competence model too.
        if last_executed_op not in self._competence_models:
            model_name = CFG.skill_competence_model
            skill_name = last_executed_op.short_str
            model = create_competence_model(model_name, skill_name)
            self._competence_models[last_executed_op] = model
        # Only update the competence model if this action was not an
        # exploratory action.
        if not exploration_indicator:
            self._competence_models[last_executed_op].observe(success)

    def _get_option_policy_for_task(self,
                                    task: Task) -> Callable[[State], _Option]:
        # Run task planning and then greedily execute.
        timeout = CFG.timeout
        task_planning_heuristic = CFG.sesame_task_planning_heuristic
        ground_op_costs = {
            o: -np.log(m.get_current_competence())
            for o, m in self._competence_models.items()
        }
        # Set large horizon for planning here because we don't want to error
        # out due to plan exceeding horizon here.
        plan, atoms_seq, _ = run_task_plan_once(
            task,
            self._nsrts,
            self._predicates,
            self._types,
            timeout,
            self._seed,
            task_planning_heuristic=task_planning_heuristic,
            ground_op_costs=ground_op_costs,
            default_cost=self._default_cost,
            max_horizon=np.inf)
        return utils.nsrt_plan_to_greedy_option_policy(
            plan, task.goal, self._rng, necessary_atoms_seq=atoms_seq)

    def _score_ground_op(
            self, ground_op: _GroundSTRIPSOperator) -> Tuple[float, ...]:
        history = self._ground_op_hist[ground_op]
        num_tries = len(history)
        success_rate = sum(history) / num_tries
        total_trials = sum(len(h) for h in self._ground_op_hist.values())
        seen_train_tasks_num = len(self._seen_train_task_idxs)
        # UCB-like bonus.
        c = CFG.active_sampler_explore_bonus
        bonus = c * np.sqrt(np.log(total_trials) / num_tries)
        if CFG.active_sampler_explore_task_strategy == "planning_progress":
            score = self._score_ground_op_planning_progress(ground_op)
            logging.info(f"[Explorer]   Base score: {score}")
            logging.info(f"[Explorer]   UCB bonus: {bonus}")
            if CFG.active_sampler_explore_use_ucb_bonus:
                # Normalize the score.
                score = score / seen_train_tasks_num
                score += bonus
        elif CFG.active_sampler_explore_task_strategy == "success_rate":
            score = (1.0 - success_rate)
            if CFG.active_sampler_explore_use_ucb_bonus:
                # Normalize the score.
                score = score / seen_train_tasks_num
                score += bonus
        elif CFG.active_sampler_explore_task_strategy == "competence_gradient":
            model = self._competence_models[ground_op]
            extrap = model.predict_competence(
                CFG.skill_competence_model_lookahead)
            competence = model.get_current_competence()
            score = extrap - competence
            logging.info(f"[Explorer] {ground_op.name}{ground_op.objects} has")
            logging.info(f"[Explorer]   posterior competence: {competence}")
            logging.info(f"[Explorer]   extrapolated competence: {extrap}")
            logging.info(f"[Explorer]   score: {score}")

            if CFG.active_sampler_explore_use_ucb_bonus:
                # Normalize the score.
                score = score / seen_train_tasks_num
                score += bonus
        elif CFG.active_sampler_explore_task_strategy == "skill_diversity":
            score = 1 / num_tries
            if CFG.active_sampler_explore_use_ucb_bonus:
                # Normalize the score.
                score = score / seen_train_tasks_num
                score += bonus
        elif CFG.active_sampler_explore_task_strategy == "random":
            # Random scores baseline.
            score = self._rng.uniform()
        else:
            raise NotImplementedError(
                "Unrecognized explore task strategy: "
                f"{CFG.active_sampler_explore_task_strategy}")
        # Break ties randomly.
        return (score, self._rng.uniform())

    def _score_ground_op_planning_progress(
            self, ground_op: _GroundSTRIPSOperator) -> float:
        # Predict the competence if we had one more data point.
        model = self._competence_models[ground_op]
        extrap = model.predict_competence(CFG.skill_competence_model_lookahead)
        competence = model.get_current_competence()
        history = self._ground_op_hist[ground_op]
        num_tries = len(history)
        success_rate = sum(history) / num_tries
        # Optimization: skip any ground op with perfect success.
        if CFG.active_sampler_explorer_skip_perfect and success_rate == 1.0:
            return -np.inf
        logging.info(f"[Explorer] {ground_op.short_str} has")
        logging.info(f"[Explorer]   success rate: {success_rate}")
        logging.info(f"[Explorer]   posterior competence: {competence}")
        logging.info(f"[Explorer]   num attempts: {num_tries}")
        logging.info(f"[Explorer]   extrapolated competence: {extrap}")
        c_hat = -np.log(extrap)
        assert c_hat >= 0
        # Update the ground op costs hypothetically.
        ground_op_costs = {
            o: -np.log(m.get_current_competence())
            for o, m in self._competence_models.items()
        }
        ground_op_costs[ground_op] = c_hat  # override
        # Make plans on some of the training tasks we've seen so far and record
        # the total plan costs.
        plan_costs: List[float] = []
        # Select an arbitrary but constant subset of the training tasks.
        # Don't randomize: would lead to noisy estimates that artificially
        # favor some operators over others.
        train_task_idxs = sorted(self._seen_train_task_idxs)
        max_num_tasks = CFG.active_sampler_explorer_planning_progress_max_tasks
        num_tasks = min(max_num_tasks, len(train_task_idxs))
        train_task_ids = [("train", i) for i in train_task_idxs[:num_tasks]]
        # Add up to a certain number of fictitious training tasks that were
        # created through re-planning. Use the most recent tasks to deal with
        # the non-stationary distribution.
        num_replan_tasks = list(range(len(self._replanning_tasks)))
        replan_task_ids = [("replan", i) for i in range(len(num_replan_tasks))]
        for task_id in train_task_ids + replan_task_ids:
            plan = self._get_task_plan_for_task(task_id, ground_op_costs)
            # If no plan can be found for a task, the task is just ignored.
            if plan is not None:
                task_plan_costs = []
                for op in plan:
                    op_cost = ground_op_costs.get(op, self._default_cost)
                    task_plan_costs.append(op_cost)
                plan_costs.append(sum(task_plan_costs))
        return -sum(plan_costs)  # higher scores are better

    def _get_task_plan_for_task(
        self, task_id: _TaskID, ground_op_costs: Dict[_GroundSTRIPSOperator,
                                                      float]
    ) -> Optional[List[_GroundSTRIPSOperator]]:
        """Returns None if no task plan can be found."""
        # Optimization: only re-plan at a certain frequency.
        replan_freq = CFG.active_sampler_explorer_replan_frequency
        if task_id not in self._task_plan_calls_since_replan or \
            self._task_plan_calls_since_replan[task_id] >= replan_freq:
            self._task_plan_calls_since_replan[task_id] = 0
            timeout = CFG.timeout
            task_planning_heuristic = CFG.sesame_task_planning_heuristic
            task_type, task_idx = task_id
            task = {
                "train": self._train_tasks,
                "replan": self._replanning_tasks,
            }[task_type][task_idx]
            try:
                plan, _, _ = run_task_plan_once(
                    task,
                    self._nsrts,
                    self._predicates,
                    self._types,
                    timeout,
                    self._seed,
                    task_planning_heuristic=task_planning_heuristic,
                    ground_op_costs=ground_op_costs,
                    default_cost=self._default_cost,
                    max_horizon=np.inf)
                self._task_plan_cache[task_id] = [n.op for n in plan]
            except (PlanningFailure, PlanningTimeout):  # pragma: no cover
                self._log("WARNING: task planning failed in the explorer.")
                self._task_plan_cache[task_id] = None

        self._task_plan_calls_since_replan[task_id] += 1
        return self._task_plan_cache[task_id]

    def _get_random_option(self, state: State) -> _Option:
        option = utils.sample_applicable_option(self._sorted_options, state,
                                                self._rng)
        assert option is not None
        return option
