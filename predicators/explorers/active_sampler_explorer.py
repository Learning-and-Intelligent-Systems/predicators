"""An explorer for active sampler learning."""

import logging
from typing import Callable, Dict, List, Optional, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.planning import run_task_plan_once
from predicators.settings import CFG
from predicators.structs import NSRT, ExplorationStrategy, GroundAtom, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, \
    _GroundSTRIPSOperator, _Option


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

    def __init__(
            self, predicates: Set[Predicate],
            options: Set[ParameterizedOption], types: Set[Type],
            action_space: Box, train_tasks: List[Task],
            max_steps_before_termination: int, nsrts: Set[NSRT],
            ground_op_hist: Dict[_GroundSTRIPSOperator, List[bool]]) -> None:

        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"

        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._nsrts = nsrts
        self._ground_op_hist = ground_op_hist
        self._last_executed_nsrt: Optional[_GroundNSRT] = None

    @classmethod
    def get_name(cls) -> str:
        return "active_sampler"

    def get_exploration_strategy(self, train_task_idx: int,
                                 timeout: int) -> ExplorationStrategy:
        """Wrap the parent termination function so that we can log the final
        outcome in ground_op_hist."""
        policy, termination_fn = super().get_exploration_strategy(
            train_task_idx, timeout)

        def wrapped_termination_fn(state: State) -> bool:
            terminate = termination_fn(state)
            if terminate:
                self._update_ground_op_hist(state)
            return terminate

        return policy, wrapped_termination_fn

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:

        assigned_task = self._train_tasks[train_task_idx]
        assigned_task_goal_reached = False
        current_policy: Optional[Callable[[State], _Option]] = None
        next_practice_nsrt: Optional[_GroundNSRT] = None

        def _option_policy(state: State) -> _Option:
            nonlocal assigned_task_goal_reached, current_policy, \
                next_practice_nsrt

            atoms = utils.abstract(state, self._predicates)

            # Record if we've reached the assigned goal; can now practice.
            if not assigned_task_goal_reached and \
                assigned_task.goal_holds(state):
                logging.info(
                    f"[Explorer] Reached assigned goal: {assigned_task.goal}")
                assigned_task_goal_reached = True
                current_policy = None

            # If we've just reached the preconditions for next_practice_nsrt,
            # then immediately execute it.
            if next_practice_nsrt is not None and \
                next_practice_nsrt.preconditions.issubset(atoms):
                g: Set[GroundAtom] = set()  # goal assumed unused
                option = next_practice_nsrt.sample_option(state, g, self._rng)
                next_practice_nsrt = None
                current_policy = None
                return option

            # Check if it's time to select a new goal and re-plan.
            if current_policy is None:
                # If the assigned goal hasn't yet been reached, try for it.
                if not assigned_task_goal_reached:
                    goal = assigned_task.goal
                # Otherwise, practice.
                else:
                    # If there are no ground NSRTs that we've tried so far,
                    # just wait until we have tried to solve some task.
                    if len(self._ground_op_hist) == 0:
                        raise utils.OptionExecutionFailure(
                            "No ground operators to practice yet")
                    next_practice_nsrt = self._get_practice_ground_nsrt()
                    goal = next_practice_nsrt.preconditions
                task = Task(state, goal)
                logging.info(f"[Explorer] Replanning to {task.goal}")
                current_policy = self._get_option_policy_for_task(task)

            # Query the current policy.
            assert current_policy is not None
            try:
                act = current_policy(state)
                return act
            except utils.OptionExecutionFailure:
                current_policy = None
            # Call recursively to trigger re-planning.
            return _option_policy(state)

        # Wrap the option policy to keep track of the executed NSRTs and if
        # they succeeded, to update the ground_op_hist.
        self._last_executed_nsrt = None

        def _wrapped_option_policy(state: State) -> _Option:
            # Update ground_op_hist.
            self._update_ground_op_hist(state)
            # Record last executed NSRT.
            option = _option_policy(state)
            ground_nsrt = utils.option_to_ground_nsrt(option, self._nsrts)
            self._last_executed_nsrt = ground_nsrt
            return option

        # Finalize policy.
        policy = utils.option_policy_to_policy(_wrapped_option_policy)

        # Never terminate.
        termination_fn = lambda _: False

        return policy, termination_fn

    def _update_ground_op_hist(self, state: State) -> None:
        """Should be called when an NSRT has just terminated."""
        nsrt = self._last_executed_nsrt
        if nsrt is None:
            return
        atoms = utils.abstract(state, self._predicates)
        success = nsrt.add_effects.issubset(atoms) and \
            not nsrt.delete_effects & atoms
        logging.info(f"[Explorer] Last NSRT: {nsrt.name}{nsrt.objects}")
        logging.info(f"[Explorer]   outcome: {success}")
        if not success:
            if nsrt.add_effects - atoms:
                logging.info(f"[Explorer]   missing: {nsrt.add_effects - atoms}")
            if nsrt.delete_effects & atoms:
                logging.info(f"[Explorer]   extra: {nsrt.delete_effects & atoms}")
        last_executed_op = nsrt.op
        if last_executed_op not in self._ground_op_hist:
            self._ground_op_hist[last_executed_op] = []
        self._ground_op_hist[last_executed_op].append(success)

    def _get_practice_ground_nsrt(self) -> _GroundNSRT:
        best_op = max(self._ground_op_hist, key=self._score_ground_op)
        logging.info(f"[Explorer] Practicing {best_op.name}{best_op.objects}")
        nsrt = [n for n in self._nsrts if n.op == best_op.parent][0]
        return nsrt.ground(best_op.objects)

    def _get_option_policy_for_task(self,
                                    task: Task) -> Callable[[State], _Option]:
        # Run task planning and then greedily execute.
        timeout = CFG.timeout
        task_planning_heuristic = CFG.sesame_task_planning_heuristic
        plan, atoms_seq, _ = run_task_plan_once(
            task,
            self._nsrts,
            self._predicates,
            self._types,
            timeout,
            self._seed,
            task_planning_heuristic=task_planning_heuristic)
        return utils.nsrt_plan_to_greedy_option_policy(
            plan, task.goal, self._rng, necessary_atoms_seq=atoms_seq)

    def _score_ground_op(self, ground_op: _GroundSTRIPSOperator) -> float:
        # Score NSRTs according to their success rate and a bonus for ones
        # that haven't been tried very much.
        history = self._ground_op_hist[ground_op]
        num_tries = len(history)
        success_rate = sum(history) / num_tries
        total_trials = sum(len(h) for h in self._ground_op_hist.values())
        logging.info(f"[Explorer] {ground_op.name}{ground_op.objects} has")
        logging.info(f"[Explorer]   success rate: {success_rate}")
        # UCB-like bonus.
        c = CFG.active_sampler_explore_bonus
        bonus = c * np.sqrt(np.log(total_trials) / num_tries)
        logging.info(f"[Explorer]   num attempts: {num_tries}")
        # Try less successful operators more often.
        score = (1.0 - success_rate) + bonus
        logging.info(f"[Explorer]   total score: {score}")
        return score
