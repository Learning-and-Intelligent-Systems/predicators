"""An explorer that uses bilevel planning with NSRTs."""

import logging
from typing import Dict, List, Set, cast

from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OptionModelBase
from predicators.planning import PlanningFailure, _MaxSkeletonsFailure, \
    run_task_plan_once, sesame_plan
from predicators.planning_with_processes import \
    task_plan_from_task as task_plan_with_processes
from predicators.settings import CFG
from predicators.structs import NSRT, CausalProcess, ExplorationStrategy, \
    ParameterizedOption, Predicate, Task, Type


class BilevelPlanningExplorer(BaseExplorer):
    """BilevelPlanningExplorer implementation.

    This explorer is abstract: subclasses decide how to use the _solve
    method implemented in this class, which calls sesame_plan().
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, nsrts: Set[NSRT],
                 option_model: _OptionModelBase) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._nsrts = nsrts
        self._option_model = option_model
        self._num_calls = 0
        # Add a dictionary to store process_plan iterators for each task
        self._process_plan_iterators: Dict = {}

    def _solve(self, task: Task, timeout: int) -> ExplorationStrategy:

        # Ensure random over successive calls.
        self._num_calls += 1
        seed = self._seed + self._num_calls
        # Note: subclasses are responsible for catching PlanningFailure and
        # PlanningTimeout and handling them accordingly.
        if CFG.bilevel_plan_without_sim:
            if isinstance(next(iter(self._nsrts)), CausalProcess):
                plan_iterator = task_plan_with_processes(
                                        task,
                                        self._predicates,
                                        cast(Set[CausalProcess], self._nsrts),
                                        seed,
                                        timeout,
                                        max_skeletons_optimized=\
                                            CFG.sesame_max_skeletons_optimized,
                                        use_visited_state_set=True
                                        )

                if CFG.bilevel_planning_explorer_enumerate_plans:
                    # Check if an iterator already exists for this task
                    if task not in self._process_plan_iterators:
                        # Create a new iterator for the task
                        self._process_plan_iterators[task] = plan_iterator
                    # Get the next process_plan from the iterator
                    try:
                        process_plan, _, _ = next(
                            self._process_plan_iterators[task])
                    except _MaxSkeletonsFailure as e:
                        # If the iterator is exhausted, raise an error or handle
                        # it
                        logging.debug(f"No more process plans available for "
                                      f"task: {e}")
                        raise PlanningFailure("No more process plans "
                                              "available for task")
                else:
                    process_plan = next(plan_iterator)

                policy = utils.process_plan_to_greedy_policy(
                    process_plan,
                    task.goal,
                    self._rng,
                    abstract_function=lambda s: utils.abstract(
                        s, self._predicates))

            else:
                plan, _, _ = run_task_plan_once(
                    task, self._nsrts, self._predicates, self._types, timeout,
                    seed, CFG.sesame_task_planning_heuristic)
                policy = utils.option_plan_to_policy(
                    plan,  # type: ignore[arg-type]
                    abstract_function=lambda s: utils.abstract(
                        s, self._predicates))

        else:
            assert not CFG.bilevel_plan_without_sim
            plan, _, _ = sesame_plan(  # type: ignore[assignment]
                task,
                self._option_model,
                self._nsrts,
                self._predicates,
                self._types,
                timeout,
                seed,
                CFG.sesame_task_planning_heuristic,
                CFG.sesame_max_skeletons_optimized,
                max_horizon=CFG.horizon,
                allow_waits=CFG.sesame_allow_waits,
                use_visited_state_set=CFG.sesame_use_visited_state_set)
            policy = utils.option_plan_to_policy(
                plan)  # type: ignore[arg-type]
        termination_function = task.goal_holds

        return policy, termination_function
