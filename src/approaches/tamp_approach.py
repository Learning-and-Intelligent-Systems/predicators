"""An abstract approach that does TAMP to solve tasks.

Uses the SeSamE planning strategy: SEarch-and-SAMple planning, then
Execution.
"""

import abc
from typing import Callable, Set, List
from gym.spaces import Box
from predicators.src.approaches import BaseApproach, ApproachFailure
from predicators.src.planning import sesame_plan
from predicators.src.structs import State, Action, Task, NSRT, \
    Predicate, ParameterizedOption, Type
from predicators.src.option_model import create_option_model
from predicators.src.settings import CFG
from predicators.src import utils


class TAMPApproach(BaseApproach):
    """TAMP approach."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        if task_planning_heuristic == "default":
            task_planning_heuristic = CFG.sesame_task_planning_heuristic
        if max_skeletons_optimized == -1:
            max_skeletons_optimized = CFG.sesame_max_skeletons_optimized
        self._task_planning_heuristic = task_planning_heuristic
        self._max_skeletons_optimized = max_skeletons_optimized
        self._option_model = create_option_model(CFG.option_model_name)
        self._num_calls = 0

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        # ensure random over successive calls
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()
        plan, metrics = sesame_plan(task,
                                    self._option_model,
                                    nsrts,
                                    preds,
                                    timeout,
                                    seed,
                                    self._task_planning_heuristic,
                                    self._max_skeletons_optimized,
                                    allow_noops=CFG.sesame_allow_noops)
        for metric in [
                "num_skeletons_optimized", "num_failures_discovered",
                "num_nodes_expanded", "num_nodes_created", "plan_length"
        ]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        self._metrics["total_num_nsrts"] += len(nsrts)
        self._metrics["total_num_preds"] += len(preds)
        self._metrics["min_num_skeletons_optimized"] = min(
            metrics["num_skeletons_optimized"],
            self._metrics["min_num_skeletons_optimized"])
        self._metrics["max_num_skeletons_optimized"] = max(
            metrics["num_skeletons_optimized"],
            self._metrics["max_num_skeletons_optimized"])
        option_policy = utils.option_plan_to_policy(plan)

        def _policy(s: State) -> Action:
            try:
                return option_policy(s)
            except utils.OptionPlanExhausted:
                raise ApproachFailure("Option plan exhausted.")

        return _policy

    def reset_metrics(self) -> None:
        super().reset_metrics()
        # Initialize min to self._max_skeletons_optimized (max gets initialized
        # to 0 by default)
        self._metrics[
            "min_num_skeletons_optimized"] = self._max_skeletons_optimized

    @abc.abstractmethod
    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        raise NotImplementedError("Override me!")

    def _get_current_predicates(self) -> Set[Predicate]:
        """Get the current set of predicates.

        Defaults to initial predicates.
        """
        return self._initial_predicates
