"""An abstract approach that does planning to solve tasks.

Uses the SeSamE bilevel planning strategy: SEarch-and-SAMple planning,
then Execution.
"""

import abc
from typing import Any, Callable, List, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    BaseApproach
from predicators.option_model import _OptionModelBase, create_option_model
from predicators.planning import PlanningFailure, PlanningTimeout, sesame_plan
from predicators.settings import CFG
from predicators.structs import NSRT, Action, Metrics, ParameterizedOption, \
    Predicate, State, Task, Type, _Option


class BilevelPlanningApproach(BaseApproach):
    """Bilevel planning approach."""

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
        self._last_plan: List[_Option] = []
        self._last_traj: List[State] = []

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        self._num_calls += 1
        # ensure random over successive calls
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()
        plan, metrics, traj = self._run_sesame_plan(task, nsrts, preds,
                                                    timeout, seed)
        self._save_metrics(metrics, nsrts, preds)
        self._last_plan = plan
        self._last_traj = traj
        option_policy = utils.option_plan_to_policy(plan)

        def _policy(s: State) -> Action:
            try:
                return option_policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy

    def _run_sesame_plan(
            self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
            timeout: float, seed: int,
            **kwargs: Any) -> Tuple[List[_Option], Metrics, List[State]]:
        """Subclasses may override.

        For example, PG4 inserts an abstract policy into kwargs.
        """
        try:
            plan, metrics, traj = sesame_plan(
                task,
                self._option_model,
                nsrts,
                preds,
                self._types,
                timeout,
                seed,
                self._task_planning_heuristic,
                self._max_skeletons_optimized,
                max_horizon=CFG.horizon,
                allow_noops=CFG.sesame_allow_noops,
                use_visited_state_set=CFG.sesame_use_visited_state_set,
                **kwargs)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        return plan, metrics, traj

    def reset_metrics(self) -> None:
        super().reset_metrics()
        # Initialize min to inf (max gets initialized to 0 by default).
        self._metrics["min_num_skeletons_optimized"] = float("inf")

    def _save_metrics(self, metrics: Metrics, nsrts: Set[NSRT],
                      predicates: Set[Predicate]) -> None:
        for metric in [
                "num_skeletons_optimized", "num_failures_discovered",
                "num_nodes_expanded", "num_nodes_created", "plan_length"
        ]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        self._metrics["total_num_nsrts"] += len(nsrts)
        self._metrics["total_num_preds"] += len(predicates)
        self._metrics["min_num_skeletons_optimized"] = min(
            metrics["num_skeletons_optimized"],
            self._metrics["min_num_skeletons_optimized"])
        self._metrics["max_num_skeletons_optimized"] = max(
            metrics["num_skeletons_optimized"],
            self._metrics["max_num_skeletons_optimized"])

    @abc.abstractmethod
    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        raise NotImplementedError("Override me!")

    def _get_current_predicates(self) -> Set[Predicate]:
        """Get the current set of predicates.

        Defaults to initial predicates.
        """
        return self._initial_predicates

    def get_option_model(self) -> _OptionModelBase:
        """For ONLY an oracle approach or BEHAVIOR, we allow the user to get
        the current option model."""
        assert self.get_name() == "oracle" or CFG.env == "behavior"
        return self._option_model

    def get_last_plan(self) -> List[_Option]:
        """For ONLY BEHAVIOR, we allow the user to get the plan that was most
        recently generated by a call to solve().

        Note that this doesn't fit into the standard API for an
        Approach, since solve() returns a policy, which abstracts away
        the details of whether that policy is actually a plan under the
        hood.
        """
        assert self.get_name() == "oracle" or CFG.env == "behavior"
        return self._last_plan

    def get_last_traj(self) -> List[State]:  # pragma: no cover
        """Note that this doesn't fit into the standard API for an Approach,
        since solve() returns a policy, which abstracts away the details of
        whether that policy is actually a plan under the hood."""
        assert self.get_name() == "oracle" or CFG.env == "behavior"
        return self._last_traj
