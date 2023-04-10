"""An explorer that uses bilevel planning with NSRTs."""

from typing import List, Set
from collections import defaultdict

from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OptionModelBase
from predicators.planning import sesame_plan
from predicators.settings import CFG
from predicators.structs import NSRT, ExplorationStrategy, \
    Metrics, ParameterizedOption, Predicate, Task, Type


class BilevelPlanningExplorer(BaseExplorer):
    """BilevelPlanningExplorer implementation.

    This explorer is abstract: subclasses decide how to use the _solve
    method implemented in this class, which calls sesame_plan().
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task], nsrts: Set[NSRT],
                 option_model: _OptionModelBase) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks)
        self._nsrts = nsrts
        self._option_model = option_model
        self._num_calls = 0
        self._metrics = defaultdict(float)

    @property
    def metrics(self) -> Metrics:
        return self._metrics.copy()

    def _solve(self, task: Task, timeout: int) -> ExplorationStrategy:

        # Ensure random over successive calls.
        self._num_calls += 1
        seed = self._seed + self._num_calls
        # Note: subclasses are responsible for catching PlanningFailure and
        # PlanningTimeout and handling them accordingly.
        plan, metrics, skeleton = sesame_plan(
            task,
            self._option_model,
            self._nsrts,
            self._predicates,
            self._types,
            timeout,
            seed,
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,#
            max_horizon=CFG.horizon,
            # max_samples_per_step=1,
            allow_noops=CFG.sesame_allow_noops,
            use_visited_state_set=CFG.sesame_use_visited_state_set,
            return_skeleton=True)
        ### Keep track of per-env metrics for planar_behavior
        for obj in task.init:
            if obj.name == "dummy":
                metrics["env"] = task.init.get(obj, "indicator")
                break
        ###

        self._save_metrics(metrics)

        policy = utils.option_plan_to_policy(plan)
        termination_function = task.goal_holds

        return policy, termination_function, skeleton

    def _save_metrics(self, metrics: Metrics) -> None:
        for metric in [
                "num_samples", "num_skeletons_optimized",
                "num_failures_discovered", "num_nodes_expanded",
                "num_nodes_created", "plan_length"
        ]:
            self._metrics[f"total_{metric}"] += metrics[metric]
        for metric in [
                "num_samples",
                "num_skeletons_optimized",
        ]:
            self._metrics[f"min_{metric}"] = min(
                metrics[metric], self._metrics[f"min_{metric}"])
            self._metrics[f"max_{metric}"] = max(
                metrics[metric], self._metrics[f"max_{metric}"])
        self._metrics["num_solved"] += 1

        if "env" in metrics:
            env = metrics["env"]
            for metric in [
                    "num_samples", "num_skeletons_optimized",
                    "num_failures_discovered", "num_nodes_expanded",
                    "num_nodes_created", "plan_length"
            ]:
                self._metrics[f"env_{env}_total_{metric}"] += metrics[metric]
            for metric in [
                    f"num_samples",
                    f"num_skeletons_optimized",
            ]:
                self._metrics[f"env_{env}_min_{metric}"] = min(
                    metrics[metric], self._metrics[f"env_{env}_min_{metric}"])
                self._metrics[f"env_{env}_max_{metric}"] = max(
                    metrics[metric], self._metrics[f"env_{env}_max_{metric}"])
            self._metrics[f"env_{env}_num_solved"] += 1
