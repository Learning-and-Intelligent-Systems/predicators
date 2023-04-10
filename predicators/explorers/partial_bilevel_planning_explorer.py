"""An explorer that explores by solving tasks with bilevel planning."""

from typing import List, Set

from gym.spaces import Box

from predicators import utils
from predicators.explorers.bilevel_planning_explorer import \
    BilevelPlanningExplorer
from predicators.explorers.random_options_explorer import RandomOptionsExplorer
from predicators.planning import PlanningFailure, PlanningTimeout
from predicators.structs import ExplorationStrategy
from predicators.settings import CFG

class PartialBilevelPlanningExplorer(BilevelPlanningExplorer):
    """PartialBilevelPlanningExplorer implementation."""

    @classmethod
    def get_name(cls) -> str:
        return "partial_planning"

    def get_exploration_strategy(self, train_task_idx: int,
                                 timeout: int) -> ExplorationStrategy:
        task = self._train_tasks[train_task_idx]
        print([task.init.get(dummy, "indicator") for dummy in task.init if dummy.name == "dummy"])
        try:
            return self._solve(task, timeout)
        except (PlanningFailure, PlanningTimeout) as e:
            # print(f'Failed to refine plan, {len(e.info["partial_refinements"])} partial partial_refinements')
            skeleton, partial_plan = e.info["partial_refinements"][0]
            policy = utils.option_plan_to_policy(partial_plan)
            # When the policy finishes, an OptionExecutionFailure is raised
            # and caught, terminating the episode.
            termination_function = lambda _: False

            self._save_failure_metrics(task)
            return policy, termination_function, skeleton

    def _save_failure_metrics(self, task) -> None:
        ### Keep track of per-env metrics for planar_behavior
        subenv = None
        for obj in task.init:
            if obj.name == "dummy":
                subenv = task.init.get(obj, "indicator")
                break
        ###
        self._metrics["total_num_samples_failed"] += CFG.sesame_max_samples_total
        self._metrics["num_unsolved"] += 1

        if subenv is not None:
            self._metrics[f"env_{subenv}_total_num_samples_failed"] += CFG.sesame_max_samples_total
            self._metrics[f"env_{subenv}_num_unsolved"] += 1
        print('saved failure metrics')   