"""A bilevel planning approach that uses a refinement cost estimator.

Generates N proposed skeletons and then ranks them based on a given
refinement cost estimation function (e.g. a heuristic, learned model),
attempting to refine them in this order.
"""

from typing import Any, List, Set, Tuple

from gym.spaces import Box

from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.refinement_estimators import create_refinement_estimator
from predicators.settings import CFG
from predicators.structs import NSRT, Metrics, ParameterizedOption, \
    Predicate, Task, Type, _Option


class RefinementEstimationApproach(BilevelPlanningApproach):
    """A bilevel planning approach that uses a refinement cost estimator."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, task_planning_heuristic,
                         max_skeletons_optimized)
        self._nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
                                   self._initial_options)
        self._refinement_estimator = create_refinement_estimator(
            CFG.refinement_estimator)

    @classmethod
    def get_name(cls) -> str:
        return "refinement_estimation"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def _run_sesame_plan(self, task: Task, nsrts: Set[NSRT],
                         preds: Set[Predicate], timeout: float, seed: int,
                         **kwargs: Any) -> Tuple[List[_Option], Metrics]:
        """Generates a plan choosing the best skeletons based on a given
        refinement cost estimator."""
        result = super()._run_sesame_plan(
            task,
            nsrts,
            preds,
            timeout,
            seed,
            refinement_estimator=self._refinement_estimator,
            **kwargs)
        return result
