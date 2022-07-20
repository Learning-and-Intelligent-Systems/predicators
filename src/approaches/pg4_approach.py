"""Policy-guided planning for generalized policy generation for planning
guidance (PG4).

PG4 requires known STRIPS operators. The command below uses oracle operators,
but it is also possible to use this approach with operators learned from
demonstrations.

Example command line:
    python src/main.py --approach pg4 --seed 0 \
        --env cover \
        --strips_learner oracle --num_train_tasks 50
"""
from __future__ import annotations

from typing import Callable, List, Set, Tuple

from predicators.src import utils
from predicators.src.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.src.approaches.pg3_approach import PG3Approach
from predicators.src.planning import sesame_plan
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, AbstractPolicy, Action, Metrics, \
    Predicate, State, Task, _Option


class PG4Approach(PG3Approach):
    """Policy-guided planning for generalized policy generation for planning
    guidance (PG4)."""

    @classmethod
    def get_name(cls) -> str:
        return "pg4"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # This is a rare case where protected access seems like the best thing
        # to do, because this approach subclasses from BilevelPlanningApproach,
        # but it's not the direct child, so we can't use super().
        return BilevelPlanningApproach._solve(self, task, timeout)  # pylint: disable=protected-access

    def _run_sesame_plan(self, task: Task, nsrts: Set[NSRT],
                         preds: Set[Predicate], timeout: float,
                         seed: int) -> Tuple[List[_Option], Metrics]:
        """Generates a plan choosing the best skeletons generated from policy-
        based skeletons and primitive successors."""
        abstract_policy: AbstractPolicy = lambda a, o, g: utils.query_ldl(
            self._current_ldl, a, o, g)
        max_policy_guided_rollout = CFG.pg3_max_policy_guided_rollout

        return sesame_plan(task,
                           self._option_model,
                           nsrts,
                           preds,
                           self._types,
                           timeout,
                           seed,
                           self._task_planning_heuristic,
                           self._max_skeletons_optimized,
                           max_horizon=CFG.horizon,
                           abstract_policy=abstract_policy,
                           max_policy_guided_rollout=max_policy_guided_rollout,
                           allow_noops=CFG.sesame_allow_noops)
