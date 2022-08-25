"""Policy-guided planning for generalized policy generation for planning
guidance (PG4).

PG4 requires known STRIPS operators. The command below uses oracle operators,
but it is also possible to use this approach with operators learned from
demonstrations.

Example command line:
    python predicators/main.py --approach pg4 --seed 0 \
        --env cover \
        --strips_learner oracle --num_train_tasks 50
"""
from __future__ import annotations

from typing import Any, Callable, List, Set, Tuple

from predicators import utils
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.approaches.pg3_approach import PG3Approach
from predicators.settings import CFG
from predicators.structs import NSRT, AbstractPolicy, Action, Metrics, \
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

    def _run_sesame_plan(
            self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
            timeout: float, seed: int,
            **kwargs: Any) -> Tuple[List[_Option], Metrics, List[State]]:
        """Generates a plan choosing the best skeletons generated from policy-
        based skeletons and primitive successors."""
        abstract_policy: AbstractPolicy = lambda a, o, g: utils.query_ldl(
            self._current_ldl, a, o, g)
        max_policy_guided_rollout = CFG.pg3_max_policy_guided_rollout
        return super()._run_sesame_plan(
            task,
            nsrts,
            preds,
            timeout,
            seed,
            abstract_policy=abstract_policy,
            max_policy_guided_rollout=max_policy_guided_rollout,
            **kwargs)
