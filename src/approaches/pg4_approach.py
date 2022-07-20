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

from typing import Callable

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.pg3_approach import PG3Approach
from predicators.src.planning import PlanningFailure, PlanningTimeout, \
    sesame_plan
from predicators.src.settings import CFG
from predicators.src.structs import AbstractPolicy, Action, State, Task


class PG4Approach(PG3Approach):
    """Policy-guided planning for generalized policy generation for planning
    guidance (PG4)."""

    @classmethod
    def get_name(cls) -> str:
        return "pg4"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Generates a plan choosing the best skeletons generated from policy-
        based skeletons and primitive successors."""
        seed = self._seed + self._num_calls
        nsrts = self._get_current_nsrts()
        preds = self._get_current_predicates()
        abstract_policy: AbstractPolicy = lambda a,o,g: \
                                            utils.query_ldl(self._current_ldl,
                                            a, o, g)
        max_policy_guided_rollout = CFG.pg3_max_policy_guided_rollout
        try:
            plan, _ = sesame_plan(
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
                abstract_policy=abstract_policy,
                max_policy_guided_rollout=max_policy_guided_rollout,
                allow_noops=CFG.sesame_allow_noops)
        except PlanningFailure as e:
            raise ApproachFailure(e.args[0], e.info)
        except PlanningTimeout as e:
            raise ApproachTimeout(e.args[0], e.info)

        option_policy = utils.option_plan_to_policy(plan)

        def _policy(s: State) -> Action:
            try:
                return option_policy(s)
            except utils.OptionExecutionFailure as e:
                raise ApproachFailure(e.args[0], e.info)

        return _policy
