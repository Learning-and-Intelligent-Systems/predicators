"""Policy-guided planning for generalized policy generation (PG3) with an
initialized policy.

PG3 requires known STRIPS operators. The command below uses oracle operators,
but it is also possible to use this approach with operators learned from
demonstrations.

Example command line:
    python predicators/main.py --approach initialized_pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10
"""
from __future__ import annotations

import dill as pkl

from predicators.approaches.pg3_approach import PG3Approach
from predicators.settings import CFG
from predicators.structs import LiftedDecisionList


class InitializedPG3Approach(PG3Approach):
    """Policy-guided planning for generalized policy generation (PG3) with
    initialized policy."""

    @classmethod
    def get_name(cls) -> str:
        return "initialized_pg3"

    @staticmethod
    def _get_policy_search_initial_ldl() -> LiftedDecisionList:
        # Initialize with initialized policy from file.
        assert CFG.pg3_init_policy
        with open(CFG.pg3_init_policy, "rb") as f:
            return pkl.load(f)
