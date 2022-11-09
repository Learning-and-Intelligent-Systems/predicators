"""Policy-guided planning for generalized policy generation (PG3) with an
initialized policy.

PG3 requires known STRIPS operators. The command below uses oracle operators,
but it is also possible to use this approach with operators learned from
demonstrations.

Example. First run:
    python predicators/main.py --approach pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10

Then run:
    python predicators/main.py --approach initialized_pg3 --seed 0 \
        --env pddl_easy_delivery_procedural_tasks \
        --strips_learner oracle --num_train_tasks 10 \
        --pg3_init_policy saved_approaches/pddl_easy_delivery_procedural_tasks__pg3__0______.saved_None.ldl \
        --pg3_init_base_env pddl_easy_delivery_procedural_tasks
"""
from __future__ import annotations

from dataclasses import dataclass
import dill as pkl
from typing import Dict

from predicators.approaches.pg3_approach import PG3Approach
from predicators.envs import get_or_create_env
from predicators.envs.base_env import BaseEnv
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.settings import CFG
from predicators.structs import LiftedDecisionList, NSRT, Predicate, Type


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
            init_policy = pkl.load(f)

        # Determine an analogical mapping between the current domain and the
        # domain that the initialized policy originates from.
        base_env_name = CFG.pg3_init_base_env
        target_env_name = CFG.env
        base_env = get_or_create_env(base_env_name)
        target_env = get_or_create_env(target_env_name)
        analogy = _find_domain_analogy(base_env, target_env)


@dataclass(frozen=True)
class _Analogy:
    # All maps are base -> target.
    predicates: Dict[Predicate, Predicate]
    nsrts: Dict[NSRT, NSRT]
    types: Dict[Type, Type]


def _find_domain_analogy(base_env: BaseEnv, target_env: BaseEnv) -> _Analogy:
    assert base_env.get_name() == target_env.get_name(), \
        "Only trivial domain mappings are implemented so far"
    predicate_map = {p: p for p in env.predicates}
    nsrt_map = {p: p for p in env.predicates}
    import ipdb; ipdb.set_trace()
