"""A hand-written LDL bridge policy."""

from typing import Set

from predicators.bridge_policies.ldl_bridge_policy import LDLBridgePolicy
from predicators.ground_truth_models import get_gt_ldl_bridge_policy
from predicators.settings import CFG
from predicators.structs import NSRT, LiftedDecisionList, \
    ParameterizedOption, Predicate, Type


class OracleBridgePolicy(LDLBridgePolicy):
    """A hand-written LDL bridge policy."""

    def __init__(self, types: Set[Type], predicates: Set[Predicate],
                 options: Set[ParameterizedOption], nsrts: Set[NSRT]) -> None:
        super().__init__(types, predicates, options, nsrts)
        all_predicates = predicates | self._failure_predicates
        self._oracle_ldl = get_gt_ldl_bridge_policy(CFG.env, self._types,
                                                    all_predicates,
                                                    self._options, self._nsrts)

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_ldl(self) -> LiftedDecisionList:
        return self._oracle_ldl
