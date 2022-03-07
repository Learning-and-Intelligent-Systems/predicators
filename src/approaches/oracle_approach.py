"""A bilevel planning approach that uses hand-specified NSRTs.

The approach is aware of the initial predicates and options. Predicates
that are not in the initial predicates are excluded from the ground
truth NSRTs. If an NSRT's option is not included, that NSRT will not be
generated at all.
"""

from typing import Set
from predicators.src.structs import NSRT
from predicators.src.approaches import BilevelPlanningApproach
from predicators.src.ground_truth_nsrts import get_gt_nsrts


class OracleApproach(BilevelPlanningApproach):
    """A bilevel planning approach that uses hand-specified NSRTs."""

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_nsrts(self) -> Set[NSRT]:
        return get_gt_nsrts(self._initial_predicates, self._initial_options)
