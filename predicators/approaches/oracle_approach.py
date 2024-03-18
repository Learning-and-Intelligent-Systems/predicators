"""A bilevel planning approach that uses hand-specified NSRTs.

The approach is aware of the initial predicates and options. Predicates
that are not in the initial predicates are excluded from the ground
truth NSRTs. If an NSRT's option is not included, that NSRT will not be
generated at all.
"""

from typing import List, Optional, Set

from gym.spaces import Box

from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.ground_truth_models import get_gt_nsrts
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import NSRT, ParameterizedOption, Predicate, Task, \
    Type


class OracleApproach(BilevelPlanningApproach):
    """A bilevel planning approach that uses hand-specified NSRTs."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 nsrts: Optional[Set[NSRT]] = None,
                 option_model: Optional[_OptionModelBase] = None) -> None:
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        if nsrts is None:
            nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
                                 self._initial_options)
        print("LENGTH OF NSRTS: ", len(nsrts))
        self._nsrts = nsrts

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts
