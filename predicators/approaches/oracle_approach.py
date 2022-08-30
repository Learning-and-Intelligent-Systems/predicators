"""A bilevel planning approach that uses hand-specified NSRTs.

The approach is aware of the initial predicates and options. Predicates
that are not in the initial predicates are excluded from the ground
truth NSRTs. If an NSRT's option is not included, that NSRT will not be
generated at all.
"""

from typing import List, Set

from gym.spaces import Box

from predicators import utils
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.envs.base_env import BaseEnv
from predicators.ground_truth_nsrts import get_gt_nsrts
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
                 max_skeletons_optimized: int = -1) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, task_planning_heuristic,
                         max_skeletons_optimized)
        self._nsrts = get_gt_nsrts(self._initial_predicates,
                                   self._initial_options)

    @classmethod
    def get_name(cls) -> str:
        return "oracle"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def recompute_nsrts(self, env: BaseEnv) -> None:
        """Function to recompute NSRTs given environment change."""
        if CFG.env == "behavior":  # pragma: no cover
            # If the env is BEHAVIOR the types and therefore
            # initial_predicates and initial_options could
            # have changed.
            self._initial_predicates, _ = \
                utils.parse_config_excluded_predicates(env)
            self._initial_options = env.options
        self._nsrts = get_gt_nsrts(self._initial_predicates,
                                   self._initial_options)
