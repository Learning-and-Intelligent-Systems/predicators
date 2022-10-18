"""An explorer that uses bilevel planning with NSRTs."""

from typing import List, Set

from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OptionModelBase
from predicators.planning import sesame_plan
from predicators.settings import CFG
from predicators.structs import NSRT, ExplorationStrategy, \
    ParameterizedOption, Predicate, Task, Type


class BilevelPlanningExplorer(BaseExplorer):
    """BilevelPlanningExplorer implementation.

    This explorer is abstract: subclasses decide how to use the _solve
    method implemented in this class, which calls sesame_plan().
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task], nsrts: Set[NSRT],
                 option_model: _OptionModelBase) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks)
        self._nsrts = nsrts
        self._option_model = option_model
        self._num_calls = 0

    def _solve(self, task: Task, timeout: int) -> ExplorationStrategy:

        # Ensure random over successive calls.
        self._num_calls += 1
        seed = self._seed + self._num_calls
        # Note: subclasses are responsible for catching PlanningFailure and
        # PlanningTimeout and handling them accordingly.
        plan, _, _ = sesame_plan(
            task,
            self._option_model,
            self._nsrts,
            self._predicates,
            self._types,
            timeout,
            seed,
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
            allow_noops=CFG.sesame_allow_noops,
            use_visited_state_set=CFG.sesame_use_visited_state_set)
        policy = utils.option_plan_to_policy(plan)
        termination_function = task.goal_holds

        return policy, termination_function
