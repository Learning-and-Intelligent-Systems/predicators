"""An explorer that uses bilevel planning with NSRTs."""

from typing import Callable, List, Set, Tuple

from gym.spaces import Box

from predicators.src import utils
from predicators.src.interaction.base_explorer import BaseExplorer
from predicators.src.option_model import _OptionModelBase
from predicators.src.planning import PlanningFailure, PlanningTimeout, \
    sesame_plan
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Action, ParameterizedOption, \
    Predicate, State, Task, Type


class BilevelPlanningExplorer(BaseExplorer):
    """BilevelPlanningExplorer implementation.

    The approach is abstract: subclasses decide how to use the bilevel
    planning _solve method to create an exploration strategy.
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task], nsrts: Set[NSRT],
                 option_model: _OptionModelBase) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks)
        self._nsrts = nsrts
        self._option_model = option_model
        self._num_calls = 0

    def _solve(
        self, task: Task, timeout: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:

        # Ensure random over successive calls.
        seed = self._seed + self._num_calls
        try:
            plan, _ = sesame_plan(task,
                                  self._option_model,
                                  self._nsrts,
                                  self._predicates,
                                  self._types,
                                  timeout,
                                  seed,
                                  CFG.sesame_task_planning_heuristic,
                                  CFG.sesame_max_skeletons_optimized,
                                  max_horizon=CFG.horizon,
                                  allow_noops=CFG.sesame_allow_noops)
        except (PlanningFailure, PlanningTimeout) as e:
            raise utils.RequestActPolicyFailure(e.args[0], e.info)

        policy = utils.option_plan_to_policy(plan)
        termination_function = task.goal_holds

        return policy, termination_function
