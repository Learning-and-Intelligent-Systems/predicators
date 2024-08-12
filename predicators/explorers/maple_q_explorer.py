"""Uses MAPLE Q function for epsilon-greedy exploration."""

from typing import List, Set

from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.ml_models import MapleQFunction
from predicators.settings import CFG
from predicators.structs import NSRT, ExplorationStrategy, \
    ParameterizedOption, Predicate, State, Task, Type, _Option


class MapleQExplorer(BaseExplorer):
    """Uses MAPLE Q function for epsilon-greedy exploration."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, nsrts: Set[NSRT],
                 q_function: MapleQFunction) -> None:

        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"

        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._nsrts = nsrts
        self._q_function = q_function

    @classmethod
    def get_name(cls) -> str:
        return "maple_q"

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:

        num_samples = CFG.active_sampler_learning_num_samples
        goal = self._train_tasks[train_task_idx].goal

        def _option_policy(state: State) -> _Option:
            return self._q_function.get_option(state, \
                                               goal, num_samples, "train")

        policy = utils.option_policy_to_policy(
            _option_policy, max_option_steps=CFG.max_num_steps_option_rollout)

        # Never terminate.
        terminal = lambda s: False

        return policy, terminal
