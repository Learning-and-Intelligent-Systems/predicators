"""An explorer that takes uses a trained RL agent that predicts both ground
NSRTs and corresponding continuous parameters."""

from typing import List, Set

from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.rl.policies import TorchStochasticPolicy
from predicators.rl.rl_utils import make_executable_maple_policy
from predicators.structs import NSRT, Action, DummyOption, \
    ExplorationStrategy, ParameterizedOption, Predicate, State, Task, Type, \
    _GroundNSRT


class MAPLEExplorer(BaseExplorer):
    """RLExplorer implementation.

    We assume that the RL agent here will output both
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, ground_nsrts: List[_GroundNSRT], exploration_policy: TorchStochasticPolicy, observations_size: int, discrete_actions_size: int, continuous_actions_size: int) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._ground_nsrts = ground_nsrts
        self._exploration_policy = exploration_policy
        self._observations_size = observations_size
        self._discrete_actions_size = discrete_actions_size
        self._continuous_actions_size = continuous_actions_size

    @classmethod
    def get_name(cls) -> str:
        return "maple_explorer"

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        
        policy = make_executable_maple_policy(self._exploration_policy, self._ground_nsrts, self._observations_size, self._discrete_actions_size, self._continuous_actions_size)
        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return policy, termination_function
