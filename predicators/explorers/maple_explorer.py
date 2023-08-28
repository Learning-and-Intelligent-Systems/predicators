"""An explorer that takes uses a trained RL agent that predicts both ground
NSRTs and corresponding continuous parameters."""

from typing import List, Set
import logging

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.rl.policies import TorchStochasticPolicy
from predicators.rl.rl_utils import env_state_to_maple_input, \
    get_ground_nsrt_and_params_from_maple
from predicators.settings import CFG
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
                 max_steps_before_termination: int,
                 ground_nsrts: List[_GroundNSRT],
                 exploration_policy: TorchStochasticPolicy,
                 observations_size: int, discrete_actions_size: int,
                 continuous_actions_size: int) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._ground_nsrts = ground_nsrts
        self._exploration_policy = exploration_policy
        self._observations_size = observations_size
        self._discrete_actions_size = discrete_actions_size
        self._continuous_actions_size = continuous_actions_size
        self._rng = np.random.default_rng(self._seed)

    @classmethod
    def get_name(cls) -> str:
        return "maple_explorer"

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        curr_ground_option = None
        num_curr_option_steps = 0

        def _policy(state: State) -> Action:
            nonlocal self, curr_ground_option, train_task_idx, num_curr_option_steps
            state_vec = env_state_to_maple_input(state)
            if curr_ground_option is None or (
                    curr_ground_option is not None
                    and curr_ground_option.terminal(state)):
                # Get the ground NSRT and continuous params predicted
                # by MAPLE.
                assert state_vec.shape[0] == self._observations_size
                num_curr_option_steps = 0
                maple_policy_action = self._exploration_policy.get_action(
                    state_vec)[0]
                maple_ground_nsrt, maple_continuous_params = get_ground_nsrt_and_params_from_maple(
                    maple_policy_action, self._ground_nsrts,
                    self._discrete_actions_size, self._continuous_actions_size)
                # Next, decide whether we're going to use these continuous
                # params or ones from the base sampler.
                curr_goal = self._train_tasks[train_task_idx].goal
                curr_ground_option = maple_ground_nsrt.sample_option(
                    state, curr_goal,
                    self._rng)  # option obtained using base sampler
                rand_val = self._rng.random()

                if rand_val < CFG.active_sampler_learning_exploration_epsilon:
                    # option obtained using the particular continuous param
                    # vals output by MAPLE.
                    curr_ground_option = maple_ground_nsrt.option.ground(
                        maple_ground_nsrt.option_objs, maple_continuous_params)
                    logging.debug(
                            f"[RL] Explorer running {maple_ground_nsrt.name}({maple_ground_nsrt.objects}) with clipped params {maple_continuous_params}"
                        )
                else:
                    logging.debug(
                            f"[RL] Explorer running {maple_ground_nsrt.name}({maple_ground_nsrt.objects}) with base sampler params."
                        )

                if not curr_ground_option.initiable(state):
                    num_curr_option_steps = 0
                    raise utils.OptionExecutionFailure(
                        "Unsound option policy.",
                        info={"last_failed_option": curr_ground_option})

            if CFG.max_num_steps_option_rollout is not None and \
                num_curr_option_steps >= CFG.max_num_steps_option_rollout:
                raise utils.OptionTimeoutFailure(
                    "Exceeded max option steps.",
                    info={"last_failed_option": curr_ground_option})

            num_curr_option_steps += 1

            return curr_ground_option.policy(state)

        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return _policy, termination_function
