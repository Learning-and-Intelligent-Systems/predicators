"""An explorer that takes uses a trained RL agent that predicts both ground
NSRTs and corresponding continuous parameters."""

import logging
from typing import Callable, List, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OracleOptionModel
from predicators.planning import run_task_plan_once, sesame_plan
from predicators.rl.policies import TorchStochasticPolicy
from predicators.rl.rl_utils import env_state_to_maple_input, \
    get_ground_nsrt_and_params_from_maple
from predicators.settings import CFG
from predicators.structs import NSRT, Action, DummyOption, \
    ExplorationStrategy, GroundAtom, ParameterizedOption, Predicate, State, \
    Task, Type, _GroundNSRT, _Option


class MAPLEExplorer(BaseExplorer):
    """RLExplorer implementation.

    We assume that the RL agent here will output both discrete (in the
    form of a ground NSRT), and continuous parameters (needed to ground
    the option associated with said NSRT).
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int,
                 ground_nsrts: List[_GroundNSRT], nsrts: Set[NSRT],
                 exploration_policy: TorchStochasticPolicy,
                 observations_size: int, discrete_actions_size: int,
                 continuous_actions_size: int) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._ground_nsrts = ground_nsrts
        self._nsrts = nsrts
        self._exploration_policy = exploration_policy
        self._observations_size = observations_size
        self._discrete_actions_size = discrete_actions_size
        self._continuous_actions_size = continuous_actions_size
        self._rng = np.random.default_rng(self._seed)
        self._option_model = _OracleOptionModel(get_or_create_env(CFG.env))

    @classmethod
    def get_name(cls) -> str:
        return "maple_explorer"

    def _get_option_policy_using_task_planner(
            self, task: Task) -> Callable[[State], _Option]:
        # Run task planning and then greedily execute.
        timeout = CFG.timeout
        task_planning_heuristic = CFG.sesame_task_planning_heuristic
        plan, atoms_seq, _ = run_task_plan_once(
            task,
            self._nsrts,
            self._predicates,
            self._types,
            timeout,
            self._seed,
            task_planning_heuristic=task_planning_heuristic)
        return utils.nsrt_plan_to_greedy_option_policy(
            plan, task.goal, self._rng, necessary_atoms_seq=atoms_seq)

    def _get_action_policy_using_bilevel_planner(self, task: Task):
        plan, _, _ = sesame_plan(
            task,
            self._option_model,
            self._nsrts,
            self._predicates,
            self._types,
            CFG.timeout,
            CFG.seed,
            CFG.sesame_task_planning_heuristic,
            CFG.sesame_max_skeletons_optimized,
            max_horizon=CFG.horizon,
            allow_noops=CFG.sesame_allow_noops,
            use_visited_state_set=CFG.sesame_use_visited_state_set)
        policy = utils.option_plan_to_policy(plan)
        return policy

    def _get_option_from_maple(self, state: State,
                               curr_goal: Set[GroundAtom]) -> _Option:
        # Get the ground NSRT and continuous params predicted
        # by MAPLE.
        state_vec = env_state_to_maple_input(state)
        assert state_vec.shape[0] == self._observations_size
        maple_policy_action = self._exploration_policy.get_action(state_vec)[0]
        maple_ground_nsrt, maple_continuous_params = get_ground_nsrt_and_params_from_maple(
            maple_policy_action, self._ground_nsrts,
            self._discrete_actions_size, self._continuous_actions_size)
        # Next, decide whether we're going to use these continuous
        # params or ones from the base sampler.
        curr_ground_option = maple_ground_nsrt.sample_option(
            state, curr_goal, self._rng)  # option obtained using base sampler
        rand_val = self._rng.random()

        if rand_val < CFG.active_sampler_learning_exploration_epsilon:
            # We select a random ground nsrt, and obtain a sample using the
            # base samplers.
            ground_nsrt = self._rng.choice(self._ground_nsrts)
            # curr_ground_option = ground_nsrt.option.ground(
            #     ground_nsrt.option_objs, maple_continuous_params)
            curr_ground_option = ground_nsrt.sample_option(
                state, curr_goal, self._rng)
            logging.debug(
                f"[RL] Explorer running {maple_ground_nsrt.name}({maple_ground_nsrt.objects}) with clipped params {maple_continuous_params}"
            )
        else:
            logging.debug(
                f"[RL] Explorer running {maple_ground_nsrt.name}({maple_ground_nsrt.objects}) with base sampler params."
            )
        return curr_ground_option

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        curr_ground_option = None
        num_curr_option_steps = 0
        curr_option_policy = None
        curr_action_policy = None
        rand_val = self._rng.random()

        def _policy(state: State) -> Action:
            nonlocal self, curr_ground_option, train_task_idx, num_curr_option_steps, curr_option_policy, curr_action_policy, rand_val
            # Get the current task goal.
            curr_goal = self._train_tasks[train_task_idx].goal
            # Use the random value to determine whether we're using
            # the planner to generate actions, or just taking random actions.
            if rand_val < 1.01:
                if curr_action_policy is None:
                    curr_action_policy = self._get_action_policy_using_bilevel_planner(
                        Task(state, curr_goal))
                return curr_action_policy(state)

            # curr_ground_option = self._get_option_from_maple(state, curr_goal)
            # if not curr_ground_option.initiable(state):
            #         raise utils.OptionExecutionFailure(
            #             "Unsound option policy.",
            #             info={"last_failed_option": curr_ground_option})

            # We select a random ground nsrt, and obtain a random sample.
            ground_nsrt = self._rng.choice(self._ground_nsrts)
            option_continuous_params = ground_nsrt.option.params_space.sample()
            curr_ground_option = ground_nsrt.option.ground(
                ground_nsrt.option_objs, option_continuous_params)
            if not curr_ground_option.initiable(state):
                num_curr_option_steps = 0
                raise utils.OptionExecutionFailure(
                    "Unsound option policy.",
                    info={"last_failed_option": curr_ground_option})

            return curr_ground_option.policy(state)

            # if curr_option_policy is None:
            #     curr_option_policy = self._get_option_policy_using_task_planner(Task(state, curr_goal))
            # assert curr_option_policy is not None

            # if curr_ground_option is None or (
            #         curr_ground_option is not None
            #         and curr_ground_option.terminal(state)):
            #     # Generate a random number and use it to decide if we're going to use
            #     # the planner or the policy.
            #     rand_val = self._rng.random()
            #     if rand_val < 1.0:
            #         try:
            #             curr_ground_option = curr_option_policy(state)
            #         except utils.OptionExecutionFailure:
            #             curr_option_policy = self._get_option_policy_using_planner(Task(state, curr_goal))
            #             curr_ground_option = curr_option_policy(state)
            #             # curr_ground_option = self._get_option_from_maple(state, curr_goal)
            #     else:
            #         curr_ground_option = self._get_option_from_maple(state, curr_goal)

            #     if not curr_ground_option.initiable(state):
            #         num_curr_option_steps = 0
            #         raise utils.OptionExecutionFailure(
            #             "Unsound option policy.",
            #             info={"last_failed_option": curr_ground_option})
            #     num_curr_option_steps = 0

            # if CFG.max_num_steps_option_rollout is not None and \
            #     num_curr_option_steps >= CFG.max_num_steps_option_rollout:
            #     raise utils.OptionTimeoutFailure(
            #         "Exceeded max option steps.",
            #         info={"last_failed_option": curr_ground_option})

            # num_curr_option_steps += 1

            # return curr_ground_option.policy(state)

        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return _policy, termination_function
