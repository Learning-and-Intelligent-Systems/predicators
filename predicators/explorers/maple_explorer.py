"""An explorer that takes uses a trained RL agent that predicts both ground
NSRTs and corresponding continuous parameters."""

from typing import Callable, List, Set

import itertools
import numpy as np
from gym.spaces import Box
from typing import Dict

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.explorers.base_explorer import BaseExplorer
from predicators.option_model import _OracleOptionModel
from predicators.planning import run_task_plan_once, sesame_plan
from predicators.rl.rl_utils import make_executable_qfunc_only_policy
from predicators.settings import CFG
from predicators.structs import NSRT, Action, DummyOption, \
    ExplorationStrategy, GroundAtom, ParameterizedOption, Predicate, State, \
    Task, Type, _GroundNSRT, _Option


_RNG_COUNTER = itertools.count()  # ensure unique but reproducible rngs


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
                 qf1, qf2, ground_nsrt_to_idx: Dict[_GroundNSRT, int],
                 observations_size: int, discrete_actions_size: int,
                 continuous_actions_size: int) -> None:

        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._ground_nsrts = ground_nsrts
        self._ground_nsrt_to_idx = ground_nsrt_to_idx
        self._qf1 = qf1
        self._qf2 = qf2
        self._nsrts = nsrts
        self._observations_size = observations_size
        self._discrete_actions_size = discrete_actions_size
        self._continuous_actions_size = continuous_actions_size
        self._rng = np.random.default_rng(self._seed + next(_RNG_COUNTER))
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

    def _get_action_policy_using_maple_q(self, task: Task):
        return make_executable_qfunc_only_policy(self._qf1, self._qf2, self._ground_nsrts, self._ground_nsrt_to_idx, self._observations_size, self._discrete_actions_size, self._continuous_actions_size, self._predicates, task.goal, self._rng, CFG.active_sampler_learning_exploration_epsilon)

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        curr_ground_option = None
        num_curr_option_steps = 0
        curr_option_policy = None
        curr_action_policy = None
        rand_val = self._rng.random()

        def _policy(state: State) -> Action:
            # nonlocal self, curr_ground_option, train_task_idx, num_curr_option_steps, curr_option_policy, curr_action_policy, rand_val
            # # Get the current task goal.
            # curr_goal = self._train_tasks[train_task_idx].goal
            # # Use the random value to determine whether we're using
            # # the planner to generate actions, or just taking random actions.
            # if rand_val < 0.75:
            #     if curr_action_policy is None:
            #         curr_action_policy = self._get_action_policy_using_bilevel_planner(
            #             Task(state, curr_goal))
            #     return curr_action_policy(state)

            # # We select a random ground nsrt, and obtain a random sample.
            # ground_nsrt = self._rng.choice(self._ground_nsrts)
            # option_continuous_params = ground_nsrt.option.params_space.sample()
            # curr_ground_option = ground_nsrt.option.ground(
            #     ground_nsrt.option_objs, option_continuous_params)
            # if not curr_ground_option.initiable(state):
            #     num_curr_option_steps = 0
            #     raise utils.OptionExecutionFailure(
            #         "Unsound option policy.",
            #         info={"last_failed_option": curr_ground_option})

            # return curr_ground_option.policy(state)

            nonlocal self, train_task_idx
            task = self._train_tasks[train_task_idx]

            return self._get_action_policy_using_maple_q(task)(state)


        # Never terminate (until the interaction budget is exceeded).
        termination_function = lambda _: False
        return _policy, termination_function
