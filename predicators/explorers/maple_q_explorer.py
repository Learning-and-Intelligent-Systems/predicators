"""TODO describe"""

import logging
from collections import deque
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.explorers.base_explorer import BaseExplorer
from predicators.planning import PlanningFailure, PlanningTimeout, \
    run_task_plan_once
from predicators.ml_models import QFunction
from predicators.settings import CFG
from predicators.structs import NSRT, Action, ExplorationStrategy, \
    GroundAtom, NSRTSampler, ParameterizedOption, Predicate, State, Task, \
    Type, _GroundNSRT, _GroundSTRIPSOperator, _Option


class MapleQExplorer(BaseExplorer):
    """TODO describe
    """

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, nsrts: Set[NSRT],
                 q_function: QFunction) -> None:

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
        
        task = self._train_tasks[train_task_idx]
        goal = task.goal
        objects = set(task.init)
        all_ground_nsrts: List[_GroundNSRT] = []
        for nsrt in self._nsrts:
            all_ground_nsrts.extend(utils.all_ground_nsrts(nsrt, objects))

        def _option_policy(state: State) -> _Option:
            candidates: List[_Option] = []
            # Find all applicable ground NSRTs.
            atoms = utils.abstract(state, self._predicates)
            applicable_ground_nsrts = utils.get_applicable_operators(all_ground_nsrts, atoms)
            # Sample candidate options.
            for ground_nsrt in applicable_ground_nsrts:
                for _ in range(CFG.active_sampler_learning_num_samples):
                    option = ground_nsrt.sample_option(state, goal, self._rng)
                    candidates.append(option)
            # Choose a random candidate with epsilon probability.
            epsilon = CFG.active_sampler_learning_exploration_epsilon
            if self._rng.uniform() < epsilon:
                idx = self._rng.choice(len(candidates))
                return candidates[idx]
            # Score the candidates using the Q function.
            scores: List[float] = []
            for option in candidates:
                score = self._q_function.predict(state, option)
                scores.append(score)
            # Select the best-scoring candidate.
            idx = np.argmax(scores)
            return candidates[idx]
        
        policy = utils.option_policy_to_policy(_option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout)
        
        # Never terminate.
        terminal = lambda s: False

        return policy, terminal