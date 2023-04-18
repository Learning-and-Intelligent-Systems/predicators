"""TODO

python predicators/main.py --env painting --approach synthetic \
        --seed 0 --num_train_tasks 10  --num_test_tasks 10\
        --painting_lid_open_prob 0.5 \
        --painting_initial_holding_prob 1.0 \
        --painting_num_objs_train '[1]' \
        --painting_num_objs_test '[1]' \
        --painting_num_objs_test '[1]' \
        --painting_goal_receptacles 'box'
"""
from __future__ import annotations

import logging
import time
from typing import Callable, List, Optional, Set

import dill as pkl
from pg3.policy_search import learn_policy

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.planning import PlanningFailure, run_low_level_search
from predicators.settings import CFG
from predicators.structs import Action, Box, Dataset, GroundAtom, \
    LiftedDecisionList, Object, ParameterizedOption, Predicate, State, Task, \
    Type, _GroundNSRT


class SyntheticNSRTLearningApproach(NSRTLearningApproach):
    """TODO."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._current_ldl = LiftedDecisionList([])

    @classmethod
    def get_name(cls) -> str:
        return "synthetic"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # TODO override
        return super()._solve(task, timeout)

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int]) -> None:

        
        
        
        # # Learn initial NSRTs without synthetic crap.
        # # TODO: skip sampler learning here in a non-terrible way.
        # sampler_learner = CFG.sampler_learner
        # utils.update_config({"sampler_learner": "oracle"})
        # super()._learn_nsrts(trajectories, online_learning_cycle)
        # nsrts = self._get_current_nsrts()
        # preds = self._get_current_predicates()

        # # Use learned operators to compute costs to go.
        # # For each irrational step, record the rational action(s) that were
        # # not taken. We will use these to add negative synthetic atoms later.
        # trajectory_optimal_ctgs: List[List[float]] = []
        # trajectory_rational_acts_not_taken: List[List[Set[_GroundNSRTs]]] = []
        # for trajectory in trajectories:
        #     optimal_ctgs: List[float] = []
        #     rational_acts_not_taken: List[List[Set[_GroundNSRTs]]] = []
        #     # TODO: segment.
        #     states = trajectory.states  # THIS IS WRONG WHEN OPTIONS ARE MULTI-STEP
        #     goal = self._train_tasks[trajectory.train_task_idx].goal

        #     for state in states:
        #         task = Task(state, goal)
        #         # Assuming optimal task planning here.
        #         assert (CFG.sesame_task_planner == "astar" and \
        #                 CFG.sesame_task_planning_heuristic == "lmcut") or \
        #                 CFG.sesame_task_planner == "fdopt"
        #         try:
        #             nsrt_plan, _, _ = self._run_task_plan(
        #                 task, nsrts, preds, CFG.timeout, self._seed)
        #             ctg: float = len(nsrt_plan)
        #         except ApproachFailure:  # pragma: no cover
        #             # Planning failed, put in infinite cost to go.
        #             ctg = float("inf")
        #         if optimal_ctgs:
        #             last_ctg = optimal_ctgs[-1]
        #             # Rational.
        #             if last_ctg == ctg + 1:
        #                 acts_not_taken = set()
        #             # Irrational.
        #             else:
        #                 import ipdb; ipdb.set_trace()
        #                 # TODO: get multiple acts?
        #                 acts_not_taken = {nsrt_plan[0]}
        #             rational_acts_not_taken.append(acts_not_taken)
        #         optimal_ctgs.append(ctg)
        #     trajectory_optimal_ctgs.append(optimal_ctgs)
        #     trajectory_rational_acts_not_taken.append(rational_acts_not_taken)

        # import ipdb; ipdb.set_trace()


        # # Rerun NSRT learning on synthetic data.
        # utils.update_config({"sampler_learner": sampler_learner})
        # super()._learn_nsrts(trajectories, online_learning_cycle)
