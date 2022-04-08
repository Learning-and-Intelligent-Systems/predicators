"""A bilevel planning approach that learns NSRTs from an offline dataset, and
continues learning options through reinforcement learning.
"""

import logging
from typing import List, Optional, Set, Callable, Sequence

import dill as pkl
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.nsrt_learning.nsrt_learning_main import \
    learn_nsrts_from_data
from predicators.src.approaches.base_approach import ApproachTimeout, \
    ApproachFailure
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Dataset, LowLevelTrajectory, \
    ParameterizedOption, Predicate, Task, Type, GroundAtom, LowLevelTrajectory, InteractionRequest, \
    InteractionResult, Action, GroundAtomsHoldQuery, GroundAtomsHoldResponse, \
    Query, DemonstrationQuery, State
from predicators.src.envs.base_env import BaseEnv
from predicators.src.nsrt_learning.nsrt_learning_main import \
    learn_pruned_nsrts_from_data
from predicators.src.utils import create_ground_atom_dataset
from predicators.src.nsrt_learning.segmentation import segment_trajectory

class ReinforcementLearningApproach(NSRTLearningApproach):
    """A bilevel planning approach that learns NSRTs."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._nsrts: Set[NSRT] = set()
        self.online_learning_cycle = 0
        # initialize option learner?

    def _learn_pruned_nsrts(self,
                initial_trajectories: List[LowLevelTrajectory],
                trajectories: List[LowLevelTrajectory],
                online_learning_cycle: Optional[int]) -> None:
        self._nsrts = learn_pruned_nsrts_from_data(
            initial_trajectories,
            trajectories,
            self._train_tasks,
            self._get_current_predicates(),
            self._action_space,
            sampler_learner=CFG.sampler_learner)
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.NSRTs", "wb") as f:
            pkl.dump(self._nsrts, f)

    @classmethod
    def get_name(cls) -> str:
        return "reinforcement_learning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # The only thing we need to do here is learn NSRTs,
        # which we split off into a different function in case
        # subclasses want to make use of it.
        self.initial_dataset = dataset.trajectories
        self.online_dataset = []
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None)

    def _make_termination_fn(
            self, goal: Set[GroundAtom]) -> Callable[[State], bool]:
        def _termination_fn(s: State) -> bool:
            return all(goal_atom.holds(s) for goal_atom in goal)
        return _termination_fn

    def get_interaction_requests(self) -> List[InteractionRequest]:
        requests = []
        for i in range(len(self._train_tasks)):
            task = self._train_tasks[i]
            try:
                _act_policy = self.solve(task, CFG.timeout)
            except (ApproachTimeout, ApproachFailure) as e:
                partial_refinements = e.info.get("partial_refinements")
                _, plan = max(partial_refinements, key=lambda x: len(x[1]))
                _act_policy = utils.option_plan_to_policy(plan)
            request = InteractionRequest(
                train_task_idx = i,
                act_policy = _act_policy,
                query_policy = lambda s: None,
                termination_function = self._make_termination_fn(task.goal)
            )
            requests.append(request)
        return requests

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        self.online_learning_cycle += 1
        # We get one result per training task.
        for i, result in enumerate(results): 
            states = result.states
            actions = result.actions
            traj = LowLevelTrajectory(
                states,
                actions,
                _is_demo=False,
                _train_task_idx = i
            )
            self.online_dataset.append(traj)

        # Replace this with an _RLOptionLearner.
        self._learn_pruned_nsrts(
            self.initial_dataset, self.initial_dataset + self.online_dataset,
            online_learning_cycle=self.online_learning_cycle
        )
