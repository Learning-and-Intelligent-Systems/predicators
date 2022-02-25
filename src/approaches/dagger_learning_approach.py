"""An approach that imlements DAgger for option learning."""

from typing import Set, List, Optional, Tuple, Callable, Sequence
import dill as pkl
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach, \
    ApproachTimeout, ApproachFailure, RandomOptionsApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Dataset, GroundAtom, LowLevelTrajectory, InteractionRequest, \
    InteractionResult, Action, GroundAtomsHoldQuery, GroundAtomsHoldResponse, \
    Query, DemonstrationQuery
from predicators.src.torch_models import LearnedPredicateClassifier, \
    MLPClassifier
from predicators.src.settings import CFG
from typing import Set, List, Sequence, Optional
import dill as pkl
from gym.spaces import Box
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, NSRT, ParameterizedOption, \
    Predicate, Type, Task, LowLevelTrajectory
from predicators.src.nsrt_learning.nsrt_learning_main import \
    learn_nsrts_from_data
from predicators.src.settings import CFG
from predicators.src import utils

class DaggerLearningApproach(NSRTLearningApproach):
    """An approach that implements DAgger for option learning."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self.online_learning_cycle = 0

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # save trajectories so that we can add more through DAgger
        self.dataset = dataset
        self._learn_nsrts(self.dataset.trajectories, online_learning_cycle=None)

    def get_interaction_requests(self) -> List[InteractionRequest]:
        requests = []

        def make_query_policy(train_task_idx: int) -> Callable[[State], Query]:
            def _query_policy(s: State) -> DemonstrationQuery:
                del s  # not used
                return DemonstrationQuery(train_task_idx)
            return _query_policy
        def make_termination_fn(goal: Set[GroundAtom]) -> Callable[[State], bool]:
            def _termination_fn(s: State) -> bool:
                return all(goal_atom.holds(s) for goal_atom in goal)
            return _termination_fn

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
                query_policy = make_query_policy(i),
                termination_function = make_termination_fn(task.goal)
            )
            requests.append(request)
        return requests

    def learn_from_interaction_results(self, results: Sequence[InteractionResult]) -> None:
        for result in results:
            actions = [a.unset_option() for a in result.teacher_traj.actions]
            traj = LowLevelTrajectory(result.teacher_traj.states,
                                      actions,
                                      is_demo=True,
                                      _train_task_idx=result.teacher_traj.train_task_idx)
            self.dataset.append(traj)
        self._learn_nsrts(self.dataset.trajectories, online_learning_cycle=self.online_learning_cycle)
        self.online_learning_cycle += 1
