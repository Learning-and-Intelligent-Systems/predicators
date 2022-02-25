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
    Query
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
        # self.task_info_to_policy = {}

    # def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
    #     policy = super()._solve(task, timeout)
    #     task_initial_state = frozenset({k: tuple(v) for k, v in task.init.data.items()})
    #     task_goal = frozenset(task.goal)
    #     self.task_info_to_policy[(task_initial_state, task_goal)] = policy
    #     return policy

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # def _dummy_query_fn(s: State) -> None:
        #     del s
        #     return None
        # def make_termination_fn(policy: Callable[[State], Action]):
        #     def _termination_fn(s: State) -> bool:
        #         try:
        #             a = policy(s)
        #             return False
        #         except utils.OptionPlanExhausted:
        #             return True
        #     return _termination_fn

        # random_options_approach = RandomOptionsApproach(
        #     self._get_current_predicates(), self._initial_options, self._types,
        #     self._action_space, self._train_tasks)
        # def _termination_fn(s: State) -> bool:
        #     # Termination is left to the environment, as in
        #     # CFG.max_num_steps_interaction_request.
        #     del s  # not used
        #     return False
        requests = []
        def _query_fn(s: State) -> Optional[DemonstrationQuery]:
            del s  # not used
            return DemonstrationQuery()
        def make_termination_fn(goal: Set[GroundAtom]) -> Callable[[State], bool]:
            def _termination_fn(s: State) -> bool:
                return all(goal_atom.holds(s) for goal_atom in task.goal)
            return _termination_fn
        for i in range(len(self._train_tasks)):
            task = self._train_tasks[i]
            # task_initial_state = frozenset({k: tuple(v) for k, v in task.init.data.items()})
            # task_goal = frozenset(task.goal)
            # act_policy = self.task_info_to_policy[(task_initial_state, task_goal)]
            act_policy = self.solve(task, CFG.timeout)
            # act_policy = random_options_approach.solve(task, CFG.timeout)
            query_policy = _dummy_query_fn
            _termination_fn = make_termination_fn(task.goal)
            # _termination_fn = make_termination_fn(act_policy)
            request = InteractionRequest(
                i,
                act_policy,
                query_policy,
                _termination_fn
            )
            requests.append(request)
            break
        return requests

    def learn_from_interaction_results(self, results: Sequence[InteractionResult]) -> None:
        pass
