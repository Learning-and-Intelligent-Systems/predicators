"""A bilevel planning approach that learns NSRTs from an offline dataset, and
continues learning through online interaction."""

from typing import Callable, List, Sequence, Set, Tuple

from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.base_approach import ApproachFailure, \
    ApproachTimeout
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.approaches.random_options_approach import \
    RandomOptionsApproach
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Action, Dataset, \
    InteractionRequest, InteractionResult, LowLevelTrajectory, \
    ParameterizedOption, Predicate, State, Task, Type


class PretendPlayLearningApproach(NSRTLearningApproach):
    """A bilevel planning approach that learns NSRTs from an offline dataset,
    and continues learning through online interaction."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._nsrts: Set[NSRT] = set()
        self._online_learning_cycle = 0
        self._initial_trajectories: List[LowLevelTrajectory] = []
        self._online_trajectories: List[LowLevelTrajectory] = []

    @classmethod
    def get_name(cls) -> str:
        return "pretend_play"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._initial_trajectories = dataset.trajectories
        super().learn_from_offline_dataset(dataset)

    def get_interaction_requests(self) -> List[InteractionRequest]:
        requests = []
        for i in self._select_interaction_train_task_idxs():
            act_policy, termination_function = self._create_action_strategy(i)
            request = InteractionRequest(
                train_task_idx=i,
                act_policy=act_policy,
                query_policy=lambda s: None,
                termination_function=termination_function)
            requests.append(request)
        return requests

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        self._online_learning_cycle += 1
        # We get one result per training task.
        for i, result in enumerate(results):
            states = result.states
            actions = result.actions
            traj = LowLevelTrajectory(states, actions)
            self._online_trajectories.append(traj)

        # Replace this with an _RLOptionLearner.
        self._learn_nsrts(self._initial_trajectories + \
            self._online_trajectories,
            online_learning_cycle=self._online_learning_cycle
        )

    def _select_interaction_train_task_idxs(self) -> List[int]:
        # At the moment, we select train task indices uniformly at
        # random, with replacement. In the future, we may want to
        # try other strategies.
        interaction_idxs = range(CFG.max_initial_demos, len(self._train_tasks))
        return self._rng.choice(interaction_idxs,
                                size=CFG.interactive_num_requests_per_cycle)

    def _create_action_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Returns an action policy and a termination function."""
        if CFG.interactive_action_strategy == "random":
            return self._create_random_interaction_strategy(train_task_idx)
        raise NotImplementedError("Unrecognized interactive_action_strategy:"
                                  f" {CFG.interactive_action_strategy}")

    def _create_random_interaction_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Sample and execute random initiable options until timeout."""

        # TODO make current options
        random_options_approach = RandomOptionsApproach(
            self._get_current_predicates(), self._initial_options, self._types,
            self._action_space, self._train_tasks)
        task = self._train_tasks[train_task_idx]
        act_policy = random_options_approach.solve(task, CFG.timeout)

        # Termination is left to the environment, as in
        # CFG.max_num_steps_interaction_request.
        termination_function = lambda _: False
        return act_policy, termination_function
