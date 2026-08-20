"""Online process learning and planning approach."""
import logging
from typing import List, Optional, Sequence, Set

from gym.spaces import Box

from predicators.approaches.pp_process_learning_approach import \
    ProcessLearningAndPlanningApproach
from predicators.explorers import BaseExplorer, create_explorer
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import Dataset, InteractionRequest, \
    InteractionResult, LowLevelTrajectory, ParameterizedOption, Predicate, \
    Task, Type


class OnlineProcessLearningAndPlanningApproach(
        ProcessLearningAndPlanningApproach):
    """A bilevel planning approach that uses hand-specified processes."""

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 task_planning_heuristic: str = "default",
                 max_skeletons_optimized: int = -1,
                 bilevel_plan_without_sim: Optional[bool] = None,
                 option_model: Optional[_OptionModelBase] = None):
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)
        self._online_dataset = Dataset([])
        self._online_learning_cycle = 0
        self._requests_train_task_idxs: Optional[List[int]] = None

    @classmethod
    def get_name(cls) -> str:
        return "online_process_learning_and_planning"

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        """Learn models from the offline datasets."""
        if len(dataset.trajectories) > 0:
            self._learn_processes(
                dataset.trajectories,
                online_learning_cycle=None,
                annotations=(dataset.annotations
                             if dataset.has_annotations else None))
        else:
            logging.info("Offline dataset is empty, skipping learning.")

    def get_interaction_requests(self) -> List[InteractionRequest]:
        """Design experiments to collect data.

        Currently the same as OnlineNSRTLearning.
        We want to collect data to learn processes for solving, for now, the
        planning tasks.
        To achieve the goal, we want to learn the conditions and effects that
        allows for efficient and effective sequencing of actions and processes.

        There are various exploration strategies:
        1. as in VisualPredicator, make plans for solving the tasks and learn
        from the failure cases.
        2. try whether removing one of the conditions of the exogenous
        process would allow the process to succeed.
        """
        explorer = self._create_explorer()

        # As in OnlineNSRTLearningApproach, do some resets.
        self._last_nsrt_plan = []
        self._last_atoms_seq = []
        self._last_plan = []

        # Create the interaction requests.
        requests = []
        # Can also just use CFG.online_nsrt_learning_requests_per_cycle
        # for _ in range(CFG.online_nsrt_learning_number_of_tasks_to_try):
        #     # Select a random task (with replacement).
        #     task_idx = self._rng.choice(len(self._train_tasks))
        #     for i in range(CFG.online_nsrt_learning_requests_per_task):
        #         logging.info(f"Getting strategy {i} for task {task_idx}")
        #         # Set up the explorer policy and termination function.
        #         policy, termination_function = \
        #             explorer.get_exploration_strategy(
        #                 task_idx, CFG.timeout)
        #         # Create the interaction request.
        #         req = InteractionRequest(
        #             train_task_idx=task_idx,
        #             act_policy=policy,
        #             query_policy=lambda s: None,
        #             termination_function=termination_function)
        #         requests.append(req)
        self._requests_train_task_idxs = []
        for i in range(CFG.online_nsrt_learning_requests_per_cycle):
            task_idx = self._rng.choice(len(self._train_tasks))
            logging.info(f"Getting strategy {i}; this is for task {task_idx}")
            self._requests_train_task_idxs.append(task_idx)
            policy, termination_function = explorer.get_exploration_strategy(
                task_idx, CFG.timeout)
            req = InteractionRequest(train_task_idx=task_idx,
                                     act_policy=policy,
                                     query_policy=lambda s: None,
                                     termination_function=termination_function)
            requests.append(req)
        return requests

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        """Learn from interaction results.

        We will organize the interaction results as follows:
        1. interaction trajectories
        2. failed initial states for options? (might not work well with weak
        option termination classifiers.)
        Old:
        For endogenous process, initial states where it succeeded and failed.
        For exogenous process, suffixes of the trajectories where that atom
        changed.
        """
        # Future: update _dataset based on the results
        # Can potentially have a positive and negative dataset
        for result in results:
            traj = LowLevelTrajectory(result.states, result.actions)
            self._online_dataset.append(traj)

        # Learn from the dataset
        annotations = None
        if self._online_dataset.has_annotations:
            annotations = self._online_dataset.annotations  # pragma: no cover
        self._learn_processes(
            self._online_dataset.trajectories,
            online_learning_cycle=self._online_learning_cycle,
            annotations=annotations)

        if CFG.learn_process_parameters:
            self._learn_process_parameters(self._offline_dataset.trajectories+\
                                           self._online_dataset.trajectories)

        self._online_learning_cycle += 1

    def _create_explorer(self) -> BaseExplorer:
        """Create a new explorer at the beginning of each interaction cycle."""
        # Note that greedy lookahead is not yet supported.
        preds = self._get_current_predicates()
        explorer = create_explorer(
            CFG.explorer,
            preds,
            self._initial_options,
            self._types,
            self._action_space,
            self._train_tasks,
            self._get_current_processes(),  # type: ignore[arg-type]
            self._option_model,
        )
        return explorer
