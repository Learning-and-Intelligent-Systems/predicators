"""Predicate invention approach for process planning."""
from typing import List, Optional, Set

from gym.spaces import Box

from predicators.approaches.pp_process_learning_approach import \
    ProcessLearningAndPlanningApproach
from predicators.option_model import _OptionModelBase
from predicators.settings import CFG
from predicators.structs import Dataset, ParameterizedOption, Predicate, \
    Task, Type


class PredicateInventionProcessPlanningApproach(
        ProcessLearningAndPlanningApproach):
    """A bilevel planning approach that invent predicates."""

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
        self._learned_predicates: Set[Predicate] = set()
        super().__init__(initial_predicates,
                         initial_options,
                         types,
                         action_space,
                         train_tasks,
                         task_planning_heuristic,
                         max_skeletons_optimized,
                         bilevel_plan_without_sim,
                         option_model=option_model)

    @classmethod
    def get_name(cls) -> str:
        return "predicate_invention_and_process_planning"

    def _get_current_predicates(self) -> Set[Predicate]:
        """Get the current predicates."""
        return self._initial_predicates | self._learned_predicates

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._offline_dataset = dataset
        # --- Invent Predicates ---
        # Check the atomic trajectory

        # ----- Predicate Proposal -----

        # ----- Predicate Selection -----

        # --- Learn Processes ---
        self._learn_processes(dataset.trajectories,
                              online_learning_cycle=None,
                              annotations=(dataset.annotations if
                                           dataset.has_annotations else None))
        if CFG.learn_process_parameters:
            self._learn_process_parameters(dataset.trajectories)
