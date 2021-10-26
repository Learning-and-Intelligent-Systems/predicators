"""An approach that iteratively invents predicates.
"""

import itertools
from dataclasses import dataclass
from typing import Set, Callable, List, Collection, Sequence
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import OperatorLearningApproach, ApproachFailure
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, GroundAtom, ActionTrajectory, Object
from predicators.src.models import MLPClassifier
from predicators.src.utils import get_object_combinations, strip_predicate
from predicators.src.operator_learning import learn_operators_from_data
from predicators.src.settings import CFG


class IterativeInventionApproach(OperatorLearningApproach):
    """An approach that iteratively invents predicates.
    """
    def __init__(self, simulator: Callable[[State, Action], State],
                 all_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task]) -> None:
        super().__init__(simulator, all_predicates, initial_options,
                         types, action_space, train_tasks)
        self._learned_predicates: Set[Predicate] = set()

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # pylint:disable-all

        # Learn operators with the predicates we have
        ops = learn_operators_from_data(
            dataset, self._get_current_predicates(), do_sampler_learning=False)

        # Invent new predicates
        # TODO: invention & remove pylint disable

        # Learn operators via superclass
        self._learn_operators(dataset)
