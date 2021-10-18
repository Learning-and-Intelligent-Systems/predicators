"""A TAMP approach that learns operators and samplers.

In contrast to other approaches, this approach does not
attempt to learn new predicates or options.
"""

import pickle as pkl
from typing import Callable, Set, List
from gym.spaces import Box
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, Operator, ParameterizedOption, \
    State, Action, Predicate, Type, Task
from predicators.src.operator_learning import learn_operators_from_data
from predicators.src.settings import get_save_path


class OperatorLearningApproach(TAMPApproach):
    """A TAMP approach that learns operators and samplers.
    """
    def __init__(self, simulator: Callable[[State, Action], State],
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task]) -> None:
        super().__init__(simulator, initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._operators: Set[Operator] = set()

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_operators(self) -> Set[Operator]:
        assert self._operators, "Operators not learned"
        return self._operators

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # The only thing we need to do here is learn operators,
        # which we split off into a different function in case
        # subclasses want to make use of it.
        self._learn_operators(dataset)

    def _learn_operators(self, dataset: Dataset) -> None:
        self._operators = learn_operators_from_data(
            dataset, self._get_current_predicates())
        save_path = get_save_path()
        with open(f"{save_path}.operators", "wb") as f:
            pkl.dump(self._operators, f)

    def load(self) -> None:
        save_path = get_save_path()
        with open(f"{save_path}.operators", "rb") as f:
            self._operators = pkl.load(f)
        print("\n\nLoaded operators:")
        for op in self._operators:
            print(op)
        print()
