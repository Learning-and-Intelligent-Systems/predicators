"""A TAMP approach that learns operators and samplers.

In contrast to other approaches, this approach does not
attempt to learn new predicates or options.
"""

from typing import Any, Set
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, Operator
from predicators.src.operator_learning import learn_operators_from_data
from predicators.src.settings import get_save_path


class OperatorLearningApproach(TAMPApproach):
    """A TAMP approach that learns operators and samplers.
    """
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._operators: Set[Operator] = set()

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_operators(self) -> Set[Operator]:
        assert self._operators, "Operators not learned"
        return self._operators

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        self._operators = learn_operators_from_data(
            dataset, self._initial_predicates)

    def save(self) -> None:
        save_path = get_save_path()
        raise NotImplementedError  # TODO: save self._operators

    def load(self) -> None:
        load_path = get_save_path()
        raise NotImplementedError  # TODO: load into self._operators
