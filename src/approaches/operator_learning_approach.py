"""A TAMP approach that learns operators and samplers.

In contrast to other approaches, this approach does not
attempt to learn new predicates or options.
"""

import pickle as pkl
from typing import Any, Set
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, Operator
from predicators.src.operator_learning import learn_operators_from_data, \
    load_sampler
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
        save_path = get_save_path()
        data = []
        for op in self._operators:
            data.append((op.name, op.parameters, op.preconditions,
                         op.add_effects, op.delete_effects, op.option.name))
        with open(f"{save_path}.operators", "wb") as f:
            pkl.dump(data, f)

    def load(self) -> None:
        save_path = get_save_path()
        with open(f"{save_path}.operators", "rb") as f:
            data = pkl.load(f)
        self._operators = set()
        for (operator_name, parameters, preconditions, add_effects,
             delete_effects, option_name) in data:
            # We'll assume the option is in the initial option set.
            # Otherwise, if it was learned, it would need to be saved.
            candidate_options = [opt for opt in self._initial_options
                                 if opt.name == option_name]
            assert len(candidate_options) == 1
            option = candidate_options[0]
            sampler = load_sampler(parameters, option, operator_name)
            self._operators.add(Operator(
                operator_name, parameters, preconditions, add_effects,
                delete_effects, option, sampler))
        print("\n\nLoaded operators:")
        for op in self._operators:
            print(op)
        print()
