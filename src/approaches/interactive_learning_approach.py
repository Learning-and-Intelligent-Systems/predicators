"""An approach that learns predicates from a teacher.
"""

from gym.spaces import Box
from typing import Set, Callable, List
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Operator, Dataset, GroundAtom
from predicators.src.operator_learning import learn_operators_from_data


class InteractiveLearningApproach(TAMPApproach):
    """An approach that learns predicates from a teacher.
    """
    def __init__(self,
                 simulator: Callable[[State, Action], State],
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task]) -> None:
        # Only the teacher is allowed to know about the initial predicates
        self._teacher = _Teacher(initial_predicates)
        # No cheating!
        self._predicates = {_strip_predicate(p) for p in initial_predicates}
        del initial_predicates
        super().__init__(simulator, self._predicates, initial_options,
                         types, action_space, train_tasks)
        self._operators: Set[Operator] = set()
        # All seen data
        # TODO: store Dataset and corresponding ground atom dataset

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_operators(self) -> Set[Operator]:
        assert self._operators, "Operators not learned"
        return self._operators

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._predicates

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, create semi-supervised dataset
        ground_atoms_data = create_teacher_dataset(dataset)
        # TODO: add to dataset
        # TODO: learn predicates
        self._operators = learn_operators_from_data(
            dataset, self._get_current_predicates())


class _Teacher:
    """Answers queries about GroundAtoms in States.
    """
    def __init__(self, predicates: Set[Predicate]) -> None:
        self._name_to_predicate = {p.name : p for p in predicates}

    def ask(self, state: State, ground_atom: GroundAtom) -> bool:
        """Returns whether the ground atom is true in the state.
        """
        # Find the predicate that has the classifier
        predicate = self._name_to_predicate[ground_atom.predicate.name]
        # Use the predicate's classifier
        return predicate.holds(state, ground_atom.objects)


def _strip_predicate(predicate: Predicate) -> Predicate:
    """Remove classifier from predicate to make new Predicate.
    """
    return Predicate(predicate.name, predicate.types, lambda s, o: False)
