"""An approach that learns predicates from a teacher.
"""

import itertools
from typing import Set, Callable, List, Collection
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import OperatorLearningApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, GroundAtom, ActionTrajectory
from predicators.src.models import MLPClassifier
from predicators.src.utils import get_object_combinations, strip_predicate
from predicators.src.settings import CFG


class InteractiveLearningApproach(OperatorLearningApproach):
    """An approach that learns predicates from a teacher.
    """
    def __init__(self, simulator: Callable[[State, Action], State],
                 all_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task]) -> None:
        # Only the teacher is allowed to know about the initial predicates
        self._known_predicates = {p for p in all_predicates
                                  if p.name in CFG.interactive_known_predicates}
        predicates_to_learn = all_predicates - self._known_predicates
        self._teacher = _Teacher(all_predicates, predicates_to_learn)
        # All seen data
        self._dataset: List[ActionTrajectory] = []
        self._ground_atom_dataset: List[List[Set[GroundAtom]]] = []
        # No cheating!
        self._predicates_to_learn = {strip_predicate(p)
                                     for p in predicates_to_learn}
        del all_predicates
        del predicates_to_learn
        super().__init__(simulator, self._predicates_to_learn, initial_options,
                         types, action_space, train_tasks)

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._known_predicates | self._predicates_to_learn

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Get data from teacher
        ground_atom_data = self._teacher.generate_data(dataset)
        self._dataset.extend(dataset)
        self._ground_atom_dataset.extend(ground_atom_data)
        # Learn predicates
        for pred in self._predicates_to_learn:
            assert pred not in self._known_predicates
            positive_examples = []
            negative_examples = []
            # Positive examples
            for i, trajectory in enumerate(self._ground_atom_dataset):
                for j, ground_atom_set in enumerate(trajectory):
                    state = self._dataset[i][0][j]
                    positives = [state.vec(ground_atom.objects)
                                 for ground_atom in ground_atom_set
                                 if ground_atom.predicate == pred]
                    positive_examples.extend(positives)
            # Negative examples - assume unlabeled is negative for now
            for (ss, _) in self._dataset:
                for state in ss:
                    possible = [state.vec(choice)
                                for choice in get_object_combinations(
                                                  list(state),
                                                  pred.types,
                                                  allow_duplicates=False)]
                    negatives = []
                    for (ex, pos) in itertools.product(possible,
                                                       positive_examples):
                        if np.array_equal(ex, pos):
                            break
                    else:
                        # It's not a positive example
                        negatives.append(ex)
                    negative_examples.extend(negatives)
            print(f"Generated {len(positive_examples)} positive and "
                  f"{len(negative_examples)} negative examples for "
                  f"predicate {pred}")

            # Train MLP
            X = np.array(positive_examples + negative_examples)
            Y = np.array([1 for _ in positive_examples] +
                         [0 for _ in negative_examples])
            model = MLPClassifier(X.shape[1])
            model.fit(X, Y)

            # Construct classifier function, create new Predicate, and save it
            classifier = utils.LearnedPredicateClassifier(model).classifier
            new_pred = Predicate(pred.name, pred.types, classifier,
                                 is_learned=True)
            self._predicates_to_learn = \
                (self._predicates_to_learn - {pred}) | {new_pred}

        # Learn operators via superclass
        self._learn_operators(dataset)

    def ask_teacher(self, state: State, ground_atom: GroundAtom) -> bool:
        """Returns whether the ground atom is true in the state.
        """
        return self._teacher.ask(state, ground_atom)


class _Teacher:
    """Answers queries about GroundAtoms in States.
    """
    def __init__(self, all_predicates: Set[Predicate],
                 predicates_to_learn: Set[Predicate]) -> None:
        self._name_to_predicate = {p.name : p for p in all_predicates}
        self._predicates_to_learn = predicates_to_learn

    def generate_data(self, dataset: Dataset) -> List[List[Set[GroundAtom]]]:
        """Creates sparse dataset of GroundAtoms.
        """
        return create_teacher_dataset(self._predicates_to_learn, dataset)

    def ask(self, state: State, ground_atom: GroundAtom) -> bool:
        """Returns whether the ground atom is true in the state.
        """
        # Find the predicate that has the classifier
        predicate = self._name_to_predicate[ground_atom.predicate.name]
        # Use the predicate's classifier
        return predicate.holds(state, ground_atom.objects)


def create_teacher_dataset(preds: Collection[Predicate],
                           dataset: Dataset) -> List[List[Set[GroundAtom]]]:
    """Create sparse dataset of GroundAtoms for interactive learning.
    """
    ratio = CFG.teacher_dataset_label_ratio
    rng = np.random.default_rng(CFG.seed)
    ground_atoms_dataset = []
    for (ss, _) in dataset:
        ground_atoms_traj = []
        for s in ss:
            ground_atoms = list(utils.abstract(s, preds))
            # select random subset to keep
            n_samples = int(len(ground_atoms) * ratio)
            subset = rng.choice(np.arange(len(ground_atoms)),
                                size=(n_samples,),
                                replace=False)
            subset_atoms = {ground_atoms[j] for j in subset}
            ground_atoms_traj.append(subset_atoms)
        ground_atoms_dataset.append(ground_atoms_traj)
    assert len(ground_atoms_dataset) == len(dataset)
    return ground_atoms_dataset
