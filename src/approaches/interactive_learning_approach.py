"""An approach that learns predicates from a teacher.
"""

import numpy as np
import torch
from gym.spaces import Box
from typing import Set, Callable, List, Sequence, Tuple, Collection

from predicators.src import utils
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Operator, Dataset, GroundAtom, ActionTrajectory, \
    Object
from predicators.src.operator_learning import learn_operators_from_data
from predicators.src.models import MLPClassifier
from predicators.src.utils import get_object_combinations
from predicators.src.settings import CFG, get_save_path


class InteractiveLearningApproach(TAMPApproach):
    """An approach that learns predicates from a teacher.
    """
    def __init__(self,
                 simulator: Callable[[State, Action], State],
                 true_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task]) -> None:
        # Only the teacher is allowed to know about the initial predicates
        self._known_predicates = {p for p in true_predicates
                                  if p.name in CFG.interactive_known_predicates}
        predicates_to_learn = true_predicates - self._known_predicates
        self._teacher = _Teacher(predicates_to_learn)
        # All seen data
        self.dataset: List[ActionTrajectory] = []
        self.ground_atom_dataset: List[List[Set[GroundAtom]]] = []
        # No cheating!
        self._predicates = {_strip_predicate(p) for p in predicates_to_learn}
        del true_predicates
        del predicates_to_learn
        super().__init__(simulator, self._predicates, initial_options,
                         types, action_space, train_tasks)
        self._operators: Set[Operator] = set()

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_operators(self) -> Set[Operator]:
        assert self._operators, "Operators not learned"
        return self._operators

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._predicates

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Get data from teacher
        data, ground_atom_data = self._teacher.generate_data_for_student(dataset)
        self.dataset.extend(data)
        self.ground_atom_dataset.extend(ground_atom_data)
        # Learn predicates
        for pred in self._predicates:
            print("Predicate:", pred)
            if pred in self._known_predicates:
                continue

            positive_examples = []
            negative_examples = []
            # Positive examples
            for i, trajectory in enumerate(self.ground_atom_dataset):
                for j, set in enumerate(trajectory):
                    state = self.dataset[i][0][j]
                    positives = [state.vec(ground_atom.objects)
                                 for ground_atom in set
                                 if ground_atom.predicate == pred]
                    positive_examples.extend(positives)
            # Negative examples - assume unlabeled is negative for now
            for (ss, _) in self.dataset:
                for state in ss:
                    possible = [state.vec(choice)
                                for choice in get_object_combinations(
                                                  list(state),
                                                  pred.types,
                                                  allow_duplicates=False)]
                    # TODO: this is ugly
                    # negatives = [ex for ex in possible
                    #              if ex not in positive_examples]
                    negatives = []
                    for ex in possible:
                        found = False
                        for pos in positive_examples:
                            if np.array_equal(ex, pos):
                                found = True
                                break
                        if not found:
                            negatives.append(ex)
                    negative_examples.extend(negatives)
            print(f"Generated {len(positive_examples)} positive and {len(negative_examples)} "
                  f"negative examples")
            save_path = get_save_path()

            # Train MLP
            print("pos example shapes:", [ex.shape for ex in positive_examples])
            print("neg example shapes:", [ex.shape for ex in negative_examples])

            X = np.array(positive_examples + negative_examples)
            print("X:", X.shape)
            Y = np.array([1 for _ in positive_examples] +
                         [0 for _ in negative_examples])
            print("Y:", Y.shape)

            model = MLPClassifier(X.shape[1])
            model.fit(X, Y)
            torch.save(model, f"{save_path}_{pred}.classifier")

            # Construct classifier function, create new Predicate, and save it
            def _classifier(state: State, objects: Sequence[Object]) -> bool:
                v = state.vec(objects)
                return model.classify(v)
            new_pred = Predicate(pred.name, pred.types, _classifier)
            self._predicates = self._predicates - {pred} | {new_pred}

        # Learn operators
        print("Learning operators...")
        self._operators = learn_operators_from_data(
            self.dataset, self._known_predicates | self._get_current_predicates())


class _Teacher:
    """Answers queries about GroundAtoms in States.
    """
    def __init__(self, predicates_to_learn: Set[Predicate]) -> None:
        self._name_to_predicate = {p.name : p for p in predicates_to_learn}

    def generate_data_for_student(self,
                                  dataset: Dataset) -> Tuple[
                                      Dataset, List[List[Set[GroundAtom]]]]:
        ground_atom_data = create_teacher_dataset(
                               self._name_to_predicate.values(),
                               dataset)
        return (dataset, ground_atom_data)

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


def _strip_predicate(predicate: Predicate) -> Predicate:
    """Remove classifier from predicate to make new Predicate.
    """
    return Predicate(predicate.name, predicate.types, lambda s, o: False)
