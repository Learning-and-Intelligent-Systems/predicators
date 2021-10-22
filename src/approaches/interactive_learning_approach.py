"""An approach that learns predicates from a teacher.
"""

import itertools
from dataclasses import dataclass
from typing import Set, Callable, List, Collection, Sequence, Dict
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import OperatorLearningApproach, \
    ApproachFailure
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, GroundAtom, ActionTrajectory, Object
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
        # Learn predicates and operators
        self.semi_supervised_learning(self._dataset)
        # Active learning
        for i in range(1, CFG.active_num_episodes+1):
            print(f"Active learning episode {i}")
            # Sample starting state from train tasks
            index = self._rng.choice(len(self._train_tasks))
            state = self._train_tasks[index].init
            # Policy for exploration
            task = glib_sample(state, self._get_current_predicates(),
                               self._ground_atom_dataset)
            policy = self.solve(task, timeout=CFG.timeout)
            print("Policy found! Collecting exploration data...")
            states = []
            actions = []
            for _ in range(CFG.active_max_steps):
                action = policy(state)
                state = self._simulator(state, action)
                states.append(state)
                actions.append(action)
            action_traj: ActionTrajectory = (states, actions)
            # Update datasets
            ground_atom_data = self._teacher.generate_data([action_traj])
            self._dataset.extend([action_traj])
            self._ground_atom_dataset.extend(ground_atom_data)
            if i % CFG.active_learning_relearn_every == 0:
                # Pick a state from the new states explored
                states_to_scores = {s: score_goal(
                                        self._ground_atom_dataset,
                                        utils.abstract(s,
                                            self._get_current_predicates()))
                                    for s in states}
                for s in self.get_states_to_ask(states_to_scores):
                    # For now, pick a random ground atom to ask about
                    ground_atoms = utils.all_ground_atoms(
                                            s, self._get_current_predicates())
                    idx = self._rng.choice(len(ground_atoms))
                    self.ask_teacher(s, ground_atoms[idx])
                # Relearn predicates and operators
                self.semi_supervised_learning(self._dataset)


    def semi_supervised_learning(self, dataset: Dataset) -> None:
        """Learns predicates and operators in a semi-supervised fashion.
        """
        print("Starting semi-supervised learning...")
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
            classifier = _LearnedPredicateClassifier(model).classifier
            new_pred = Predicate(pred.name, pred.types, classifier)
            self._predicates_to_learn = \
                (self._predicates_to_learn - {pred}) | {new_pred}

        # Learn operators via superclass
        self._learn_operators(dataset)


    def get_states_to_ask(self,
                        #   goal_state: Set[GroundAtom],
                          states_to_scores: Dict[State, float]) -> Set[State]:
        """Gets set of states to ask about, according to ask_strategy.
        """
        # if CFG.ask_strategy == "goal_state_only":
            # return {goal_state}
        if CFG.ask_strategy == "all_seen_states":
            return set(states_to_scores.keys())
        elif CFG.ask_strategy == "threshold":
            assert isinstance(CFG.ask_strategy_threshold, float)
            return {s for (s, score) in states_to_scores.items()
                    if score >= CFG.ask_strategy_threshold}
        else:
            raise NotImplementedError(f"Ask strategy {CFG.ask_strategy} "
                                      "not supported")


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
            ground_atoms = sorted(utils.abstract(s, preds))
            # select random subset to keep
            n_samples = int(len(ground_atoms) * ratio)
            if n_samples < 1:
                raise ApproachFailure("Need at least 1 ground atom sample")
            subset = rng.choice(np.arange(len(ground_atoms)),
                                size=(n_samples,),
                                replace=False)
            subset_atoms = {ground_atoms[j] for j in subset}
            ground_atoms_traj.append(subset_atoms)
        ground_atoms_dataset.append(ground_atoms_traj)
    assert len(ground_atoms_dataset) == len(dataset)
    return ground_atoms_dataset


def glib_sample(initial_state: State,
                   predicates: Set[Predicate],
                   ground_atom_dataset: List[List[Set[GroundAtom]]]) -> Task:
    """Sample a task via the GLIB approach.
    """
    print("Sampling a task using GLIB approach...")
    assert CFG.atom_type_babbled == "ground"
    rng = np.random.default_rng(CFG.seed)
    ground_atoms = utils.all_ground_atoms(initial_state, predicates)
    best_score = 0.0
    best_goal: Set[GroundAtom] = set()
    for _ in range(CFG.active_num_babbles):
        # Sample num atoms to babble
        num_atoms = 1 + rng.choice(CFG.max_num_atoms_babbled)
        # Sample goal (a set of atoms)
        idxs = rng.choice(np.arange(len(ground_atoms)),
                          size=(num_atoms,),
                          replace=False)
        goal = {ground_atoms[i] for i in idxs}
        # Score and remember best goal
        score = score_goal(ground_atom_dataset, goal)
        if score > best_score:
            best_score = score
            best_goal = goal
    return Task(initial_state, best_goal)


def score_goal(ground_atom_dataset: List[List[Set[GroundAtom]]],
               goal: Set[GroundAtom]) -> float:
    """Score a goal as inversely proportional to the number of examples seen
    during training.
    """
    count = 1  # Avoid division by 0
    for trajectory in ground_atom_dataset:
        for ground_atom_set in trajectory:
            count += 1 if goal.issubset(ground_atom_set) else 0
    return 1.0 / count


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedPredicateClassifier:
    """A convenience class for holding the model underlying a learned predicate.
    Prefer to use this because it is pickleable.
    """
    _model: MLPClassifier

    def classifier(self, state: State, objects: Sequence[Object]) -> bool:
        """The classifier corresponding to the given model. May be used
        as the _classifier field in a Predicate.
        """
        v = state.vec(objects)
        return self._model.classify(v)
