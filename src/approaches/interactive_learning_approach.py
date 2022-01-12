"""An approach that learns predicates from a teacher."""

from typing import Set, Callable, List, Collection
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, GroundAtom, GroundAtomTrajectory, \
    LowLevelTrajectory
from predicators.src.torch_models import LearnedPredicateClassifier, \
    MLPClassifier
from predicators.src.utils import get_object_combinations, strip_predicate
from predicators.src.settings import CFG


class InteractiveLearningApproach(NSRTLearningApproach):
    """An approach that learns predicates from a teacher."""

    def __init__(self, simulator: Callable[[State, Action], State],
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box) -> None:
        # Predicates should not be ablated
        assert not CFG.excluded_predicates
        # Only the teacher is allowed to know about the initial predicates
        self._known_predicates = {
            p
            for p in initial_predicates
            if p.name in CFG.interactive_known_predicates
        }
        predicates_to_learn = initial_predicates - self._known_predicates
        self._teacher = _Teacher(initial_predicates, predicates_to_learn)
        # All seen data
        self._dataset_with_atoms: List[GroundAtomTrajectory] = []
        # No cheating!
        self._predicates_to_learn = {
            strip_predicate(p)
            for p in predicates_to_learn
        }
        del initial_predicates
        del predicates_to_learn
        super().__init__(simulator, self._predicates_to_learn, initial_options,
                         types, action_space)

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._known_predicates | self._predicates_to_learn

    def _load_dataset(self, dataset: Dataset) -> None:
        """Stores dataset and corresponding ground atom dataset."""
        self._dataset.extend(dataset)
        self._dataset_with_atoms.extend(self._teacher.generate_data(dataset))

    def learn_from_offline_dataset(self, dataset: Dataset,
                                   train_tasks: List[Task]) -> None:
        self._load_dataset(dataset)
        del dataset
        # Learn predicates and NSRTs
        self._relearn_predicates_and_nsrts()
        # Track score of best state seen so far
        best_score = 0.0
        # Active learning
        for i in range(1, CFG.interactive_num_episodes + 1):
            print(f"\nActive learning episode {i}")
            # Sample initial state from train tasks
            index = self._rng.choice(len(train_tasks))
            s = train_tasks[index].init
            # Find policy for exploration
            task_list = glib_sample(s, self._get_current_predicates(),
                                    self._dataset_with_atoms)
            assert task_list
            task = task_list[0]
            for task in task_list:
                try:
                    print("Solving for policy...")
                    policy = self.solve(task, timeout=CFG.timeout)
                    break
                except (ApproachTimeout, ApproachFailure) \
                        as e:  # pragma: no cover
                    print(f"Approach failed to solve with error: {e}")
                    continue
            else:  # No policy found
                raise ApproachFailure("Failed to sample a task that approach "
                                      "can solve.")  # pragma: no cover
            # Roll out policy
            traj, _, _ = utils.run_policy_on_task(
                policy,
                task,
                self._simulator,
                self._get_current_predicates(),
                max_steps=CFG.interactive_max_steps)
            # Decide whether to ask at each state during exploration
            for s in traj.states:
                abstract_state = utils.abstract(s,
                                                self._get_current_predicates())
                score = score_goal(self._dataset_with_atoms, abstract_state)
                # Ask about this state if it is the best seen so far
                if score > best_score:
                    # Pick best ground atom in the state
                    ground_atoms = utils.all_possible_ground_atoms(
                        s, self._predicates_to_learn)
                    atom_set = get_best_goal_by_score(self._dataset_with_atoms,
                                                      [{ga}
                                                       for ga in ground_atoms])
                    assert len(atom_set) == 1
                    atom = atom_set.pop()
                    if self._ask_teacher(s, atom):
                        # Add this atom if it's a positive example
                        self._dataset_with_atoms.append(  # pragma: no cover
                            (LowLevelTrajectory([s], []), [{atom}]))
                        # Still need to implement a way to use negative examples
                    best_score = score
            if i % CFG.interactive_relearn_every == 0:
                self._relearn_predicates_and_nsrts()

    def _relearn_predicates_and_nsrts(self) -> None:
        """Learns predicates and NSRTs in a semi-supervised fashion."""
        print("\nStarting semi-supervised learning...")
        # Learn predicates
        for pred in self._predicates_to_learn:
            assert pred not in self._known_predicates
            positive_examples = []
            negative_examples = []
            # Positive examples
            for (traj, ground_atom_sets) in self._dataset_with_atoms:
                assert len(traj.states) == len(ground_atom_sets)
                for (state, ground_atom_set) in zip(traj.states,
                                                    ground_atom_sets):
                    if len(ground_atom_set) == 0:
                        continue
                    positives = [
                        state.vec(ground_atom.objects)
                        for ground_atom in ground_atom_set
                        if ground_atom.predicate == pred
                    ]
                    positive_examples.extend(positives)
            # Negative examples - assume unlabeled is negative for now
            for (traj, _) in self._dataset_with_atoms:
                for state in traj.states:
                    possible = [
                        state.vec(choice)
                        for choice in get_object_combinations(
                            list(state), pred.types)
                    ]
                    negatives = []
                    for ex in possible:
                        for pos in positive_examples:
                            if np.array_equal(ex, pos):
                                break
                        else:  # It's not a positive example
                            negatives.append(ex)
                    negative_examples.extend(negatives)
            print(f"Generated {len(positive_examples)} positive and "
                  f"{len(negative_examples)} negative examples for "
                  f"predicate {pred}")

            # Train MLP
            X = np.array(positive_examples + negative_examples)
            Y = np.array([1 for _ in positive_examples] +
                         [0 for _ in negative_examples])
            model = MLPClassifier(X.shape[1],
                                  CFG.predicate_mlp_classifier_max_itr)
            model.fit(X, Y)

            # Construct classifier function, create new Predicate, and save it
            classifier = LearnedPredicateClassifier(model).classifier
            new_pred = Predicate(pred.name, pred.types, classifier)
            self._predicates_to_learn = \
                (self._predicates_to_learn - {pred}) | {new_pred}

        # Learn NSRTs via superclass
        self._learn_nsrts()

    def _ask_teacher(self, state: State, ground_atom: GroundAtom) -> bool:
        """Returns whether the ground atom is true in the state."""
        return self._teacher.ask(state, ground_atom)


class _Teacher:
    """Answers queries about GroundAtoms in States."""

    def __init__(self, initial_predicates: Set[Predicate],
                 predicates_to_learn: Set[Predicate]) -> None:
        self._name_to_predicate = {p.name: p for p in initial_predicates}
        self._predicates_to_learn = predicates_to_learn
        self._has_generated_data = False

    def generate_data(self, dataset: Dataset) -> List[GroundAtomTrajectory]:
        """Creates sparse dataset of GroundAtoms."""
        # No cheating!
        assert not self._has_generated_data
        self._has_generated_data = True
        return create_teacher_dataset(self._predicates_to_learn, dataset)

    def ask(self, state: State, ground_atom: GroundAtom) -> bool:
        """Returns whether the ground atom is true in the state."""
        # Find the predicate that has the classifier
        predicate = self._name_to_predicate[ground_atom.predicate.name]
        # Use the predicate's classifier
        return predicate.holds(state, ground_atom.objects)


def create_teacher_dataset(preds: Collection[Predicate],
                           dataset: Dataset) -> List[GroundAtomTrajectory]:
    """Create sparse dataset of GroundAtoms for interactive learning."""
    ratio = CFG.teacher_dataset_label_ratio
    rng = np.random.default_rng(CFG.seed)
    teacher_dataset: List[GroundAtomTrajectory] = []
    for traj in dataset:
        ground_atoms_traj: List[Set[GroundAtom]] = []
        for s in traj.states:
            ground_atoms = sorted(utils.abstract(s, preds))
            # select random subset to keep
            n_samples = int(len(ground_atoms) * ratio)
            subset_atoms = set()
            if n_samples > 0:
                subset = rng.choice(np.arange(len(ground_atoms)),
                                    size=(n_samples, ),
                                    replace=False)
                subset_atoms = {ground_atoms[j] for j in subset}
            ground_atoms_traj.append(subset_atoms)
        assert len(traj.states) == len(ground_atoms_traj)
        teacher_dataset.append((traj, ground_atoms_traj))
    assert len(teacher_dataset) == len(dataset)
    return teacher_dataset


def glib_sample(
    initial_state: State,
    predicates: Set[Predicate],
    dataset_with_atoms: List[GroundAtomTrajectory],
) -> List[Task]:
    """Sample some tasks via the GLIB approach."""
    print("Sampling a task using GLIB approach...")
    assert CFG.interactive_atom_type_babbled == "ground"
    rng = np.random.default_rng(CFG.seed)
    ground_atoms = utils.all_possible_ground_atoms(initial_state, predicates)
    goals = []  # list of (goal, score) tuples
    for _ in range(CFG.interactive_num_babbles):
        # Sample num atoms to babble
        num_atoms = 1 + rng.choice(CFG.interactive_max_num_atoms_babbled)
        # Sample goal (a set of atoms)
        idxs = rng.choice(np.arange(len(ground_atoms)),
                          size=(num_atoms, ),
                          replace=False)
        goal = {ground_atoms[i] for i in idxs}
        goals.append((goal, score_goal(dataset_with_atoms, goal)))
    goals.sort(key=lambda tup: tup[1], reverse=True)
    return [Task(initial_state, g) for (g, _) in \
            goals[:CFG.interactive_num_tasks_babbled]]


def score_goal(dataset_with_atoms: List[GroundAtomTrajectory],
               goal: Set[GroundAtom]) -> float:
    """Score a goal as inversely proportional to the number of examples seen
    during training."""
    count = 1  # Avoid division by 0
    for (_, trajectory) in dataset_with_atoms:
        for ground_atom_set in trajectory:
            count += 1 if goal.issubset(ground_atom_set) else 0
    return 1.0 / count


def get_best_goal_by_score(dataset_with_atoms: List[GroundAtomTrajectory],
                           goals: List[Set[GroundAtom]]) -> Set[GroundAtom]:
    """Return the best goal out of `goals` using the score function."""
    scores = [score_goal(dataset_with_atoms, g) for g in goals]
    goals_and_scores = list(zip(goals, scores))
    goals_and_scores.sort(key=lambda tup: tup[1], reverse=True)
    return goals_and_scores[0][0]
