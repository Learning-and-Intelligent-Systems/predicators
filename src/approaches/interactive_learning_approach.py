"""An approach that learns predicates from a teacher."""

from typing import Set, List, Optional
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Dataset, GroundAtom, LowLevelTrajectory
from predicators.src.torch_models import LearnedPredicateClassifier, \
    MLPClassifier
from predicators.src.option_model import _OracleOptionModel
from predicators.src.utils import get_object_combinations, strip_predicate
from predicators.src.settings import CFG


class InteractiveLearningApproach(NSRTLearningApproach):
    """An approach that learns predicates from a teacher."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        # Predicates should not be ablated
        assert not CFG.excluded_predicates
        # Only the teacher is allowed to know about the initial predicates
        known_predicates = set(CFG.interactive_known_predicates.split(","))
        self._known_predicates = {
            p
            for p in initial_predicates if p.name in known_predicates
        }
        predicates_to_learn = initial_predicates - self._known_predicates
        self._teacher = _Teacher(initial_predicates, predicates_to_learn)
        # No cheating!
        self._predicates_to_learn = {
            strip_predicate(p)
            for p in predicates_to_learn
        }
        del initial_predicates
        del predicates_to_learn
        super().__init__(self._predicates_to_learn, initial_options, types,
                         action_space, train_tasks)

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._known_predicates | self._predicates_to_learn

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Learn predicates and NSRTs
        self._relearn_predicates_and_nsrts(dataset, online_learning_cycle=None)
        # Track score of best atom seen so far
        best_score = 0.0
        # Active learning
        for i in range(1, CFG.interactive_num_episodes + 1):
            print(f"\nActive learning episode {i}")
            # Sample initial state from train tasks
            index = self._rng.choice(len(self._train_tasks))
            state = dataset.trajectories[index].states[0]
            # Detect and filter out static predicates
            static_preds = utils.get_static_preds(
                self._nsrts, self._get_current_predicates())
            preds = self._get_current_predicates() - static_preds
            # Find policy for exploration
            task_list = glib_sample(state, preds, dataset)
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
            # Roll out policy via a temporary hack
            assert isinstance(self._option_model, _OracleOptionModel)
            simulator = self._option_model._simulator  # pylint:disable=protected-access
            traj, _, _ = utils.run_policy_on_task(
                policy,
                task,
                simulator,
                max_num_steps=CFG.interactive_max_num_steps)
            # Decide whether to ask about each possible atom during exploration
            for s in traj.states:
                ground_atoms = utils.all_possible_ground_atoms(
                    s, self._predicates_to_learn)
                for atom in ground_atoms:
                    # Note: future score functions will use the state s
                    score = score_atom(dataset, atom)
                    # Ask about this atom if it is the best seen so far
                    if score > best_score:
                        if self._ask_teacher(s, atom):
                            # Add this atom if it's a positive example
                            traj = LowLevelTrajectory([s], [])
                            dataset.append(traj, [{atom}])
                            # Still need a way to use negative examples
                        best_score = score
            if i % CFG.interactive_relearn_every == 0:
                self._relearn_predicates_and_nsrts(dataset,
                                                   online_learning_cycle=i - 1)

    def _relearn_predicates_and_nsrts(
            self, dataset: Dataset,
            online_learning_cycle: Optional[int]) -> None:
        """Learns predicates and NSRTs in a semi-supervised fashion."""
        print("\nStarting semi-supervised learning...")
        # Learn predicates
        for pred in self._predicates_to_learn:
            assert pred not in self._known_predicates
            positive_examples = []
            negative_examples = []
            # Positive examples
            for (traj, ground_atom_sets) in zip(dataset.trajectories,
                                                dataset.annotations):
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
            for traj in dataset.trajectories:
                for state in traj.states:
                    possible = [
                        state.vec(choice)
                        for choice in get_object_combinations(
                            list(state), pred.types)
                    ]
                    negatives = []
                    # TODO: I think this logic is wrong. Confirm and fix in a
                    # separate PR before merging this one.
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
        self._learn_nsrts(dataset.trajectories, online_learning_cycle)

    def _ask_teacher(self, state: State, ground_atom: GroundAtom) -> bool:
        """Returns whether the ground atom is true in the state."""
        return self._teacher.ask(state, ground_atom)


class _Teacher:
    """Answers queries about GroundAtoms in States."""

    def __init__(self, initial_predicates: Set[Predicate],
                 predicates_to_learn: Set[Predicate]) -> None:
        self._name_to_predicate = {p.name: p for p in initial_predicates}
        self._predicates_to_learn = predicates_to_learn

    def ask(self, state: State, ground_atom: GroundAtom) -> bool:
        """Returns whether the ground atom is true in the state."""
        # Find the predicate that has the classifier
        predicate = self._name_to_predicate[ground_atom.predicate.name]
        # Use the predicate's classifier
        return predicate.holds(state, ground_atom.objects)


def glib_sample(
    initial_state: State,
    predicates: Set[Predicate],
    dataset: Dataset,
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
        goals.append((goal, score_goal(dataset, goal)))
    goals.sort(key=lambda tup: tup[1], reverse=True)
    return [Task(initial_state, g) for (g, _) in \
            goals[:CFG.interactive_num_tasks_babbled]]


def score_goal(dataset: Dataset, goal: Set[GroundAtom]) -> float:
    """Score a goal as inversely proportional to the number of examples seen
    during training."""
    count = 1  # Avoid division by 0
    for ground_atom_traj in dataset.annotations:
        for ground_atom_set in ground_atom_traj:
            count += 1 if goal.issubset(ground_atom_set) else 0
    return 1.0 / count


def score_atom(dataset: Dataset, atom: GroundAtom) -> float:
    """Score an atom as inversely proportional to the number of examples seen
    during training."""
    count = 1  # Avoid division by 0
    for ground_atom_traj in dataset.annotations:
        for ground_atom_set in ground_atom_traj:
            count += 1 if atom in ground_atom_set else 0
    return 1.0 / count
