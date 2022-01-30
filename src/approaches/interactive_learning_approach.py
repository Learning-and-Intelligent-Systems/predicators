"""An approach that learns predicates from a teacher."""

from typing import Set, List, Optional, Tuple, Callable, Sequence, Dict
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Dataset, GroundAtom, LowLevelTrajectory, InteractionRequest, \
    InteractionResult, Action
from predicators.src.torch_models import LearnedPredicateClassifier, \
    MLPClassifier
from predicators.src.teacher import GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse
from predicators.src.settings import CFG


class InteractiveLearningApproach(NSRTLearningApproach):
    """An approach that learns predicates from a teacher."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Track score of best atom seen so far.
        self._best_score = 0.0
        # Initialize things that will be set correctly in offline learning.
        self._dataset = Dataset([], [])
        self._predicates_to_learn: Set[Predicate] = set()
        self._online_learning_cycle = 0

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._predicates_to_learn

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First, go through the dataset's annotations and figure out the
        # set of predicates to learn. Note that their classifiers were
        # stripped away during the creation of the annotations.
        for ground_atom_sets in dataset.annotations:
            for ground_atom_set in ground_atom_sets:
                for atom in ground_atom_set:
                    assert atom.predicate not in self._initial_predicates
                    self._predicates_to_learn.add(atom.predicate)
        # Next, convert the dataset with positive annotations only into a
        # dataset with positive and unlabeled annotations.
        new_annotations = []
        for traj, ground_atom_sets in zip(dataset.trajectories,
                                          dataset.annotations):
            new_traj_annotation = []
            # Get all possible ground atoms given the objects in traj.
            possible = set(utils.all_possible_ground_atoms(
                traj.states[0], self._predicates_to_learn))
            for positives in ground_atom_sets:
                unlabeled = possible - positives
                new_traj_annotation.append({
                    "positive": positives,
                    "unlabeled": unlabeled,
                    "negative": set(),
                })
            new_annotations.append(new_traj_annotation)
        dataset = Dataset(dataset.trajectories, new_annotations)
        # Learn predicates and NSRTs.
        self._relearn_predicates_and_nsrts(dataset, online_learning_cycle=None)
        # Save dataset, to be used for online interaction.
        self._dataset = dataset

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # Sample a train task.
        train_task_idx = self._rng.choice(len(self._train_tasks))
        init = self._train_tasks[train_task_idx].init
        # Detect and filter out static predicates.
        static_preds = utils.get_static_preds(self._nsrts,
                                              self._predicates_to_learn)
        preds = self._predicates_to_learn - static_preds
        # Find acting policy for the request.
        task_list = glib_sample(init, preds, self._dataset)
        assert task_list
        task, act_policy = self._find_first_solvable(task_list)
        assert task.init is init

        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            # Decide whether to ask about each possible atom.
            ground_atoms = utils.all_possible_ground_atoms(
                s, self._predicates_to_learn)
            atoms_to_query = set()
            for atom in ground_atoms:
                score = score_atom(self._dataset, atom)
                # Ask about this atom if it is the best seen so far.
                if score > self._best_score:
                    atoms_to_query.add(atom)
                    self._best_score = score
            return GroundAtomsHoldQuery(atoms_to_query)

        def _termination_function(s: State) -> bool:
            # Stop the episode if we reach the goal that we babbled.
            return all(goal_atom.holds(s) for goal_atom in task.goal)

        return [
            InteractionRequest(train_task_idx, act_policy, _query_policy,
                               _termination_function)
        ]

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        assert len(results) == 1
        result = results[0]
        for state, response in zip(result.states, result.responses):
            assert isinstance(response, GroundAtomsHoldResponse)
            state_annotation: Dict[str, Set[GroundAtom]] = {
                "positive": set(),
                "negative": set(),
                "unlabeled": set()
            }
            for query_atom, atom_holds in response.holds.items():
                label = "positive" if atom_holds else "negative"
                state_annotation[label].add(query_atom)
            traj = LowLevelTrajectory([state], [])
            self._dataset.append(traj, [state_annotation])
        self._relearn_predicates_and_nsrts(
            self._dataset, online_learning_cycle=self._online_learning_cycle)
        self._online_learning_cycle += 1

    def _find_first_solvable(
            self,
            task_list: List[Task]) -> Tuple[Task, Callable[[State], Action]]:
        for task in task_list:
            try:
                print("Solving for policy...")
                policy = self.solve(task, timeout=CFG.timeout)
                return task, policy
            except (ApproachTimeout, ApproachFailure) \
                    as e:  # pragma: no cover
                print(f"Approach failed to solve with error: {e}")
                continue
        raise ApproachFailure("Failed to sample a task that approach "
                              "can solve.")  # pragma: no cover

    def _relearn_predicates_and_nsrts(
            self, dataset: Dataset,
            online_learning_cycle: Optional[int]) -> None:
        """Learns predicates and NSRTs in a semi-supervised fashion."""
        print("\nRelearning predicates and NSRTs...")
        # Learn predicates
        for pred in self._predicates_to_learn:
            input_examples = []
            output_examples = []
            for (traj, traj_annotations) in zip(dataset.trajectories,
                                                dataset.annotations):
                assert len(traj.states) == len(traj_annotations)
                for (state, state_annotation) in zip(traj.states,
                                                     traj_annotations):
                    # Here we make the (wrong in general!) assumption that
                    # unlabeled ground atoms are negative. In the future, we
                    # may want to modify this, e.g., downweight or remove
                    # the unlabeled examples once we collect enough negatives.
                    for label, target_class in [("positive", 1),
                                                ("unlabeled", 0),
                                                ("negative", 0)]:
                        for atom in state_annotation[label]:
                            if not atom.predicate == pred:
                                continue
                            x = state.vec(atom.objects)
                            input_examples.append(x)
                            output_examples.append(target_class)
            num_positives = sum(y == 1 for y in output_examples)
            num_negatives = sum(y == 0 for y in output_examples)
            assert num_positives + num_negatives == len(output_examples)
            print(f"Generated {num_positives} positive and "
                  f"{num_negatives} negative examples for "
                  f"predicate {pred}")

            # Train MLP
            X = np.array(input_examples)
            Y = np.array(output_examples)
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
