"""An approach that learns predicates from a teacher."""

from typing import Set, List, Optional, Tuple, Callable, Sequence
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
from predicators.src.option_model import _OracleOptionModel
from predicators.src.utils import get_object_combinations, strip_predicate
from predicators.src.teacher import GroundAtomHoldsQuery, \
    GroundAtomHoldsResponse
from predicators.src.settings import CFG


class InteractiveLearningApproach(NSRTLearningApproach):
    """An approach that learns predicates from a teacher."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Track score of best atom seen so far
        self._best_score = 0.0
        # Initialize things that will be set correctly in offline learning
        self._dataset = Dataset([])
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
        # Learn predicates and NSRTs
        self._relearn_predicates_and_nsrts(dataset, online_learning_cycle=None)
        # Save dataset, to be used for online interaction
        self._dataset = dataset

    def get_interaction_requests(self) -> List[InteractionRequest]:
        # Sample a train task
        train_task_idx = self._rng.choice(len(self._train_tasks))
        init = self._train_tasks[train_task_idx].init
        # Detect and filter out static predicates
        static_preds = utils.get_static_preds(
            self._nsrts, self._get_current_predicates())
        preds = self._get_current_predicates() - static_preds
        # Find acting policy for the request
        task_list = glib_sample(init, preds, self._dataset)
        assert task_list
        task, act_policy = self._find_first_solvable(task_list)
        assert task.init is init
        def _query_policy(s: State) -> Optional[GroundAtomHoldsQuery]:
            # Decide whether to ask about each possible atom
            ground_atoms = utils.all_possible_ground_atoms(
                s, self._predicates_to_learn)
            for atom in ground_atoms:
                score = score_atom(self._dataset, atom)
                # Ask about this atom if it is the best seen so far
                if score >= self._best_score:
                    # TODO: the previous line is a > on master, changed to >=
                    # because otherwise i wasn't seeing any new data get added
                    # (due mostly to the next TODO)
                    self._best_score = score
                    # TODO: one difference from current master is that
                    # we can now only ask one Query per state, whereas
                    # before you could ask the teacher multiple things
                    # about the same state. hopefully this is still okay...
                    # otherwise we can change the InteractionRequest object's
                    # query object to allow a set of Querys per timestep
                    return GroundAtomHoldsQuery(
                        atom.predicate.name, atom.objects)
            return None
        def _termination_function(s: State) -> bool:
            return all(goal_atom.holds(s) for goal_atom in task.goal)
        return [InteractionRequest(train_task_idx, act_policy, _query_policy,
                                   _termination_function)]

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        assert len(results) == 1
        result = results[0]
        pred_name_to_pred = {pred.name: pred for pred
                             in self._predicates_to_learn}
        for state, response in zip(result.states, result.responses):
            if response is None:
                continue  # we didn't ask a query on this timestep
            assert isinstance(response, GroundAtomHoldsResponse)
            assert isinstance(response.query, GroundAtomHoldsQuery)
            if response.holds:
                # Add this atom if it's a positive example
                traj = LowLevelTrajectory([state], [])
                pred = pred_name_to_pred[response.query.predicate_name]
                atom = GroundAtom(pred, response.query.objects)
                self._dataset.append(traj, [{atom}])
                # Still need a way to use negative examples
        self._relearn_predicates_and_nsrts(
            self._dataset, online_learning_cycle=self._online_learning_cycle)
        self._online_learning_cycle += 1

    def _find_first_solvable(self, task_list: List[Task]) -> Tuple[
            Task, Callable[[State], Action]]:
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
        print("\nStarting semi-supervised learning...")
        # Learn predicates
        for pred in self._predicates_to_learn:
            positive_examples = []
            negative_examples = []
            for (traj, ground_atom_sets) in zip(dataset.trajectories,
                                                dataset.annotations):
                assert len(traj.states) == len(ground_atom_sets)
                for (state, ground_atom_set) in zip(traj.states,
                                                    ground_atom_sets):
                    # Object tuples that appear as the arguments to a ground
                    # atom where the predicate is pred.
                    positive_args = {
                        tuple(atom.objects)
                        for atom in ground_atom_set if atom.predicate == pred
                    }
                    # Loop through all possible examples. If an example appears
                    # in the ground atom set, it's positive. Otherwise, we make
                    # the (wrong in general!) assumption that it's negative.
                    for choice in get_object_combinations(
                            list(state), pred.types):
                        x = state.vec(choice)
                        if tuple(choice) in positive_args:
                            positive_examples.append(x)
                        else:
                            negative_examples.append(x)
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
