"""An approach that learns predicates from a teacher."""

import logging
from typing import Callable, Dict, List, Optional, Sequence, Set

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.explorers import create_explorer
from predicators.ml_models import BinaryClassifierEnsemble, \
    KNeighborsClassifier, LearnedPredicateClassifier, MLPBinaryClassifier
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtom, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, InteractionRequest, InteractionResult, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Query, State, Task, \
    Type


class InteractiveLearningApproach(NSRTLearningApproach):
    """An approach that learns predicates from a teacher."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Track score of best atom seen so far.
        self._best_score = -np.inf
        # Initialize things that will be set correctly in offline learning.
        self._dataset = Dataset([], [])
        self._predicates_to_learn: Set[Predicate] = set()
        self._online_learning_cycle = 0
        self._pred_to_ensemble: Dict[str, BinaryClassifierEnsemble] = {}

    @classmethod
    def get_name(cls) -> str:
        return "interactive_learning"

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._predicates_to_learn

    def load(self, online_learning_cycle: Optional[int]) -> None:
        super().load(online_learning_cycle)
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "rb") as f:
            save_dict = pkl.load(f)
        self._dataset = save_dict["dataset"]
        self._predicates_to_learn = save_dict["predicates_to_learn"]
        self._pred_to_ensemble = save_dict["pred_to_ensemble"]
        self._best_score = save_dict["best_score"]

    ######################## Semi-supervised learning #########################

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Special case: empty offline dataset. Annotations may be None.
        if not dataset.trajectories:
            return
        # First, go through the dataset's annotations and figure out the
        # set of predicates to learn. Note that their classifiers were
        # stripped away during the creation of the annotations.
        for ground_atom_traj in dataset.annotations:
            for ground_atom_sets in ground_atom_traj:
                assert len(ground_atom_sets) == 2
                for atom in ground_atom_sets[0] | ground_atom_sets[1]:
                    assert atom.predicate not in self._initial_predicates
                    self._predicates_to_learn.add(atom.predicate)
        self._dataset = Dataset(dataset.trajectories, dataset.annotations)
        # Learn predicates and NSRTs.
        self._relearn_predicates_and_nsrts(online_learning_cycle=None)

    def _relearn_predicates_and_nsrts(
            self, online_learning_cycle: Optional[int]) -> None:
        """Learns predicates and NSRTs in a semi-supervised fashion."""
        logging.info("\nRelearning predicates and NSRTs...")
        # Learn predicates
        for pred in self._predicates_to_learn:
            input_examples = []
            output_examples = []
            for (traj, traj_annotations) in zip(self._dataset.trajectories,
                                                self._dataset.annotations):
                assert len(traj.states) == len(traj_annotations)
                for (state, state_annotation) in zip(traj.states,
                                                     traj_annotations):
                    assert len(state_annotation) == 2
                    for target_class, examples in enumerate(state_annotation):
                        for atom in examples:
                            if not atom.predicate == pred:
                                continue
                            x = state.vec(atom.objects)
                            input_examples.append(x)
                            output_examples.append(target_class)
            num_positives = sum(y == 1 for y in output_examples)
            num_negatives = sum(y == 0 for y in output_examples)
            assert num_positives + num_negatives == len(output_examples)
            logging.info(f"Generated {num_positives} positive and "
                         f"{num_negatives} negative examples for "
                         f"predicate {pred}")

            # Train MLP
            X = np.array(input_examples)
            Y = np.array(output_examples)
            if CFG.predicate_classifier_model == "mlp":
                model = BinaryClassifierEnsemble(
                    seed=CFG.seed,
                    ensemble_size=CFG.interactive_num_ensemble_members,
                    member_cls=MLPBinaryClassifier,
                    balance_data=CFG.mlp_classifier_balance_data,
                    max_train_iters=CFG.predicate_mlp_classifier_max_itr,
                    learning_rate=CFG.learning_rate,
                    n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
                    hid_sizes=CFG.mlp_classifier_hid_sizes,
                    n_reinitialize_tries=CFG.
                    predicate_mlp_classifier_n_reinitialize_tries,
                    weight_init=CFG.predicate_mlp_classifier_init)
            elif CFG.predicate_classifier_model == "knn":
                model = BinaryClassifierEnsemble(
                    seed=CFG.seed,
                    ensemble_size=CFG.interactive_num_ensemble_members,
                    member_cls=KNeighborsClassifier,
                    n_neighbors=CFG.predicate_knn_classifier_n_neighbors)
            else:
                raise ValueError("Unrecognized predicate_classifier_model")
            model.fit(X, Y)

            # Save the ensemble
            self._pred_to_ensemble[pred.name] = model
            # Construct classifier function, create new Predicate, and save it
            classifier = LearnedPredicateClassifier(model).classifier
            new_pred = Predicate(pred.name, pred.types, classifier)
            self._predicates_to_learn = \
                (self._predicates_to_learn - {pred}) | {new_pred}

        # Learn NSRTs via superclass
        self._learn_nsrts(self._dataset.trajectories, online_learning_cycle)

        # Save the things we need other than the NSRTs, which were already
        # saved in the above call to self._learn_nsrts()
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "wb") as f:
            pkl.dump(
                {
                    "dataset": self._dataset,
                    "predicates_to_learn": self._predicates_to_learn,
                    "pred_to_ensemble": self._pred_to_ensemble,
                    "best_score": self._best_score,
                }, f)

    ########################### Active learning ###############################

    def get_interaction_requests(self) -> List[InteractionRequest]:
        requests = []
        # Create the explorer.
        explorer = create_explorer(
            CFG.explorer,
            self._get_current_predicates(),
            self._initial_options,
            self._types,
            self._action_space,
            self._train_tasks,
            self._get_current_nsrts(),
            self._option_model,
            babble_predicates=self._predicates_to_learn,
            atom_score_fn=self._score_atom_set_frequency,
            state_score_fn=self._score_atom_set)
        for train_task_idx in self._select_interaction_train_task_idxs():
            # Determine the action policy and termination function.
            act_policy, termination_fn = explorer.get_exploration_strategy(
                train_task_idx, CFG.timeout)
            # Determine the query policy.
            query_policy = self._create_interaction_query_policy(
                train_task_idx)
            request = InteractionRequest(train_task_idx, act_policy,
                                         query_policy, termination_fn)
            requests.append(request)
        assert len(requests) == CFG.interactive_num_requests_per_cycle
        return requests

    def _score_atom_set(self, atom_set: Set[GroundAtom],
                        state: State) -> float:
        """Score an atom set based on how much we would like to know the values
        of all the atoms in the set in the given state.

        Higher scores are better.
        """
        # Filter out any atoms that are not being learned.
        learned_atom_set = atom_set.copy()
        for atom in atom_set:
            if atom.predicate not in self._predicates_to_learn:
                assert atom.predicate in self._initial_predicates
                learned_atom_set.discard(atom)
        if CFG.interactive_score_function == "frequency":
            return self._score_atom_set_frequency(learned_atom_set)
        if CFG.interactive_score_function == "trivial":
            return 0.0  # always return the same score
        if CFG.interactive_score_function == "entropy":
            return self._score_atom_set_entropy(learned_atom_set, state)
        if CFG.interactive_score_function == "BALD":
            return self._score_atom_set_bald(learned_atom_set, state)
        if CFG.interactive_score_function == "variance":
            return self._score_atom_set_variance(learned_atom_set, state)
        raise NotImplementedError("Unrecognized interactive_score_function:"
                                  f" {CFG.interactive_score_function}.")

    def _select_interaction_train_task_idxs(self) -> List[int]:
        # At the moment, we select train task indices uniformly at
        # random, with replacement. In the future, we may want to
        # try other strategies.
        return self._rng.choice(len(self._train_tasks),
                                size=CFG.interactive_num_requests_per_cycle)

    def _create_interaction_query_policy(
            self, train_task_idx: int) -> Callable[[State], Optional[Query]]:
        """Returns a query policy."""
        del train_task_idx  # unused right now, but future policies may use
        if CFG.interactive_query_policy == "strict_best_seen":
            return self._create_best_seen_query_policy(strict=True)
        if CFG.interactive_query_policy == "nonstrict_best_seen":
            return self._create_best_seen_query_policy(strict=False)
        if CFG.interactive_query_policy == "threshold":
            return self._create_threshold_query_policy()
        if CFG.interactive_query_policy == "random":
            return self._create_random_query_policy()
        raise NotImplementedError("Unrecognized interactive_query_policy:"
                                  f" {CFG.interactive_query_policy}")

    def _create_best_seen_query_policy(
            self, strict: bool) -> Callable[[State], Optional[Query]]:
        """Only query if the atom has the best score seen so far."""

        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            # Decide whether to ask about each possible atom.
            ground_atoms = utils.all_possible_ground_atoms(
                s, self._predicates_to_learn)
            atoms_to_query = set()
            for atom in ground_atoms:
                score = self._score_atom_set({atom}, s)
                # Ask about this atom if it is the best seen so far.
                if (strict and score > self._best_score) or \
                   (not strict and score >= self._best_score):
                    atoms_to_query.add(atom)
                    self._best_score = score
            return GroundAtomsHoldQuery(atoms_to_query)

        return _query_policy

    def _create_threshold_query_policy(
            self) -> Callable[[State], Optional[Query]]:
        """Only query if the atom has score above the set threshold."""

        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            ground_atoms = utils.all_possible_ground_atoms(
                s, self._predicates_to_learn)
            atoms_to_query = set()
            for atom in ground_atoms:
                score = self._score_atom_set({atom}, s)
                if score > CFG.interactive_score_threshold:
                    atoms_to_query.add(atom)
            return GroundAtomsHoldQuery(atoms_to_query)

        return _query_policy

    def _create_random_query_policy(
            self) -> Callable[[State], Optional[Query]]:
        """Query each possible atom with a certain probability."""

        def _query_policy(s: State) -> Optional[GroundAtomsHoldQuery]:
            ground_atoms = utils.all_possible_ground_atoms(
                s, self._predicates_to_learn)
            atoms_to_query = set()
            for atom in ground_atoms:
                if self._rng.random() < CFG.interactive_random_query_prob:
                    atoms_to_query.add(atom)
            return GroundAtomsHoldQuery(atoms_to_query)

        return _query_policy

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        for result in results:
            state_annotations: List[List[Set[GroundAtom]]] = []
            for response in result.responses:
                assert isinstance(response, GroundAtomsHoldResponse)
                state_annotation: List[Set[GroundAtom]] = [set(), set()]
                for query_atom, atom_holds in response.holds.items():
                    state_annotation[atom_holds].add(query_atom)
                state_annotations.append(state_annotation)
            traj = LowLevelTrajectory(result.states, result.actions)
            self._dataset.append(traj, state_annotations)
        self._relearn_predicates_and_nsrts(
            online_learning_cycle=self._online_learning_cycle)
        self._online_learning_cycle += 1

    def _score_atom_set_frequency(self, atom_set: Set[GroundAtom]) -> float:
        """Score an atom set as inversely proportional to the number of
        examples seen during training."""
        count = 1  # Avoid division by 0
        for ground_atom_traj in self._dataset.annotations:
            for ground_atom_sets in ground_atom_traj:
                assert len(ground_atom_sets) == 2
                _, pos_examples = ground_atom_sets
                count += 1 if atom_set.issubset(pos_examples) else 0
        return 1.0 / count

    def _score_atom_set_entropy(self, atom_set: Set[GroundAtom],
                                state: State) -> float:
        """Score an atom set as the sum of the entropies of each atom's
        predicate classifier."""
        entropy_sum = 0.0
        for atom in atom_set:
            x = state.vec(atom.objects)
            ps = self._pred_to_ensemble[
                atom.predicate.name].predict_member_probas(x)
            entropy_sum += utils.entropy(float(np.mean(ps)))
        return entropy_sum

    def _score_atom_set_bald(self, atom_set: Set[GroundAtom],
                             state: State) -> float:
        """Score an atom set as the sum of the BALD objectives of each atom's
        predicate classifier."""
        objective = 0.0
        for atom in atom_set:
            x = state.vec(atom.objects)
            ps = self._pred_to_ensemble[
                atom.predicate.name].predict_member_probas(x)
            entropy = utils.entropy(float(np.mean(ps)))
            mean_entropy = float(np.mean([utils.entropy(p) for p in ps]))
            objective += entropy - mean_entropy
        return objective

    def _score_atom_set_variance(self, atom_set: Set[GroundAtom],
                                 state: State) -> float:
        """Score an atom set as the sum of the variances of the ensemble
        predictions for each atom's classifier."""
        objective = 0.0
        for atom in atom_set:
            x = state.vec(atom.objects)
            ps = self._pred_to_ensemble[
                atom.predicate.name].predict_member_probas(x)
            objective += float(np.var(ps))
        return objective
