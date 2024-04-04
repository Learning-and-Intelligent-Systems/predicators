"""An approach that learns predicates from a teacher."""

import logging
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.approaches.random_options_approach import \
    RandomOptionsApproach
from predicators.src.ml_models import LearnedPredicateClassifier, \
    MLPBinaryClassifierEnsemble
from predicators.src.settings import CFG
from predicators.src.structs import Action, Dataset, GroundAtom, \
    GroundAtomsHoldQuery, GroundAtomsHoldResponse, InteractionRequest, \
    InteractionResult, LowLevelTrajectory, ParameterizedOption, Predicate, \
    Query, State, Task, Type, _GroundNSRT


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
        self._pred_to_ensemble: Dict[str, MLPBinaryClassifierEnsemble] = {}

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
            model = MLPBinaryClassifierEnsemble(
                seed=CFG.seed,
                balance_data=CFG.mlp_classifier_balance_data,
                max_train_iters=CFG.predicate_mlp_classifier_max_itr,
                learning_rate=CFG.learning_rate,
                n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
                hid_sizes=CFG.mlp_classifier_hid_sizes,
                ensemble_size=CFG.interactive_num_ensemble_members)
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
        for train_task_idx in self._select_interaction_train_task_idxs():
            # Determine the action policy and termination function.
            act_policy, termination_function = \
                self._create_interaction_action_strategy(train_task_idx)
            # Determine the query policy.
            query_policy = self._create_interaction_query_policy(
                train_task_idx)
            request = InteractionRequest(train_task_idx, act_policy,
                                         query_policy, termination_function)
            requests.append(request)
        assert len(requests) == CFG.interactive_num_requests_per_cycle
        return requests

    def _score_atom_set(self, atom_set: Set[GroundAtom],
                        state: State) -> float:
        """Score an atom set based on how much we would like to know the values
        of all the atoms in the set in the given state.

        Higher scores are better.
        """
        if CFG.interactive_score_function == "frequency":
            return self._score_atom_set_frequency(atom_set)
        if CFG.interactive_score_function == "trivial":
            return 0.0  # always return the same score
        if CFG.interactive_score_function == "entropy":
            return self._score_atom_set_entropy(atom_set, state)
        if CFG.interactive_score_function == "BALD":
            return self._score_atom_set_bald(atom_set, state)
        if CFG.interactive_score_function == "variance":
            return self._score_atom_set_variance(atom_set, state)
        raise NotImplementedError("Unrecognized interactive_score_function:"
                                  f" {CFG.interactive_score_function}.")

    def _select_interaction_train_task_idxs(self) -> List[int]:
        # At the moment, we select train task indices uniformly at
        # random, with replacement. In the future, we may want to
        # try other strategies.
        return self._rng.choice(len(self._train_tasks),
                                size=CFG.interactive_num_requests_per_cycle)

    def _create_interaction_action_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Returns an action policy and a termination function."""
        if CFG.interactive_action_strategy == "glib":
            return self._create_glib_interaction_strategy(train_task_idx)
        if CFG.interactive_action_strategy == "greedy_lookahead":
            return self._create_greedy_lookahead_interaction_strategy(
                train_task_idx)
        if CFG.interactive_action_strategy == "random":
            return self._create_random_interaction_strategy(train_task_idx)
        if CFG.interactive_action_strategy == "do_nothing":
            return self._create_do_nothing_interaction_strategy(train_task_idx)
        raise NotImplementedError("Unrecognized interactive_action_strategy:"
                                  f" {CFG.interactive_action_strategy}")

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

    def _create_glib_interaction_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Find the most interesting reachable ground goal and plan to it."""
        init = self._train_tasks[train_task_idx].init
        # Detect and filter out static predicates.
        static_preds = utils.get_static_preds(self._nsrts,
                                              self._predicates_to_learn)
        preds = self._predicates_to_learn - static_preds
        # Sample possible goals to plan toward.
        ground_atom_universe = utils.all_possible_ground_atoms(init, preds)
        # If there are no possible goals, fall back to random immediately.
        if not ground_atom_universe:
            logging.info("No possible goals, falling back to random")
            return self._create_random_interaction_strategy(train_task_idx)
        possible_goals = utils.sample_subsets(
            ground_atom_universe,
            num_samples=CFG.interactive_num_babbles,
            min_set_size=1,
            max_set_size=CFG.interactive_max_num_atoms_babbled,
            rng=self._rng)
        # Exclude goals that hold in the initial state to prevent trivial
        # interaction requests.
        possible_goal_lst = [
            g for g in possible_goals if not all(a.holds(init) for a in g)
        ]
        # Sort the possible goals based on how interesting they are.
        # Note: we're using _score_atom_set_frequency here instead of
        # _score_atom_set because _score_atom_set in general could depend
        # on the current state. While babbling goals, we don't have any
        # current state because we don't know what the state will be if and
        # when we get to the goal.
        goal_list = sorted(possible_goal_lst,
                           key=self._score_atom_set_frequency,
                           reverse=True)  # largest to smallest
        task_list = [Task(init, goal) for goal in goal_list]
        try:
            task, act_policy = self._find_first_solvable(task_list)
        except ApproachFailure:
            # Fall back to a random exploration strategy if no solvable task
            # can be found.
            logging.info("No solvable task found, falling back to random")
            return self._create_random_interaction_strategy(train_task_idx)
        assert task.init is init

        logging.info(f"GLIB found a plan to task with goal {task.goal}.")

        # Stop the episode if we reach the goal that we babbled.
        termination_function = task.goal_holds
        return act_policy, termination_function

    def _create_greedy_lookahead_interaction_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Sample a certain number of max-length trajectories and pick the one
        that has the highest cumulative score."""
        init = self._train_tasks[train_task_idx].init
        # Create all applicable ground NSRTs
        ground_nsrts: List[_GroundNSRT] = []
        for nsrt in sorted(self._get_current_nsrts()):
            ground_nsrts.extend(utils.all_ground_nsrts(nsrt, list(init)))
        # Sample trajectories by sampling random sequences of NSRTs.
        best_score = -np.inf
        best_options = []
        for _ in range(CFG.interactive_max_num_trajectories):
            state = init.copy()
            options = []
            trajectory_length = 0
            total_score = 0.0
            while trajectory_length < CFG.interactive_max_trajectory_length:
                # Sample an NSRT that has preconditions satisfied in the
                # current state.
                ground_nsrt = self._sample_applicable_ground_nsrt(
                    state, ground_nsrts)
                if ground_nsrt is None:  # No applicable NSRTs
                    break
                assert all(a.holds for a in ground_nsrt.preconditions)
                # Sample an option. Note that goal is assumed not used.
                assert not CFG.sampler_learning_use_goals
                option = ground_nsrt.sample_option(state,
                                                   goal=set(),
                                                   rng=self._rng)
                # Assume for now that options will be initiable when the
                # preconditions of the NSRT are satisfied.
                assert option.initiable(state)
                state, num_actions = \
                    self._option_model.get_next_state_and_num_actions(state,
                                                                      option)
                # Special case: if the num actions is 0, something went wrong,
                # and we don't want to use this option after all. To prevent
                # possible infinite loops, just break immediately in this case.
                if num_actions == 0:
                    break
                options.append(option)
                trajectory_length += num_actions
                # Update the total score.
                atoms = utils.abstract(state, self._predicates_to_learn)
                total_score += self._score_atom_set(atoms, state)
            if total_score > best_score:
                best_score = total_score
                best_options = options
        assert not np.isinf(best_score)
        act_policy = utils.option_plan_to_policy(best_options)
        # When the act policy finishes, an OptionExecutionFailure is raised
        # and caught, terminating the episode.
        termination_function = lambda s: False

        return act_policy, termination_function

    def _sample_applicable_ground_nsrt(
            self, state: State,
            ground_nsrts: Sequence[_GroundNSRT]) -> Optional[_GroundNSRT]:
        """Choose uniformly among the ground NSRTs that are applicable in the
        state."""
        atoms = utils.abstract(state, self._get_current_predicates())
        applicable_nsrts = sorted(
            utils.get_applicable_operators(ground_nsrts, atoms))
        if len(applicable_nsrts) == 0:
            return None
        idx = self._rng.choice(len(applicable_nsrts))
        return applicable_nsrts[idx]

    def _create_random_interaction_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Sample and execute random initiable options until timeout."""

        random_options_approach = RandomOptionsApproach(
            self._get_current_predicates(), self._initial_options, self._types,
            self._action_space, self._train_tasks)
        task = self._train_tasks[train_task_idx]
        act_policy = random_options_approach.solve(task, CFG.timeout)

        # Termination is left to the environment, as in
        # CFG.max_num_steps_interaction_request.
        termination_function = lambda _: False
        return act_policy, termination_function

    def _create_do_nothing_interaction_strategy(
        self, train_task_idx: int
    ) -> Tuple[Callable[[State], Action], Callable[[State], bool]]:
        """Do nothing until timeout."""
        del train_task_idx  # unused
        # Action policy is practically unused because we terminate immediately.
        act_policy = lambda s: Action(self._action_space.sample())
        termination_function = lambda _: True
        return act_policy, termination_function

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

    def _find_first_solvable(
            self,
            task_list: List[Task]) -> Tuple[Task, Callable[[State], Action]]:
        for task in task_list:
            try:
                logging.info("Solving for policy...")
                policy = self.solve(task, timeout=CFG.timeout)
                return task, policy
            except (ApproachTimeout, ApproachFailure) as e:
                logging.info(f"Approach failed to solve with error: {e}")
                continue
        raise ApproachFailure("Failed to sample a task that approach "
                              "can solve.")

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
            entropy_sum += utils.entropy(np.mean(ps))
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
            entropy = utils.entropy(np.mean(ps))
            objective += entropy - np.mean([utils.entropy(p) for p in ps])
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
            objective += np.var(ps)
        return objective
