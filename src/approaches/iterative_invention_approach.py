"""An approach that iteratively invents predicates."""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach
from predicators.src.nsrt_learning.strips_learning import (
    learn_strips_operators, segment_trajectory)
from predicators.src.settings import CFG
from predicators.src.structs import (Array, Dataset, Datastore, GroundAtom,
                                     LiftedAtom, OptionSpec,
                                     ParameterizedOption, Predicate, Segment,
                                     STRIPSOperator, Task, Type)
from predicators.src.torch_models import (LearnedPredicateClassifier,
                                          MLPClassifier)


class IterativeInventionApproach(NSRTLearningApproach):
    """An approach that iteratively invents predicates."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._learned_predicates: Set[Predicate] = set()
        self._num_inventions = 0

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Use the current predicates to segment dataset.
        predicates = self._get_current_predicates()
        # Apply predicates to dataset.
        ground_atom_dataset = utils.create_ground_atom_dataset(
            dataset.trajectories, predicates)
        # Segment transitions based on changes in predicates.
        segments = [
            seg for traj in ground_atom_dataset
            for seg in segment_trajectory(traj)
        ]
        assert CFG.option_learner == "no_learning", \
            "Iterative invention assumes that options are given."
        for segment in segments:
            assert segment.has_option()
        while True:
            print(f"\n\nInvention iteration {self._num_inventions}")
            # Invent predicates one at a time (iteratively).
            new_predicate = self._invent_for_some_op(segments)
            if new_predicate is None:
                # No new invention for any operator, terminate.
                print("\tFound no new predicates, terminating invention\n")
                break
            # Add the new predicate and its negation to our predicate set.
            neg_new_predicate = new_predicate.get_negation()
            self._learned_predicates.add(new_predicate)
            self._learned_predicates.add(neg_new_predicate)
            self._num_inventions += 1
            # Add the new predicate and its negation to all the transitions.
            new_preds = {new_predicate, neg_new_predicate}
            for segment in segments:
                segment.init_atoms.update(
                    utils.abstract(segment.states[0], new_preds))
                segment.final_atoms.update(
                    utils.abstract(segment.states[-1], new_preds))
        # Finally, learn NSRTs via superclass, using all the predicates.
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None)

    def _invent_for_some_op(
            self, segments: Sequence[Segment]) -> Optional[Predicate]:
        # Iterate over parameterized options in a random order.
        # Run operator learning.
        pnads = learn_strips_operators(segments)
        strips_ops = [pnad.op for pnad in pnads]
        datastores = [pnad.datastore for pnad in pnads]
        option_specs = [pnad.option_spec for pnad in pnads]
        # Iterate over operators in a random order.
        for idx in self._rng.permutation(len(strips_ops)):
            op = strips_ops[idx]
            option_spec = option_specs[idx]
            new_predicate = self._invent_for_op(op, option_spec, datastores)
            if new_predicate is not None:
                # Halt on ANY successful invention.
                return new_predicate
        return None

    def _invent_for_op(self, op: STRIPSOperator, option_spec: OptionSpec,
                       datastores: Sequence[Datastore]) -> Optional[Predicate]:
        """Go through the data, splitting it into positives and negatives based
        on whether the operator correctly predicts each transition or not.

        If there is any negative data, we have a classification problem,
        which we solve to produce a new predicate.
        """
        if not op.parameters:
            # We can't learn 0-arity predicates since the vectorized
            # states would be empty, i.e. the X matrix has no features.
            return None
        param_option, option_vars = option_spec
        opt_arg_pred = Predicate("OPT-ARGS",
                                 param_option.types,
                                 _classifier=lambda s, o: False)  # dummy
        lifted_opt_atom = LiftedAtom(opt_arg_pred, option_vars)
        op_pre = utils.wrap_atom_predicates(op.preconditions, "PRE-")
        op_add_effs = utils.wrap_atom_predicates(op.add_effects, "ADD-")
        op_del_effs = utils.wrap_atom_predicates(op.delete_effects, "DEL-")
        lifteds = frozenset(op_pre | op_add_effs | op_del_effs
                            | {lifted_opt_atom})
        # Organize segments by the set of objects that are in each one.
        segments_by_objects = defaultdict(list)
        for datastore in datastores:
            for (segment, _) in datastore:
                # Exclude if options don't match.
                if segment.get_option().parent != param_option:
                    continue
                objects = frozenset(segment.states[0])
                segments_by_objects[objects].append(segment)
        del datastores
        # Figure out which transitions the op makes wrong predictions on.
        # Keep track of data for every subset of op parameters, so that
        # we can do pruning later.
        data: Dict[Sequence[Any], Dict[str, List[Array]]] = {}
        for params in utils.powerset(op.parameters, exclude_empty=True):
            data[params] = {"pos": [], "neg": []}
        for objects in segments_by_objects:
            for grounding, sub in utils.get_all_groundings(lifteds, objects):
                for segment in segments_by_objects[objects]:
                    option = segment.get_option()
                    ground_opt_atom = GroundAtom(opt_arg_pred, option.objects)
                    trans_atoms = utils.wrap_atom_predicates(
                        segment.init_atoms, "PRE-")
                    trans_add_effs = utils.wrap_atom_predicates(
                        segment.add_effects, "ADD-")
                    trans_del_effs = utils.wrap_atom_predicates(
                        segment.delete_effects, "DEL-")
                    # Check whether the grounding holds for the atoms & option.
                    # If not, continue.
                    pre_opt_grounding = {
                        atom
                        for atom in grounding
                        if atom.predicate.name == "OPT-ARGS"
                        or atom.predicate.name.startswith("PRE-")
                    }
                    if not pre_opt_grounding.issubset(trans_atoms
                                                      | {ground_opt_atom}):
                        continue
                    # Since we made it past the above check, we know that the
                    # preconditions of the operator can be bound to this
                    # transition. So, this transition belongs in our dataset.
                    # Assign it to either positive_data or negative_data
                    # depending on whether the effects hold.
                    grounding_add_effects = {
                        atom.ground(sub)
                        for atom in op_add_effs
                    }
                    grounding_delete_effects = {
                        atom.ground(sub)
                        for atom in op_del_effs
                    }
                    state = segment.states[0]
                    for params, params_data in data.items():
                        predicate_objects = [sub[v] for v in params]
                        vec = state.vec(predicate_objects)
                        if trans_add_effs == grounding_add_effects and \
                           trans_del_effs == grounding_delete_effects:
                            params_data["pos"].append(vec)
                        else:
                            params_data["neg"].append(vec)
        all_params = tuple(op.parameters)
        assert data[all_params]["pos"], "How was this operator learned...?"
        if not data[all_params]["neg"]:
            print(f"\tNo wrong predictions for operator {op.name}")
            return None
        print(f"\tFound a classification problem for operator {op.name}")
        print(f"\t\tData: {len(data[all_params]['pos'])} positives, "
              f"{len(data[all_params]['neg'])} negatives")
        # For every subset of op parameters, try to fit an MLP classifier.
        # Score based on how well the classifier fits the data, regularized
        # by the number of parameters in this subset.
        best_pred = None
        best_params = None
        best_score = float("-inf")
        for params, params_data in data.items():
            X = np.array(params_data["pos"] + params_data["neg"])
            Y = np.array([1 for _ in params_data["pos"]] +
                         [0 for _ in params_data["neg"]])
            model = MLPClassifier(X.shape[1],
                                  CFG.predicate_mlp_classifier_max_itr)
            model.fit(X, Y)
            fit_score = np.sum([model.classify(x) for x in X] == Y) / len(Y)
            if fit_score < CFG.iterative_invention_accept_score:
                print(f"\t\tFor parameters {params}, rejecting predicate due "
                      f"to fit score: {fit_score:.5f}")
                continue
            score = fit_score - 0.1 * len(params)  # regularize
            print(f"\t\tFor parameters {params}, got regularized score: "
                  f"{score:.5f}")
            # Construct classifier function & create new Predicate.
            types = [param.type for param in params]
            classifier = LearnedPredicateClassifier(model).classifier
            pred = Predicate(f"InventedPredicate-{self._num_inventions}",
                             types, classifier)
            # Update bests.
            if score > best_score:
                best_pred = pred
                best_params = params
                best_score = score
        # Note that we could get here with both best_params and best_pred
        # being None, meaning that all predicates were rejected due to the
        # accept_score threshold.
        print(f"\t\tChose parameters {best_params}")
        return best_pred
