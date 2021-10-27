"""An approach that iteratively invents predicates.
"""

from collections import defaultdict
from typing import Set, Callable, List, Optional, DefaultDict
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import OperatorLearningApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, GroundAtom, Transition, Operator, LiftedAtom
from predicators.src.models import LearnedPredicateClassifier, MLPClassifier
from predicators.src.operator_learning import generate_transitions, \
    learn_operators_for_option
from predicators.src.settings import CFG


class IterativeInventionApproach(OperatorLearningApproach):
    """An approach that iteratively invents predicates.
    """
    def __init__(self, simulator: Callable[[State, Action], State],
                 all_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task]) -> None:
        super().__init__(simulator, all_predicates, initial_options,
                         types, action_space, train_tasks)
        self._learned_predicates: Set[Predicate] = set()
        self._num_inventions = 0

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Use the current predicates to generate transitions.
        transitions_by_option = generate_transitions(
            dataset, self._get_current_predicates())
        while True:
            print(f"\n\nInvention iteration {self._num_inventions}")
            # Invent predicates one at a time (iteratively).
            new_predicate = self._invent_for_some_operator(
                transitions_by_option)
            if new_predicate is None:
                # No new invention for any operator, terminate.
                print("\tFound no new predicates, terminating invention\n")
                break
            # Add the new predicate to our set. Also add the negation.
            neg_new_predicate = new_predicate.get_negation()
            self._learned_predicates.add(new_predicate)
            self._learned_predicates.add(neg_new_predicate)
            self._num_inventions += 1
            # Add the new predicate to all the transitions.
            for transitions in transitions_by_option.values():
                for i, (state, next_state, atoms, option, next_atoms,
                        _, _) in enumerate(transitions):
                    atoms = atoms | utils.abstract(
                        state, {new_predicate, neg_new_predicate})
                    next_atoms = next_atoms | utils.abstract(
                        next_state, {new_predicate, neg_new_predicate})
                    add_effects = next_atoms - atoms
                    delete_effects = atoms - next_atoms
                    transitions[i] = (state, next_state, atoms, option,
                                      next_atoms, add_effects, delete_effects)
        # Finally, learn operators via superclass, using all the predicates.
        self._learn_operators(dataset)

    def _invent_for_some_operator(self, transitions_by_option: DefaultDict[
            ParameterizedOption, List[Transition]]) -> Optional[Predicate]:
        # Iterate over parameterized options in a random order.
        for param_option in self._rng.permutation(list(transitions_by_option)):
            option_transitions = transitions_by_option[param_option]
            # Run operator learning.
            operators = learn_operators_for_option(
                param_option, option_transitions, do_sampler_learning=False)
            # Iterate over operators in a random order.
            for operator in self._rng.permutation(operators):
                new_predicate = self._invent_for_operator(
                    operator, option_transitions)
                if new_predicate is not None:
                    # Halt on ANY successful invention.
                    return new_predicate
        return None

    def _invent_for_operator(self, operator: Operator,
                             transitions: List[Transition]
                             ) -> Optional[Predicate]:
        opt_arg_pred = Predicate("OPT-ARGS", operator.option.types,
                                 _classifier=lambda s, o: False)  # dummy
        lifted_opt_atom = LiftedAtom(opt_arg_pred, operator.option_vars)
        operator_pre = utils.wrap_atom_predicates_lifted(
            operator.preconditions, "PRE-")
        operator_add_effs = utils.wrap_atom_predicates_lifted(
            operator.add_effects, "ADD-")
        operator_del_effs = utils.wrap_atom_predicates_lifted(
            operator.delete_effects, "DEL-")
        # Organize transitions by the set of objects that are in each one.
        transitions_by_objects = defaultdict(list)
        for transition in transitions:
            state, _, _, option, _, _, _ = transition
            assert operator.option == option.parent
            objects = frozenset(list(state))
            transitions_by_objects[objects].append(transition)
        del transitions
        positive_data = []
        negative_data = []
        # Figure out which transitions the operator makes wrong predictions on.
        for objects in transitions_by_objects:
            lifteds = frozenset(operator_pre | operator_add_effs |
                                operator_del_effs | {lifted_opt_atom})
            for grounding, sub in utils.get_all_groundings(lifteds, objects):
                for (state, _, atoms, option, _, add_effs,
                     del_effs) in transitions_by_objects[objects]:
                    ground_opt_atom = GroundAtom(opt_arg_pred, option.objects)
                    trans_atoms = utils.wrap_atom_predicates_ground(
                        atoms, "PRE-")
                    trans_add_effs = utils.wrap_atom_predicates_ground(
                        add_effs, "ADD-")
                    trans_del_effs = utils.wrap_atom_predicates_ground(
                        del_effs, "DEL-")
                    # Check whether the grounding holds for the atoms & option.
                    # If not, continue.
                    pre_opt_grounding = {atom for atom in grounding if
                                         atom.predicate.name == "OPT-ARGS" or
                                         atom.predicate.name.startswith("PRE-")}
                    if not pre_opt_grounding.issubset(
                            trans_atoms | {ground_opt_atom}):
                        continue
                    # Since we made it past the above check, we know that the
                    # preconditions of the operator and the option arguments
                    # can be bound to this transition. So, this transition
                    # belongs in our dataset. Assign it to either positive_data
                    # or negative_data depending on whether the effects hold.
                    grounding_add_effects = {atom.ground(sub) for atom in
                                             operator_add_effs}
                    grounding_delete_effects = {atom.ground(sub) for atom in
                                                operator_del_effs}
                    predicate_objects = [sub[v] for v in operator.parameters]
                    if trans_add_effs == grounding_add_effects and \
                       trans_del_effs == grounding_delete_effects:
                        positive_data.append(state.vec(predicate_objects))
                    else:
                        negative_data.append(state.vec(predicate_objects))
        assert positive_data, "How was this operator learned...?"
        if not negative_data:
            print(f"\tNo wrong predictions for operator {operator.name}")
            return None
        print(f"\tFound a classification problem for operator {operator.name}")
        print(f"\t\tData: {len(positive_data)} positives, "
              f"{len(negative_data)} negatives")
        # Fit MLP classifier and score predicate
        X = np.array(positive_data + negative_data)
        Y = np.array([1 for _ in positive_data] +
                     [0 for _ in negative_data])
        model = MLPClassifier(X.shape[1])
        model.fit(X, Y)
        score = sum([model.classify(x) for x in X] == Y) / len(Y)
        if score < CFG.iterative_invention_accept_score:
            print(f"\t\tRejecting predicate with score: {score:.5f}")
            return None
        print(f"\t\tAccepting predicate with score: {score:.5f}")
        # Construct classifier function & create new Predicate
        types = [param.type for param in operator.parameters]
        classifier = LearnedPredicateClassifier(model).classifier
        return Predicate(f"InventedPredicate-{self._num_inventions}",
                         types, classifier)
