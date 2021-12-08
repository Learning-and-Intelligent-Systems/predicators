"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar.
"""

import abc
from dataclasses import dataclass
from functools import cached_property
import itertools
from operator import ge, le
from typing import Set, Callable, List, Sequence, FrozenSet, Iterator, Tuple, \
    Dict
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach
from predicators.src.nsrt_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, Object, GroundAtomTrajectory, STRIPSOperator, \
    OptionSpec, Segment
from predicators.src.settings import CFG


################################################################################
#                          Programmatic classifiers                            #
################################################################################

class _ProgrammaticClassifier(abc.ABC):
    """A classifier implemented as an arbitrary program.
    """
    @abc.abstractmethod
    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        """All programmatic classifiers are functions of state and objects.

        The objects are the predicate arguments.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError("Override me!")


class _NullaryClassifier(_ProgrammaticClassifier):
    """A classifier on zero objects.
    """
    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        assert len(o) == 0
        return self._classify_state(s)

    @abc.abstractmethod
    def _classify_state(self, s: State) -> bool:
        raise NotImplementedError("Override me!")


class _UnaryClassifier(_ProgrammaticClassifier):
    """A classifier on one object.
    """
    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        assert len(o) == 1
        return self._classify_object(s, o[0])

    @abc.abstractmethod
    def _classify_object(self, s: State, obj: Object) -> bool:
        raise NotImplementedError("Override me!")


@dataclass(frozen=True, eq=False, repr=False)
class _SingleAttributeCompareClassifier(_UnaryClassifier):
    """Compare a single feature value with a constant value.
    """
    object_index: int
    object_type: Type
    attribute_name: str
    constant: float
    compare: Callable[[float, float], bool]
    compare_str: str

    def _classify_object(self, s: State, obj: Object) -> bool:
        assert obj.type == self.object_type
        return self.compare(s.get(obj, self.attribute_name), self.constant)

    def __str__(self) -> str:
        return (f"(({self.object_index}:{self.object_type.name})."
                f"{self.attribute_name}{self.compare_str}{self.constant:.3})")


@dataclass(frozen=True, eq=False, repr=False)
class _NegationClassifier(_ProgrammaticClassifier):
    """Negate a given classifier.
    """
    body: Predicate

    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        return not self.body.holds(s, o)

    def __str__(self) -> str:
        return f"NOT-{self.body}"


@dataclass(frozen=True, eq=False, repr=False)
class _ForallClassifier(_NullaryClassifier):
    """Apply a predicate to all objects.
    """
    body: Predicate

    def _classify_state(self, s: State) -> bool:
        for o in utils.get_object_combinations(set(s), self.body.types):
            if not self.body.holds(s, o):
                return False
        return True

    def __str__(self) -> str:
        types = self.body.types
        type_sig = ",".join(f"{i}:{t.name}" for i, t in enumerate(types))
        objs = ",".join(str(i) for i in range(len(types)))
        return f"Forall[{type_sig}].[{str(self.body)}({objs})]"


@dataclass(frozen=True, eq=False, repr=False)
class _UnaryFreeForallClassifier(_UnaryClassifier):
    """Universally quantify all but one variable in a multi-arity predicate.

    Examples:
        - ForAll ?x. On(?x, ?y)
        - Forall ?y. On(?x, ?y)
        - ForAll ?x, ?y. Between(?x, ?z, ?y)
    """
    body: Predicate  # Must be arity 2 or greater.
    free_variable_idx: int

    def __post_init__(self) -> None:
        assert self.body.arity >= 2
        assert self.free_variable_idx < self.body.arity

    @cached_property
    def _quantified_types(self) -> List[Type]:
        return [t for i, t in enumerate(self.body.types)
                if i != self.free_variable_idx]

    def _classify_object(self, s: State, obj: Object) -> bool:
        assert obj.type == self.body.types[self.free_variable_idx]
        for o in utils.get_object_combinations(set(s), self._quantified_types):
            o_lst = list(o)
            o_lst.insert(self.free_variable_idx, obj)
            if not self.body.holds(s, o_lst):
                return False
        return True

    def __str__(self) -> str:
        types = self.body.types
        type_sig = ",".join(f"{i}:{t.name}" for i, t in enumerate(types)
                            if i != self.free_variable_idx)
        objs = ",".join(str(i) for i in range(len(types)))
        return f"Forall[{type_sig}].[{str(self.body)}({objs})]"


################################################################################
#                             Predicate grammars                               #
################################################################################

@dataclass(frozen=True, eq=False, repr=False)
class _PredicateGrammar:
    """A grammar for generating predicate candidates.
    """
    def generate(self, max_num: int) -> Dict[Predicate, float]:
        """Generate candidate predicates from the grammar.
        The dict values are costs, e.g., negative log prior probability for the
        predicate in a PCFG.
        """
        candidates: Dict[Predicate, float] = {}
        if max_num == 0:
            return candidates
        assert max_num > 0
        for candidate, cost in self.enumerate():
            candidates[candidate] = cost
            if len(candidates) == max_num:
                break
        return candidates

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        """Iterate over candidate predicates from less to more cost.
        """
        raise NotImplementedError("Override me!")


@dataclass(frozen=True, eq=False, repr=False)
class _DataBasedPredicateGrammar(_PredicateGrammar):
    """A predicate grammar that uses a dataset.
    """
    dataset: Dataset

    @cached_property
    def types(self) -> Set[Type]:
        """Infer types from the dataset.
        """
        types: Set[Type] = set()
        for (states, _) in self.dataset:
            types.update(o.type for o in states[0])
        return types

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        """Iterate over candidate predicates from less to more cost.
        """
        raise NotImplementedError("Override me!")


@dataclass(frozen=True, eq=False, repr=False)
class _HoldingDummyPredicateGrammar(_DataBasedPredicateGrammar):
    """A hardcoded cover-specific grammar.

    Good for testing with:
        python src/main.py --env cover --approach grammar_search_invention \
            --seed 0 --excluded_predicates Holding
    """
    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # A necessary predicate.
        block_type = [t for t in self.types if t.name == "block"][0]
        types = [block_type]
        classifier = _SingleAttributeCompareClassifier(
            0, block_type, "grasp", -0.9, ge, ">=")
        # The name of the predicate is derived from the classifier.
        # In this case, the name will be (0.grasp>=-0.9). The "0" at the
        # beginning indicates that the classifier is indexing into the
        # first object argument and looking at its grasp feature. For
        # example, (0.grasp>=-0.9)(block1) would look be a function of
        # state.get(block1, "grasp").
        yield (Predicate(str(classifier), types, classifier), 1.)

        # An unnecessary predicate (because it's redundant).
        classifier = _SingleAttributeCompareClassifier(
            0, block_type, "is_block", 0.5, ge, ">=")
        yield (Predicate(str(classifier), types, classifier), 1.)


def _halving_constant_generator(lo: float, hi: float) -> Iterator[float]:
    mid = (hi + lo) / 2.
    yield mid
    left_gen = _halving_constant_generator(lo, mid)
    right_gen = _halving_constant_generator(mid, hi)
    for l, r in zip(left_gen, right_gen):
        yield l
        yield r


@dataclass(frozen=True, eq=False, repr=False)
class _SingleFeatureInequalitiesPredicateGrammar(_DataBasedPredicateGrammar):
    """Generates features of the form "0.feature >= c" or "0.feature <= c".
    """
    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # Get ranges of feature values from data.
        feature_ranges = self._get_feature_ranges()
        # 0.5, 0.25, 0.75, 0.125, 0.375, ...
        constant_generator = _halving_constant_generator(0., 1.)
        for c in constant_generator:
            for t in sorted(self.types):
                for f in t.feature_names:
                    lb, ub = feature_ranges[t][f]
                    # Optimization: if lb == ub, there is no variation
                    # among this feature, so there's no point in trying to
                    # learn a classifier with it. So, skip the feature.
                    if abs(lb - ub) < 1e-6:
                        continue
                    # Scale the constant by the feature range.
                    k = (c + lb) / (ub - lb)
                    # Only need one of (ge, le) because we can use negations
                    # to get the other (modulo equality, which we shouldn't
                    # rely on anyway because of precision issues).
                    comp, comp_str = le, "<="
                    classifier = _SingleAttributeCompareClassifier(
                        0, t, f, k, comp, comp_str)
                    name = str(classifier)
                    types = [t]
                    yield (Predicate(name, types, classifier), 1.)


    def _get_feature_ranges(self) -> Dict[Type, Dict[str, Tuple[float, float]]]:
        feature_ranges: Dict[Type, Dict[str, Tuple[float, float]]] = {}
        for (states, _) in self.dataset:
            for state in states:
                for obj in state:
                    if obj.type not in feature_ranges:
                        feature_ranges[obj.type] = {}
                        for f in obj.type.feature_names:
                            v = state.get(obj, f)
                            feature_ranges[obj.type][f] = (v, v)
                    else:
                        for f in obj.type.feature_names:
                            mn, mx = feature_ranges[obj.type][f]
                            v = state.get(obj, f)
                            feature_ranges[obj.type][f] = (min(mn, v),
                                                           max(mx, v))
        return feature_ranges


@dataclass(frozen=True, eq=False, repr=False)
class _GivenPredicateGrammar(_PredicateGrammar):
    """Enumerates a given set of predicates.
    """
    given_predicates: Set[Predicate]

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for predicate in sorted(self.given_predicates):
            yield (predicate, 1.0)


@dataclass(frozen=True, eq=False, repr=False)
class _ChainPredicateGrammar(_PredicateGrammar):
    """Chains together multiple predicate grammars in sequence.
    """
    base_grammars: Sequence[_PredicateGrammar]

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        return itertools.chain.from_iterable(
            g.enumerate() for g in self.base_grammars)


@dataclass(frozen=True, eq=False, repr=False)
class _SkipGrammar(_PredicateGrammar):
    """A grammar that omits given predicates from being enumerated.
    """
    base_grammar: _PredicateGrammar
    omitted_predicates: Set[Predicate]

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for (predicate, cost) in self.base_grammar.enumerate():
            if predicate in self.omitted_predicates:
                continue
            yield (predicate, cost)


@dataclass(frozen=True, eq=False, repr=False)
class _NegationPredicateGrammarWrapper(_PredicateGrammar):
    """For each x generated by the base grammar, also generates not(x).
    """
    base_grammar: _PredicateGrammar

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for (predicate, cost) in self.base_grammar.enumerate():
            yield (predicate, cost)
            classifier = _NegationClassifier(predicate)
            negated_predicate = Predicate(str(classifier), predicate.types,
                                          classifier)
            yield (negated_predicate, cost)


@dataclass(frozen=True, eq=False, repr=False)
class _ForallPredicateGrammarWrapper(_PredicateGrammar):
    """For each x generated by the base grammar, also generates forall(x).
    """
    base_grammar: _PredicateGrammar

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for (predicate, cost) in self.base_grammar.enumerate():
            yield (predicate, cost)
            classifier = _ForallClassifier(predicate)
            yield (Predicate(str(classifier), [], classifier), cost)
            if predicate.arity >= 2:
                for idx in range(predicate.arity):
                    uff_classifier = _UnaryFreeForallClassifier(predicate, idx)
                    uff_predicate = Predicate(str(uff_classifier),
                                              [predicate.types[idx]],
                                              uff_classifier)
                    yield (uff_predicate, cost)


### Useful for debugging ###

# # Replace these strings with anything you want to exclusively enumerate.
# # Make sure to uncomment return _DebugGrammar(skip_grammar) in
# # _create_grammar and the call to _run_debug_analysis.
# _DEBUG_PREDICATE_STRS = [
#     "((0:obj).wetness<=0.5)",
#     "NOT-((0:obj).wetness<=0.5)",
# ]

# @dataclass(frozen=True, eq=False, repr=False)
# class _DebugGrammar(_PredicateGrammar):
#     """A grammar that generates only predicates in _DEBUG_PREDICATE_STRS.
#     """
#     base_grammar: _PredicateGrammar

#     def generate(self, max_num: int) -> Dict[Predicate, float]:
#         del max_num
#         return super().generate(len(_DEBUG_PREDICATE_STRS))

#     def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
#         for (predicate, cost) in self.base_grammar.enumerate():
#             if str(predicate) in _DEBUG_PREDICATE_STRS:
#                 yield (predicate, cost)


# def _run_debug_analysis(candidates: Dict[Predicate, float],
#                         atom_dataset: List[GroundAtomTrajectory],
#                         initial_predicates: Set[Predicate]) -> None:
#     """Some helpful debugging stuff.
#     """
#     print("All candidates:", sorted(candidates))
#     print("Running learning & scoring with ALL predicates.")
#     all_segments = [seg for traj in atom_dataset
#                     for seg in segment_trajectory(traj)]
#     all_strips_ops, all_partitions = learn_strips_operators(all_segments,
#                                                             verbose=False)
#     all_option_specs = [p.option_spec for p in all_partitions]
#     all_num_tps, all_num_fps, _, all_fp_idxs = \
#         _count_positives_for_ops(all_strips_ops, all_option_specs,
#                                  all_segments)
#     print("TP/FP:", all_num_tps, all_num_fps)
#     print("Running learning & scoring with INITIAL predicates.")
#     pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset,
#                                                        initial_predicates)
#     init_segments = [seg for traj in pruned_atom_data
#                      for seg in segment_trajectory(traj)]
#     assert len(all_segments) == len(init_segments), \
#         "This analysis assumes that segmentation does not change."
#     init_strips_ops, init_partitions = learn_strips_operators(init_segments,
#                                                               verbose=False)
#     init_option_specs = [p.option_spec for p in init_partitions]
#     # Score based on how well the operators fit the data.
#     init_num_tps, init_num_fps, _, init_fp_idxs = \
#         _count_positives_for_ops(init_strips_ops, init_option_specs,
#                                  init_segments)
#     print("TP/FP:", init_num_tps, init_num_fps)
#     # Generally we would expect false positives to go down with the extra
#     # predicates. But we're debugging, and there may be some false positives
#     # that appear with the learned predicates that did not appear initially.
#     all_combined_fps = {idx for fp_idxs in all_fp_idxs for idx in fp_idxs}
#     init_combined_fps = {idx for fp_idxs in init_fp_idxs for idx in fp_idxs}
#     for idx in sorted(all_combined_fps - init_combined_fps):
#         all_segment = all_segments[idx]
#         init_segment = init_segments[idx]
#         assert all_segment.get_option() == init_segment.get_option()
#         print("The following segment is a FP for ALL but not INITIAL.")
#         print("Start of segment:", all_segment.init_atoms)
#         print("Option:", all_segment.get_option())
#         print("Add effects of segment:", all_segment.add_effects)
#         print("Delete effects of segment:", all_segment.delete_effects)
#         print("Here is the operator(s) that it is a FP for:")
#         for i, op in enumerate(all_strips_ops):
#             if idx in all_fp_idxs[i]:
#                 print(op)
#                 print("    Option Spec:", all_option_specs[i])
#                 import ipdb; ipdb.set_trace()

### End debugging ###


def _create_grammar(grammar_name: str, dataset: Dataset,
                    given_predicates: Set[Predicate]) -> _PredicateGrammar:
    if grammar_name == "holding_dummy":
        return _HoldingDummyPredicateGrammar(dataset)
    if grammar_name == "single_feat_ineqs":
        sfi_grammar = _SingleFeatureInequalitiesPredicateGrammar(dataset)
        return _NegationPredicateGrammarWrapper(sfi_grammar)
    if grammar_name == "forall_single_feat_ineqs":
        # We start with the given predicates because we want to allow
        # negated and quantified versions of the given predicates, in
        # addition to negated and quantified versions of new predicates.
        given_grammar = _GivenPredicateGrammar(given_predicates)
        sfi_grammar = _SingleFeatureInequalitiesPredicateGrammar(dataset)
        # This chained grammar has the effect of enumerating first the
        # given predicates, then the single feature inequality ones.
        chained_grammar = _ChainPredicateGrammar([given_grammar, sfi_grammar])
        # For each predicate enumerated by the chained grammar, we also
        # enumerate the negation of that predicate.
        negated_grammar = _NegationPredicateGrammarWrapper(chained_grammar)
        # For each predicate enumerated, we also enumerate foralls for
        # that predicate.
        forall_grammar = _ForallPredicateGrammarWrapper(negated_grammar)
        # Finally, we don't actually need to enumerate the given predicates
        # because we already have them in the initial predicate set,
        # so we just filter them out from actually being enumerated.
        # But remember that we do want to enumerate their negations
        # and foralls, which is why they're included originally.
        skip_grammar = _SkipGrammar(forall_grammar, given_predicates)
        return skip_grammar
        # For debugging, uncomment this, and comment out the previous line.
        # return _DebugGrammar(skip_grammar)
    raise NotImplementedError(f"Unknown grammar name: {grammar_name}.")


################################################################################
#                                 Approach                                     #
################################################################################

class GrammarSearchInventionApproach(NSRTLearningApproach):
    """An approach that invents predicates by searching over candidate sets,
    with the candidates proposed from a grammar.
    """
    def __init__(self, simulator: Callable[[State, Action], State],
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box) -> None:
        super().__init__(simulator, initial_predicates, initial_options,
                         types, action_space)
        self._learned_predicates: Set[Predicate] = set()
        self._num_inventions = 0

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    def learn_from_offline_dataset(self, dataset: Dataset,
                                   train_tasks: List[Task]) -> None:
        self._dataset.extend(dataset)
        del dataset
        # Generate a candidate set of predicates.
        print("Generating candidate predicates...")
        grammar = _create_grammar(CFG.grammar_search_grammar_name,
                                  self._dataset, self._initial_predicates)
        candidates = grammar.generate(max_num=CFG.grammar_search_max_predicates)
        print(f"Done: created {len(candidates)} candidates:")
        for predicate in candidates:
            print(predicate)
        # Apply the candidate predicates to the data.
        print("Applying predicates to data...")
        atom_dataset = utils.create_ground_atom_dataset(
            self._dataset, set(candidates) | self._initial_predicates)
        print("Done.")
        # Useful for debugging in combination with _DebugGrammar.
        # _run_debug_analysis(candidates, atom_dataset,
        #                     self._initial_predicates)
        # Select a subset of the candidates to keep.
        print("Selecting a subset...")
        self._learned_predicates = self._select_predicates_to_keep(
            candidates, atom_dataset)
        print("Done.")
        # Finally, learn NSRTs via superclass, using all the kept predicates.
        self._learn_nsrts()

    def _select_predicates_to_keep(self, candidates: Dict[Predicate, float],
                                   atom_dataset: List[GroundAtomTrajectory]
                                   ) -> Set[Predicate]:
        # Perform a greedy search over predicate sets.

        # The heuristic is where the action happens...
        def _heuristic(s: FrozenSet[Predicate]) -> float:
            print("Scoring predicates:", s)
            # Relearn operators with the current predicates.
            kept_preds = s | self._initial_predicates
            pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset,
                                                               kept_preds)
            segments = [seg for traj in pruned_atom_data
                        for seg in segment_trajectory(traj)]
            strips_ops, partitions = learn_strips_operators(segments,
                                                            verbose=False)
            option_specs = [p.option_spec for p in partitions]

            # Score based on how well the operators fit the data.
            num_true_positives, num_false_positives, _, _ = \
                _count_positives_for_ops(strips_ops, option_specs, segments)
            # Also add a size penalty.
            op_size = _get_operators_size(strips_ops)
            # Also add a penalty based on predicate complexity.
            pred_complexity = sum(candidates[p] for p in s)
            total_score = \
                CFG.grammar_search_false_pos_weight * num_false_positives + \
                CFG.grammar_search_true_pos_weight * (-num_true_positives) + \
                CFG.grammar_search_size_weight * op_size + \
                CFG.grammar_search_pred_complexity_weight * pred_complexity
            # Useful for debugging:
            # print("TP/FP/S/C/Total:", num_true_positives, num_false_positives,
            #       op_size, pred_complexity, total_score)
            # Lower is better.
            return total_score

        # There are no goal states for this search; run until exhausted.
        def _check_goal(s: FrozenSet[Predicate]) -> bool:
            del s  # unused
            return False

        if CFG.grammar_search_direction == "largetosmall":
            # Successively consider smaller predicate sets.
            def _get_successors(s: FrozenSet[Predicate]
                    ) -> Iterator[Tuple[None, FrozenSet[Predicate], float]]:
                for predicate in sorted(s):  # sorting for determinism
                    # Actions not needed. Frozensets for hashing.
                    # The cost of 1. is irrelevant because we're doing GBFS
                    # and not A* (because we don't care about the path).
                    yield (None, frozenset(s - {predicate}), 1.)

            # Start the search with all of the candidates.
            init = frozenset(candidates)
        else:
            assert CFG.grammar_search_direction == "smalltolarge"
            # Successively consider larger predicate sets.
            def _get_successors(s: FrozenSet[Predicate]
                    ) -> Iterator[Tuple[None, FrozenSet[Predicate], float]]:
                for predicate in sorted(set(candidates) - s):  # determinism
                    # Actions not needed. Frozensets for hashing.
                    # The cost of 1. is irrelevant because we're doing GBFS
                    # and not A* (because we don't care about the path).
                    yield (None, frozenset(s | {predicate}), 1.)

            # Start the search with no candidates.
            init = frozenset()

        # Greedy best first search.
        path, _ = utils.run_gbfs(
            init, _check_goal, _get_successors, _heuristic,
            max_evals=CFG.grammar_search_max_evals)
        kept_predicates = path[-1]

        print(f"Selected {len(kept_predicates)} predicates out of "
              f"{len(candidates)} candidates:")
        for pred in kept_predicates:
            print(pred)

        return set(kept_predicates)


def _count_positives_for_ops(strips_ops: List[STRIPSOperator],
                             option_specs: List[OptionSpec],
                             segments: List[Segment],
                             ) -> Tuple[int, int,
                                        List[Set[int]], List[Set[int]]]:
    """Returns num true positives, num false positives, and for each strips op,
    lists of segment indices that contribute true or false positives.

    The lists of segment indices are useful only for debugging; they are
    otherwise redundant with num_true_positives/num_false_positives.
    """
    assert len(strips_ops) == len(option_specs)
    num_true_positives = 0
    num_false_positives = 0
    # The following two lists are just useful for debugging with
    # _run_debug_analysis.
    true_positive_idxs : List[Set[int]] = [set() for _ in strips_ops]
    false_positive_idxs : List[Set[int]] = [set() for _ in strips_ops]
    for idx, segment in enumerate(segments):
        objects = set(segment.states[0])
        segment_option = segment.get_option()
        option_objects = segment_option.objects
        covered_by_some_op = False
        # Ground only the operators with a matching option spec.
        for op_idx, (op, option_spec) in enumerate(zip(strips_ops,
                                                       option_specs)):
            # If the parameterized options are different, not relevant.
            if option_spec[0] != segment_option.parent:
                continue
            option_vars = option_spec[1]
            assert len(option_vars) == len(option_objects)
            option_var_to_obj = dict(zip(option_vars, option_objects))
            # We want to get all ground operators whose corresponding
            # substitution is consistent with the option vars for this
            # segment. So, determine all of the operator variables
            # that are not in the option vars, and consider all
            # groundings of them.
            for ground_op in utils.all_ground_operators_given_partial(
                op, objects, option_var_to_obj):
                # Check the ground_op against the segment.
                if not ground_op.preconditions.issubset(
                    segment.init_atoms):
                    continue
                if ground_op.add_effects == segment.add_effects and \
                   ground_op.delete_effects == segment.delete_effects:
                    covered_by_some_op = True
                    true_positive_idxs[op_idx].add(idx)
                else:
                    false_positive_idxs[op_idx].add(idx)
                    num_false_positives += 1
        if covered_by_some_op:
            num_true_positives += 1
    return num_true_positives, num_false_positives, \
        true_positive_idxs, false_positive_idxs


def _get_operators_size(strips_ops: List[STRIPSOperator]) -> int:
    size = 0
    for op in strips_ops:
        size += len(op.parameters) + len(op.preconditions) + \
                len(op.add_effects) + len(op.delete_effects)
    return size
