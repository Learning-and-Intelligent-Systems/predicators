"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar."""

from __future__ import annotations

import abc
import itertools
import logging
from dataclasses import dataclass, field
from functools import cached_property
from operator import le
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Sequence, \
    Set, Tuple
from collections import namedtuple

import numpy as np
from gym.spaces import Box
from scipy.stats import kstest
from sklearn.mixture import GaussianMixture as GMM

from predicators import utils
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.predicate_search_score_functions import \
    _PredicateSearchScoreFunction, create_score_function
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtomTrajectory, Object, \
    ParameterizedOption, Predicate, Segment, State, Task, Type

################################################################################
#                          Programmatic classifiers                            #
################################################################################


def _create_grammar(dataset: Dataset,
                    given_predicates: Set[Predicate]) -> _PredicateGrammar:
    # We start with considering various ways to split either single or
    # two feature values across our dataset.
    grammar: _PredicateGrammar = _SingleFeatureInequalitiesPredicateGrammar(
        dataset)
    if CFG.grammar_search_grammar_use_diff_features:
        diff_grammar = _FeatureDiffInequalitiesPredicateGrammar(dataset)
        grammar = _ChainPredicateGrammar([grammar, diff_grammar],
                                         alternate=True)
    # We next optionally add in the given predicates because we want to allow
    # negated and quantified versions of the given predicates, in
    # addition to negated and quantified versions of new predicates.
    # The chained grammar has the effect of enumerating first the
    # given predicates, then the single feature inequality ones.
    if CFG.grammar_search_grammar_includes_givens:
        given_grammar = _GivenPredicateGrammar(given_predicates)
        grammar = _ChainPredicateGrammar([given_grammar, grammar])
    # Now, the grammar will undergo a series of transformations.
    # For each predicate enumerated by the grammar, we also
    # enumerate the negation of that predicate.
    grammar = _NegationPredicateGrammarWrapper(grammar)
    # For each predicate enumerated, we also optionally enumerate foralls
    # for that predicate, along with appropriate negations.
    if CFG.grammar_search_grammar_includes_foralls:
        grammar = _ForallPredicateGrammarWrapper(grammar)
    # Prune proposed predicates by checking if they are equivalent to
    # any already-generated predicates with respect to the dataset.
    # Note that we want to do this before the skip grammar below,
    # because if any predicates are equivalent to the given predicates,
    # we would not want to generate them. Don't do this if we're using
    # DebugGrammar, because we don't want to prune things that are in there.
    if not CFG.grammar_search_use_handcoded_debug_grammar:
        grammar = _PrunedGrammar(dataset, grammar)
    # We don't actually need to enumerate the given predicates
    # because we already have them in the initial predicate set,
    # so we just filter them out from actually being enumerated.
    # But remember that we do want to enumerate their negations
    # and foralls, which is why they're included originally.
    grammar = _SkipGrammar(grammar, given_predicates)
    # If we're using the DebugGrammar, filter out all other predicates.
    if CFG.grammar_search_use_handcoded_debug_grammar:
        grammar = _DebugGrammar(grammar)
    # We're done! Return the final grammar.
    return grammar


class _ProgrammaticClassifier(abc.ABC):
    """A classifier implemented as an arbitrary program."""

    @abc.abstractmethod
    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        """All programmatic classifiers are functions of state and objects.

        The objects are the predicate arguments.
        """
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def pretty_str(self) -> Tuple[str, str]:
        """Display the classifier in a nice human-readable format.

        Returns a tuple of (variables string, body string).
        """
        raise NotImplementedError("Override me!")


class _NullaryClassifier(_ProgrammaticClassifier):
    """A classifier on zero objects."""

    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        assert len(o) == 0
        return self._classify_state(s)

    @abc.abstractmethod
    def _classify_state(self, s: State) -> bool:
        raise NotImplementedError("Override me!")


class _UnaryClassifier(_ProgrammaticClassifier):
    """A classifier on one object."""

    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        assert len(o) == 1
        return self._classify_object(s, o[0])

    @abc.abstractmethod
    def _classify_object(self, s: State, obj: Object) -> bool:
        raise NotImplementedError("Override me!")


class _BinaryClassifier(_ProgrammaticClassifier):
    """A classifier on two objects."""

    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        assert len(o) == 2
        o0, o1 = o
        return self._classify_object(s, o0, o1)

    @abc.abstractmethod
    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        raise NotImplementedError("Override me!")


@dataclass(frozen=True, eq=False, repr=False)
class _SingleAttributeCompareClassifier(_UnaryClassifier):
    """Compare a single feature value with a constant value."""
    object_index: int
    object_type: Type
    attribute_name: str
    constant: float
    constant_idx: int
    compare: Callable[[float, float], bool]
    compare_str: str

    def _classify_object(self, s: State, obj: Object) -> bool:
        assert obj.type == self.object_type
        return self.compare(s.get(obj, self.attribute_name), self.constant)

    def __str__(self) -> str:
        return (
            f"(({self.object_index}:{self.object_type.name})."
            f"{self.attribute_name}{self.compare_str}[idx {self.constant_idx}]"
            f"{self.constant:.3})")

    def pretty_str(self) -> Tuple[str, str]:
        name = CFG.grammar_search_classifier_pretty_str_names[
            self.object_index]
        vars_str = f"{name}:{self.object_type.name}"
        body_str = (f"({name}.{self.attribute_name} "
                    f"{self.compare_str} {self.constant:.3})")
        return vars_str, body_str


@dataclass(frozen=True, eq=False, repr=False)
class _AttributeDiffCompareClassifier(_BinaryClassifier):
    """Compare the difference between two feature values with a constant
    value."""
    object1_index: int
    object1_type: Type
    attribute1_name: str
    object2_index: int
    object2_type: Type
    attribute2_name: str
    constant: float
    constant_idx: int
    compare: Callable[[float, float], bool]
    compare_str: str

    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        assert obj1.type == self.object1_type
        assert obj2.type == self.object2_type
        return self.compare(
            abs(
                s.get(obj1, self.attribute1_name) -
                s.get(obj2, self.attribute2_name)), self.constant)

    def __str__(self) -> str:
        return (f"(|({self.object1_index}:{self.object1_type.name})."
                f"{self.attribute1_name} - ({self.object2_index}:"
                f"{self.object2_type.name}).{self.attribute2_name}|"
                f"{self.compare_str}[idx {self.constant_idx}]"
                f"{self.constant:.3})")

    def pretty_str(self) -> Tuple[str, str]:
        name1 = CFG.grammar_search_classifier_pretty_str_names[
            self.object1_index]
        name2 = CFG.grammar_search_classifier_pretty_str_names[
            self.object2_index]
        vars_str = (f"{name1}:{self.object1_type.name}, "
                    f"{name2}:{self.object2_type.name}")
        body_str = (f"(|{name1}.{self.attribute1_name} - "
                    f"{name2}.{self.attribute2_name}| "
                    f"{self.compare_str} {self.constant:.3})")
        return vars_str, body_str


@dataclass(frozen=True, eq=False, repr=False)
class _NegationClassifier(_ProgrammaticClassifier):
    """Negate a given classifier."""
    body: Predicate

    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        return not self.body.holds(s, o)

    def __str__(self) -> str:
        return f"NOT-{self.body}"

    def pretty_str(self) -> Tuple[str, str]:
        vars_str, body_str = self.body.pretty_str()
        return vars_str, f"¬{body_str}"


@dataclass(frozen=True, eq=False, repr=False)
class _ForallClassifier(_NullaryClassifier):
    """Apply a predicate to all objects."""
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

    def pretty_str(self) -> Tuple[str, str]:
        types = self.body.types
        _, body_str = self.body.pretty_str()
        head = ", ".join(
            f"{CFG.grammar_search_classifier_pretty_str_names[i]}:{t.name}"
            for i, t in enumerate(types))
        vars_str = ""  # there are no variables
        return vars_str, f"(∀ {head} . {body_str})"


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
        return [
            t for i, t in enumerate(self.body.types)
            if i != self.free_variable_idx
        ]

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

    def pretty_str(self) -> Tuple[str, str]:
        types = self.body.types
        _, body_str = self.body.pretty_str()
        head = ", ".join(
            f"{CFG.grammar_search_classifier_pretty_str_names[i]}:{t.name}"
            for i, t in enumerate(types) if i != self.free_variable_idx)
        name = CFG.grammar_search_classifier_pretty_str_names[
            self.free_variable_idx]
        vars_str = f"{name}:{types[self.free_variable_idx].name}"
        return vars_str, f"(∀ {head} . {body_str})"


################################################################################
#                             Predicate grammars                               #
################################################################################


@dataclass(frozen=True, eq=False, repr=False)
class _PredicateGrammar(abc.ABC):
    """A grammar for generating predicate candidates."""

    def generate(self, max_num: int) -> Dict[Predicate, float]:
        """Generate candidate predicates from the grammar.

        The dict values are costs, e.g., negative log prior probability
        for the predicate in a PCFG.
        """
        candidates: Dict[Predicate, float] = {}
        if max_num == 0:
            return candidates
        assert max_num > 0
        for candidate, cost in self.enumerate():
            assert cost > 0
            if cost >= CFG.grammar_search_predicate_cost_upper_bound:
                break
            candidates[candidate] = cost
            if len(candidates) == max_num:
                break
        return candidates

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        """Iterate over candidate predicates from less to more cost."""
        raise NotImplementedError("Override me!")


_DEBUG_PREDICATE_PREFIXES = {
    "tools": [
        "NOT-((0:robot).fingers<=[idx 0]0.5)",  # HandEmpty
        "NOT-((0:screw).is_held<=[idx 0]0.5)",  # HoldingScrew
        "NOT-((0:screwdriver).is_held<=[idx 0]0.5)",  # HoldingScrewdriver
        "NOT-((0:nail).is_held<=[idx 0]0.5)",  # HoldingNail
        "NOT-((0:hammer).is_held<=[idx 0]0.5)",  # HoldingHammer
        "NOT-((0:bolt).is_held<=[idx 0]0.5)",  # HoldingBolt
        "NOT-((0:wrench).is_held<=[idx 0]0.5)",  # HoldingWrench
        "((0:screwdriver).size<=[idx 0]",  # ScrewdriverGraspable
        "((0:hammer).size<=[idx 0]",  # HammerGraspable
    ],
    "painting": [
        "NOT-((0:robot).fingers<=[idx 0]0.5)",  # GripperOpen
        "((0:obj).pose_y<=[idx 2]",  # OnTable
        "NOT-((0:obj).grasp<=[idx 0]0.5)",  # HoldingTop
        "((0:obj).grasp<=[idx 1]0.25)",  # HoldingSide
        "NOT-((0:obj).held<=[idx 0]0.5)",  # Holding
        "NOT-((0:obj).wetness<=[idx 0]0.5)",  # IsWet
        "((0:obj).wetness<=[idx 0]0.5)",  # IsDry
        "NOT-((0:obj).dirtiness<=[idx 0]",  # IsDirty
        "((0:obj).dirtiness<=[idx 0]",  # IsClean
        "Forall[0:lid].[NOT-((0:lid).is_open<=[idx 0]0.5)(0)]",  # AllLidsOpen
        # "NOT-((0:lid).is_open<=[idx 0]0.5)",  # LidOpen (doesn't help)
    ],
    "cover": [
        "NOT-((0:block).grasp<=[idx 0]",  # Holding
        "Forall[0:block].[((0:block).grasp<=[idx 0]",  # HandEmpty
    ],
    "cover_regrasp": [
        "NOT-((0:block).grasp<=[idx 0]",  # Holding
        "Forall[0:block].[((0:block).grasp<=[idx 0]",  # HandEmpty
    ],
    "cover_multistep_options": [
        "NOT-((0:block).grasp<=[idx 0]",  # Holding
        "Forall[0:block].[((0:block).grasp<=[idx 0]",  # HandEmpty
    ],
    "blocks": [
        "NOT-((0:robot).fingers<=[idx 0]",  # GripperOpen
        "Forall[0:block].[NOT-On(0,1)]",  # Clear
        "NOT-((0:block).pose_z<=[idx 0]",  # Holding
    ],
    "repeated_nextto_single_option": [
        "(|(0:dot).x - (1:robot).x|<=[idx 7]6.25)",  # NextTo
    ],
    "unittest": [
        "((0:robot).hand<=[idx 0]0.65)", "((0:block).grasp<=[idx 0]0.0)",
        "NOT-Forall[0:block].[((0:block).width<=[idx 0]0.085)(0)]"
    ],
}


@dataclass(frozen=True, eq=False, repr=False)
class _DebugGrammar(_PredicateGrammar):
    """A grammar that generates only predicates starting with some string in
    _DEBUG_PREDICATE_PREFIXES[CFG.env]."""
    base_grammar: _PredicateGrammar

    def generate(self, max_num: int) -> Dict[Predicate, float]:
        del max_num
        env_name = (CFG.env if not CFG.env.startswith("pybullet") else
                    CFG.env[CFG.env.index("_") + 1:])
        expected_len = len(_DEBUG_PREDICATE_PREFIXES[env_name])
        result = super().generate(expected_len)
        assert len(result) == expected_len
        return result

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        env_name = (CFG.env if not CFG.env.startswith("pybullet") else
                    CFG.env[CFG.env.index("_") + 1:])
        for (predicate, cost) in self.base_grammar.enumerate():
            if any(
                    str(predicate).startswith(debug_str)
                    for debug_str in _DEBUG_PREDICATE_PREFIXES[env_name]):
                yield (predicate, cost)


@dataclass(frozen=True, eq=False, repr=False)
class _DataBasedPredicateGrammar(_PredicateGrammar):
    """A predicate grammar that uses a dataset."""
    dataset: Dataset

    @cached_property
    def types(self) -> Set[Type]:
        """Infer types from the dataset."""
        types: Set[Type] = set()
        for traj in self.dataset.trajectories:
            types.update(o.type for o in traj.states[0])
        return types

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        """Iterate over candidate predicates in an arbitrary order."""
        raise NotImplementedError("Override me!")


def _halving_constant_generator(
        lo: float,
        hi: float,
        cost: float = 1.0) -> Iterator[Tuple[float, float]]:
    """The second element of the tuple is a cost. For example, the first
    several tuples yielded will be:

    (0.5, 1.0), (0.25, 2.0), (0.75, 2.0), (0.125, 3.0), ...
    """
    mid = (hi + lo) / 2.
    yield (mid, cost)
    left_gen = _halving_constant_generator(lo, mid, cost + 1)
    right_gen = _halving_constant_generator(mid, hi, cost + 1)
    for l, r in zip(left_gen, right_gen):
        yield l
        yield r


@dataclass(frozen=True, eq=False, repr=False)
class _SingleFeatureInequalitiesPredicateGrammar(_DataBasedPredicateGrammar):
    """Generates features of the form "0.feature >= c" or "0.feature <= c"."""

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # Get ranges of feature values from data.
        feature_ranges = self._get_feature_ranges()
        # Edge case: if there are no features at all, return immediately.
        if not any(r for r in feature_ranges.values()):
            return
        # 0.5, 0.25, 0.75, 0.125, 0.375, ...
        constant_generator = _halving_constant_generator(0.0, 1.0)
        for constant_idx, (constant, cost) in enumerate(constant_generator):
            for t in sorted(self.types):
                for f in t.feature_names:
                    lb, ub = feature_ranges[t][f]
                    # Optimization: if lb == ub, there is no variation
                    # among this feature, so there's no point in trying to
                    # learn a classifier with it. So, skip the feature.
                    if abs(lb - ub) < 1e-6:
                        continue
                    # Scale the constant by the feature range.
                    k = constant * (ub - lb) + lb
                    # Only need one of (ge, le) because we can use negations
                    # to get the other (modulo equality, which we shouldn't
                    # rely on anyway because of precision issues).
                    comp, comp_str = le, "<="
                    classifier = _SingleAttributeCompareClassifier(
                        0, t, f, k, constant_idx, comp, comp_str)
                    name = str(classifier)
                    types = [t]
                    pred = Predicate(name, types, classifier)
                    assert pred.arity == 1
                    yield (pred, 1 + cost)  # cost = arity + cost from constant

    def _get_feature_ranges(
            self) -> Dict[Type, Dict[str, Tuple[float, float]]]:
        feature_ranges: Dict[Type, Dict[str, Tuple[float, float]]] = {}
        for traj in self.dataset.trajectories:
            for state in traj.states:
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
                            feature_ranges[obj.type][f] = (min(mn,
                                                               v), max(mx, v))
        return feature_ranges


@dataclass(frozen=True, eq=False, repr=False)
class _FeatureDiffInequalitiesPredicateGrammar(
        _SingleFeatureInequalitiesPredicateGrammar):
    """Generates features of the form "|0.feature - 1.feature| <= c"."""

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # Get ranges of feature values from data.
        feature_ranges = self._get_feature_ranges()
        # Edge case: if there are no features at all, return immediately.
        if not any(r for r in feature_ranges.values()):
            return
        # 0.5, 0.25, 0.75, 0.125, 0.375, ...
        constant_generator = _halving_constant_generator(0.0, 1.0)
        for constant_idx, (constant, cost) in enumerate(constant_generator):
            for (t1, t2) in itertools.combinations_with_replacement(
                    sorted(self.types), 2):
                for f1 in t1.feature_names:
                    for f2 in t2.feature_names:
                        # To create our classifier, we need to leverage the
                        # upper and lower bounds of its features.
                        # First, we extract these and move on if these
                        # bounds are relatively close together.
                        lb1, ub1 = feature_ranges[t1][f1]
                        if abs(lb1 - ub1) < 1e-6:
                            continue
                        lb2, ub2 = feature_ranges[t2][f2]
                        if abs(lb2 - ub2) < 1e-6:
                            continue
                        # Now, we must compute the upper and lower bounds of
                        # the expression |t1.f1 - t2.f2|. If the intervals
                        # [lb1, ub1] and [lb2, ub2] overlap, then the lower
                        # bound of the expression is just 0. Otherwise, if
                        # lb2 > ub1, the lower bound is |ub1 - lb2|, and if
                        # ub2 < lb1, the lower bound is |lb1 - ub2|.
                        if utils.f_range_intersection(lb1, ub1, lb2, ub2):
                            lb = 0.0
                        else:
                            lb = min(abs(lb2 - ub1), abs(lb1 - ub2))
                        # The upper bound for the expression can be
                        # computed in a similar fashion.
                        ub = max(abs(ub2 - lb1), abs(ub1 - lb2))

                        # Scale the constant by the correct range.
                        k = constant * (ub - lb) + lb
                        # Create classifier.
                        comp, comp_str = le, "<="
                        diff_classifier = _AttributeDiffCompareClassifier(
                            0, t1, f1, 1, t2, f2, k, constant_idx, comp,
                            comp_str)
                        name = str(diff_classifier)
                        types = [t1, t2]
                        pred = Predicate(name, types, diff_classifier)
                        assert pred.arity == 2
                        yield (pred, 2 + cost
                               )  # cost = arity + cost from constant


@dataclass(frozen=True, eq=False, repr=False)
class _GivenPredicateGrammar(_PredicateGrammar):
    """Enumerates a given set of predicates."""
    given_predicates: Set[Predicate]

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for predicate in sorted(self.given_predicates):
            yield (predicate, predicate.arity + 1)


@dataclass(frozen=True, eq=False, repr=False)
class _ChainPredicateGrammar(_PredicateGrammar):
    """Chains together multiple predicate grammars in sequence."""
    base_grammars: Sequence[_PredicateGrammar]
    alternate: bool = False

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        if not self.alternate:
            return itertools.chain.from_iterable(g.enumerate()
                                                 for g in self.base_grammars)
        return utils.roundrobin([g.enumerate() for g in self.base_grammars])


@dataclass(frozen=True, eq=False, repr=False)
class _SkipGrammar(_PredicateGrammar):
    """A grammar that omits given predicates from being enumerated."""
    base_grammar: _PredicateGrammar
    omitted_predicates: Set[Predicate]

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for (predicate, cost) in self.base_grammar.enumerate():
            if predicate in self.omitted_predicates:
                continue
            # No change to costs when skipping.
            yield (predicate, cost)


@dataclass(frozen=True, eq=False, repr=False)
class _PrunedGrammar(_DataBasedPredicateGrammar):
    """A grammar that prunes redundant predicates."""
    base_grammar: _PredicateGrammar
    _state_sequences: List[List[State]] = field(init=False,
                                                default_factory=list)

    def __post_init__(self) -> None:
        if CFG.segmenter != "atom_changes":
            # If the segmenter doesn't depend on atoms, we can be very
            # efficient during pruning by pre-computing the segments.
            # Then, we only need to care about the initial and final
            # states in each segment, which we store into
            # self._state_sequence.
            for traj in self.dataset.trajectories:
                # The init_atoms and final_atoms are not used.
                seg_traj = segment_trajectory(traj, predicates=set())
                state_seq = utils.segment_trajectory_to_state_sequence(
                    seg_traj)
                self._state_sequences.append(state_seq)

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # Predicates are identified based on their evaluation across
        # all states in the dataset.
        seen: Dict[FrozenSet[Tuple[int, int, FrozenSet[Tuple[Object, ...]]]],
                   Predicate] = {}  # keys are from _get_predicate_identifier()
        for (predicate, cost) in self.base_grammar.enumerate():
            if cost >= CFG.grammar_search_predicate_cost_upper_bound:
                return
            pred_id = self._get_predicate_identifier(predicate)
            if pred_id in seen:
                logging.debug(f"Pruning {predicate} b/c equal to "
                              f"{seen[pred_id]}")
                continue
            # Found a new predicate.
            seen[pred_id] = predicate
            yield (predicate, cost)

    def _get_predicate_identifier(
        self, predicate: Predicate
    ) -> FrozenSet[Tuple[int, int, FrozenSet[Tuple[Object, ...]]]]:
        """Returns frozenset identifiers for each data point."""
        raw_identifiers = set()
        if CFG.segmenter == "atom_changes":
            # Get atoms for this predicate alone on the dataset, and then
            # go through the entire dataset.
            atom_dataset = utils.create_ground_atom_dataset(
                self.dataset.trajectories, {predicate})
            for traj_idx, (_, atom_traj) in enumerate(atom_dataset):
                for t, atoms in enumerate(atom_traj):
                    atom_args = frozenset(tuple(a.objects) for a in atoms)
                    raw_identifiers.add((traj_idx, t, atom_args))
        else:
            # This list may expand in the future if we add other segmentation
            # methods, but leaving this assertion in as a safeguard anyway.
            assert CFG.segmenter in ("option_changes", "contacts")
            # Make use of the pre-computed segment-level state sequences.
            for traj_idx, state_seq in enumerate(self._state_sequences):
                for t, state in enumerate(state_seq):
                    atoms = utils.abstract(state, {predicate})
                    atom_args = frozenset(tuple(a.objects) for a in atoms)
                    raw_identifiers.add((traj_idx, t, atom_args))
        return frozenset(raw_identifiers)


@dataclass(frozen=True, eq=False, repr=False)
class _NegationPredicateGrammarWrapper(_PredicateGrammar):
    """For each x generated by the base grammar, also generates not(x)."""
    base_grammar: _PredicateGrammar

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for (predicate, cost) in self.base_grammar.enumerate():
            yield (predicate, cost)
            classifier = _NegationClassifier(predicate)
            negated_predicate = Predicate(str(classifier), predicate.types,
                                          classifier)
            # No change to costs when negating.
            yield (negated_predicate, cost)


@dataclass(frozen=True, eq=False, repr=False)
class _ForallPredicateGrammarWrapper(_PredicateGrammar):
    """For each x generated by the base grammar, also generates forall(x) and
    the negation not-forall(x).

    If x has arity at least 2, also generates UnaryFreeForallClassifiers
    over x, along with negations.
    """
    base_grammar: _PredicateGrammar

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for (predicate, cost) in self.base_grammar.enumerate():
            yield (predicate, cost)
            if predicate.arity == 0:
                continue
            # Generate Forall(x)
            forall_classifier = _ForallClassifier(predicate)
            forall_predicate = Predicate(str(forall_classifier), [],
                                         forall_classifier)
            assert forall_predicate.arity == 0
            yield (forall_predicate, cost + 1)  # add arity + 1 to cost
            # Generate NOT-Forall(x)
            notforall_classifier = _NegationClassifier(forall_predicate)
            notforall_predicate = Predicate(str(notforall_classifier),
                                            forall_predicate.types,
                                            notforall_classifier)
            assert notforall_predicate.arity == 0
            yield (notforall_predicate, cost + 1)  # add arity + 1 to cost
            # Generate UFFs
            if predicate.arity >= 2:
                for idx in range(predicate.arity):
                    # Positive UFF
                    uff_classifier = _UnaryFreeForallClassifier(predicate, idx)
                    uff_predicate = Predicate(str(uff_classifier),
                                              [predicate.types[idx]],
                                              uff_classifier)
                    assert uff_predicate.arity == 1
                    yield (uff_predicate, cost + 2)  # add arity + 1 to cost
                    # Negated UFF
                    notuff_classifier = _NegationClassifier(uff_predicate)
                    notuff_predicate = Predicate(str(notuff_classifier),
                                                 uff_predicate.types,
                                                 notuff_classifier)
                    assert notuff_predicate.arity == 1
                    yield (notuff_predicate, cost + 2)  # add arity + 1 to cost


################################################################################
#                                 Approach                                     #
################################################################################


class GrammarSearchInventionApproach(NSRTLearningApproach):
    """An approach that invents predicates by searching over candidate sets,
    with the candidates proposed from a grammar."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._learned_predicates: Set[Predicate] = set()
        self._num_inventions = 0

    @classmethod
    def get_name(cls) -> str:
        return "grammar_search_invention"

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Generate a candidate set of predicates.
        logging.info("Generating candidate predicates...")
        grammar = _create_grammar(dataset, self._initial_predicates)
        candidates = grammar.generate(
            max_num=CFG.grammar_search_max_predicates)
        logging.info(f"Done: created {len(candidates)} candidates:")
        for predicate, cost in candidates.items():
            logging.info(f"{predicate} {cost}")
        # Apply the candidate predicates to the data.
        logging.info("Applying predicates to data...")
        atom_dataset = utils.create_ground_atom_dataset(
            dataset.trajectories,
            set(candidates) | self._initial_predicates)
        logging.info("Done.")
        # Select a subset of the candidates to keep.
        logging.info("Selecting a subset...")
        if CFG.grammar_search_pred_selection_approach == "score_optimization":
            # Create the score function that will be used to guide search.
            score_function = create_score_function(
                CFG.grammar_search_score_function, self._initial_predicates,
                atom_dataset, candidates, self._train_tasks)
            self._learned_predicates = \
                self._select_predicates_by_score_hillclimbing(
                candidates, score_function, self._initial_predicates,
                atom_dataset, self._train_tasks)
        elif CFG.grammar_search_pred_selection_approach == "clustering":
            # self._learned_predicates = self._select_predicates_by_clustering(
            #     candidates, self._initial_predicates, dataset, atom_dataset)
            self._learned_predicates = self._select_predicates_and_learn_operators_by_clustering(
                candidates, self._initial_predicates, dataset, atom_dataset
            )
        logging.info("Done.")
        # Finally, learn NSRTs via superclass, using all the kept predicates.
        annotations = None
        if dataset.has_annotations:
            annotations = dataset.annotations
        self._learn_nsrts(dataset.trajectories,
                          online_learning_cycle=None,
                          annotations=annotations)

    def _select_predicates_by_score_hillclimbing(
            self, candidates: Dict[Predicate, float],
            score_function: _PredicateSearchScoreFunction,
            initial_predicates: Set[Predicate],
            atom_dataset: List[GroundAtomTrajectory],
            train_tasks: List[Task]) -> Set[Predicate]:
        """Perform a greedy search over predicate sets."""

        # There are no goal states for this search; run until exhausted.
        def _check_goal(s: FrozenSet[Predicate]) -> bool:
            del s  # unused
            return False

        # Successively consider larger predicate sets.
        def _get_successors(
            s: FrozenSet[Predicate]
        ) -> Iterator[Tuple[None, FrozenSet[Predicate], float]]:
            for predicate in sorted(set(candidates) - s):  # determinism
                # Actions not needed. Frozensets for hashing. The cost of
                # 1.0 is irrelevant because we're doing GBFS / hill
                # climbing and not A* (because we don't care about the
                # path).
                yield (None, frozenset(s | {predicate}), 1.0)

        # Start the search with no candidates.
        init: FrozenSet[Predicate] = frozenset()

        # Greedy local hill climbing search.
        if CFG.grammar_search_search_algorithm == "hill_climbing":
            path, _, heuristics = utils.run_hill_climbing(
                init,
                _check_goal,
                _get_successors,
                score_function.evaluate,
                enforced_depth=CFG.grammar_search_hill_climbing_depth,
                parallelize=CFG.grammar_search_parallelize_hill_climbing)
            logging.info("\nHill climbing summary:")
            for i in range(1, len(path)):
                new_additions = path[i] - path[i - 1]
                assert len(new_additions) == 1
                new_addition = next(iter(new_additions))
                h = heuristics[i]
                prev_h = heuristics[i - 1]
                logging.info(f"\tOn step {i}, added {new_addition}, with "
                             f"heuristic {h:.3f} (an improvement of "
                             f"{prev_h - h:.3f} over the previous step)")
        elif CFG.grammar_search_search_algorithm == "gbfs":
            path, _ = utils.run_gbfs(
                init,
                _check_goal,
                _get_successors,
                score_function.evaluate,
                max_evals=CFG.grammar_search_gbfs_num_evals)
        else:
            raise NotImplementedError(
                "Unrecognized grammar_search_search_algorithm: "
                f"{CFG.grammar_search_search_algorithm}.")
        kept_predicates = path[-1]
        # The total number of predicate sets evaluated is just the
        # ((number of candidates selected) + 1) * total number of candidates.
        # However, since 'path' always has length one more than the
        # number of selected candidates (since it evaluates the empty
        # predicate set first), we can just compute it as below.
        assert self._metrics.get("total_num_predicate_evaluations") is None
        self._metrics["total_num_predicate_evaluations"] = len(path) * len(
            candidates)

        # Filter out predicates that don't appear in some operator
        # preconditions.
        logging.info("\nFiltering out predicates that don't appear in "
                     "preconditions...")
        preds = kept_predicates | initial_predicates
        pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, preds)
        segmented_trajs = [
            segment_trajectory(ll_traj, set(preds), atom_seq=atom_seq)
            for (ll_traj, atom_seq) in pruned_atom_data
        ]
        low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        preds_in_preconds = set()
        for pnad in learn_strips_operators(low_level_trajs,
                                           train_tasks,
                                           set(kept_predicates
                                               | initial_predicates),
                                           segmented_trajs,
                                           verify_harmlessness=False,
                                           annotations=None,
                                           verbose=False):
            for atom in pnad.op.preconditions:
                preds_in_preconds.add(atom.predicate)
        kept_predicates &= preds_in_preconds

        logging.info(f"\nSelected {len(kept_predicates)} predicates out of "
                     f"{len(candidates)} candidates:")
        for pred in kept_predicates:
            logging.info(f"\t{pred}")
        score_function.evaluate(kept_predicates)  # log useful numbers

        return set(kept_predicates)

    @staticmethod
    def _get_consistent_predicates(
        predicates: Set[Predicate], clusters: List[List[Segment]]
    ) -> Tuple[Set[Predicate], Set[Predicate]]:
        """Returns all predicates that are consistent with respect to a set of
        segment clusters.

        A consistent predicate is is either an add effect, a delete
        effect, or doesn't change, within each cluster, for all
        clusters.
        """

        consistent: Set[Predicate] = set()
        inconsistent: Set[Predicate] = set()
        for pred in predicates:
            keep_pred = True
            for seg_list in clusters:
                segment_0 = seg_list[0]
                pred_in_add_effs_0 = pred in [
                    atom.predicate for atom in segment_0.add_effects
                ]
                pred_in_del_effs_0 = pred in [
                    atom.predicate for atom in segment_0.delete_effects
                ]
                for seg in seg_list[1:]:
                    pred_in_curr_add_effs = pred in [
                        atom.predicate for atom in seg.add_effects
                    ]
                    pred_in_curr_del_effs = pred in [
                        atom.predicate for atom in seg.delete_effects
                    ]
                    not_consis_add = pred_in_add_effs_0 != pred_in_curr_add_effs
                    not_consis_del = pred_in_del_effs_0 != pred_in_curr_del_effs
                    if not_consis_add or not_consis_del:
                        keep_pred = False
                        inconsistent.add(pred)
                        logging.info(f"Inconsistent predicate: {pred.name}")
                        break
                if not keep_pred:
                    break
            else:
                consistent.add(pred)
        return consistent, inconsistent

    def _select_predicates_and_learn_operators_by_clustering(
            self, candidates: Dict[Predicate, float],
            initial_predicates: Set[Predicate], dataset: Dataset,
            atom_dataset: List[GroundAtomTrajectory]) -> Set[Predicate]:
        """Assume that the demonstrator used a TAMP theory to generate the
        demonstration data and try to reverse-engineer that theory (operators,
        and in turn, the predicates used in the operator definitions) with the
        following strategy:

            (1) Cluster the segments in the atom_dataset -- an operator (that
            we don't know the definition of yet) exists for each cluster and
            "describes" that cluster. That is, the operator's preconditions and
            effects are consistent, for some grounding, with the initial atoms
            and effects of each segment in the cluster.

            (2) Note the predicates that occur in the initial atoms and
            effects of all the segments in a cluster, for each cluster. This
            set of predicates will be much smaller than what we initially
            enumerate from the entire grammar while also likely being a
            superset of the smallest set of predicates you could use and still
            do well on the test tasks with.

            (3) Choose the definitions (preconditions and effects -- step 1 & 2
            pin down the parameters) of the operators by reasoning backwards
            from the goal in each demonstration.

            (4) Do steps 1-3 above for various clusterings, and pick the theory
            (operators) that achieves the best score, for some
            scoring function.

        # In other words, this procedure chooses
        # With this procedure, we learn predicates and operators somewhat "in the
        # same step", rather than one after another, in two separate stages of a
        # pipeline. Afterwards, we only have to learn samplers, since we assume
        # the options are given.
        # This procedure tries to reverse engineer the clusters of segments
        # that correspond to the oracle operators, and from those clusters,
        # learns operator definitions that achieve harmlessness on the
        # demonstrations.
        """
        if CFG.grammar_search_pred_clusterer == "option-type-number-sample":

            # first, look at every operator in the last frontier.
            # from the potential add effects, initialize the goal atoms as definitely add effects.



            # Algorithm:
            # Step 1: cluster segments according to which option was executed
            # Step 2: in each of clusters from the previous step, further cluster
            # segments according to the unique set of object types involved in
            # the segment's add effects
            # Step 3: in each of the clusters from the previous step, further
            # cluster segments according to the (maximum) number of unique objects
            # involved in the segment's add effects
            # Step 4: in each of the clusters from the previous step, further
            # cluster segments by assignment from a gaussian mixture model fit
            # to samples associated with the options of all the segments in the
            # cluster, but only if a sample exists
            # Step 5: remove predicates that are not consistent
            # Step 6: get the final set of predicates via a pure intersection of
            # add effect predicates across segments in each cluster
            from predicators.nsrt_learning.segmentation import segment_trajectory
            assert CFG.segmenter == "option_changes"
            # segmented_trajs = [segment_trajectory(traj) for traj in atom_dataset]
            segmented_trajs = [
                segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in atom_dataset
            ]

            # for seg_traj in segmented_trajs:
            #     l = [seg.get_option().name for seg in seg_traj]
            #     b = [seg.get_option() for seg in seg_traj]
            #     if "Pick" in l:
            #         import pdb; pdb.set_trace()


            from functools import reduce
            flattened_segmented_trajs = reduce(lambda a, b: a+b, segmented_trajs)

            # import pdb; pdb.set_trace()

            # Step 1:
            option_to_segments = {}
            for s in flattened_segmented_trajs:
                n = s.get_option().name
                if n in option_to_segments:
                    option_to_segments[n].append(s)
                else:
                    option_to_segments[n] = [s]
            logging.info(f"STEP 1: generated {len(option_to_segments.values())} option-based clusters.")

            # Step 2:
            all_clusters = []
            for j, pair in enumerate(option_to_segments.items()):
                option, segments = pair
                clusters = {}
                for seg in segments:
                    all_types = [set(a.predicate.types) for a in seg.add_effects]
                    if len(all_types) == 0 or len(set.union(*all_types)) == 0:
                        # Either there are no add effects, or the object
                        # arguments for all add effects are empty (which would
                        # happen e.g. if the add effects only involved Forall
                        # predicates with no object arguments). The former
                        # happens in repeated_nextto.
                        continue
                    types = tuple(sorted(list(set.union(*all_types))))
                    if types in clusters:
                        clusters[types].append(seg)
                    else:

                        clusters[types] = [seg]
                        # print("New unique type: ", types) # to detect the issue in cover where it infers an incorrect cluster with just type (block, )
                        # if len(types) == 1:
                            # import pdb; pdb.set_trace()
                logging.info(f"STEP 2: generated {len(clusters.values())} type-based clusters for for {j+1}th cluster from STEP 1 involving option {option}.")
                for _, c in clusters.items():
                    all_clusters.append(c)

            # Step 3:
            next_clusters = []
            for j, cluster in enumerate(all_clusters):
                clusters = {}
                for seg in cluster:

                    # debugs = list(seg.add_effects)
                    # import pdb; pdb.set_trace()

                    max_num_unique_objs = max(len(eff.objects) for eff in seg.add_effects)
                    if max_num_unique_objs in clusters:
                        clusters[max_num_unique_objs].append(seg)
                    else:
                        clusters[max_num_unique_objs] = [seg]
                for c in clusters.values():
                    next_clusters.append(c)
                logging.info(f"STEP 3: generated {len(clusters.values())} num-object-based clusters for the {j+1}th cluster from STEP 2 involving option {seg.get_option().name}.")
            # final_clusters = next_clusters
            all_clusters = next_clusters

            # Step 4:
            # final_clusters = []
            actual_final_clusters = []
            for j, cluster in enumerate(all_clusters):
                example_segment = cluster[0]
                option_name = example_segment.get_option().name

                if len(example_segment.get_option().params) == 0:
                    actual_final_clusters.append(cluster)
                    logging.info(f"STEP 4: generated no further sample-based clusters (no parameter!) for the {j+1}th cluster from STEP 3 involving option {option_name}.")
                else:
                    # try all sub-clusterings from 2 to K
                    clusters_to_test = []
                    for x in range(len(all_clusters)):
                        if x != j:
                            clusters_to_test.append(all_clusters[x])

                    scores = {}

                    for k in range(1, CFG.grammar_search_max_sub_clustering_components):
                        import copy
                        clusters_to_test_k = copy.deepcopy(clusters_to_test)
                        # learn a theory and check the score
                        import numpy as np
                        from sklearn.mixture import GaussianMixture as GMM
                        data = np.array([seg.get_option().params for seg in cluster])
                        # cluster on just the output distribution of the sampler, might want to
                        # more generally cluster on both (input, output) concatenated data
                        model = GMM(k, covariance_type="full", random_state=0).fit(data)
                        assignments = model.predict(data)
                        sub_clusters = {}
                        for a, assignment in enumerate(assignments):
                            if assignment in sub_clusters:
                                sub_clusters[assignment].append(cluster[a])
                            else:
                                sub_clusters[assignment] = [cluster[a]]
                        for c in sub_clusters.values():
                            clusters_to_test_k.append(c)

                        final_clusters = clusters_to_test_k

                        # now, do operator learning with clusters_to_test_k and report the score
                        ddd = {}
                        for i, c in enumerate(final_clusters):
                            op_name = "Op"+str(i)+"-"+str(c[0].get_option().name)
                            # preconditions, add effects, delete effects, segments
                            ddd[op_name] = [set(), set(), set(), c]

                        add_effects_per_cluster = []
                        all_add_effects = set()
                        for j, c in enumerate(final_clusters):

                            # add effects
                            add_effects_per_segment = [s.add_effects for s in c]
                            ungrounded_add_effects_per_segment = []
                            for add_effects in add_effects_per_segment:
                                ungrounded_add_effects_per_segment.append(set(a.predicate for a in add_effects))
                            add_effects = set.intersection(*ungrounded_add_effects_per_segment)
                            add_effects_per_cluster.append(add_effects)

                            # delete effects
                            del_effects_per_segment = [s.delete_effects for s in c]
                            ungrounded_del_effects_per_segment = []
                            for del_effects in del_effects_per_segment:
                                ungrounded_del_effects_per_segment.append(set(a.predicate for a in del_effects))
                            del_effects = set.intersection(*ungrounded_del_effects_per_segment)

                            # preconditions
                            init_atoms_per_segment = [s.init_atoms for s in c]
                            ungrounded_init_atoms_per_segment = []
                            for init_atoms in init_atoms_per_segment:
                                ungrounded_init_atoms_per_segment.append(set(a.predicate for a in init_atoms))
                            init_atoms = set.intersection(*ungrounded_init_atoms_per_segment)

                            op_name = "Op"+str(j)+"-"+str(c[0].get_option().name)
                            ddd[op_name][0] = init_atoms
                            ddd[op_name][1] = add_effects
                            ddd[op_name][2] = del_effects


                            print(f"Cluster {j} with option {c[0].get_option().name}, predicates:")
                            for a in add_effects:
                                print(a)
                            print()
                            all_add_effects |= add_effects

                        predicates_to_keep = all_add_effects
                        ##########
                        # Remove inconsistent predicates.
                        inconsistent_preds = set()

                        # Old way to remove inconsistent predicates
                        predicates_to_keep: Set[Predicate] = set()
                        for pred in all_add_effects:
                            keep_pred = True
                            for j, seg_list in enumerate(final_clusters):
                                seg_0 = seg_list[0]
                                pred_in_add_effs_0 = pred in [
                                    atom.predicate for atom in seg_0.add_effects
                                ]
                                pred_in_del_effs_0 = pred in [
                                    atom.predicate for atom in seg_0.delete_effects
                                ]
                                for seg in seg_list[1:]:
                                    pred_in_curr_add_effs = pred in [
                                        atom.predicate for atom in seg.add_effects
                                    ]
                                    pred_in_curr_del_effs = pred in [
                                        atom.predicate for atom in seg.delete_effects
                                    ]
                                    A = pred_in_add_effs_0 != pred_in_curr_add_effs
                                    B = pred_in_del_effs_0 != pred_in_curr_del_effs
                                    if A or B:
                                    # if not ((pred_in_add_effs_0 == pred_in_curr_add_effs)
                                    #         and
                                    #         (pred_in_del_effs_0 == pred_in_curr_del_effs)):
                                        keep_pred = False
                                        print("INCONSISTENT: ", pred.name)

                                        inconsistent_preds.add(pred)
                                        # if pred.name == "NOT-((0:obj).grasp<=[idx 0]0.5)" or pred.name == "((0:obj).grasp<=[idx 1]0.25)":
                                        #     import pdb; pdb.set_trace()
                                        break
                                if not keep_pred:
                                    break
                            if keep_pred:
                                predicates_to_keep.add(pred)
                            else:
                                inconsistent_preds.add(pred)

                        # Re-add inconsistent predicates that were necessary to disambiguate two clusters.
                        # That is, if any two clusters's predicates look the same after removing inconsistent predicates from both,
                        # keep all those predicates.
                        new_add_effects_per_cluster = []
                        for effs_cluster in add_effects_per_cluster:
                            new_add_effects_per_cluster.append(effs_cluster - inconsistent_preds)
                        # Check all pairs of clusters, if they are the same, add their original cluster's predicates back.
                        # import pdb; pdb.set_trace()
                        num_clusters = len(final_clusters)
                        for i in range(num_clusters):
                            for j in range(num_clusters):
                                if i == j:
                                    continue
                                else:
                                    if new_add_effects_per_cluster[i] == new_add_effects_per_cluster[j]:
                                        print(f"MATCH! : {i}, {j}")
                                        predicates_to_keep = predicates_to_keep.union(add_effects_per_cluster[i])
                                        predicates_to_keep = predicates_to_keep.union(add_effects_per_cluster[j])
                        ####################
                        # remove predicates we aren't keeping
                        for i, c in enumerate(final_clusters):
                            if len(c) < 10:
                                # ddd.pop("Op"+str(i))
                                op_name = "Op"+str(i)+"-"+str(c[0].get_option().name)
                        print()
                        print()
                        for op, stuff in ddd.items():
                            print()
                            ddd[op][0] = ddd[op][0].intersection(predicates_to_keep)
                            ddd[op][1] = ddd[op][1].intersection(predicates_to_keep)
                            ddd[op][2] = ddd[op][2].intersection(predicates_to_keep)

                            preconditions, add_effects, del_effects, _ = stuff
                            print(op + ": ")
                            print("preconditions: ")
                            for p in preconditions:
                                print(p)
                            print()
                            print("add effects: ")
                            for p in add_effects:
                                print(p)
                            print()
                            print("delete effects: ")
                            for p in del_effects:
                                print(p)
                            print()

                        # import pdb; pdb.set_trace()
                        logging.info("Performing backchaining to decide operator definitions.")
                        #####################
                        # backchaining
                        #####################

                        def seg_in_cluster(seg, cluster):
                            for seg_2 in cluster:
                                if len(seg.states) != len(seg_2.states):
                                    continue

                                all_match = True
                                for i in range(len(seg.states)):
                                    if not seg.states[i].allclose(seg_2.states[i]):
                                        all_match = False
                                        break
                                if all_match:
                                    return True
                            return False

                        def seg_to_op(segment, clusters):
                            for i, c in enumerate(clusters):
                                # if segment in c:
                                if seg_in_cluster(seg, c):
                                    # return f"Op{i}-{c[0].get_option().name} with objects ({c[0].get_option().objects})"
                                    return f"Op{i}-{c[0].get_option().name}"
                            print("couldn't find segment...")

                        def get_story2(segmented_traj):
                            story = []
                                # each entry is a list of lists
                            for k, seg in enumerate(segmented_traj):
                                # if k == 2:
                                #     import pdb; pdb.set_trace()
                                entry = []
                                op_name = seg_to_op(seg, final_clusters)

                                # init_atoms = set(p for p in seg.init_atoms if p.predicate in ddd[op_name][0])
                                # objects = (set(o for a in seg.add_effects for o in a.objects) | set(o for a in seg.delete_effects for o in a.objects))
                                # preconditions = set(p for p in init_atoms if set(p.objects).issubset(objects))
                                # add_effects = set(p for p in seg.add_effects if p.predicate in ddd[op_name][1])
                                # delete_effects = set(p for p in seg.delete_effects if p.predicate in ddd[op_name][2])

                                opt_objs = tuple(seg.get_option().objects)
                                objects = (set(o for a in seg.add_effects for o in a.objects) | set(o for a in seg.delete_effects for o in a.objects) | set(opt_objs))
                                preconditions_hypothesis = set(p for p in seg.init_atoms if p.predicate in ddd[op_name][0])
                                add_effects_hypothesis = set(p for p in seg.add_effects if p.predicate in ddd[op_name][1])
                                delete_effects_hypothesis = set(p for p in seg.delete_effects if p.predicate in ddd[op_name][2])
                                preconditions = set(p for p in preconditions_hypothesis if set(p.objects).issubset(objects))
                                add_effects = set(p for p in add_effects_hypothesis if set(p.objects).issubset(objects))
                                delete_effects = set(p for p in delete_effects_hypothesis if set(p.objects).issubset(objects))

                                relevant_objects = (set(o for a in add_effects for o in a.objects) | set(o for a in delete_effects for o in a.objects))

                                entry = [
                                    op_name,
                                    preconditions,
                                    add_effects,
                                    delete_effects,
                                    relevant_objects
                                ]
                                story.append(entry)
                            return story

                        def process_adds2(potential_ops, remaining_story, goal):
                            curr_entry = remaining_story[0]
                            name = curr_entry[0]
                            to_add = []
                            for ground_atom in curr_entry[2]: # add effects
                                # can we find this atom in the preconditions of any future segment
                                for entry in remaining_story[1:]:
                                    if ground_atom in entry[1]: # preconditions
                                        # then we add it!
                                        to_add.append(ground_atom) # for debugging
                                        potential_ops[name]["add"].add(ground_atom.predicate)
                                        if "pre" not in potential_ops[entry[0]].keys():
                                            import pdb; pdb.set_trace()
                                        potential_ops[entry[0]]["pre"].add(ground_atom.predicate)
                                if ground_atom in goal:
                                    # then we add it!
                                    to_add.append(ground_atom) # for debugging
                                    potential_ops[name]["add"].add(ground_atom.predicate)

                        def process_dels(potential_ops, remaining_story):
                            curr_entry = remaining_story[0]
                            name = curr_entry[0]
                            to_del = []
                            for ground_atom in curr_entry[3]: # delete effects
                                # is this atom every added in the future?
                                # *** do we need to do this with an updated potential_ops after process_adds runs on all the trajectories?
                                for entry in remaining_story[1:]:
                                    if ground_atom in entry[2]: # add effects
                                        to_del.append(ground_atom) # for debugging
                                        potential_ops[name]["del"].add(ground_atom.predicate)
                                        potential_ops[entry[0]]["add"].add(ground_atom.predicate)

                        all_potential_ops = []
                        # TODO: change it so that you fill up a unique potential_ops at each iteration, and then take the union for the final one
                        # potential_ops = {}
                        # for op_name in ddd.keys():
                        #     potential_ops[op_name] = {
                        #         "pre": set(),
                        #         "add": set(),
                        #         "del": set()
                        #     }

                        for j, traj in enumerate(segmented_trajs):
                            print(f"Segmented trajectory #{j}")
                            potential_ops = {}
                            for op_name in ddd.keys():
                                potential_ops[op_name] = {
                                    "pre": set(),
                                    "add": set(),
                                    "del": set()
                                }

                            story = get_story2(traj)
                            # if j == 2:
                            #     story = get_story2(traj)
                            #     import pdb; pdb.set_trace()
                            reversed_story = list(reversed(story))
                            for i, entry in enumerate(reversed_story):
                                remaining_story = story[len(story)-i-1:]
                                name, preconds, add_effs, del_effs, relevant_objs = entry
                                # TODO: do some more thinking about what is a relevant object
                                if i == 0:
                                    # For the last segment (in the non-reversed story), we can only evaluate add_effects.
                                    # The initial predicates are the goal predicates.
                                    for p in add_effs:
                                        if p in self._train_tasks[j].goal:
                                            potential_ops[name]["add"].add(p.predicate)
                                else:
                                    process_adds2(potential_ops, remaining_story, self._train_tasks[j].goal)
                                    process_dels(potential_ops, remaining_story)

                            all_potential_ops.append(potential_ops)


                        # final_potential_ops = {
                        #     op_name: {
                        #         "pre": set(),
                        #         "add": set(),
                        #         "del": set()
                        #     } for op_name in all_potential_ops[0].keys()
                        # }
                        final_potential_ops = {
                            op_name: {
                                "pre": ddd[op_name][0], # take all the precondition predicates
                                "add": set(),
                                "del": ddd[op_name][2]
                            } for op_name in all_potential_ops[0].keys()
                        }

                        # for op_name in final_potential_ops.keys():
                        #     # get all preconditions that aren't empty
                        #
                        #     # if op_name == "Op0-Pick":
                        #     #     temp = []
                        #     #     for h, e in enumerate(all_potential_ops):
                        #     #         s = e[op_name]["pre"]
                        #     #         if len(s) > 0:
                        #     #             temp.append(s)
                        #     #         name_ps = [p.name for p in s]
                        #     #         if "OnTable" not in name_ps:
                        #     #             import pdb; pdb.set_trace()
                        #
                        #     # temp = [e[op_name]["pre"] for e in all_potential_ops if len(e[op_name]["pre"]) > 0]
                        #
                        #     final_potential_ops[op_name]["pre"] = set.intersection(*[e[op_name]["pre"] for e in all_potential_ops if len(e[op_name]["pre"]) > 0])
                        #     final_potential_ops[op_name]["add"] = set.intersection(*[e[op_name]["add"] for e in all_potential_ops if len(e[op_name]["add"]) > 0])
                        #     final_potential_ops[op_name]["del"] = set.intersection(*[e[op_name]["del"] for e in all_potential_ops if len(e[op_name]["del"]) > 0])

                        for k, po in enumerate(all_potential_ops):
                            for op_name in po.keys():
                                final_potential_ops[op_name]["pre"] = final_potential_ops[op_name]["pre"].union(po[op_name]["pre"])
                                final_potential_ops[op_name]["add"] = final_potential_ops[op_name]["add"].union(po[op_name]["add"])
                                final_potential_ops[op_name]["del"] = final_potential_ops[op_name]["del"].union(po[op_name]["del"])
                                # if len(po[op_name]["pre"]) > 0:
                                #     final_potential_ops[op_name]["pre"] = final_potential_ops[op_name]["pre"].intersection(po[op_name]["pre"])
                                # if len(po[op_name]["add"]) > 0:
                                #     final_potential_ops[op_name]["add"] = final_potential_ops[op_name]["add"].intersection(po[op_name]["add"])
                                # if len(po[op_name]["del"]) > 0:
                                #     final_potential_ops[op_name]["del"] = final_potential_ops[op_name]["del"].intersection(po[op_name]["del"])

                        # print the operators nicely to see what we missed
                        for k, v in final_potential_ops.items():
                            print(k)
                            print("====")
                            print("preconditions:")
                            for p in sorted(list(v["pre"])):
                                print(p)
                            print("====")
                            print("add effects:")
                            for p in sorted(list(v["add"])):
                                print(p)
                            print("====")
                            print("delete effects:")
                            for p in sorted(list(v["del"])):
                                print(p)

                        fff = {}
                        for op in ddd.keys():
                            fff[op] = []
                            fff[op].append(
                                set(p for p in ddd[op][0] if p in final_potential_ops[op]["pre"])
                            )

                            # if op == "Op0-Pick":
                            #     pred_to_manually_add = [p for p in ddd[op][0] if p.name == "OnTable"][0]
                            #     print("MANUALLY ADDING: ", pred_to_manually_add)
                            #     fff[op][-1].add(pred_to_manually_add)

                            fff[op].append(
                                set(p for p in ddd[op][1] if p in final_potential_ops[op]["add"])
                            )

                            fff[op].append(
                                set(p for p in ddd[op][2] if p in final_potential_ops[op]["del"])
                            )
                            # if op == "Op2-Stack":
                            #     pred_to_manually_add = [p for p in ddd[op][2] if p.name == "Forall[0:block].[NOT-On(0,1)]"][0]
                            #     print("MANUALLY ADDING: ", pred_to_manually_add)
                            #     fff[op][-1].add(pred_to_manually_add)

                            fff[op].append(ddd[op][3])

                        self._clusters = fff
                        self._stuff_needed = (initial_predicates, atom_dataset, candidates, self._train_tasks, "num_nodes_expanded", predicates_to_keep)


                        ########################################################
                        from predicators.structs import STRIPSOperator, Variable, PNAD
                        pnads: List[PNAD] = []
                        ops_to_print = []
                        for name, v in self._clusters.items():
                            preconds, add_effects, del_effects, segments = v
                            seg_0 = segments[0]
                            opt_objs = tuple(seg_0.get_option().objects)
                            relevant_add_effects = [a for a in seg_0.add_effects if a.predicate in add_effects]
                            relevant_del_effects = [a for a in seg_0.delete_effects if a.predicate in del_effects]
                            objects = {o for atom in relevant_add_effects + relevant_del_effects for o in atom.objects} | set(opt_objs)
                            objects_list = sorted(objects)

                            params = utils.create_new_variables([o.type for o in objects_list])
                            obj_to_var = dict(zip(objects_list, params))
                            var_to_obj = dict(zip(params, objects_list))

                            relevant_preconds = [a for a in seg_0.init_atoms if (a.predicate in preconds and set(a.objects).issubset(set(objects_list)))]
                            op_add_effects = {atom.lift(obj_to_var) for atom in relevant_add_effects}
                            op_del_effects = {atom.lift(obj_to_var) for atom in relevant_del_effects}
                            op_preconds = {atom.lift(obj_to_var) for atom in relevant_preconds}

                            # if name == "Op2-Stack":
                            #     import pdb; pdb.set_trace()
                            #     block_to_not_del = [p for p in op_preconds if p.predicate.name=="NOT-((0:block).pose_z<=[idx 0]0.461)"][0].entities[0]
                            #     new_op_preconds = set()
                            #     for p in op_preconds:
                            #         if p.predicate.name == "NOT-OnTable" and block_to_not_del not in p.entities:
                            #             continue
                            #         new_op_preconds.add(p)
                            #     op_preconds = new_op_preconds
                            #     # this is dealing with the fact that segment 0 happens to be stacking onto a block
                            #     # that's not on the table, but this isn't always true. you can be stacking onto a block that
                            #     # is on the table.
                            #     # need to fix this more generally without hardcoding it in.
                            #     # also, if the map isn't 1:1 obvious, it's not clear how exactly how to correspond
                            #     # obj_to_var in one segment versus another.
                            #     import pdb; pdb.set_trace()

                            option_vars = [obj_to_var[o] for o in opt_objs]
                            option_spec = [seg_0.get_option().parent, option_vars]

                            # if name == "Op2-Stack":
                            #     import pdb; pdb.set_trace()

                            op_ignore_effects = set()
                            op = STRIPSOperator(name, params, op_preconds, op_add_effects, op_del_effects, op_ignore_effects)


                            from itertools import permutations, product
                            def get_mapping_between_params(params1):
                                unique_types_old = sorted(set(elem.type for elem in params1))
                                # unique types with same order as params, don't want to sort it because of issue in painting with robby:robot and receptacle_shelf:shelf
                                unique_types = []
                                unique_types_set = set()
                                for param in params1:
                                    if param.type not in unique_types_set:
                                        unique_types.append(param.type)
                                        unique_types_set.add(param.type)

                                # if len(params) == 3:
                                #     import pdb; pdb.set_trace()

                                # import pdb; pdb.set_trace()

                                group_params_by_type = []
                                for elem_type in unique_types:
                                    elements_of_type = [elem for elem in params1 if elem.type == elem_type]
                                    group_params_by_type.append(elements_of_type)

                                all_mappings = list(product(*list(permutations(l) for l in group_params_by_type)))
                                squash = []
                                for m in all_mappings:
                                    a = []
                                    for i in m:
                                        a.extend(i)
                                    squash.append(a)

                                return squash

                            ops = []
                            for seg in segments:
                                opt_objs = tuple(seg.get_option().objects)
                                relevant_add_effects = [a for a in seg.add_effects if a.predicate in add_effects]
                                relevant_del_effects = [a for a in seg.delete_effects if a.predicate in del_effects]
                                objects = {o for atom in relevant_add_effects + relevant_del_effects for o in atom.objects} | set(opt_objs)
                                objects_list = sorted(objects)
                                # objects_list = sorted(objects, key=lambda x: (x.type.name, x.name))
                                # have to do this otherwise robby:robot and receptacle_shelf:shelf get swapped later after they are lifted and sort by type and not name
                                params = utils.create_new_variables([o.type for o in objects_list])
                                obj_to_var = dict(zip(objects_list, params))
                                var_to_obj = dict(zip(params, objects_list))
                                relevant_preconds = [a for a in seg.init_atoms if (a.predicate in preconds and set(a.objects).issubset(set(objects_list)))]

                                op_add_effects = {atom.lift(obj_to_var) for atom in relevant_add_effects}
                                op_del_effects = {atom.lift(obj_to_var) for atom in relevant_del_effects}
                                op_preconds = {atom.lift(obj_to_var) for atom in relevant_preconds}
                                # t = (params, var_to_obj, obj_to_var, op_preconds, op_add_effects, op_del_effects)
                                t = (params, objects_list, relevant_preconds, relevant_add_effects, relevant_del_effects)
                                ops.append(t)


                            # We would like to take the intersection of preconditions, add effects, and delete effects
                            # between operators in a particular cluster to weed out ones that do not generalize, e.g. that
                            # the block you are stacking on (in Op2-Stack), is also on another block (ratherr than on the table).
                            # But the object -> variable mapping is not consistent, so this takes some extra effort. For example,
                            # consider Op2-Stack, which operates on two blocks and one robot. Sometimes, blockn is put on blockn+1,
                            # but other times, blockn+1 is put on blockn. Because the object -> variable mapping is done with sorted
                            # objects (which sort by name and then by type), the stack operators created from some segments will have
                            # the first parameter be the top block, while others will have the first operator be the second block. So,
                            # if we just took an intersection of lifted atoms between the two operators, the predicates would would not
                            # correspond to each other properly.
                            # if name in  ["Op2-Stack"]:
                            # if name in  ["Op0-Pick", "Op1-PutOnTable", "Op2-Stack", "Op3-Pick"]:
                            if True:
                                print(f"DOING THIS FOR {name}")
                                op1 = ops[0]
                                op1_params = op1[0]
                                op1_objs_list = op1[1]
                                op1_obj_to_var = dict(zip(op1_objs_list, op1_params))
                                op1_preconds = {atom.lift(op1_obj_to_var) for atom in op1[2]}
                                op1_add_effects = {atom.lift(op1_obj_to_var) for atom in op1[3]}
                                op1_del_effects = {atom.lift(op1_obj_to_var) for atom in op1[4]}

                                op1_preconds_str = set(str(a) for a in op1_preconds)
                                op1_adds_str = set(str(a) for a in op1_add_effects)
                                op1_dels_str = set(str(a) for a in op1_del_effects)

                                # debug:
                                # maybe explicitly go find operators where n+1 is put on n, and then where n is put on n+1
                                # look at add effects and see the numbers
                                # demo 7 seems to have n on n+1 at some point in it
                                # demo 0 has n+1 on n
                                import re
                                def extract_numbers_from_string(input_string):
                                    # Use regular expression to find all numeric sequences in the 'blockX' format
                                    numbers = re.findall(r'\bblock(\d+)\b', input_string)
                                    # Convert the found strings to integers and return a set
                                    return set(map(int, numbers))
                                def print_predicates(preds):
                                    l = sorted(list(preds))
                                    print("Printing set: ")
                                    for p in l:
                                        print(p)
                                    print()
                                # for x, seg in enumerate(segments):
                                #     relevant = [str(a) for a in seg.add_effects if a.predicate.name == "On"]
                                #     # now see if n+1 on n or n on n+1
                                #     assert len(relevant) == 1
                                #     z = relevant[0]
                                #     nums = extract_numbers_from_string(z)
                                #     assert len(nums) == 2
                                #     nums_sorted = sorted(list(nums))
                                #     higher_on_top = z.index(str(nums_sorted[1])) < z.index(str(nums_sorted[0]))
                                    # if not higher_on_top:
                                    #     print("HIGHER NOT ON TOP")
                                    #     import pdb; pdb.set_trace()

                                for i in range(1, len(ops)):
                                    op2 = ops[i]
                                    op2_params = op2[0]
                                    op2_objs_list = op2[1]

                                    mappings = get_mapping_between_params(op2_params)
                                    mapping_scores = []
                                    for m in mappings:

                                        mapping = dict(zip(op2_params, m))

                                        overlap = 0

                                        # Get Operator 2's preconditions, add effects, and delete effects
                                        # in terms of a particular object -> variable mapping.
                                        new_op2_params = [mapping[p] for p in op2_params]
                                        new_op2_obj_to_var = dict(zip(op2_objs_list, new_op2_params))
                                        # import pdb; pdb.set_trace()
                                        try:
                                            op2_preconds = {atom.lift(new_op2_obj_to_var) for atom in op2[2]}
                                        except:
                                            import pdb; pdb.set_trace()
                                        op2_preconds = {atom.lift(new_op2_obj_to_var) for atom in op2[2]}
                                        op2_add_effects = {atom.lift(new_op2_obj_to_var) for atom in op2[3]}
                                        op2_del_effects = {atom.lift(new_op2_obj_to_var) for atom in op2[4]}

                                        # Take the intersection of lifted atoms across both operators, and
                                        # count the overlap.
                                        op2_preconds_str = set(str(a) for a in op2_preconds)
                                        op2_adds_str = set(str(a) for a in op2_add_effects)
                                        op2_dels_str = set(str(a) for a in op2_del_effects)

                                        score1 = len(op1_preconds_str.intersection(op2_preconds_str))
                                        score2 = len(op1_adds_str.intersection(op2_adds_str))
                                        score3 = len(op1_dels_str.intersection(op2_dels_str))
                                        score = score1 + score2 + score3

                                        new_preconds = set(a for a in op1_preconds if str(a) in op1_preconds_str.intersection(op2_preconds_str))
                                        new_adds = set(a for a in op1_add_effects if str(a) in op1_adds_str.intersection(op2_adds_str))
                                        new_dels = set(a for a in op1_del_effects if str(a) in op1_dels_str.intersection(op2_dels_str))

                                        mapping_scores.append((score, new_preconds, new_adds, new_dels))

                                    s, a, b, c = max(mapping_scores, key=lambda x: x[0])
                                    op1_preconds = a
                                    op1_add_effects = b
                                    op1_del_effects = c
                                    op1_preconds_str = set(str(a) for a in op1_preconds)
                                    op1_adds_str = set(str(a) for a in op1_add_effects)
                                    op1_dels_str = set(str(a) for a in op1_del_effects)

                                    # import pdb; pdb.set_trace()

                                # import pdb; pdb.set_trace()

                                # NOT-Forall[0:block].[NOT-On(0,1)](?x0:block) --> there exists a block that is on top of ?x0
                                # NOT-Forall[1:block].[NOT-On(0,1)](?x1:block) --> there exists a block that ?x1 is on top of

                                # op = STRIPSOperator(name, params, op_preconds, op_add_effects, op_del_effects, op_ignore_effects)
                                op = STRIPSOperator(name, op1_params, op1_preconds, op1_add_effects, op1_del_effects, set())
                                ops_to_print.append(op)
                                # import pdb; pdb.set_trace()


                            datastore = []
                            counter = 0
                            for seg in segments:
                                seg_opt_objs = tuple(seg.get_option().objects)
                                var_to_obj = {v: o for v, o in zip(option_vars, seg_opt_objs)}


                                relevant_add_effects = [a for a in seg.add_effects if a.predicate in add_effects]
                                relevant_del_effects = [a for a in seg.delete_effects if a.predicate in del_effects]

                                seg_objs = {o for atom in relevant_add_effects + relevant_del_effects for o in atom.objects} | set(seg_opt_objs)

                                seg_objs_list = sorted(seg_objs)
                                # seg_objs_list = sorted(seg_objs, key=lambda x: (x.type.name, x.name))

                                remaining_objs = [o for o in seg_objs_list if o not in seg_opt_objs]
                                # if you do this, then there's an issue in sampler learning, because it uses
                                # pre.variables for pre in preconditions -- so it will look for ?x0 but not find it
                                # and there is a key error
                                # remaining_params = utils.create_new_variables(
                                #     [o.type for o in remaining_objs], existing_vars = list(var_to_obj.keys()))

                                from predicators.structs import Variable
                                def diff_create_new_variables(types, existing_vars, var_prefix: str = "?x"):
                                    pre_len = len(var_prefix)
                                    existing_var_nums = set()
                                    if existing_vars:
                                        for v in existing_vars:
                                            if v.name.startswith(var_prefix) and v.name[pre_len:].isdigit():
                                                existing_var_nums.add(int(v.name[pre_len:]))
                                    def get_next_num(used):
                                        counter = 0
                                        while True:
                                            if counter in used:
                                                counter += 1
                                            else:
                                                return counter
                                    new_vars = []
                                    for t in types:
                                        num = get_next_num(existing_var_nums)
                                        existing_var_nums.add(num)
                                        new_var_name = f"{var_prefix}{num}"
                                        new_var = Variable(new_var_name, t)
                                        new_vars.append(new_var)
                                    return new_vars
                                remaining_params = diff_create_new_variables(
                                    [o.type for o in remaining_objs], existing_vars = list(var_to_obj.keys())
                                )

                                var_to_obj2 = dict(zip(remaining_params, remaining_objs))
                                # var_to_obj = dict(zip(seg_params, seg_objs_list))
                                var_to_obj = {**var_to_obj, **var_to_obj2}
                                datastore.append((seg, var_to_obj))
                                # if name == "Op2-Stack" and counter == 10:
                                    # normally, block n+1 is stacked on block n, but here
                                    # block2 is stacked on block3.
                                    # so, when we sort the seg_objs_list, we have [block2, block3, robot]
                                    # the operator params are such that [?x0:block, "?x1:block, "?x2: robot]
                                    # ?x1 is stacked on ?x0.
                                    # so, later, in "learn_option_specs()", we get the error: assert option_args == option.objects
                                    # because option_args are [block2, robot], while the gt option objects are [block3, robot]
                                    # so: how do we order var_to_obj here correctly?
                                    # we want a consistent map - take one of the predicates that involves two blocks, and
                                    # make sure the assignment of variables is the same (order-wise) as was used in the construction of
                                    # params?
                                    # that is, if we saw On(x1, x0) in params, then we must also have that here.
                                    # how do you choose On to do this for?
                                    # or, you can ensure the sub is correct for the option spec

                                    # hardcode it for now

                                    # import pdb; pdb.set_trace()
                                counter += 1

                            option_vars = [obj_to_var[o] for o in opt_objs]
                            option_spec = [seg_0.get_option().parent, option_vars]
                            # if name == "Op2-Stack":
                            #     import pdb; pdb.set_trace()
                            pnads.append(PNAD(op, datastore, option_spec))

                            # might be able to handle ignore effects by something like what
                            # _compute_pnad_ignore_effects in base_strips_learner.py does.
                            # basically you can try to see if the ops would be used anywhere in
                            # the demos, and and narrow it down via the demos. but you might
                            # pick up some extraneous predicates in some of them, so you would
                            # have to be careful and do some inference when trying them out on
                            # the demos.

                        def print_ops(op):
                            print("====")
                            print(op.name)
                            print(op.parameters)
                            print("preconditions:")
                            for p in sorted(op.preconditions):
                                print(p)
                            print("add effects:")
                            for p in sorted(op.add_effects):
                                print(p)
                            print("delete effects:")
                            for p in sorted(op.delete_effects):
                                print(p)
                            print("====")


                        for operator in ops_to_print:
                            print_ops(operator)

                        self._pnads = pnads
                        return predicates_to_keep

                        ########################################################################
                        # Check Score
                        ########################################################################
                        from predicators.predicate_search_score_functions import _ExpectedNodesScoreFunction
                        # self._stuff_needed = (initial_predicates, atom_dataset, candidates, self._train_tasks, "num_nodes_expanded", predicates_to_keep)
                        z = self._stuff_needed
                        # score_function = _ExpectedNodesScoreFunction(initial_predicates, atom_dataset, candidates, train_tasks, metric_name)
                        score_function = _ExpectedNodesScoreFunction(z[0], z[1], z[2], z[3], z[4])

                        # pruned_atom_data = utils.prune_ground_atom_dataset(
                        #     self._atom_dataset,
                        #     candidate_predicates | self._initial_predicates)
                        pruned_atom_data = utils.prune_ground_atom_dataset(z[1], z[5] | z[0])
                        # segmented_trajs = [
                        #     segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in atom_dataset
                        # ]
                        segmented_trajs = [segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in pruned_atom_data]
                        low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
                        strips_ops = [pnad.op for pnad in pnads]
                        option_specs = [pnad.option_spec for pnad in pnads]
                        op_score = score_function.evaluate_with_operators(
                            z[5],
                            low_level_trajs,
                            segmented_trajs,
                            strips_ops,
                            option_specs
                        )
                        scores[k] = (op_score, list(sub_clusters.values()))

                        import pdb; pdb.set_trace()

                    import pdb; pdb.set_trace()

                    ########################################################################


                    import pdb; pdb.set_trace()
                    self._pnads = pnads
                    return predicates_to_keep
