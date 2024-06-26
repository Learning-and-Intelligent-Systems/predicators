"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar."""

from __future__ import annotations

import abc
import itertools
import logging
from dataclasses import dataclass, field
from functools import cached_property
from operator import le
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, \
    Sequence, Set, Tuple

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
from predicators.structs import Dataset, GroundAtom, GroundAtomTrajectory, \
    Object, ParameterizedOption, Predicate, Segment, State, Task, Type

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
    if CFG.grammar_search_grammar_use_euclidean_dist:
        for (t1_f1, t1_f2, t2_f1,
             t2_f2) in CFG.grammar_search_euclidean_feature_names:
            euclidean_dist_grammar = _EuclideanDistancePredicateGrammar(
                dataset, t1_f1, t2_f1, t1_f2, t2_f2)
            grammar = _ChainPredicateGrammar([grammar, euclidean_dist_grammar],
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
    # Also don't do this if we explicitly don't want to prune such
    # predicates.
    if not CFG.grammar_search_use_handcoded_debug_grammar and \
        CFG.grammar_search_prune_redundant_preds:
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
class _EuclideanAttributeDiffCompareClassifier(_BinaryClassifier):
    """Compare the euclidean distance between two feature values with a
    constant value."""
    object1_index: int
    object1_type: Type
    obj1_attr1_name: str
    obj1_attr2_name: str
    object2_index: int
    object2_type: Type
    obj2_attr1_name: str
    obj2_attr2_name: str
    constant: float
    constant_idx: int
    compare: Callable[[float, float], bool]
    compare_str: str

    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        assert obj1.type == self.object1_type
        assert obj2.type == self.object2_type
        return self.compare((s.get(obj1, self.obj1_attr1_name) -
                             s.get(obj2, self.obj2_attr1_name))**2 +
                            (s.get(obj1, self.obj1_attr2_name) -
                             s.get(obj2, self.obj2_attr2_name))**2,
                            self.constant)

    def __str__(self) -> str:
        return (f"((({self.object1_index}:{self.object1_type.name})."
                f"{self.obj1_attr1_name} - ({self.object2_index}:"
                f"{self.object2_type.name}).{self.obj2_attr1_name})^2"
                f" + (({self.object1_index}:{self.object1_type.name})."
                f"{self.obj1_attr2_name} - ({self.object2_index}:"
                f"{self.object2_type.name}).{self.obj2_attr2_name})^2)"
                f"{self.compare_str}[idx {self.constant_idx}]"
                f"{self.constant:.3})")

    def pretty_str(self) -> Tuple[str, str]:
        name1 = CFG.grammar_search_classifier_pretty_str_names[
            self.object1_index]
        name2 = CFG.grammar_search_classifier_pretty_str_names[
            self.object2_index]
        vars_str = (f"{name1}:{self.object1_type.name}, "
                    f"{name2}:{self.object2_type.name}")
        body_str = (f"(({name1}.{self.obj1_attr1_name} - "
                    f"{name2}.{self.obj2_attr1_name})^2 "
                    f" + (({name1}.{self.obj1_attr2_name} - "
                    f"{name2}.{self.obj2_attr2_name})^2 "
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
        return f"Forall[{type_sig}].[{str(self.body)}[{objs}]]"

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
        # assert obj.type == self.body.types[self.free_variable_idx]
        assert obj.is_instance(self.body.types[self.free_variable_idx])
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
        return f"Forall[{type_sig}].[{str(self.body)}[{objs}]]"

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
    "stick_button_move": [
        # NOTE: changing the demonstration data slightly causes
        # the value of the constant to change. Need to uncomment these
        # as necessary.
        # StickAboveButton
        "(((0:button).x - (1:stick).tip_x)^2 + ((0:button).y - " + \
            "(1:stick).tip_y)^2)<=[idx 0]0.18)",
        # RobotAboveButton
        "(((0:button).x - (1:robot).x)^2 + ((0:button).y - " + \
            "(1:robot).y)^2)<=[idx 0]0.194)",
        "((0:stick).held<=[idx 0]0.5)",  # Handempty
        "NOT-((0:stick).held<=[idx 0]0.5)",  # Grasped
        "((0:button).y<=[idx 0]3.01)",  # ButtonReachableByRobot
        "NOT-((0:button).y<=[idx 0]3.01)",  # ButtonNotReachableByRobot
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

    def _yield_pred_given_const(
            self, feature_ranges: Dict[Type, Dict[str, Tuple[float, float]]],
            constant_idx: int, constant: float,
            cost: float) -> Iterator[Tuple[Predicate, float]]:
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
                    lb, ub = utils.compute_abs_range_given_two_ranges(
                        lb1, ub1, lb2, ub2)
                    # Scale the constant by the correct range.
                    k = constant * (ub - lb) + lb
                    # Create classifier.
                    comp, comp_str = le, "<="
                    diff_classifier = _AttributeDiffCompareClassifier(
                        0, t1, f1, 1, t2, f2, k, constant_idx, comp, comp_str)
                    name = str(diff_classifier)
                    types = [t1, t2]
                    pred = Predicate(name, types, diff_classifier)
                    assert pred.arity == 2
                    yield (pred, cost)

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # Get ranges of feature values from data.
        feature_ranges = self._get_feature_ranges()
        # Edge case: if there are no features at all, return immediately.
        if not any(r for r in feature_ranges.values()):
            return
        # Start by generating predicates such that the two features are
        # very close together. The reason we can't just set the constant
        # to 1e-6 is because objects have some amount of "size", and so even
        # when they're touching, it's not like their centers overlap.
        # E.g. in stick button, when the robot touches the button, the center
        # of the robot and the object might still be offset by a bit.
        for ret_val in self._yield_pred_given_const(
                feature_ranges, 0,
                CFG.grammar_search_diff_features_const_multiplier, 4.0):
            yield ret_val
        # 0.5, 0.25, 0.75, 0.125, 0.375, ...
        constant_generator = _halving_constant_generator(0.0, 1.0)
        for constant_idx, (constant, cost) in enumerate(constant_generator):
            for ret_val in self._yield_pred_given_const(
                    feature_ranges, constant_idx, constant, cost):
                yield ret_val


@dataclass(frozen=True, eq=False, repr=False)
class _EuclideanDistancePredicateGrammar(
        _SingleFeatureInequalitiesPredicateGrammar):
    """Generates predicates of the form "|0.x - 1.x|^2 + |0.y - 1.y|^2 <= c^2".
    Importantly, this only operates over types that have features
    named f1_name and f2_name.
    """
    t1_f1_name: str
    t2_f1_name: str
    t1_f2_name: str
    t2_f2_name: str

    def _compute_xy_bounds(self, feature_ranges: Dict[Type,
                                                      Dict[str, Tuple[float,
                                                                      float]]],
                           t1: Type, t2: Type) -> Tuple[float, float]:
        # To create our classifier, we need to leverage the
        # upper and lower bounds of its x, y features.
        lbx1, ubx1 = feature_ranges[t1][self.t1_f1_name]
        lbx2, ubx2 = feature_ranges[t2][self.t2_f1_name]
        lby1, uby1 = feature_ranges[t1][self.t1_f2_name]
        lby2, uby2 = feature_ranges[t2][self.t2_f2_name]
        # Compute the upper and lower bounds of each feature range.
        lbx, ubx = utils.compute_abs_range_given_two_ranges(
            lbx1, ubx1, lbx2, ubx2)
        lby, uby = utils.compute_abs_range_given_two_ranges(
            lby1, uby1, lby2, uby2)
        # Now, use these to compute the upper and lower bounds of
        # the squared expression of interest.
        lb = lbx**2 + lby**2
        ub = ubx**2 + uby**2
        return (lb, ub)

    def _generate_pred_given_constant(self, constant_idx: int, constant: float,
                                      t1: Type, t2: Type, t1_f1_name: str,
                                      t1_f2_name: str, t2_f1_name: str,
                                      t2_f2_name: str) -> Predicate:
        # Create classifier.
        comp, comp_str = le, "<="
        diff_classifier = _EuclideanAttributeDiffCompareClassifier(
            0, t1, t1_f1_name, t1_f2_name, 1, t2, t2_f1_name, t2_f2_name,
            constant, constant_idx, comp, comp_str)
        name = str(diff_classifier)
        types = [t1, t2]
        pred = Predicate(name, types, diff_classifier)
        return pred

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # Get ranges of feature values from data.
        feature_ranges = self._get_feature_ranges()
        # Edge case: if there are no features at all, return immediately.
        if not any(r for r in feature_ranges.values()):
            return

        # Start by generating predicates with a very small constant,
        # to indicate that the objects are touching/overlapped.
        for (t1, t2) in itertools.combinations_with_replacement(
                sorted(self.types), 2):
            if (self.t1_f1_name in t1.feature_names
                    and self.t2_f1_name in t2.feature_names
                    and self.t1_f2_name in t1.feature_names
                    and self.t2_f2_name in t2.feature_names):
                lb, ub = self._compute_xy_bounds(feature_ranges, t1, t2)
                constant = ((ub - lb) *
                            CFG.grammar_search_euclidean_const_multiplier) + lb
                pred = self._generate_pred_given_constant(
                    0, constant, t1, t2, self.t1_f1_name, self.t1_f2_name,
                    self.t2_f1_name, self.t2_f2_name)
                assert pred.arity == 2
                yield (pred, 3.0)  # cost = arity + cost from constant

        # 0.5, 0.25, 0.75, 0.125, 0.375, ...
        constant_generator = _halving_constant_generator(0.0, 1.0)
        for constant_idx, (constant, cost) in enumerate(constant_generator):
            for (t1, t2) in itertools.combinations_with_replacement(
                    sorted(self.types), 2):
                if (self.t1_f1_name in t1.feature_names
                        and self.t2_f1_name in t2.feature_names
                        and self.t1_f2_name in t1.feature_names
                        and self.t2_f2_name in t2.feature_names):
                    lb, ub = self._compute_xy_bounds(feature_ranges, t1, t2)
                    # Scale the constant by the correct range.
                    k = constant * (ub - lb) + lb
                    pred = self._generate_pred_given_constant(
                        constant_idx + 1, k, t1, t2, self.t1_f1_name,
                        self.t1_f2_name, self.t2_f1_name, self.t2_f2_name)
                    assert pred.arity == 2
                    yield (pred, 2 + cost)  # cost = arity + cost from constant


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
                state_seq = utils.segment_trajectory_to_start_end_state_sequence(  # pylint:disable=line-too-long
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

    def _generate_atom_dataset_via_grammar(
        self, dataset: Dataset
    ) -> Tuple[List[GroundAtomTrajectory], Dict[Predicate, float]]:
        """Generates predicates from a grammar, and applies them to the
        dataset."""
        # Generate a candidate set of predicates.
        logging.info("Generating candidate predicates...")
        grammar = _create_grammar(dataset, self._initial_predicates)
        candidates = grammar.generate(
            max_num=CFG.grammar_search_max_predicates)
        logging.info(f"Done: created {len(candidates)} candidates:")
        self._metrics["grammar_size"] = len(candidates)
        for predicate, cost in candidates.items():
            logging.info(f"{predicate} {cost}")
        # Apply the candidate predicates to the data.
        logging.info("Applying predicates to data...")

        # Get the template str for the dataset filename for saving
        # a ground atom dataset.
        dataset_fname, _ = utils.create_dataset_filename_str(True)
        # Add a bunch of things relevant to grammar search to the
        # dataset filename string.
        dataset_fname = dataset_fname[:-5] + \
            f"_{CFG.grammar_search_max_predicates}" + \
            f"_{CFG.grammar_search_grammar_includes_givens}" + \
            f"_{CFG.grammar_search_grammar_includes_foralls}" + \
            f"_{CFG.grammar_search_grammar_use_diff_features}" + \
            f"_{CFG.grammar_search_use_handcoded_debug_grammar}" + \
            dataset_fname[-5:]

        # Load pre-saved data if the CFG.load_atoms flag is set.
        atom_dataset: Optional[List[GroundAtomTrajectory]] = None
        if CFG.load_atoms:
            atom_dataset = utils.load_ground_atom_dataset(
                dataset_fname, dataset.trajectories)
        else:
            atom_dataset = utils.create_ground_atom_dataset(
                dataset.trajectories,
                set(candidates) | self._initial_predicates)
            # Save this atoms dataset if the save_atoms flag is set.
            if CFG.save_atoms:
                utils.save_ground_atom_dataset(atom_dataset, dataset_fname)
        logging.info("Done.")
        assert atom_dataset is not None
        return (atom_dataset, candidates)

    def _parse_atom_dataset_from_annotated_dataset(
        self, dataset: Dataset
    ) -> Tuple[List[GroundAtomTrajectory], Dict[Predicate, float]]:
        """Uses a dataset with annotations to create a candidate predicate set
        and atoms trajectories."""
        # We rely on the annotations as our ground atom datasets.
        assert dataset.annotations is not None
        # We now turn these into GroundAtomTrajectories.
        atom_dataset = []
        for traj, atoms in zip(dataset.trajectories, dataset.annotations):
            atom_dataset.append((traj, atoms))
        # Also generate the grammar by ripping out all the Predicates
        # associated with each of the atoms in our sets.
        candidates = {}
        for ano_traj in dataset.annotations:
            for ground_atom_state in ano_traj:
                for ground_atom in ground_atom_state:
                    assert isinstance(ground_atom, GroundAtom)
                    if ground_atom.predicate not in candidates:
                        # The cost of this predicate is simply its arity.
                        candidates[ground_atom.predicate] = float(
                            len(ground_atom.objects))
        logging.debug(f"All candidate predicates: {candidates.keys()}")
        return (atom_dataset, candidates)

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        if CFG.offline_data_method in [
                "geo_and_demo_with_vlm_imgs", "geo_and_demo+labelled_atoms",
                "geo_and_saved_vlm_img_demos_folder"
        ]:
            atom_dataset_from_grammar, candidates_from_grammar = \
                self._generate_atom_dataset_via_grammar(dataset)
            atom_dataset_from_vlm, candidates_from_vlm = \
                self._parse_atom_dataset_from_annotated_dataset(dataset)
            atom_dataset = utils.merge_ground_atom_datasets(
                atom_dataset_from_grammar, atom_dataset_from_vlm)
            candidates = candidates_from_grammar | candidates_from_vlm
        elif not CFG.offline_data_method in [
                "demo+labelled_atoms", "saved_vlm_img_demos_folder",
                "demo_with_vlm_imgs"
        ]:
            atom_dataset, candidates = self._generate_atom_dataset_via_grammar(
                dataset)
        else:
            # In this case, we're inventing over already-labelled atoms.
            atom_dataset, candidates = \
                self._parse_atom_dataset_from_annotated_dataset(dataset)
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
            self._learned_predicates = self._select_predicates_by_clustering(
                candidates, self._initial_predicates, dataset, atom_dataset)
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

    def _select_predicates_by_clustering(
            self, candidates: Dict[Predicate, float],
            initial_predicates: Set[Predicate], dataset: Dataset,
            atom_dataset: List[GroundAtomTrajectory]) -> Set[Predicate]:
        """Cluster segments from the atom_dataset into clusters corresponding
        to operators and use this to select predicates."""

        if CFG.grammar_search_pred_clusterer == "option-type-number-sample":
            # This procedure tries to reverse engineer the clusters of segments
            # that correspond to the oracle operators and selects predicates
            # that are add effects in those clusters, letting pnad_search
            # downstream handle chainability.

            assert CFG.segmenter == "option_changes"
            segments = [
                seg for ll_traj, atom_seq in atom_dataset for seg in
                segment_trajectory(ll_traj, initial_predicates, atom_seq)
            ]

            # Step 1:
            # Cluster segments by the option that generated them. We know that
            # at the very least, operators are 1 to 1 with options.
            option_to_segments: Dict[Any, Any] = {}  # Dict[str, List[Segment]]
            for seg in segments:
                name = seg.get_option().name
                option_to_segments.setdefault(name, []).append(seg)
            logging.info(f"STEP 1: generated {len(option_to_segments.keys())} "
                         f"option-based clusters.")
            clusters = option_to_segments.copy()  # Tree-like structure.

            # Step 2:
            # Further cluster by the types that appear in a segment's add
            # effects. Operators have a fixed number of typed arguments.
            for i, pair in enumerate(option_to_segments.items()):
                option, segments = pair
                types_to_segments: Dict[Tuple[Type, ...], List[Segment]] = {}
                for seg in segments:
                    types_in_effects = [
                        set(a.predicate.types) for a in seg.add_effects
                    ]
                    # To cluster on type, there must be types. That is, there
                    # must be add effects in the segment and the object
                    # arguments for at least one add effect must be nonempty.
                    assert len(types_in_effects) > 0 and len(
                        set.union(*types_in_effects)) > 0
                    types = tuple(sorted(list(set.union(*types_in_effects))))
                    types_to_segments.setdefault(types, []).append(seg)
                logging.info(
                    f"STEP 2: generated {len(types_to_segments.keys())} "
                    f"type-based clusters for cluster {i+1} from STEP 1 "
                    f"involving option {option}.")
                clusters[option] = types_to_segments

            # Step 3:
            # Further cluster by the maximum number of objects that appear in a
            # segment's add effects. Note that the use of maximum here is
            # somewhat arbitrary. Alternatively, you could cluster for every
            # possible number of objects and not the max among what you see in
            # the add effects of a particular segment.
            for i, (option, types_to_segments) in enumerate(clusters.items()):
                for j, (types,
                        segments) in enumerate(types_to_segments.items()):
                    num_to_segments: Dict[int, List[Segment]] = {}
                    for seg in segments:
                        max_num_objs = max(
                            len(a.objects) for a in seg.add_effects)
                        num_to_segments.setdefault(max_num_objs,
                                                   []).append(seg)
                    logging.info(
                        f"STEP 3: generated {len(num_to_segments.keys())} "
                        f"num-object-based clusters for cluster {i+j+1} from "
                        f"STEP 2 involving option {option} and type {types}.")
                    clusters[option][types] = num_to_segments

            # Step 4:
            # Further cluster by sample, if a sample is present. The idea here
            # is to separate things like PickFromTop and PickFromSide.
            for i, (option, types_to_num) in enumerate(clusters.items()):
                for j, (types,
                        num_to_segments) in enumerate(types_to_num.items()):
                    for k, (max_num_objs,
                            segments) in enumerate(num_to_segments.items()):
                        # If the segments in this cluster have no sample, then
                        # don't cluster further.
                        if len(segments[0].get_option().params) == 0:
                            clusters[option][types][max_num_objs] = [segments]
                            logging.info(
                                f"STEP 4: generated no further sample-based "
                                f"clusters (no parameter) for cluster {i+j+k+1}"
                                f" from STEP 3 involving option {option}, type "
                                f"{types}, and max num objects {max_num_objs}."
                            )
                            continue
                        # pylint: disable=line-too-long
                        # If the parameters are described by a uniform
                        # distribution, then don't cluster further. This
                        # helps prevent overfitting. A proper implementation
                        # would do a multi-dimensional test
                        # (https://ieeexplore.ieee.org/document/4767477,
                        # https://ui.adsabs.harvard.edu/abs/1987MNRAS.225..155F/abstract,
                        # https://stats.stackexchange.com/questions/30982/how-to-test-uniformity-in-several-dimensions)
                        # but for now we will only check each dimension
                        # individually to keep the implementation simple.
                        # pylint: enable=line-too-long
                        samples = np.array(
                            [seg.get_option().params for seg in segments])
                        each_dim_uniform = True
                        for d in range(samples.shape[1]):
                            col = samples[:, d]
                            minimum = col.min()
                            maximum = col.max()
                            null_hypothesis = np.random.uniform(
                                minimum, maximum, len(col))
                            p_value = kstest(col, null_hypothesis).pvalue

                            # We use a significance value of 0.05.
                            if p_value < 0.05:
                                each_dim_uniform = False
                                break
                        if each_dim_uniform:
                            clusters[option][types][max_num_objs] = [segments]
                            logging.info(
                                f"STEP 4: generated no further sample-based"
                                f" clusters (uniformly distributed "
                                f"parameter) for cluster {i+j+k+1} "
                                f"from STEP 3 involving option {option}, "
                                f" type {types}, and max num objects "
                                f"{max_num_objs}.")
                            continue
                        # Determine clusters by assignment from a
                        # Gaussian Mixture Model. The number of
                        # components and the negative weighting on the
                        # complexity of the model (chosen by BIC here)
                        # are hyperparameters.
                        max_components = min(
                            len(samples), len(np.unique(samples)),
                            CFG.grammar_search_clustering_gmm_num_components)
                        n_components = np.arange(1, max_components + 1)
                        models = [
                            GMM(n, covariance_type="full",
                                random_state=0).fit(samples)
                            for n in n_components
                        ]
                        bic = [m.bic(samples) for m in models]
                        best = models[np.argmin(bic)]
                        assignments = best.predict(samples)
                        label_to_segments: Dict[int, List[Segment]] = {}
                        for l, assignment in enumerate(assignments):
                            label_to_segments.setdefault(
                                assignment, []).append(segments[l])
                        clusters[option][types][max_num_objs] = list(
                            label_to_segments.values())
                        logging.info(f"STEP 4: generated "
                                     f"{len(label_to_segments.keys())}"
                                     f"sample-based clusters for cluster "
                                     f"{i+j+k+1} from STEP 3 involving option "
                                     f"{option}, type {types}, and max num "
                                     f"objects {max_num_objs}.")

            # We could avoid these loops by creating the final set of clusters
            # as part of STEP 4, but this is not prohibitively slow and serves
            # to clarify the nested dictionary structure, which we may make use
            # of in follow-up work that modifies the clusters more.
            final_clusters = []
            for option in clusters.keys():
                for types in clusters[option].keys():
                    for max_num_objs in clusters[option][types].keys():
                        for cluster in clusters[option][types][max_num_objs]:
                            final_clusters.append(cluster)
            logging.info(f"Total {len(final_clusters)} final clusters.")

            # Step 5:
            # Extract predicates from the pure intersection of the add effects
            # in each cluster.
            extracted_preds = set()
            shared_add_effects_per_cluster = []
            for c in final_clusters:
                grounded_add_effects_per_segment = [
                    seg.add_effects for seg in c
                ]
                ungrounded_add_effects_per_segment = []
                for effs in grounded_add_effects_per_segment:
                    ungrounded_add_effects_per_segment.append(
                        set(a.predicate for a in effs))
                shared_add_effects_in_cluster = set.intersection(
                    *ungrounded_add_effects_per_segment)
                shared_add_effects_per_cluster.append(
                    shared_add_effects_in_cluster)
                extracted_preds |= shared_add_effects_in_cluster

            # Step 6:
            # Remove inconsistent predicates except if removing them prevents us
            # from disambiguating two or more clusters (i.e. their add effect
            # sets are the same after removing the inconsistent predicates). The
            # idea here is that HoldingTop and HoldingSide are inconsistent
            # within the PlaceOnTable cluster in painting, but we don't want to
            # remove them, since we had generated them specifically to
            # disambiguate segments in the cluster with the Pick option.
            # A consistent predicate is either an add effect, a delete
            # effect, or doesn't change, within each cluster, for all clusters.
            # Note that it is possible that when 2 inconsistent predicates are
            # removed, then two clusters cannot be disambiguated, but if you
            # keep either of the two, then you can disambiguate the clusters.
            # For now, we just add both back, which is not ideal.
            consistent, inconsistent = self._get_consistent_predicates(
                extracted_preds, list(final_clusters))
            predicates_to_keep: Set[Predicate] = consistent
            consistent_shared_add_effects_per_cluster = [
                add_effs - inconsistent
                for add_effs in shared_add_effects_per_cluster
            ]
            num_clusters = len(final_clusters)
            for i in range(num_clusters):
                for j in range(num_clusters):
                    if i == j:
                        continue
                    if consistent_shared_add_effects_per_cluster[
                            i] == consistent_shared_add_effects_per_cluster[j]:
                        logging.info(
                            f"Final clusters {i} and {j} cannot be "
                            f"disambiguated after removing the inconsistent"
                            f" predicates.")
                        predicates_to_keep |= \
                            shared_add_effects_per_cluster[i]
                        predicates_to_keep |= \
                            shared_add_effects_per_cluster[j]

            # Remove the initial predicates.
            predicates_to_keep -= initial_predicates

            logging.info(
                f"\nSelected {len(predicates_to_keep)} predicates out of "
                f"{len(candidates)} candidates:")
            for pred in sorted(predicates_to_keep):
                logging.info(f"{pred}")
            return predicates_to_keep

        if CFG.grammar_search_pred_clusterer == "oracle":
            assert CFG.offline_data_method == "demo+gt_operators"
            assert dataset.annotations is not None and len(
                dataset.annotations) == len(dataset.trajectories)
            assert CFG.segmenter == "option_changes"
            preds = set(initial_predicates) | initial_predicates
            segmented_trajs = [
                segment_trajectory(ll_traj, preds, atom_seq)
                for ll_traj, atom_seq in atom_dataset
            ]
            assert len(segmented_trajs) == len(dataset.annotations)
            # First, get the set of all ground truth operator names.
            all_gt_ops = set(ground_nsrt.parent
                             for anno_list in dataset.annotations
                             for ground_nsrt in anno_list)
            # Next, make a dictionary mapping operator name to segments
            # where that operator was used.
            gt_op_to_segments: Dict[str, List[Segment]] = {
                op_name: []
                for op_name in all_gt_ops
            }
            for op_list, seg_list in zip(dataset.annotations, segmented_trajs):
                assert len(seg_list) == len(op_list)
                for ground_nsrt, segment in zip(op_list, seg_list):
                    gt_op_to_segments[ground_nsrt.parent].append(segment)

            # Before selecting some subset of predicates to keep, we first
            # get the set of all predicates that ever change.
            non_static_predicates: Set[Predicate] = set()
            for segment in {x for v in gt_op_to_segments.values() for x in v}:
                for add_eff in segment.add_effects:
                    non_static_predicates.add(add_eff.predicate)
                for del_eff in segment.delete_effects:
                    non_static_predicates.add(del_eff.predicate)

            # Finally, select predicates that are consistent (either, it is
            # an add effect, or a delete effect, or doesn't change)
            # within all demos.
            predicates_to_keep, _ = self._get_consistent_predicates(
                non_static_predicates, list(gt_op_to_segments.values()))

            # Before returning, remove all the initial predicates.
            predicates_to_keep -= initial_predicates
            logging.info(
                f"\nSelected {len(predicates_to_keep)} predicates out of "
                f"{len(candidates)} candidates:")
            for pred in predicates_to_keep:
                logging.info(f"\t{pred}")
            return predicates_to_keep

        raise NotImplementedError(
            "Unrecognized clusterer for predicate " +
            f"invention {CFG.grammar_search_pred_clusterer}.")
