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
    if CFG.grammar_search_grammar_use_euclidean_dist:
        euclidean_dist_grammar = _EuclideanDistancePredicateGrammar(dataset)
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
class _EuclideanAttributeDiffCompareClassifier(_AttributeDiffCompareClassifier
                                               ):
    """Compare the euclidean distance between two feature values with a
    constant value."""

    def _classify_object(self, s: State, obj1: Object, obj2: Object) -> bool:
        assert obj1.type == self.object1_type
        assert obj2.type == self.object2_type
        return self.compare((s.get(obj1, self.attribute1_name) -
                             s.get(obj2, self.attribute1_name))**2 +
                            (s.get(obj1, self.attribute2_name) -
                             s.get(obj2, self.attribute2_name))**2,
                            self.constant)

    def __str__(self) -> str:
        return (f"((({self.object1_index}:{self.object1_type.name})."
                f"{self.attribute1_name} - ({self.object2_index}:"
                f"{self.object2_type.name}).{self.attribute1_name})^2"
                f" + (({self.object1_index}:{self.object1_type.name})."
                f"{self.attribute2_name} - ({self.object2_index}:"
                f"{self.object2_type.name}).{self.attribute2_name})^2)"
                f"{self.compare_str}[idx {self.constant_idx}]"
                f"{self.constant:.3})")

    def pretty_str(self) -> Tuple[str, str]:
        name1 = CFG.grammar_search_classifier_pretty_str_names[
            self.object1_index]
        name2 = CFG.grammar_search_classifier_pretty_str_names[
            self.object2_index]
        vars_str = (f"{name1}:{self.object1_type.name}, "
                    f"{name2}:{self.object2_type.name}")
        body_str = (f"(({name1}.{self.attribute1_name} - "
                    f"{name2}.{self.attribute2_name})^2 "
                    f" + (({name1}.{self.attribute1_name} - "
                    f"{name2}.{self.attribute2_name})^2 "
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
    "stick_button_move": [
        # NOTE: sometimes (especially with more demos) we need
        # to use 0.146 and 0.207 for the first two constants
        # respectively.
        # "(|(0:button).x - (1:stick).x|<=[idx 0]0.144)",
        "(|(0:button).x - (1:stick).x|<=[idx 0]0.145)",  # StickAboveButton
        # RobotAboveButton
        # "(|(0:button).x - (1:stick).x|<=[idx 0]0.146)",
        "(((0:button).x - (1:robot).x)^2 + ((0:button).y - (1:robot).y)^2)<=[idx 0]0.204)",
        # "(((0:button).x - (1:robot).x)^2 + ((0:button).y - (1:robot).y)^2)<=[idx 0]0.206)",
        # "(((0:button).x - (1:robot).x)^2 + ((0:button).y - (1:robot).y)^2)"+ \
        #     "<=[idx 0]0.207)",
        "((0:stick).held<=[idx 0]0.5)",  # Handempty
        "NOT-((0:stick).held<=[idx 0]0.5)",  # Grasped
        "((0:button).y<=[idx 0]2.96)",  # ButtonReachable
        "Forall[0:button,1:stick].[NOT-(|(0:button).x - (1:stick).x|<=[idx 0]0.145)(0,1)]"
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
                    lb, ub = utils.compute_abs_bounds_given_frange(
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
        # very close together.
        for ret_val in self._yield_pred_given_const(feature_ranges, 0,
                                                    (1 / 60.0), 4.0):
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
    named "x" and "y".
    """

    def _compute_xy_bounds(self, feature_ranges: Dict[Type,
                                                      Dict[str, Tuple[float,
                                                                      float]]],
                           t1: Type, t2: Type) -> Tuple[float, float]:
        # To create our classifier, we need to leverage the
        # upper and lower bounds of its x, y features.
        lbx1, ubx1 = feature_ranges[t1]["x"]
        lbx2, ubx2 = feature_ranges[t2]["x"]
        lby1, uby1 = feature_ranges[t1]["y"]
        lby2, uby2 = feature_ranges[t2]["y"]
        # Compute the upper and lower bounds of each feature range.
        lbx, ubx = utils.compute_abs_bounds_given_frange(
            lbx1, ubx1, lbx2, ubx2)
        lby, uby = utils.compute_abs_bounds_given_frange(
            lby1, uby1, lby2, uby2)
        # Now, use these to compute the upper and lower bounds of
        # the squared expression of interest.
        lb = lbx**2 + lby**2
        ub = ubx**2 + uby**2
        return (lb, ub)

    def _generate_pred_given_constant(self, constant_idx: int, constant: float,
                                      t1: Type, t2: Type) -> Predicate:
        # Create classifier.
        comp, comp_str = le, "<="
        diff_classifier = _EuclideanAttributeDiffCompareClassifier(
            0, t1, "x", 1, t2, "y", constant, constant_idx, comp, comp_str)
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
            if t1 == t2:
                continue
            if ("x" in t1.feature_names and "x" in t2.feature_names
                    and "y" in t1.feature_names and "y" in t2.feature_names):
                lb, ub = self._compute_xy_bounds(feature_ranges, t1, t2)
                constant = ((ub - lb) / 500) + lb
                pred = self._generate_pred_given_constant(0, constant, t1, t2)
                assert pred.arity == 2
                yield (pred, 3.0)  # cost = arity + cost from constant

        # 0.5, 0.25, 0.75, 0.125, 0.375, ...
        constant_generator = _halving_constant_generator(0.0, 1.0)
        for constant_idx, (constant, cost) in enumerate(constant_generator):
            for (t1, t2) in itertools.combinations_with_replacement(
                    sorted(self.types), 2):
                if t1 == t2:
                    continue
                if ("x" in t1.feature_names and "x" in t2.feature_names
                        and "y" in t1.feature_names
                        and "y" in t2.feature_names):
                    lb, ub = self._compute_xy_bounds(feature_ranges, t1, t2)
                    # Scale the constant by the correct range.
                    k = constant * (ub - lb) + lb
                    pred = self._generate_pred_given_constant(
                        constant_idx + 1, k, t1, t2)
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

    def _check_single_demo_preservation(
            self, seg_traj: List[Segment], init_state: State,
            atoms_seq: List[Set[GroundAtom]], traj_goal: Set[GroundAtom],
            strips_ops: List[STRIPSOperator],
            option_specs: List[OptionSpec],
            predicates_we_kept: Set[Predicate]) -> bool:
        """Function to check whether a given set of operators preserves a
        single training trajectory."""
        from predicators.planning import task_plan_with_option_plan_constraint
        # init_atoms = utils.abstract(init_state, self._predicates)
        init_atoms = utils.abstract(init_state, predicates_we_kept)
        objects = set(init_state)
        option_plan = []
        for seg in seg_traj:
            if seg.has_option():
                option = seg.get_option()
            else:
                option = DummyOption
            option_plan.append((option.parent, option.objects))
        ground_nsrt_plan = task_plan_with_option_plan_constraint(objects, predicates_we_kept, strips_ops, option_specs, init_atoms, traj_goal, option_plan, atoms_seq)

        # if ground_nsrt_plan is None:
        #     print("112")
        #     ground_nsrt_plan = task_plan_with_option_plan_constraint2(objects, predicates_we_kept, strips_ops, option_specs, init_atoms, traj_goal, option_plan, atoms_seq)
        #     print("312")
        #     import pdb; pdb.set_trace()
        #
        #     for k, v in init_state.data.items(): print(f"{k}: {v.tolist()}")
        #
        #     Op0Pick = [op for op in strips_ops if op.name=="Op0-Pick"][0]
        #     Op1Pick = [op for op in strips_ops if op.name=="Op1-Pick"][0]
        #     Op2Stack = [op for op in strips_ops if op.name=="Op2-Stack"][0]
        #     Op3PutOnTable = [op for op in strips_ops if op.name=="Op3-PutOnTable"][0]
        #     # Op4PutOnTable = [op for op in strips_ops if op.name=="Op4-PutOnTable"][0]
        #
        #     objs = list(init_state.data.keys())
        #     robot = [o for o in objs if o.name == "robby"][0]
        #     block0 = [b for b in objs if b.name == "block0"][0]
        #     block1 = [b for b in objs if b.name == "block1"][0]
        #     block2 = [b for b in objs if b.name == "block2"][0]
        #     block3 = [b for b in objs if b.name == "block3"][0]
        #     # block4 = [b for b in objs if b.name == "block4"][0]
        #     # block5 = [b for b in objs if b.name == "block5"][0]
        #
        #     first = Op0Pick.ground((block2, block3, robot))
        #     second = Op3PutOnTable.ground((block3, robot))
        #     third = Op0Pick.ground((block1, block2, robot))
        #     fourth = Op2Stack.ground((block2, block3, robot))
        #     fifth = Op1Pick.ground((block1, robot))
        #     sixth = Op2Stack.ground((block1, block0, robot))
        #     seventh = Op0Pick.ground((block3, block2, robot))
        #     eighth = Op2Stack.ground((block2, block1, robot))
        #
            # first.preconditions.issubset(init_atoms)
            # after_first = (init_atoms | first.add_effects) - first.delete_effects
            # second.preconditions.issubset(after_first)
            # after_second = (after_first | second.add_effects) - second.delete_effects
            # third.preconditions.issubset(after_second)
            # after_third = (after_second | third.add_effects) - third.delete_effects
            # fourth.preconditions.issubset(after_third)
            # after_fourth = (after_third | fourth.add_effects) - fourth.delete_effects
            # fifth.preconditions.issubset(after_fourth)
            # after_fifth = (after_fourth | fifth.add_effects) - fifth.delete_effects
            # sixth.preconditions.issubset(after_fifth)
            # after_sixth = (after_fifth | sixth.add_effects) - sixth.delete_effects
            # seventh.preconditions.issubset(after_sixth)
            # after_seventh = (after_sixth | seventh.add_effects) - seventh.delete_effects
            # eighth.preconditions.issubset(after_sixth)
            # after_eighth = (after_seventh | eighth.add_effects) - eighth.delete_effects
            # traj_goal.issubset(after_eighth)
            # my_plan = [first, second, third, fourth, fifth, sixth, seventh, eighth]

        return ground_nsrt_plan is not None

    def get_score(self, all_clusters, example_segment, segmented_trajs, jjj, kkk, candidates, initial_predicates, dataset, atom_dataset):
        ####
        # for cluster in all_clusters:
        #     for segment in cluster:
        #         # make sure you can find this in some trajectory
        #         assert any(segment in seg for seg in segmented_trajs)
        # import pdb; pdb.set_trace()
        ####
        indices_of_sub_clusters = set()

        option_name = example_segment.get_option().name
        final_clusters2 = []
        for x, c in enumerate(all_clusters):
            if x != jjj:
                final_clusters2.append(c)
        final_clusters = final_clusters2
        # import pdb; pdb.set_trace()

        if len(example_segment.get_option().params) == 0:
            final_clusters.append(all_clusters[jjj])
            # final_clusters.append(cluster)
            logging.info(f"STEP 4: generated no further sample-based clusters (no parameter!) for the {jjj+1}th cluster from STEP 3 involving option {option_name}.")
            return None
        else:
            import numpy as np
            from sklearn.mixture import GaussianMixture as GMM
            from scipy.stats import kstest
            # data = np.array([seg.get_option().params for seg in cluster])
            data = np.array([seg.get_option().params for seg in all_clusters[jjj]])

            if len(data) == 1:
                # Can't fit GMM to 1 data point, need at least 2
                # ValueError: Found array with 1 sample(s) (shape=(1, 1)) while a minimum of 2 is required. for GMM.fit() method
                # Don't subcluster
                final_clusters.append(all_clusters[jjj])
                indices_of_sub_clusters.add(len(final_clusters) - 1)
            else:
                model = GMM(kkk, covariance_type="full", random_state=0).fit(data)
                assignments = model.predict(data)
                sub_clusters = {}
                for i, assignment in enumerate(assignments):
                    if assignment in sub_clusters:
                        # sub_clusters[assignment].append(cluster[i])
                        sub_clusters[assignment].append(all_clusters[jjj][i])
                    else:
                        sub_clusters[assignment] = [all_clusters[jjj][i]]
                for c in sub_clusters.values():
                    final_clusters.append(c)
                    indices_of_sub_clusters.add(len(final_clusters) - 1)

        # import pdb; pdb.set_trace()

        logging.info(f"Total {len(final_clusters)} final clusters, evaluating k={kkk} for {option_name}.")

        ####
        # this fails, as expected
        # for traj in segmented_trajs:
        #     for seg in traj:
        #         assert any(seg in c for c in final_clusters)
        # import pdb; pdb.set_trace()
        ####

        ########################################################
        ########################################################
        # Everything put here to get the best k
        ########################################################
        ########################################################
        ddd = {}
        for i, c in enumerate(final_clusters):
            op_name = "Op"+str(i)+"-"+str(c[0].get_option().name)
            # preconditions, add effects, delete effects, segments
            ddd[op_name] = [set(), set(), set(), c]

        # For clusters that appear at the end,
        # we can narrow down their add effects as:
        # they DEFINITELY have an add effect that whatever goal atoms
        # were not true in the pre-image.
        # But beyond that, they may have other effects?
        #


        ####

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

            # if "Forall[0:block].[((0:block).pose_z<=[idx 1]0.342)(0)]" in [str(p) for p in add_effects]:
            #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

            #
            # ex = c[0]
            # print(f"Cluster {j} with option {ex.get_option().name}:")
            # for a in add_effects:
            #     print(a)
            #

        # import pdb; pdb.set_trace()

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
                        # if str(pred) == "Forall[0:block].[((0:block).pose_z<=[idx 1]0.342)(0)]":
                        #     import pdb; pdb.set_trace()
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
        print("got to hereee")
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
                        for p in add_effects_per_cluster[i] | add_effects_per_cluster[j]:
                            if p in inconsistent_preds:
                                print(f"ADDING BACK: {p}")

        # Remove the initial predicates.
        # predicates_to_keep -= initial_predicates

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

        # import pdb; pdb.set_trace()
        logging.info("Performing backchaining to decide operator definitions.")

        # print("DOING CHECK SEEING IF DDD IS THE SAME")
        # for op_name, v in final_predicate_operator_definitions.items():
        #     print(f"doing ddd check for {op_name}")
        #     clus = op_pred_def_superset[op_name][3]
        #     assert ddd[op_name][0] == v["pre"]
        #     assert ddd[op_name][1] == v["add"]
        #     assert ddd[op_name][2] == v["del"]
        #     assert set(ddd[op_name][3]) == set(clus)
        #
        # print("DONE WITH CHECK SEEING IF DDD IS THE SAME")

        #####################
        # backchaining
        #####################
        def seg_to_op(segment, clusters):
            for i, c in enumerate(clusters):
                if segment in c:
                    # return f"Op{i}-{c[0].get_option().name} with objects ({c[0].get_option().objects})"
                    return f"Op{i}-{c[0].get_option().name}"
            return None

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

        def seg_to_op2(segment, clusters):
            for i, c in enumerate(clusters):
                # if segment in c:
                if seg_in_cluster(segment, c):
                    # return f"Op{i}-{c[0].get_option().name} with objects ({c[0].get_option().objects})"
                    return f"Op{i}-{c[0].get_option().name}"
            print("couldn't find segment...")

        # Go through demos
        temp = []
        for a, segmented_traj in enumerate(segmented_trajs):
            # if a == 10:
            #     break
            traj = []
            for seg in segmented_traj:
                # if seg_to_op(seg, final_clusters) is None:
                #     import pdb; pdb.set_trace()
                traj.append(seg_to_op(seg, final_clusters))
            temp.append(traj)

            print(traj)
            # if a == 1:
            #     import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()

        ttt = list(predicates_to_keep)
        for p in ttt:
            print(p)

        def print_demo2(segmented_traj, filename):

            max_length = 0
            with open(filename, 'w') as file:
                # preconditions
                preconds = []
                for seg in segmented_traj:
                    op_name = seg_to_op(seg, final_clusters)

                    ps = sorted([p.name for p in ddd[op_name][0]])
                    init_atoms = set(p for p in seg.init_atoms if p.predicate.name in ps)
                    # we assume we only have to look at objects involved in the operator,
                    # so we can ignore all other ground atoms.
                    # can figure out which objects change in terms of predicates.
                    objects = set(o for a in seg.add_effects for o in a.objects) |set(o for a in seg.delete_effects for o in a.objects)
                    relevant_init_atoms = set(p for p in init_atoms if set(p.objects).issubset(objects))
                    relevant_init_atoms_str = sorted([str(p) for p in relevant_init_atoms])
                    for p in relevant_init_atoms_str:
                        max_length = max(max_length, len(p))
                    to_print = [op_name, '----', 'preconditions', '===='] + relevant_init_atoms_str
                    preconds.append(to_print)

                    # if op_name == "Op3-FastenScrewByHand":
                    #     import pdb; pdb.set_trace()

                # add effects
                add_effs = []
                for seg in segmented_traj:
                    op_name = seg_to_op(seg, final_clusters)
                    adds = sorted([str(p) for p in seg.add_effects if p.predicate in ddd[op_name][1]])
                    for p in adds:
                        max_length = max(max_length, len(p))
                    adds = ['add effects', '===='] + adds
                    add_effs.append(adds)

                # delete effects
                del_effs = []
                for seg in segmented_traj:
                    op_name = seg_to_op(seg, final_clusters)
                    dels = sorted([str(p) for p in seg.delete_effects if p.predicate in ddd[op_name][2]])
                    for p in dels:
                        max_length = max(max_length, len(p))
                    dels = ['delete effects', '===='] + dels
                    del_effs.append(dels)

                max_length += 3

                max_lst_len = max(len(lst) for lst in preconds)
                for i in range(max_lst_len):
                    s = " ".join(f"{lst[i]:<{max_length}}" if i < len(lst) else " " * max_length for lst in preconds)
                    file.write(s + '\n')
                file.write('\n')
                max_lst_len = max(len(lst) for lst in add_effs)
                for i in range(max_lst_len):
                    s = " ".join(f"{lst[i]:<{max_length}}" if i < len(lst) else " " * max_length for lst in add_effs)
                    file.write(s + '\n')
                file.write('\n')
                max_lst_len = max(len(lst) for lst in del_effs)
                for i in range(max_lst_len):
                    s = " ".join(f"{lst[i]:<{max_length}}" if i < len(lst) else " " * max_length for lst in del_effs)
                    file.write(s + '\n')
                file.write('\n')

        # print_demo2(segmented_trajs[8][0:3], "demo8_part1.txt")
        # print_demo2(segmented_trajs[8][3:], "demo8_part12.txt")
        # print_demo2(segmented_trajs[0], "demo0_blocks.txt")
        # import pdb; pdb.set_trace()

        # for i, t in enumerate(temp):
        #     for o in t:
        #         if o in ["Op4-Place", "Op3-Place"]:
        #             print(f"demo {i} has {o}")
        #
        # print_demo2(segmented_trajs[2], "demo2.txt")
        # import pdb; pdb.set_trace()

        ###################
        # ALGORITHM
        #

        def get_story(segmented_traj):
            story = []
                # each entry is a list of lists
            for seg in segmented_traj:
                entry = []
                op_name = seg_to_op(seg, final_clusters)
                init_atoms = set(p for p in seg.init_atoms if p.predicate in ddd[op_name][0])
                objects = (set(o for a in seg.add_effects for o in a.objects) | set(o for a in seg.delete_effects for o in a.objects))
                preconditions = set(p for p in init_atoms if set(p.objects).issubset(objects))
                add_effects = set(p for p in seg.add_effects if p.predicate in ddd[op_name][1])
                delete_effects = set(p for p in seg.delete_effects if p.predicate in ddd[op_name][2])

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
        def get_story2(segmented_traj):
            story = []
                # each entry is a list of lists
            for k, seg in enumerate(segmented_traj):
                # if k == 2:
                #     import pdb; pdb.set_trace()
                entry = []
                op_name = seg_to_op(seg, final_clusters)
                # op_name2 = seg_to_op2(seg, final_clusters)
                # if op_name != op_name2:
                #     import pdb; pdb.set_trace()

                # init_atoms = set(p for p in seg.init_atoms if p.predicate in ddd[op_name][0])
                # objects = (set(o for a in seg.add_effects for o in a.objects) | set(o for a in seg.delete_effects for o in a.objects))
                # preconditions = set(p for p in init_atoms if set(p.objects).issubset(objects))
                # add_effects = set(p for p in seg.add_effects if p.predicate in ddd[op_name][1])
                # delete_effects = set(p for p in seg.delete_effects if p.predicate in ddd[op_name][2])

                opt_objs = tuple(seg.get_option().objects)
                objects = (set(o for a in seg.add_effects for o in a.objects) | set(o for a in seg.delete_effects for o in a.objects) | set(opt_objs))
                try:
                    preconditions_hypothesis = set(p for p in seg.init_atoms if p.predicate in ddd[op_name][0])
                except:
                    import pdb; pdb.set_trace()
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

        # import pdb; pdb.set_trace()

        def process_adds(potential_ops, remaining_story, goal):
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
                        # if "pre" not in potential_ops[entry[0]].keys():
                        #     import pdb; pdb.set_trace()
                        potential_ops[entry[0]]["pre"].add(ground_atom.predicate)
                if ground_atom in goal:
                    # then we add it!
                    to_add.append(ground_atom) # for debugging
                    potential_ops[name]["add"].add(ground_atom.predicate)

        def process_adds2(potential_ops, remaining_story, goal):
            # import pdb; pdb.set_trace()
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
                        # if "pre" not in potential_ops[entry[0]].keys():
                        #     import pdb; pdb.set_trace()
                        potential_ops[entry[0]]["pre"].add(ground_atom.predicate)
                if ground_atom in goal:
                    # then we add it!
                    to_add.append(ground_atom) # for debugging
                    potential_ops[name]["add"].add(ground_atom.predicate)

        def process_dels(potential_ops, potential_ops2, remaining_story):
            curr_entry = remaining_story[0]
            name = curr_entry[0]
            to_del = []
            for ground_atom in curr_entry[3]: # delete effects
                # is this atom every added in the future?
                # *** do we need to do this with an updated potential_ops after process_adds runs on all the trajectories?
                for entry in remaining_story[1:]:
                    if ground_atom in entry[2]: # add effects
                        to_del.append(str(ground_atom.predicate)) # for debugging
                        potential_ops[name]["del"].add(ground_atom.predicate)
                        potential_ops[entry[0]]["add"].add(ground_atom.predicate)
                        potential_ops2[entry[0]]["add"].add(ground_atom.predicate)

                        # if name == "Op0-Pick" and "Forall[0:block].[((0:block).pose_z<=[idx 1]0.342)(0)]" in to_del:
                        #     print(f"to_del: {to_del}")
                        #     import pdb; pdb.set_trace()


        all_potential_ops = []
        # TODO: change it so that you fill up a unique potential_ops at each iteration, and then take the union for the final one
        # potential_ops = {}
        # for op_name in ddd.keys():
        #     potential_ops[op_name] = {
        #         "pre": set(),
        #         "add": set(),
        #         "del": set()
        #     }

        all_potential_ops2 = []
        for j, traj in enumerate(segmented_trajs):
            print(f"Segmented trajectory #{j}")
            potential_ops = {}
            for op_name in ddd.keys():
                potential_ops[op_name] = {
                    "pre": set(),
                    "add": set(),
                    "del": set()
                }
            potential_ops2 = {}
            for op_name in ddd.keys():
                potential_ops2[op_name] = {
                    "pre": set(),
                    "add": set(),
                    "del": set()
                }

            skip = False
            for seg in traj:
                if seg_to_op(seg, final_clusters) is None:
                    skip = True
                    break
            if skip:
                logging.info(f"Skipping this trajectory because it has a segment that isn't in any cluster.")
                continue
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
                    if name == "Op12-Place" and j == 2:
                        process_adds2(potential_ops, remaining_story, self._train_tasks[j].goal)
                    else:
                        process_adds(potential_ops, remaining_story, self._train_tasks[j].goal)
                    process_dels(potential_ops, potential_ops2, remaining_story)


            all_potential_ops.append(potential_ops)
            all_potential_ops2.append(potential_ops2)

        # Leave out delete effects for now.
        final_potential_ops2 = {
            op_name: {
                "pre": ddd[op_name][0], # take all the precondition predicates
                "add": set(),
                "del": ddd[op_name][2]
            } for op_name in all_potential_ops[0].keys()
        }
        for k, po in enumerate(all_potential_ops):
            for op_name in po.keys():

                final_potential_ops2[op_name]["pre"] = final_potential_ops2[op_name]["pre"].union(po[op_name]["pre"])
                final_potential_ops2[op_name]["add"] = final_potential_ops2[op_name]["add"].union(po[op_name]["add"])
                final_potential_ops2[op_name]["del"] = final_potential_ops2[op_name]["del"].union(po[op_name]["del"])

        # Leave out delete effects for now.
        final_potential_ops = {
            op_name: {
                "pre": ddd[op_name][0], # take all the precondition predicates
                "add": set(),
                # "del": ddd[op_name][2]
                "del": set()
            } for op_name in all_potential_ops[0].keys()
        }
        for k, po in enumerate(all_potential_ops):
            for op_name in po.keys():
                final_potential_ops[op_name]["pre"] = final_potential_ops[op_name]["pre"].union(po[op_name]["pre"])
                final_potential_ops[op_name]["add"] = final_potential_ops[op_name]["add"].union(po[op_name]["add"])

        fff = {}
        for op in ddd.keys():
            fff[op] = []
            fff[op].append(
                set(p for p in ddd[op][0] if p in final_potential_ops2[op]["pre"])
            )
            fff[op].append(
                set(p for p in ddd[op][1] if p in final_potential_ops2[op]["add"])
            )
            fff[op].append(
                set(p for p in ddd[op][2] if p in final_potential_ops2[op]["del"])
            )
            fff[op].append(ddd[op][3])

        ########################################################
        # Choose delete effects that do not violate harmlessness
        ########################################################
        # Keep all delete effects
        # Learn nsrts
        # Go through each demo
        # use segment -> nsrt map to get the nsrt
        # pick the grounding that has the most add effects holding true?
        # => doing this will let us see if we recover the correct sequence of grounded nsrts, as a first test of this
        # then, from left to right in a demo, walk through -- and keep track of which atoms are deleted at each step (and which nsrt that was associated with)
        # if a precondition check ever fails, see if what is missing is from something that was deleted earlier
        # if so, don't include that delete effect in that operator



        ########################################################

        self._clusters = fff
        # import pdb; pdb.set_trace()

        # Learn pnads.
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

            option_vars = [obj_to_var[o] for o in opt_objs]
            option_spec = [seg_0.get_option().parent, option_vars]

            # if "Stack" in name:
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
                t = (params, objects_list, relevant_preconds, relevant_add_effects, relevant_del_effects, seg, opt_objs)
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

            # print(f"DOING THIS FOR {name}")
            # if name == "Op0-Pick":
            #     import pdb; pdb.set_trace()
            # if "Stack" in name:
            #     import pdb; pdb.set_trace()
            op1 = ops[0]
            op1_opt_objs = op1[-1]
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

                counter += 1

            option_vars = [op1_obj_to_var[o] for o in op1_opt_objs]
            option_spec = [seg_0.get_option().parent, option_vars]
            pnads.append(PNAD(op, datastore, option_spec))


        # import pdb; pdb.set_trace()
        logging.info(f"Right before delete effects harmlessness check.")
        ##################################
        # IDENTIFY HARMLESS DELETE EFFECTS
        ##################################
        # want a list, not a set, as the return value
        def _ops_and_specs_to_dummy_nsrts(strips_ops: Sequence[STRIPSOperator], option_specs: Sequence[OptionSpec]) -> Set[NSRT]:
            """Create NSRTs from strips operators and option specs with dummy
            samplers."""
            assert len(strips_ops) == len(option_specs)
            nsrts = []
            for op, (param_option, option_vars) in zip(strips_ops, option_specs):
                nsrt = op.make_nsrt(
                    param_option,
                    option_vars,  # dummy sampler
                    lambda s, g, rng, o: np.zeros(1, dtype=np.float32))
                nsrts.append(nsrt)

            return nsrts

        predicates_we_kept = set()
        for op_name in final_potential_ops2.keys():
            predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["pre"])
            predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["add"])
            predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["del"])
        pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, predicates_we_kept | initial_predicates)
        pruned_segmented_trajs = [segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in pruned_atom_data]
        pruned_low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        pruned_atom_seq = [atom_seq for _, atom_seq in pruned_atom_data]

        strips_ops = [pnad.op for pnad in pnads]
        option_specs = [pnad.option_spec for pnad in pnads]
        dummy_nsrts = _ops_and_specs_to_dummy_nsrts(strips_ops, option_specs)
        pnad_to_nsrt = dict(zip(pnads, dummy_nsrts))

        harmful_del_effs_per_nsrt = {p.op.name: set() for p in pnads}
        harmful_preconds_per_nsrt = {p.op.name: set() for p in pnads}

        low_level_trajs = [llt for llt, _ in atom_dataset]
        for ccc, tup in enumerate(zip(low_level_trajs, segmented_trajs)):
            ll_traj, seg_traj = tup
            # this segmented data has atoms from the entire grammar
            ground_nsrt_plan = []
            for seg in seg_traj:
                for pnad in pnads:
                    for seg2, sub in pnad.datastore:
                        if seg2 == seg:
                            nsrt = pnad_to_nsrt[pnad]
                            grounding = tuple(sub[p] for p in nsrt.parameters)
                            ground_nsrt = nsrt.ground(grounding)
                            ground_nsrt_plan.append(ground_nsrt)

            init_state = ll_traj.states[0]
            init_atoms = utils.abstract(init_state, predicates_we_kept)
            curr_abstract_state = init_atoms
            prev_nsrts = []
            for h, nsrt in enumerate(ground_nsrt_plan):
                if not nsrt.preconditions.issubset(curr_abstract_state):
                    missing = nsrt.preconditions - nsrt.preconditions.intersection(curr_abstract_state)
                    for miss in missing:
                        miss_caused_by_delete_effect = False
                        for prev in prev_nsrts[::-1]:
                            if miss in prev.delete_effects:
                                harmful_del_effs_per_nsrt[prev.op.name].add(miss.predicate)
                                miss_caused_by_delete_effect = True
                                break
                                # only care about most recent one => may have to do multiple passes,
                                # if two previous nsrts had it as a deleted effect (which may be unlikely or impossible, idk)

                        # TODO: do something about the condition where you don't find the miss anywhere
                        if not miss_caused_by_delete_effect:
                            # we should remove this predicate
                            harmful_preconds_per_nsrt[nsrt.op.name].add(miss.predicate)

                curr_abstract_state = (curr_abstract_state | nsrt.add_effects) - nsrt.delete_effects
                prev_nsrts.append(nsrt)

        logging.info(f"Harmful delete effects per nsrt: {harmful_del_effs_per_nsrt}")
        logging.info(f"Harmful preconditions per nsrt: {harmful_preconds_per_nsrt}")
        for k, v in harmful_del_effs_per_nsrt.items():
            for pred in v:
                final_potential_ops2[k]["del"].remove(pred)
                fff[k][2].remove(pred)
        for k, v in harmful_preconds_per_nsrt.items():
            for pred in v:
                final_potential_ops2[k]["pre"].remove(pred)
                fff[k][0].remove(pred)
        self._clusters = fff
        predicates_we_kept = set()
        for op_name in final_potential_ops.keys():
            predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["pre"])
            predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["add"])
            predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["del"])

        # import pdb; pdb.set_trace()
        logging.info(f"Right after delete effects harmlessness check.")
        # ########################################################################
        # ####################
        # # OLD HARMLESS CHECK
        # ####################
        # # Now check harmlessness. If not harmless, do not include the delete effect.
        # # If harmless, keep it as is.
        # strips_ops = [pnad.op for pnad in pnads]
        # option_specs = [pnad.option_spec for pnad in pnads]
        # predicates_we_kept = set()
        # for op_name in final_potential_ops.keys():
        #     predicates_we_kept = predicates_we_kept.union(final_potential_ops[op_name]["pre"])
        #     predicates_we_kept = predicates_we_kept.union(final_potential_ops[op_name]["add"])
        #     predicates_we_kept = predicates_we_kept.union(final_potential_ops[op_name]["del"])
        # pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, predicates_we_kept | initial_predicates)
        # segmented_trajs = [segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in pruned_atom_data]
        # low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        #
        # harmless = True
        # # ccc = 0
        # for ll_traj, seg_traj in zip(low_level_trajs, segmented_trajs):
        #     # ccc += 1
        #     # logging.info(f"Checking harmlessness in traj {ccc-1}")
        #     if not ll_traj.is_demo:
        #         import pdb; pdb.set_trace()
        #     atoms_seq = utils.segment_trajectory_to_atoms_sequence(seg_traj)
        #     task = self._train_tasks[ll_traj.train_task_idx]
        #     traj_goal = task.goal
        #     if not traj_goal.issubset(atoms_seq[-1]):
        #         # In this case, the goal predicates are not correct (e.g.,
        #         # we are learning them), so we skip this demonstration.
        #         continue
        #     demo_preserved = self._check_single_demo_preservation(seg_traj, ll_traj.states[0], atoms_seq, traj_goal, strips_ops, option_specs, predicates_we_kept)
        #     if not demo_preserved:
        #         harmless = False
        #
        # if not harmless:
        #     print(f"Predicate {str(delete_effect)} in {op_name2} is not harmless.")
        #     final_potential_ops[op_name2]["del"].remove(delete_effect)
        #     fff[op_name2][2].remove(delete_effect)
        #     self._clusters = fff
        #     predicates_we_kept = set()
        #     for op_name in final_potential_ops.keys():
        #         predicates_we_kept = predicates_we_kept.union(final_potential_ops[op_name]["pre"])
        #         predicates_we_kept = predicates_we_kept.union(final_potential_ops[op_name]["add"])
        #         predicates_we_kept = predicates_we_kept.union(final_potential_ops[op_name]["del"])
        #
        # if harmless:
        #     print(f"Predicate {str(delete_effect)} in {op_name2} is harmless!")
        # count += 1
        #
        # ########################################################################
        # import pdb; pdb.set_trace()

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

            option_vars = [obj_to_var[o] for o in opt_objs]
            option_spec = [seg_0.get_option().parent, option_vars]

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
                t = (params, objects_list, relevant_preconds, relevant_add_effects, relevant_del_effects, seg, opt_objs)
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

            # print(f"DOING THIS FOR {name}")
            # if name == "Op0-Pick":
            #     import pdb; pdb.set_trace()
            op1 = ops[0]
            op1_opt_objs = op1[-1]
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

                counter += 1

            option_vars = [op1_obj_to_var[o] for o in op1_opt_objs]
            option_spec = [seg_0.get_option().parent, option_vars]
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

        # import pdb; pdb.set_trace()

        # prune duplicate pnads (happens in cover with k=2)
        # probably want to check this for the mapping with the highest overlap?
        # since the variables may match? or will they match in the cases where it ends up
        # being the equivalent operator?
        pruned_pnads = []
        sub_cluster_pnads_set = set()
        # how many effective clusters do we have for the thing we are exploring sub-clustering for?
        effective_clusters = 0
        # the only duplicates we can have are among the ones that we sub-clustered
        # out of the pnads we created from the subcluster, only keep the unique ones
        for i, pnad in enumerate(pnads):
            if i in indices_of_sub_clusters:
                temp = pnad.op.copy_with(name="null")
                if temp in sub_cluster_pnads_set:
                    continue
                else:
                    sub_cluster_pnads_set.add(temp)
                    pruned_pnads.append(pnad)
                if pnad.option_spec[0].name == example_segment.get_option().name:
                    effective_clusters += 1
            else:
                pruned_pnads.append(pnad)

        # import pdb; pdb.set_trace()

        # indices_considered = set()
        # for i, pnad in enumerate(sub_cluster_pnads):
        #     if i in indices_considered:
        #         continue
        #     # identify all indicies of pnads that are equivalent to this one
        #     for j, pnad_2 in enumerate(sub_cluster_pnads):
        #         if j in indices_considered:
        #             continue
        #         if ops_are_equivalent(pnad.op, pnad_2.op):
        #             indices_considered.add(j)
        #     indices_considered.add(i)
        #     pruned_pnads.append(pnad)

        # import pdb; pdb.set_trace()
        print("ashay")

        pnads = pruned_pnads

        # # how many effective clusters do we have for the thing we are exploring sub-clustering for?
        # effective_clusters = 0
        # for p in pnads:
        #     if p.option_spec[0].name == example_segment.get_option().name:
        #         effective_clusters += 1

        # import pdb; pdb.set_trace()
        print("ashay2")

        from predicators.predicate_search_score_functions import _ExpectedNodesScoreFunction
        score_function = _ExpectedNodesScoreFunction(initial_predicates, atom_dataset, candidates, self._train_tasks, "num_nodes_expanded")
        pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, predicates_we_kept | initial_predicates)
        segmented_trajs = [segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in pruned_atom_data]
        low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
        strips_ops = [pnad.op for pnad in pnads]
        option_specs = [pnad.option_spec for pnad in pnads]
        op_score = score_function.evaluate_with_operators(
            predicates_we_kept,
            low_level_trajs,
            segmented_trajs,
            strips_ops,
            option_specs
        )
        return op_score, effective_clusters

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
        # ############
        # # Clustering
        # ############
        # import numpy as np
        # from sklearn.mixture import GaussianMixture as GMM
        # from scipy.stats import kstest
        # assert CFG.segmenter == "option_changes"
        # segments = [
        #     seg for ll_traj, atom_seq in atom_dataset for seg in
        #     segment_trajectory(ll_traj, initial_predicates, atom_seq)
        # ]
        #
        # # Step 1:
        # # Cluster segments by the option that generated them. We know that
        # # at the very least, operators are 1 to 1 with options.
        # option_to_segments: Dict[Any, Any] = {}  # Dict[str, List[Segment]]
        # for seg in segments:
        #     name = seg.get_option().name
        #     option_to_segments.setdefault(name, []).append(seg)
        # logging.info(f"STEP 1: generated {len(option_to_segments.keys())} "
        #              f"option-based clusters.")
        # clusters = option_to_segments.copy()  # Tree-like structure.
        #
        # # Step 2:
        # # Further cluster by the types that appear in a segment's add
        # # effects. Operators have a fixed number of typed arguments.
        # for i, pair in enumerate(option_to_segments.items()):
        #     option, segments = pair
        #     types_to_segments: Dict[Tuple[Type, ...], List[Segment]] = {}
        #     for seg in segments:
        #         types_in_effects = [
        #             set(a.predicate.types) for a in seg.add_effects
        #         ]
        #         # To cluster on type, there must be types. That is, there
        #         # must be add effects in the segment and the object
        #         # arguments for at least one add effect must be nonempty.
        #         assert len(types_in_effects) > 0 and len(
        #             set.union(*types_in_effects)) > 0
        #         types = tuple(sorted(list(set.union(*types_in_effects))))
        #         types_to_segments.setdefault(types, []).append(seg)
        #     logging.info(
        #         f"STEP 2: generated {len(types_to_segments.keys())} "
        #         f"type-based clusters for cluster {i+1} from STEP 1 "
        #         f"involving option {option}.")
        #     clusters[option] = types_to_segments
        #
        # # Step 3:
        # # Further cluster by the maximum number of objects that appear in a
        # # segment's add effects. Note that the use of maximum here is
        # # somewhat arbitrary. Alternatively, you could cluster for every
        # # possible number of objects and not the max among what you see in
        # # the add effects of a particular segment.
        # for i, (option, types_to_segments) in enumerate(clusters.items()):
        #     for j, (types,
        #             segments) in enumerate(types_to_segments.items()):
        #         num_to_segments: Dict[int, List[Segment]] = {}
        #         for seg in segments:
        #             max_num_objs = max(
        #                 len(a.objects) for a in seg.add_effects)
        #             num_to_segments.setdefault(max_num_objs,
        #                                        []).append(seg)
        #         logging.info(
        #             f"STEP 3: generated {len(num_to_segments.keys())} "
        #             f"num-object-based clusters for cluster {i+j+1} from "
        #             f"STEP 2 involving option {option} and type {types}.")
        #         clusters[option][types] = num_to_segments
        #
        # # Step 4:
        # # Further cluster by sample, if a sample is present. The idea here
        # # is to separate things like PickFromTop and PickFromSide.
        # for i, (option, types_to_num) in enumerate(clusters.items()):
        #     for j, (types,
        #             num_to_segments) in enumerate(types_to_num.items()):
        #         for k, (max_num_objs,
        #                 segments) in enumerate(num_to_segments.items()):
        #             # If the segments in this cluster have no sample, then
        #             # don't cluster further.
        #             if len(segments[0].get_option().params) == 0:
        #                 clusters[option][types][max_num_objs] = [segments]
        #                 logging.info(
        #                     f"STEP 4: generated no further sample-based "
        #                     f"clusters (no parameter) for cluster {i+j+k+1}"
        #                     f" from STEP 3 involving option {option}, type "
        #                     f"{types}, and max num objects {max_num_objs}."
        #                 )
        #                 continue
        #             # pylint: disable=line-too-long
        #             # If the parameters are described by a uniform
        #             # distribution, then don't cluster further. This
        #             # helps prevent overfitting. A proper implementation
        #             # would do a multi-dimensional test
        #             # (https://ieeexplore.ieee.org/document/4767477,
        #             # https://ui.adsabs.harvard.edu/abs/1987MNRAS.225..155F/abstract,
        #             # https://stats.stackexchange.com/questions/30982/how-to-test-uniformity-in-several-dimensions)
        #             # but for now we will only check each dimension
        #             # individually to keep the implementation simple.
        #             # pylint: enable=line-too-long
        #             samples = np.array(
        #                 [seg.get_option().params for seg in segments])
        #             each_dim_uniform = True
        #             for d in range(samples.shape[1]):
        #                 col = samples[:, d]
        #                 minimum = col.min()
        #                 maximum = col.max()
        #                 null_hypothesis = np.random.uniform(
        #                     minimum, maximum, len(col))
        #                 p_value = kstest(col, null_hypothesis).pvalue
        #
        #                 # We use a significance value of 0.05.
        #                 if p_value < 0.05:
        #                     each_dim_uniform = False
        #                     break
        #             if each_dim_uniform:
        #                 clusters[option][types][max_num_objs] = [segments]
        #                 logging.info(
        #                     f"STEP 4: generated no further sample-based"
        #                     f" clusters (uniformly distributed "
        #                     f"parameter) for cluster {i+j+k+1} "
        #                     f"from STEP 3 involving option {option}, "
        #                     f" type {types}, and max num objects "
        #                     f"{max_num_objs}.")
        #                 continue
        #             # Determine clusters by assignment from a
        #             # Gaussian Mixture Model. The number of
        #             # components and the negative weighting on the
        #             # complexity of the model (chosen by BIC here)
        #             # are hyperparameters.
        #             max_components = min(
        #                 len(samples), len(np.unique(samples)),
        #                 CFG.grammar_search_clustering_gmm_num_components)
        #             n_components = np.arange(1, max_components + 1)
        #             models = [
        #                 GMM(n, covariance_type="full",
        #                     random_state=0).fit(samples)
        #                 for n in n_components
        #             ]
        #             bic = [m.bic(samples) for m in models]
        #             best = models[np.argmin(bic)]
        #             assignments = best.predict(samples)
        #             label_to_segments: Dict[int, List[Segment]] = {}
        #             for l, assignment in enumerate(assignments):
        #                 label_to_segments.setdefault(
        #                     assignment, []).append(segments[l])
        #             clusters[option][types][max_num_objs] = list(
        #                 label_to_segments.values())
        #             logging.info(f"STEP 4: generated "
        #                          f"{len(label_to_segments.keys())}"
        #                          f"sample-based clusters for cluster "
        #                          f"{i+j+k+1} from STEP 3 involving option "
        #                          f"{option}, type {types}, and max num "
        #                          f"objects {max_num_objs}.")
        #
        # # We could avoid these loops by creating the final set of clusters
        # # as part of STEP 4, but this is not prohibitively slow and serves
        # # to clarify the nested dictionary structure, which we may make use
        # # of in follow-up work that modifies the clusters more.
        # final_clusters = []
        # for option in clusters.keys():
        #     for types in clusters[option].keys():
        #         for max_num_objs in clusters[option][types].keys():
        #             for cluster in clusters[option][types][max_num_objs]:
        #                 final_clusters.append(cluster)
        # logging.info(f"Total {len(final_clusters)} final clusters.")
        #
        # # Step 4.5:
        # # For debugging purposes, here is code to use the oracle clusters.
        # assert CFG.offline_data_method == "demo+gt_operators"
        # assert dataset.annotations is not None and len(
        #     dataset.annotations) == len(dataset.trajectories)
        # assert CFG.segmenter == "option_changes"
        # segmented_trajs = [
        #     segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in atom_dataset
        # ]
        # assert len(segmented_trajs) == len(dataset.annotations)
        # # First, get the set of all ground truth operator names.
        # all_gt_op_names = set(ground_nsrt.parent.name
        #                       for anno_list in dataset.annotations
        #                       for ground_nsrt in anno_list)
        # # Next, make a dictionary mapping operator name to segments
        # # where that operator was used.
        # gt_op_to_segments: Dict[str, List[Segment]] = {
        #     op_name: []
        #     for op_name in all_gt_op_names
        # }
        # for op_list, seg_list in zip(dataset.annotations, segmented_trajs):
        #     assert len(seg_list) == len(op_list)
        #     for ground_nsrt, segment in zip(op_list, seg_list):
        #         gt_op_to_segments[ground_nsrt.parent.name].append(segment)
        # final_clusters = list(gt_op_to_segments.values())
        #
        # # Step 5:
        # # Extract predicates from the pure intersection of the add effects
        # # in each cluster.
        # extracted_preds = set()
        # shared_add_effects_per_cluster = []
        # for c in final_clusters:
        #     grounded_add_effects_per_segment = [
        #         seg.add_effects for seg in c
        #     ]
        #     ungrounded_add_effects_per_segment = []
        #     for effs in grounded_add_effects_per_segment:
        #         ungrounded_add_effects_per_segment.append(
        #             set(a.predicate for a in effs))
        #     shared_add_effects_in_cluster = set.intersection(
        #         *ungrounded_add_effects_per_segment)
        #     shared_add_effects_per_cluster.append(
        #         shared_add_effects_in_cluster)
        #     extracted_preds |= shared_add_effects_in_cluster
        #
        # # Step 6:
        # # Remove inconsistent predicates except if removing them prevents us
        # # from disambiguating two or more clusters (i.e. their add effect
        # # sets are the same after removing the inconsistent predicates). The
        # # idea here is that HoldingTop and HoldingSide are inconsistent
        # # within the PlaceOnTable cluster in painting, but we don't want to
        # # remove them, since we had generated them specifically to
        # # disambiguate segments in the cluster with the Pick option.
        # # A consistent predicate is either an add effect, a delete
        # # effect, or doesn't change, within each cluster, for all clusters.
        # # Note that it is possible that when 2 inconsistent predicates are
        # # removed, then two clusters cannot be disambiguated, but if you
        # # keep either of the two, then you can disambiguate the clusters.
        # # For now, we just add both back, which is not ideal.
        # consistent, inconsistent = self._get_consistent_predicates(
        #     extracted_preds, list(final_clusters))
        # predicates_to_keep: Set[Predicate] = consistent
        # consistent_shared_add_effects_per_cluster = [
        #     add_effs - inconsistent
        #     for add_effs in shared_add_effects_per_cluster
        # ]
        # num_clusters = len(final_clusters)
        # for i in range(num_clusters):
        #     for j in range(num_clusters):
        #         if i == j:
        #             continue
        #         if consistent_shared_add_effects_per_cluster[
        #                 i] == consistent_shared_add_effects_per_cluster[j]:
        #             logging.info(
        #                 f"Final clusters {i} and {j} cannot be "
        #                 f"disambiguated after removing the inconsistent"
        #                 f" predicates.")
        #             predicates_to_keep |= \
        #                 shared_add_effects_per_cluster[i]
        #             predicates_to_keep |= \
        #                 shared_add_effects_per_cluster[j]
        #
        # logging.info(
        #     f"\nSelected {len(predicates_to_keep)} predicates out of "
        #     f"{len(candidates)} candidates:")
        # for pred in sorted(predicates_to_keep):
        #     logging.info(f"{pred}")
        #
        # ###################################
        # # Operator (and predicate) learning
        # ###################################
        # # Now we can learn operators (and predicates) given clusters. First, we
        # # will decide what predicates to include in the preconditions, add
        # # effects, and delete effects or each operator. Second, we will
        # # finalize the exact lifted atoms in the operators' definitions.
        #
        # # Note that step 4 of clustering (clustering on the sampled parameter)
        # # is prone to error (in terms of recovering the clustering that
        # # the demonstrator had). For this reason, we an operator/predicate
        # # learning procedure on each of many candidate clusterings, score each,
        # # and take the one with the best score. One way we can score them is
        # # by passing the operators into _ExpectedNodesScoreFunction.evaluate_with_operators(),
        # # which is one of the possible score functions used in the hill climbing
        # # predicate invention algorithm.
        #
        # # TODO: loop over clusterings
        #
        # # Step 1: decide what predicates to include in the operator's definition
        # # As a starting point, we can make a draft operator definition for each
        # # cluster from the predicates that appear in the intersection of the
        # # preconditions, the intersection of the add effects, and the
        # # intersection of the delete effects of the segments in that cluster.
        # # The definition of an operator involves lifted atoms, but in this first
        # # step we will only identify predicates.
        # op_pred_def_superset = {} # operator predicate definition superset
        # for i, c in enumerate(final_clusters):
        #     op_name = f"Op{i}-{c[0].get_option().name}"
        #
        #     # preconditions
        #     init_atoms_per_segment = [s.init_atoms for s in c]
        #     ungrounded_init_atoms_per_segment = []
        #     for init_atoms in init_atoms_per_segment:
        #         ungrounded_init_atoms_per_segment.append(set(a.predicate for a in init_atoms if a.predicate in predicates_to_keep))
        #     init_atoms = set.intersection(*ungrounded_init_atoms_per_segment)
        #
        #     # add effects
        #     add_effects_per_segment = [s.add_effects for s in c]
        #     ungrounded_add_effects_per_segment = []
        #     for add_effects in add_effects_per_segment:
        #         ungrounded_add_effects_per_segment.append(set(a.predicate for a in add_effects if a.predicate in predicates_to_keep))
        #     add_effects = set.intersection(*ungrounded_add_effects_per_segment)
        #
        #     # delete effects
        #     delete_effects_per_segment = [s.delete_effects for s in c]
        #     ungrounded_delete_effects_per_segment = []
        #     for delete_effects in delete_effects_per_segment:
        #         ungrounded_delete_effects_per_segment.append(set(a.predicate for a in delete_effects if a.predicate in predicates_to_keep))
        #     delete_effects = set.intersection(*ungrounded_delete_effects_per_segment)
        #
        #     op_pred_def_superset[op_name] = [
        #         init_atoms,
        #         add_effects,
        #         delete_effects,
        #         c
        #     ]
        #
        # # Now we can reason backwards from the goal to further narrow down the
        # # preconditions and effects in each operator and also ensure that the
        # # operators can chain together properly. The idea is that, for each
        # # operator in a trajectory, working backwards from the last operator,
        # # - if an add effect of the operator is a goal atom or a precondition
        # # of a future operator, include the predicate in the operator's
        # # add effects, and include the predicate in the future operator's
        # # preconditions.
        # #  - if a delete effect of the operator is added in a future operator,
        # # include the predicate in the operator's delete effect, and include
        # # the predicate in the future operator's add effects.
        #
        # # Later, we'll iterate through a segmented trajectory and will need to
        # # know which cluster we had placed a particular segment in. Clusters
        # # are identified by the name of the operator we assign to them.
        # def seg_to_op_name(segment, clusters):
        #     for i, c in enumerate(clusters):
        #         if segment in c:
        #             return f"Op{i}-{c[0].get_option().name}"
        #
        # Entry = namedtuple('Entry', ['op_name', 'preconditions', 'add_effects', 'delete_effects'])
        #
        # # To perform the backchaining described above, we need an annotated
        # # trajectory for each demonstration, where each each operator is
        # # annotated with its potential preconditions and effects.
        # def get_ground_pred_annotated_traj(segmented_traj):
        #     annotated_traj = []
        #     for i, seg in enumerate(segmented_traj):
        #         option_objects = tuple(seg.get_option().objects)
        #         objects = (set(o for a in seg.add_effects for o in a.objects) | set(o for a in seg.delete_effects for o in a.objects) | set(option_objects))
        #         preconditions = set(p for p in seg.init_atoms if p.predicate in op_pred_def_superset[op_name][0] and set(p.objects).issubset(objects))
        #         add_effects = set(p for p in seg.add_effects if p.predicate in op_pred_def_superset[op_name][1] and set(p.objects).issubset(objects))
        #         delete_effects = set(p for p in seg.delete_effects if p.predicate in op_pred_def_superset[op_name][2] and set(p.objects).issubset(objects))
        #         # Note that these may include some ground atoms that we DO NOT
        #         # want to include in the definition of the operator. For
        #         # example, the cluster for option Unstack in blocks env has some
        #         # precondition, NOT-Forall[1:block].[NOT-On(0,1)](block1:block)
        #         # (you see this in demonstration #8). We want to include it when
        #         # it's applied to the top block (the one getting unstacked), but
        #         # not when it's applied to the bottom block (the block below the
        #         # block getting unstacked). (This predicate means "there exists
        #         # some block that the block in question is on top of" -- which
        #         # always is true for the unstacked block, but not always true
        #         # for the block below it.) But, both the unstacked block and
        #         # the block below it are parameters for this operator, so
        #         # not including ground atoms that involve objects that aren't
        #         # in the operator's parameters will not help deal with this.
        #         # We'll deal with this problem -- what exact lifted atoms to
        #         # include in our operator definition -- later. In the step where
        #         # this function gets used, we only care about what predicates
        #         # we'll use in the operator's definition.
        #
        #         e = Entry(op_name, preconditions, add_effects, delete_effects)
        #
        #     annotated_traj.append(e)
        #     return annotated_traj
        #
        # def process_add_effects(potential_ops, remaining_traj, goal):
        #     curr_entry = remaining_traj[0]
        #     # If we can find this ground atom in the preconditions of any future segment
        #     # or in the goal, then we'll include it in this operator's add effects.
        #     for ground_atom in curr_entry.add_effects:
        #         for entry in remaining_traj[1:]:
        #             if ground_atom in entry.preconditions:
        #                 potential_ops[curr_entry.op_name]["add"].add(ground_atom.predicate)
        #                 potential_ops[entry.op_name]["pre"].add(ground_atom.predicate)
        #         if ground_atom in goal:
        #             potential_ops[curr_entry.op_name]["add"].add(ground_atom.predicate)
        #
        # def process_delete_effects(potential_ops, remaining_traj):
        #     curr_entry = remaining_traj[0]
        #     # If this ground atom is added in any future segment, then
        #     # we'll include it in this operator's delete effects.
        #     for ground_atom in curr_entry.delete_effects:
        #         for entry in remaining_traj[1:]:
        #             # Alternatively, we could run process_add_effects()
        #             # on all the trajectories before this function, and
        #             # then check if the ground atom is in the add effects
        #             # of the potential operator. Haven't explored this idea.
        #             if ground_atom in entry.add_effects:
        #                 potential_ops[curr_entry.op_name]["del"].add(ground_atom.predicate)
        #                 potential_ops[entry.op_name]["add"].add(ground_atom.predicate)
        #
        # # As we iterate through each demo trajectory, we will generate an operator definition hypothesis
        # # for each segment in that demo trajectory, and append it to this list to further process later.
        # possible_operator_definitions_per_demo = []
        #
        # for i, traj in enumerate(segmented_trajs):
        #     possible_operator_definitions = {}
        #     for op_name in op_pred_def_superset.keys():
        #         possible_operator_definitions[op_name] = {
        #             "pre": set(),
        #             "add": set(),
        #             "del": set()
        #         }
        #
        #     annotated_traj = get_ground_pred_annotated_traj(traj)
        #     reversed_annotated_traj = list(reversed(annotated_traj))
        #     for j, entry in enumerate(reversed_annotated_traj):
        #         remaining_annotated_traj = annotated_traj[len(annotated_traj)-i-1:]
        #         name, preconds, add_effs, del_effs = entry
        #         if j == 0:
        #             for p in add_effs:
        #                 if p in train_tasks[i].goal:
        #                     possible_operator_definitions[name]["add"].add(p.predicate)
        #         else:
        #             process_add_effects(possible_operator_definitions, remaining_annotated_traj, train_tasks[j].goal)
        #             process_delete_effects(possible_operator_definitions, remaining_annotated_traj)
        #
        #     possible_operator_definitions_per_demo.append(possible_operator_definitions)
        #
        # # We'll take all the precondition predicates and delete effects in the
        # # intersection of the segments in the operator's clusteer because
        # # backchaining doesn't identify all of them. For example, for certain
        # # operators involving the Place option in painting env, no other
        # # operator adds their preconditions (their preconditions are True at the
        # # beginning of the task). As another example, for the operators
        # # involving the Pick option in blocks env, that have "OnTable" as a
        # # possible delete effect, we don't include it in backchaining because
        # # "OnTable" is never added again after the object is picked up (it's
        # # never picked up from the table and put back on the table).
        # final_predicate_operator_definitions = {
        #     op_name: {
        #         "pre": op_pred_def_superset[op_name][0],
        #         "add": set(),
        #         "del": op_pred_def_superset[op_name][2]
        #     } for op_name in possible_operator_definitions_per_demo[0].keys()
        # }
        # for i, ops in enumerate(possible_operator_definitions_per_demo):
        #     for op_name in ops.keys():
        #         # We take the union because, different demos have different
        #         # goals, and, for example, certain add effects won't appear
        #         # relevant only by looking at certain demos. Taking the union is
        #         # akin to including the add effects that we identified as
        #         # important across all the demos.
        #         final_predicate_operator_definitions[op_name]["pre"] = final_predicate_operator_definitions[op_name]["pre"].union(ops[op_name]["pre"])
        #         final_predicate_operator_definitions[op_name]["add"] = final_predicate_operator_definitions[op_name]["add"].union(ops[op_name]["add"])
        #         final_predicate_operator_definitions[op_name]["del"] = final_predicate_operator_definitions[op_name]["del"].union(ops[op_name]["del"])
        #
        # # Step 2: decide what exact lifted atoms to include in the operator's
        # # definition.
        # from predicators.structs import PNAD, Datastore, STRIPSOperator
        # pnads: List[PNAD] = []

        ########################################################################

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

            assert CFG.segmenter == "option_changes"
            # segmented_trajs = [segment_trajectory(traj) for traj in atom_dataset]
            segmented_trajs = [
                segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in atom_dataset
            ]
            # import pdb; pdb.set_trace()

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

            # TODO: ******************
            # Only keep sufficiently sized clusters.
            # TODO: add logging info, and maybe make this step optional via a flag.
            # all_clusters = [c for c in all_clusters if len(c) > 20]
            # import pdb; pdb.set_trace()

            # # Step 4:
            # final_clusters = []
            #
            # clustering_scores = {}
            # for z, cluster in enumerate(all_clusters):
            #     op_name = f"Op{z}-{cluster[0].get_option().name}"
            #     clustering_scores[op_name] = [] # will hold scores for 1, 2, etc. clusterings
            #
            # for jjj, cluster in enumerate(all_clusters):
            #     op_name = f"Op{jjj}-{cluster[0].get_option().name}"
            #     # if "PutOnTable" not in op_name:
            #     #     continue
            #     # final_clusters2 = []
            #     # for x, c in enumerate(all_clusters):
            #     #     if x != j:
            #     #         final_clusters2.append(c)
            #     example_segment = cluster[0]
            #     option_name = example_segment.get_option().name
            #     if len(example_segment.get_option().params) == 0:
            #         final_clusters.append(cluster)
            #         logging.info(f"STEP 4: generated no further sample-based clusters (no parameter!) for the {jjj+1}th cluster from STEP 3 involving option {option_name}.")
            #
            #     else:
            #         for kkk in range(1, 4):
            #             logging.info(f"Evaluating kkk {kkk} for op {op_name}.")
            #             example_segment = cluster[0]
            #             option_name = example_segment.get_option().name
            #             score, eff_kkk = self.get_score(all_clusters, example_segment, segmented_trajs, jjj, kkk, candidates, initial_predicates, dataset, atom_dataset)
            #             if score is None:
            #                 break
            #             clustering_scores[op_name].append((kkk, score, eff_kkk))
            #
            #             # import pdb; pdb.set_trace()
            #             logging.info(f"Clustering scores: {clustering_scores}")
            #             print("asdf")
            #
            #         # best_k = min(clustering_scores[op_name], key=lambda x: (x[1], x[0]))[0]
            #         # best_k = 1
            #         best_k = min(clustering_scores[op_name], key=lambda x: (x[1], x[0]))
            #         logging.info(f"BEST K: {best_k[0]}, BEST EFFECIVE K: {best_k[-1]}")
            #
            #         import numpy as np
            #         from sklearn.mixture import GaussianMixture as GMM
            #         from scipy.stats import kstest
            #         data = np.array([seg.get_option().params for seg in cluster])
            #
            #
            #         if len(data) == 1:
            #             # Can't fit GMM to 1 data point, need at least 2
            #             # ValueError: Found array with 1 sample(s) (shape=(1, 1)) while a minimum of 2 is required. for GMM.fit() method
            #             # Don't subcluster
            #             final_clusters.append(cluster)
            #         else:
            #             model = GMM(best_k[-1], covariance_type="full", random_state=0).fit(data)
            #             # if "Pick" in option_name:
            #             #     best_k = 2
            #             # elif "PutOnTable" in option_name:
            #             #     best_k = 1
            #             # model = GMM(best_k, covariance_type="full", random_state=0).fit(data)
            #             assignments = model.predict(data)
            #             sub_clusters = {}
            #             for i, assignment in enumerate(assignments):
            #                 if assignment in sub_clusters:
            #                     sub_clusters[assignment].append(cluster[i])
            #                 else:
            #                     sub_clusters[assignment] = [cluster[i]]
            #             for c in sub_clusters.values():
            #                 final_clusters.append(c)
            #
            # logging.info(f"Clustering best k scores: {clustering_scores}")
            # # import pdb; pdb.set_trace()
            # # print("hecz")

            ###
            # Stuff from oracle learning to test if the stuff is working.
            ###
            assert CFG.offline_data_method == "demo+gt_operators"
            assert dataset.annotations is not None and len(
                dataset.annotations) == len(dataset.trajectories)
            assert CFG.segmenter == "option_changes"
            # segmented_trajs = [
            #     segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in atom_dataset
            # ]
            assert len(segmented_trajs) == len(dataset.annotations)
            # First, get the set of all ground truth operator names.
            all_gt_op_names = set(ground_nsrt.parent.name
                                  for anno_list in dataset.annotations
                                  for ground_nsrt in anno_list)
            # import pdb; pdb.set_trace()
            # Next, make a dictionary mapping operator name to segments
            # where that operator was used.
            gt_op_to_segments: Dict[str, List[Segment]] = {
                op_name: []
                for op_name in all_gt_op_names
            }
            for op_list, seg_list in zip(dataset.annotations, segmented_trajs):
                assert len(seg_list) == len(op_list)
                for ground_nsrt, segment in zip(op_list, seg_list):
                    gt_op_to_segments[ground_nsrt.parent.name].append(segment)
            final_clusters = list(gt_op_to_segments.values())
            ###

            # ###
            # take the clusters involving option paint from our computed ones,
            # but the rest from oracle
            # import pdb; pdb.set_trace()
            # new_final_clusters = []
            # for c in final_clusters:
            #     if c[0].get_option().name != "Paint":
            #         new_final_clusters.append(c)
            #     elif c[0].get_option().name
            # for c in final_clusters2:
            #     if c[0].get_option().name == "Paint":
            #         new_final_clusters.append(c)
            # import pdb; pdb.set_trace()
            # ###

            # new_final_clusters = []
            # for i, clus in enumerate(final_clusters):
            #     if i == 6:
            #         new_clus = []
            #         for segg in clus:
            #             if len(segg.add_effects) > 0:
            #                 new_clus.append(segg)
            #         new_final_clusters.append(new_clus)
            #     else:
            #         new_final_clusters.append(clus)
            # final_clusters = new_final_clusters

            # operator to preconditions, and add effects
            # filter out an operator that barely ever appears
            ddd = {}
            for i, c in enumerate(final_clusters):
                op_name = "Op"+str(i)+"-"+str(c[0].get_option().name)
                # preconditions, add effects, delete effects, segments
                ddd[op_name] = [set(), set(), set(), c]
                # if op_name == "Op6-StickMoveToButton":
                #     new_c = []
                #     for segg in c:
                #         print("segg add effects: ", segg.add_effects)
                #         if len(segg.add_effects) > 0:
                #             new_c.append(segg)
                #     import pdb; pdb.set_trace()
                #     ddd[op_name] = [set(), set(), set(), new_c]
                # else:
                #     ddd[op_name] = [set(), set(), set(), c]

            # found = []
            # for operator in ddd.keys():
            #     # if "RobotMoveToButton" in operator:
            #     if operator == 'Op8-RobotMoveToButton':
            #         c = ddd[operator][3]
            #         init_atoms_per_segment = [s.init_atoms for s in c]
            #         found = []
            #         for per in init_atoms_per_segment:
            #             names = list(a.predicate.name for a in per)
            #             # if "(|(0:button).x - (1:stick).x|<=[idx 0]0.145" in names:
            #             # if '(|(0:button).x - (1:stick).x|<=[idx 0]0.145)' in names:
            #             if 'Forall[0:button,1:stick].[NOT-(|(0:button).x - (1:stick).x|<=[idx 0]0.145)(0,1)]' in names:
            #                 found.append(True)
            #             else:
            #                 found.append(False)
            #
            #         import pdb; pdb.set_trace()
            #         print("aimbot222")

            # (|(0:button).x - (1:stick).x|<=[idx 0]0.145

            # For clusters that appear at the end,
            # we can narrow down their add effects as:
            # they DEFINITELY have an add effect that whatever goal atoms
            # were not true in the pre-image.
            # But beyond that, they may have other effects?
            #


            ####

            # import pdb; pdb.set_trace()
            # print("star wars")

            add_effects_per_cluster = []
            all_add_effects = set()
            all_preconditions = set()
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
                all_preconditions |= init_atoms

                # if "Forall[0:block].[((0:block).pose_z<=[idx 1]0.342)(0)]" in [str(p) for p in add_effects]:
                #     import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()

                #
                # ex = c[0]
                # print(f"Cluster {j} with option {ex.get_option().name}:")
                # for a in add_effects:
                #     print(a)
                #

            # import pdb; pdb.set_trace()

            predicates_to_keep = all_add_effects | all_preconditions

            import pdb; pdb.set_trace()
            print("keyboard")

            ##########
            # Remove inconsistent predicates.
            inconsistent_preds = set()

            # Old way to remove inconsistent predicates
            predicates_to_keep: Set[Predicate] = set()
            for pred in all_add_effects | all_preconditions:
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
                            # if str(pred) == "Forall[0:block].[((0:block).pose_z<=[idx 1]0.342)(0)]":
                            #     import pdb; pdb.set_trace()
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

            import pdb; pdb.set_trace()
            # only remove inconsistent add effects -- keep all predicates in preconditions
            predicates_to_keep |= all_preconditions
            print("screen")

            # Re-add inconsistent predicates that were necessary to disambiguate two clusters.
            # That is, if any two clusters's predicates look the same after removing inconsistent predicates from both,
            # keep all those predicates.
            new_add_effects_per_cluster = []
            for effs_cluster in add_effects_per_cluster:
                new_add_effects_per_cluster.append(effs_cluster - inconsistent_preds)
            # Check all pairs of clusters, if they are the same, add their original cluster's predicates back.
            print("got to hereee")
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
                            for p in add_effects_per_cluster[i] | add_effects_per_cluster[j]:
                                if p in inconsistent_preds:
                                    print(f"ADDING BACK: {p}")

            # Remove the initial predicates.
            # predicates_to_keep -= initial_predicates

            import pdb; pdb.set_trace()
            print("mouse")

            # remove predicates we aren't keeping
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

            import pdb; pdb.set_trace()
            logging.info("Performing backchaining to decide operator definitions.")

            # print("DOING CHECK SEEING IF DDD IS THE SAME")
            # for op_name, v in final_predicate_operator_definitions.items():
            #     print(f"doing ddd check for {op_name}")
            #     clus = op_pred_def_superset[op_name][3]
            #     assert ddd[op_name][0] == v["pre"]
            #     assert ddd[op_name][1] == v["add"]
            #     assert ddd[op_name][2] == v["del"]
            #     assert set(ddd[op_name][3]) == set(clus)
            #
            # print("DONE WITH CHECK SEEING IF DDD IS THE SAME")

            #####################
            # backchaining
            #####################
            def seg_to_op(segment, clusters):
                for i, c in enumerate(clusters):
                    if segment in c:
                        # return f"Op{i}-{c[0].get_option().name} with objects ({c[0].get_option().objects})"
                        return f"Op{i}-{c[0].get_option().name}"

            # def seg_to_op(segment, clusters):
            #     found = ""
            #     for i, c in enumerate(clusters):
            #         if segment in c:
            #             # return f"Op{i}-{c[0].get_option().name} with objects ({c[0].get_option().objects})"
            #             found = f"Op{i}-{c[0].get_option().name}"
            #             break
            #     if len(found) > 0:
            #         return found
            #     else:
            #         return f"Something-{segment.get_option().name}"

            # Go through demos
            logging.info(f"Printing demos!!!!!")
            temp = []
            for a, segmented_traj in enumerate(segmented_trajs):
                # if a == 10:
                #     break
                traj = []
                for seg in segmented_traj:
                    # if seg_to_op(seg, final_clusters) is None:
                    #     import pdb; pdb.set_trace()
                    traj.append(seg_to_op(seg, final_clusters))
                temp.append(traj)

                print(traj)
                # if a == 1:
                #     import pdb; pdb.set_trace()
            # import pdb; pdb.set_trace()

            ttt = list(predicates_to_keep)
            for p in ttt:
                print(p)

            def print_demo2(segmented_traj, filename):

                max_length = 0
                with open(filename, 'w') as file:
                    # preconditions
                    preconds = []
                    for seg in segmented_traj:
                        op_name = seg_to_op(seg, final_clusters)

                        ps = sorted([p.name for p in ddd[op_name][0]])
                        init_atoms = set(p for p in seg.init_atoms if p.predicate.name in ps)
                        # we assume we only have to look at objects involved in the operator,
                        # so we can ignore all other ground atoms.
                        # can figure out which objects change in terms of predicates.
                        objects = set(o for a in seg.add_effects for o in a.objects) |set(o for a in seg.delete_effects for o in a.objects)
                        relevant_init_atoms = set(p for p in init_atoms if set(p.objects).issubset(objects))
                        relevant_init_atoms_str = sorted([str(p) for p in relevant_init_atoms])
                        for p in relevant_init_atoms_str:
                            max_length = max(max_length, len(p))
                        to_print = [op_name, '----', 'preconditions', '===='] + relevant_init_atoms_str
                        preconds.append(to_print)

                        # if op_name == "Op3-FastenScrewByHand":
                        #     import pdb; pdb.set_trace()

                    # add effects
                    add_effs = []
                    for seg in segmented_traj:
                        op_name = seg_to_op(seg, final_clusters)
                        adds = sorted([str(p) for p in seg.add_effects if p.predicate in ddd[op_name][1]])
                        for p in adds:
                            max_length = max(max_length, len(p))
                        adds = ['add effects', '===='] + adds
                        add_effs.append(adds)

                    # delete effects
                    del_effs = []
                    for seg in segmented_traj:
                        op_name = seg_to_op(seg, final_clusters)
                        dels = sorted([str(p) for p in seg.delete_effects if p.predicate in ddd[op_name][2]])
                        for p in dels:
                            max_length = max(max_length, len(p))
                        dels = ['delete effects', '===='] + dels
                        del_effs.append(dels)

                    max_length += 3

                    max_lst_len = max(len(lst) for lst in preconds)
                    for i in range(max_lst_len):
                        s = " ".join(f"{lst[i]:<{max_length}}" if i < len(lst) else " " * max_length for lst in preconds)
                        file.write(s + '\n')
                    file.write('\n')
                    max_lst_len = max(len(lst) for lst in add_effs)
                    for i in range(max_lst_len):
                        s = " ".join(f"{lst[i]:<{max_length}}" if i < len(lst) else " " * max_length for lst in add_effs)
                        file.write(s + '\n')
                    file.write('\n')
                    max_lst_len = max(len(lst) for lst in del_effs)
                    for i in range(max_lst_len):
                        s = " ".join(f"{lst[i]:<{max_length}}" if i < len(lst) else " " * max_length for lst in del_effs)
                        file.write(s + '\n')
                    file.write('\n')

            # print_demo2(segmented_trajs[8][0:3], "demo8_part1.txt")
            # print_demo2(segmented_trajs[8][3:], "demo8_part12.txt")
            # print_demo2(segmented_trajs[0], "demo0_blocks.txt")
            # import pdb; pdb.set_trace()

            # for i, t in enumerate(temp):
            #     for o in t:
            #         if o in ["Op4-Place", "Op3-Place"]:
            #             print(f"demo {i} has {o}")
            #
            # print_demo2(segmented_trajs[2], "demo2.txt")
            # import pdb; pdb.set_trace()

            ###################
            # ALGORITHM
            #

            def get_story(segmented_traj):
                story = []
                    # each entry is a list of lists
                for seg in segmented_traj:
                    entry = []
                    op_name = seg_to_op(seg, final_clusters)
                    init_atoms = set(p for p in seg.init_atoms if p.predicate in ddd[op_name][0])
                    objects = (set(o for a in seg.add_effects for o in a.objects) | set(o for a in seg.delete_effects for o in a.objects))
                    preconditions = set(p for p in init_atoms if set(p.objects).issubset(objects))
                    add_effects = set(p for p in seg.add_effects if p.predicate in ddd[op_name][1])
                    delete_effects = set(p for p in seg.delete_effects if p.predicate in ddd[op_name][2])

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

            # import pdb; pdb.set_trace()

            def process_adds(potential_ops, remaining_story, goal):
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
                            # if "pre" not in potential_ops[entry[0]].keys():
                            #     import pdb; pdb.set_trace()
                            potential_ops[entry[0]]["pre"].add(ground_atom.predicate)
                    if ground_atom in goal:
                        # then we add it!
                        to_add.append(ground_atom) # for debugging
                        potential_ops[name]["add"].add(ground_atom.predicate)

            def process_adds2(potential_ops, remaining_story, goal):
                # import pdb; pdb.set_trace()
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
                            # if "pre" not in potential_ops[entry[0]].keys():
                            #     import pdb; pdb.set_trace()
                            potential_ops[entry[0]]["pre"].add(ground_atom.predicate)
                    if ground_atom in goal:
                        # then we add it!
                        to_add.append(ground_atom) # for debugging
                        potential_ops[name]["add"].add(ground_atom.predicate)

            def process_dels(potential_ops, potential_ops2, remaining_story):
                curr_entry = remaining_story[0]
                name = curr_entry[0]
                to_del = []
                for ground_atom in curr_entry[3]: # delete effects
                    # is this atom every added in the future?
                    # *** do we need to do this with an updated potential_ops after process_adds runs on all the trajectories?
                    for entry in remaining_story[1:]:
                        if ground_atom in entry[2]: # add effects
                            to_del.append(str(ground_atom.predicate)) # for debugging
                            potential_ops[name]["del"].add(ground_atom.predicate)
                            potential_ops[entry[0]]["add"].add(ground_atom.predicate)
                            potential_ops2[entry[0]]["add"].add(ground_atom.predicate)

                            # if name == "Op0-Pick" and "Forall[0:block].[((0:block).pose_z<=[idx 1]0.342)(0)]" in to_del:
                            #     print(f"to_del: {to_del}")
                            #     import pdb; pdb.set_trace()


            all_potential_ops = []
            # TODO: change it so that you fill up a unique potential_ops at each iteration, and then take the union for the final one
            # potential_ops = {}
            # for op_name in ddd.keys():
            #     potential_ops[op_name] = {
            #         "pre": set(),
            #         "add": set(),
            #         "del": set()
            #     }

            all_potential_ops2 = []
            for j, traj in enumerate(segmented_trajs):
                print(f"Segmented trajectory #{j}")
                potential_ops = {}
                for op_name in ddd.keys():
                    potential_ops[op_name] = {
                        "pre": set(),
                        "add": set(),
                        "del": set()
                    }
                potential_ops2 = {}
                for op_name in ddd.keys():
                    potential_ops2[op_name] = {
                        "pre": set(),
                        "add": set(),
                        "del": set()
                    }

                skip = False
                for seg in traj:
                    if seg_to_op(seg, final_clusters) is None:
                        skip = True
                        break
                if skip:
                    logging.info(f"Skipping this trajectory because it has a segment that isn't in any cluster.")
                    continue
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
                        if name == "Op12-Place" and j == 2:
                            process_adds2(potential_ops, remaining_story, self._train_tasks[j].goal)
                        else:
                            process_adds(potential_ops, remaining_story, self._train_tasks[j].goal)
                        process_dels(potential_ops, potential_ops2, remaining_story)


                all_potential_ops.append(potential_ops)
                all_potential_ops2.append(potential_ops2)

            # Leave out delete effects for now.
            final_potential_ops2 = {
                op_name: {
                    "pre": ddd[op_name][0], # take all the precondition predicates
                    "add": set(),
                    "del": ddd[op_name][2]
                } for op_name in all_potential_ops[0].keys()
            }
            for k, po in enumerate(all_potential_ops):
                for op_name in po.keys():
                    final_potential_ops2[op_name]["pre"] = final_potential_ops2[op_name]["pre"].union(po[op_name]["pre"])
                    final_potential_ops2[op_name]["add"] = final_potential_ops2[op_name]["add"].union(po[op_name]["add"])
                    final_potential_ops2[op_name]["del"] = final_potential_ops2[op_name]["del"].union(po[op_name]["del"])

            # Leave out delete effects for now.
            final_potential_ops = {
                op_name: {
                    "pre": ddd[op_name][0], # take all the precondition predicates
                    "add": set(),
                    # "del": ddd[op_name][2]
                    "del": set()
                } for op_name in all_potential_ops[0].keys()
            }
            for k, po in enumerate(all_potential_ops):
                for op_name in po.keys():
                    final_potential_ops[op_name]["pre"] = final_potential_ops[op_name]["pre"].union(po[op_name]["pre"])
                    final_potential_ops[op_name]["add"] = final_potential_ops[op_name]["add"].union(po[op_name]["add"])

            fff = {}
            for op in ddd.keys():
                fff[op] = []
                fff[op].append(
                    set(p for p in ddd[op][0] if p in final_potential_ops2[op]["pre"])
                )
                fff[op].append(
                    set(p for p in ddd[op][1] if p in final_potential_ops2[op]["add"])
                )
                fff[op].append(
                    set(p for p in ddd[op][2] if p in final_potential_ops2[op]["del"])
                )
                fff[op].append(ddd[op][3])

            import pdb; pdb.set_trace()
            print("mandalorian")
            self._clusters = fff

            # Check each delete effect of each operator to make sure it satisfies harmlessness if we include it
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

                option_vars = [obj_to_var[o] for o in opt_objs]
                option_spec = [seg_0.get_option().parent, option_vars]

                op_ignore_effects = set()
                op = STRIPSOperator(name, params, op_preconds, op_add_effects, op_del_effects, op_ignore_effects)

                # if op.name == "Op0-StickPressButton":
                #     import pdb; pdb.set_trace()
                #     print("slot")

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
                    t = (params, objects_list, relevant_preconds, relevant_add_effects, relevant_del_effects, seg, opt_objs)
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

                # print(f"DOING THIS FOR {name}")
                # if name == "Op0-Pick":
                #     import pdb; pdb.set_trace()
                op1 = ops[0]
                op1_opt_objs = op1[-1]
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
                        # if op.name == "Op0-StickPressButton":
                        #     print(f"Op2 preconds: {op2_preconds}")
                        op2_adds_str = set(str(a) for a in op2_add_effects)
                        op2_dels_str = set(str(a) for a in op2_del_effects)

                        score1 = len(op1_preconds_str.intersection(op2_preconds_str))
                        score2 = len(op1_adds_str.intersection(op2_adds_str))
                        score3 = len(op1_dels_str.intersection(op2_dels_str))
                        score = score1 + score2 + score3

                        new_preconds = set(a for a in op1_preconds if str(a) in op1_preconds_str.intersection(op2_preconds_str))
                        # if op.name == "Op0-StickPressButton":
                        #     print(f"New preconds: {new_preconds}")
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
                # if op.name == "Op0-StickPressButton":
                #     import pdb; pdb.set_trace()
                #     print("camera")
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

                    counter += 1

                option_vars = [op1_obj_to_var[o] for o in op1_opt_objs]
                option_spec = [seg_0.get_option().parent, option_vars]
                pnads.append(PNAD(op, datastore, option_spec))
                # if op.name == "Op0-StickPressButton":
                #     import pdb; pdb.set_trace()
                #     print("power button")

            for p in pnads:
                print(p)

            ##################################
            # IDENTIFY HARMLESS DELETE EFFECTS
            ##################################
            logging.info(f"Right before delete effects harmlessness check.")
            # want a list, not a set, as the return value
            def _ops_and_specs_to_dummy_nsrts(strips_ops: Sequence[STRIPSOperator], option_specs: Sequence[OptionSpec]) -> Set[NSRT]:
                """Create NSRTs from strips operators and option specs with dummy
                samplers."""
                assert len(strips_ops) == len(option_specs)
                nsrts = []
                for op, (param_option, option_vars) in zip(strips_ops, option_specs):
                    nsrt = op.make_nsrt(
                        param_option,
                        option_vars,  # dummy sampler
                        lambda s, g, rng, o: np.zeros(1, dtype=np.float32))
                    nsrts.append(nsrt)

                return nsrts

            predicates_we_kept = set()
            for op_name in final_potential_ops2.keys():
                predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["pre"])
                predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["add"])
                predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["del"])
            pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, predicates_we_kept | initial_predicates)
            pruned_segmented_trajs = [segment_trajectory(ll_traj, initial_predicates, atom_seq) for ll_traj, atom_seq in pruned_atom_data]
            pruned_low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
            pruned_atom_seq = [atom_seq for _, atom_seq in pruned_atom_data]

            strips_ops = [pnad.op for pnad in pnads]
            option_specs = [pnad.option_spec for pnad in pnads]
            dummy_nsrts = _ops_and_specs_to_dummy_nsrts(strips_ops, option_specs)
            pnad_to_nsrt = dict(zip(pnads, dummy_nsrts))

            harmful_del_effs_per_nsrt = {p.op.name: set() for p in pnads}
            harmful_preconds_per_nsrt = {p.op.name: set() for p in pnads}

            low_level_trajs = [llt for llt, _ in atom_dataset]
            for ccc, tup in enumerate(zip(low_level_trajs, segmented_trajs)):
                ll_traj, seg_traj = tup
                # this segmented data has atoms from the entire grammar
                ground_nsrt_plan = []
                for seg in seg_traj:
                    for pnad in pnads:
                        for seg2, sub in pnad.datastore:
                            if seg2 == seg:
                                nsrt = pnad_to_nsrt[pnad]
                                try:
                                    grounding = tuple(sub[p] for p in nsrt.parameters)
                                except:
                                    import pdb; pdb.set_trace()
                                ground_nsrt = nsrt.ground(grounding)
                                ground_nsrt_plan.append(ground_nsrt)

                init_state = ll_traj.states[0]
                init_atoms = utils.abstract(init_state, predicates_we_kept)
                curr_abstract_state = init_atoms
                prev_nsrts = []
                logging.info(f"ccc: {ccc}, len(ground_nsrt_plan): {len(ground_nsrt_plan)}")
                for h, nsrt in enumerate(ground_nsrt_plan):
                    # if ccc == 0 and h == len(ground_nsrt_plan) - 1:
                    # # if ccc == 0:
                    #     import pdb; pdb.set_trace()
                    #     print("aimbot123")
                    if not nsrt.preconditions.issubset(curr_abstract_state):
                        missing = nsrt.preconditions - nsrt.preconditions.intersection(curr_abstract_state)
                        for miss in missing:
                            miss_caused_by_delete_effect = False
                            for prev in prev_nsrts[::-1]:
                                if miss in prev.delete_effects:
                                    harmful_del_effs_per_nsrt[prev.op.name].add(miss.predicate)
                                    miss_caused_by_delete_effect = True
                                    break
                                    # only care about most recent one => may have to do multiple passes,
                                    # if two previous nsrts had it as a deleted effect (which may be unlikely or impossible, idk)

                            # TODO: do something about the condition where you don't find the miss anywhere
                            if not miss_caused_by_delete_effect:
                                # we should remove this predicate
                                harmful_preconds_per_nsrt[nsrt.op.name].add(miss.predicate)

                    curr_abstract_state = (curr_abstract_state | nsrt.add_effects) - nsrt.delete_effects
                    prev_nsrts.append(nsrt)

            logging.info(f"Harmful delete effects per nsrt: {harmful_del_effs_per_nsrt}")
            logging.info(f"Harmful preconditions per nsrt: {harmful_preconds_per_nsrt}")
            for k, v in harmful_del_effs_per_nsrt.items():
                for pred in v:
                    final_potential_ops2[k]["del"].remove(pred)
                    fff[k][2].remove(pred)
            for k, v in harmful_preconds_per_nsrt.items():
                for pred in v:
                    final_potential_ops2[k]["pre"].remove(pred)
                    fff[k][0].remove(pred)
            self._clusters = fff
            predicates_we_kept = set()
            for op_name in final_potential_ops.keys():
                predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["pre"])
                predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["add"])
                predicates_we_kept = predicates_we_kept.union(final_potential_ops2[op_name]["del"])

            import pdb; pdb.set_trace()
            logging.info(f"Right after delete effects harmlessness check.")
            ####################################################################

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

                option_vars = [obj_to_var[o] for o in opt_objs]
                option_spec = [seg_0.get_option().parent, option_vars]

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
                    t = (params, objects_list, relevant_preconds, relevant_add_effects, relevant_del_effects, seg, opt_objs)
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

                # print(f"DOING THIS FOR {name}")
                # if name == "Op0-Pick":
                #     import pdb; pdb.set_trace()
                op1 = ops[0]
                op1_opt_objs = op1[-1]
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

                temp = []

                datastore = []
                counter = 0
                for seg in segments:
                    seg_opt_objs = tuple(seg.get_option().objects)
                    var_to_obj = {v: o for v, o in zip(option_vars, seg_opt_objs)}


                    relevant_add_effects = [a for a in seg.add_effects if a.predicate in add_effects]
                    relevant_del_effects = [a for a in seg.delete_effects if a.predicate in del_effects]

                    seg_objs = {o for atom in relevant_add_effects + relevant_del_effects for o in atom.objects} | set(seg_opt_objs)

                    seg_objs_list = sorted(seg_objs)

                    # if op.name == "Op6-StickMoveToButton":
                    #     button = seg_objs_list[0]
                    #     stick = seg_objs_list[-1]
                    #     button_x = seg.states[-1].get(button, "x")
                    #     stick_x = seg.states[-1].get(stick, "x")
                    #     temp.append(abs(button_x - stick_x))

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

                    counter += 1

                # if op.name == "Op6-StickMoveToButton":
                #     print(f"max of temp: {max(temp)}")
                #     print(f"min of temp: {min(temp)}")
                #     import pdb; pdb.set_trace()

                option_vars = [op1_obj_to_var[o] for o in op1_opt_objs]
                option_spec = [seg_0.get_option().parent, option_vars]
                pnads.append(PNAD(op, datastore, option_spec))

                # might be able to handle ignore effects by something like what
                # _compute_pnad_ignore_effects in base_strips_learner.py does.
                # basically you can try to see if the ops would be used anywhere in
                # the demos, and and narrow it down via the demos. but you might
                # pick up some extraneous predicates in some of them, so you would
                # have to be careful and do some inference when trying them out on
                # the demos.

            # # prune duplicate pnads (happens in cover with k=2)
            # # probably want to check this for the mapping with the highest overlap?
            # # since the variables may match? or will they match in the cases where it ends up
            # # being the equivalent operator?
            # def ops_are_equivalent(op1, op2):
            #     return op1.preconditions==op2.preconditions and op1.add_effects==op2.add_effects and op1.delete_effects==op2.delete_effects
            #
            # pruned_pnads = []
            # indices_considered = set()
            # for i, pnad in enumerate(pnads):
            #     if i in indices_considered:
            #         continue
            #     # identify all indicies of pnads that are equivalent to this one
            #     for j, pnad_2 in enumerate(pnads):
            #         if j in indices_considered:
            #             continue
            #         if ops_are_equivalent(pnad.op, pnad_2.op):
            #             indices_considered.add(j)
            #     indices_considered.add(i)
            #     pruned_pnads.append(pnad)
            #
            # pnads = pruned_pnads

            # import pdb; pdb.set_trace()

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

            # import pdb; pdb.set_trace()
            self._pnads = pnads
            return predicates_we_kept
            # return predicates_to_keep
