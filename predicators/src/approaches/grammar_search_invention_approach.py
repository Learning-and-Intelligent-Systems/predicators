"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar."""

from __future__ import annotations

import abc
import itertools
import logging
from dataclasses import dataclass, field
from functools import cached_property
from operator import le
from typing import Callable, Dict, FrozenSet, Iterator, List, Sequence, Set, \
    Tuple

from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.nsrt_learning.segmentation import segment_trajectory
from predicators.src.nsrt_learning.strips_learning import \
    learn_strips_operators
from predicators.src.predicate_search_score_functions import \
    _PredicateSearchScoreFunction, create_score_function
from predicators.src.settings import CFG
from predicators.src.structs import Dataset, GroundAtom, \
    GroundAtomTrajectory, Object, ParameterizedOption, Predicate, State, \
    Task, Type

################################################################################
#                          Programmatic classifiers                            #
################################################################################


def _create_grammar(dataset: Dataset,
                    given_predicates: Set[Predicate]) -> _PredicateGrammar:
    # We start with considering various ways to split single feature values
    # across our dataset.
    grammar: _PredicateGrammar = _SingleFeatureInequalitiesPredicateGrammar(
        dataset)
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
    "blocks": [
        "NOT-((0:robot).fingers<=[idx 0]",  # GripperOpen
        "Forall[0:block].[NOT-On(0,1)]",  # Clear
        "NOT-((0:block).pose_z<=[idx 0]",  # Holding
    ],
    "unittest": [
        "((0:robot).hand<=[idx 0]0.65)",
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

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        return itertools.chain.from_iterable(g.enumerate()
                                             for g in self.base_grammars)


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
        if CFG.segmenter == "option_changes":
            # If the segmenter is based on changes in options, then
            # because it doesn't depend on atoms, we can be very
            # efficient during pruning by pre-computing the segments.
            # Then, we only need to care about the initial and final
            # states in each segment, which we store into
            # self._state_sequence.
            for traj in self.dataset.trajectories:
                dummy_atoms_seq: List[Set[GroundAtom]] = [
                    set() for _ in range(len(traj.states))
                ]
                seg_traj = segment_trajectory((traj, dummy_atoms_seq))
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
        if CFG.segmenter == "option_changes":
            # Make use of the pre-computed segment-level state sequences.
            for traj_idx, state_seq in enumerate(self._state_sequences):
                for t, state in enumerate(state_seq):
                    atoms = utils.abstract(state, {predicate})
                    atom_args = frozenset(tuple(a.objects) for a in atoms)
                    raw_identifiers.add((traj_idx, t, atom_args))
        else:
            assert CFG.segmenter == "atom_changes"
            # Get atoms for this predicate alone on the dataset, and then
            # go through the entire dataset.
            atom_dataset = utils.create_ground_atom_dataset(
                self.dataset.trajectories, {predicate})
            for traj_idx, (_, atom_traj) in enumerate(atom_dataset):
                for t, atoms in enumerate(atom_traj):
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
        # Create the score function that will be used to guide search.
        score_function = create_score_function(
            CFG.grammar_search_score_function, self._initial_predicates,
            atom_dataset, candidates, self._train_tasks)
        # Select a subset of the candidates to keep.
        logging.info("Selecting a subset...")
        self._learned_predicates = _select_predicates_to_keep(
            candidates, score_function, self._initial_predicates, atom_dataset,
            self._train_tasks)
        logging.info("Done.")
        # Finally, learn NSRTs via superclass, using all the kept predicates.
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None)


def _select_predicates_to_keep(candidates: Dict[Predicate, float],
                               score_function: _PredicateSearchScoreFunction,
                               initial_predicates: set[Predicate],
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
        path, _ = utils.run_gbfs(init,
                                 _check_goal,
                                 _get_successors,
                                 score_function.evaluate,
                                 max_evals=CFG.grammar_search_gbfs_num_evals)
    else:
        raise NotImplementedError(
            "Unrecognized grammar_search_search_algorithm: "
            f"{CFG.grammar_search_search_algorithm}.")
    kept_predicates = path[-1]

    # Filter out predicates that don't appear in some operator preconditions.
    logging.info("\nFiltering out predicates that don't appear in "
                 "preconditions...")
    pruned_atom_data = utils.prune_ground_atom_dataset(
        atom_dataset, kept_predicates | initial_predicates)
    segmented_trajs = [segment_trajectory(traj) for traj in pruned_atom_data]
    low_level_trajs = [ll_traj for ll_traj, _ in pruned_atom_data]
    preds_in_preconds = set()
    for pnad in learn_strips_operators(low_level_trajs,
                                       train_tasks,
                                       set(kept_predicates
                                           | initial_predicates),
                                       segmented_trajs,
                                       verify_harmlessness=False,
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
