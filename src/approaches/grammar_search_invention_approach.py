"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar.
"""

from __future__ import annotations
import time
import abc
from dataclasses import dataclass, field
from functools import cached_property
import itertools
from operator import le
from typing import Set, Callable, List, Sequence, FrozenSet, Iterator, Tuple, \
    Dict, Collection
from gym.spaces import Box
import numpy as np
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach, ApproachFailure, \
    ApproachTimeout
from predicators.src.nsrt_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src.planning import task_plan
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, Object, GroundAtomTrajectory, STRIPSOperator, \
    OptionSpec, Segment, GroundAtom, _GroundSTRIPSOperator
from predicators.src.settings import CFG


################################################################################
#                          Programmatic classifiers                            #
################################################################################

def _create_grammar(dataset: Dataset, given_predicates: Set[Predicate]
                    ) -> _PredicateGrammar:
    # We start with the given predicates because we want to allow
    # negated and quantified versions of the given predicates, in
    # addition to negated and quantified versions of new predicates.
    given_grammar = _GivenPredicateGrammar(given_predicates)
    # Next, we consider various ways to split single feature values
    # across our dataset.
    sfi_grammar = _SingleFeatureInequalitiesPredicateGrammar(dataset)
    # This chained grammar has the effect of enumerating first the
    # given predicates, then the single feature inequality ones.
    chained_grammar = _ChainPredicateGrammar([given_grammar, sfi_grammar])
    # Now, the chained grammar will undergo a series of transformations.
    # For each predicate enumerated by the chained grammar, we also
    # enumerate the negation of that predicate.
    negated_grammar = _NegationPredicateGrammarWrapper(chained_grammar)
    # For each predicate enumerated, we also enumerate foralls for
    # that predicate, along with appropriate negations.
    forall_grammar = _ForallPredicateGrammarWrapper(negated_grammar)
    # Prune proposed predicates by checking if they are equivalent to
    # any already-generated predicates with respect to the dataset.
    # Note that we want to do this before the skip grammar below,
    # because if any predicates are equivalent to the given predicates,
    # we would not want to generate them.
    pruned_grammar = _PrunedGrammar(dataset, forall_grammar)
    # We don't actually need to enumerate the given predicates
    # because we already have them in the initial predicate set,
    # so we just filter them out from actually being enumerated.
    # But remember that we do want to enumerate their negations
    # and foralls, which is why they're included originally.
    final_grammar = _SkipGrammar(pruned_grammar, given_predicates)
    # We're done! Return the final grammar.
    return final_grammar


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
            assert cost > 0
            if cost >= CFG.grammar_search_predicate_cost_upper_bound:
                break
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
        for traj in self.dataset:
            types.update(o.type for o in traj.states[0])
        return types

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        """Iterate over candidate predicates in an arbitrary order.
        """
        raise NotImplementedError("Override me!")


def _halving_constant_generator(lo: float, hi: float, cost: float = 1.0
                                ) -> Iterator[Tuple[float, float]]:
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
    """Generates features of the form "0.feature >= c" or "0.feature <= c".
    """
    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # Get ranges of feature values from data.
        feature_ranges = self._get_feature_ranges()
        # 0.5, 0.25, 0.75, 0.125, 0.375, ...
        constant_generator = _halving_constant_generator(0.0, 1.0)
        for constant, cost in constant_generator:
            for t in sorted(self.types):
                for f in t.feature_names:
                    lb, ub = feature_ranges[t][f]
                    # Optimization: if lb == ub, there is no variation
                    # among this feature, so there's no point in trying to
                    # learn a classifier with it. So, skip the feature.
                    if abs(lb - ub) < 1e-6:
                        continue
                    # Scale the constant by the feature range.
                    k = (constant + lb) / (ub - lb)
                    # Only need one of (ge, le) because we can use negations
                    # to get the other (modulo equality, which we shouldn't
                    # rely on anyway because of precision issues).
                    comp, comp_str = le, "<="
                    classifier = _SingleAttributeCompareClassifier(
                        0, t, f, k, comp, comp_str)
                    name = str(classifier)
                    types = [t]
                    pred = Predicate(name, types, classifier)
                    assert pred.arity == 1
                    yield (pred, 1 + cost)  # cost = arity + cost from constant


    def _get_feature_ranges(self) -> Dict[Type, Dict[str, Tuple[float, float]]]:
        feature_ranges: Dict[Type, Dict[str, Tuple[float, float]]] = {}
        for traj in self.dataset:
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
            yield (predicate, predicate.arity + 1)


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
            # No change to costs when skipping.
            yield (predicate, cost)


@dataclass(frozen=True, eq=False, repr=False)
class _PrunedGrammar(_DataBasedPredicateGrammar):
    """A grammar that prunes redundant predicates.
    """
    base_grammar: _PredicateGrammar

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        # Predicates are identified based on their evaluation across
        # all states in the dataset.
        seen = {}  # maps identifier to previous predicate
        for (predicate, cost) in self.base_grammar.enumerate():
            if cost >= CFG.grammar_search_predicate_cost_upper_bound:
                return
            pred_id = self._get_predicate_identifier(predicate)
            if pred_id in seen:
                # Useful for debugging
                # print("Pruning", predicate, "b/c equal to", seen[pred_id])
                continue
            # Found a new predicate.
            seen[pred_id] = predicate
            yield (predicate, cost)

    def _get_predicate_identifier(self, predicate: Predicate
            ) -> FrozenSet[Tuple[int, int, FrozenSet[Tuple[Object, ...]]]]:
        """Returns frozensets of groundatoms for each data point.
        """
        # Get atoms for this predicate alone on the dataset.
        atom_dataset = utils.create_ground_atom_dataset(self.dataset,
                                                        {predicate})
        raw_identifiers = set()
        for traj_idx, (_, atom_traj) in enumerate(atom_dataset):
            for t, atoms in enumerate(atom_traj):
                atom_args = frozenset(tuple(a.objects) for a in atoms)
                raw_identifiers.add((traj_idx, t, atom_args))
        return frozenset(raw_identifiers)


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
            # No change to costs when negating.
            yield (negated_predicate, cost)


@dataclass(frozen=True, eq=False, repr=False)
class _ForallPredicateGrammarWrapper(_PredicateGrammar):
    """For each x generated by the base grammar, also generates forall(x)
    and the negation not-forall(x). If x has arity at least 2, also
    generates UnaryFreeForallClassifiers over x, along with negations.
    """
    base_grammar: _PredicateGrammar

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for (predicate, cost) in self.base_grammar.enumerate():
            yield (predicate, cost)
            if predicate.arity == 0:
                continue
            # Generate Forall(x)
            forall_classifier = _ForallClassifier(predicate)
            forall_predicate = Predicate(
                str(forall_classifier), [], forall_classifier)
            assert forall_predicate.arity == 0
            yield (forall_predicate, cost + 1)  # add arity + 1 to cost
            # Generate NOT-Forall(x)
            notforall_classifier = _NegationClassifier(forall_predicate)
            notforall_predicate = Predicate(
                str(notforall_classifier), forall_predicate.types,
                notforall_classifier)
            assert notforall_predicate.arity == 0
            yield (notforall_predicate, cost + 1)  # add arity + 1 to cost
            # Generate UFFs
            if predicate.arity >= 2:
                for idx in range(predicate.arity):
                    # Positive UFF
                    uff_classifier = _UnaryFreeForallClassifier(predicate, idx)
                    uff_predicate = Predicate(
                        str(uff_classifier), [predicate.types[idx]],
                        uff_classifier)
                    assert uff_predicate.arity == 1
                    yield (uff_predicate, cost + 2)  # add arity + 1 to cost
                    # Negated UFF
                    notuff_classifier = _NegationClassifier(uff_predicate)
                    notuff_predicate = Predicate(
                        str(notuff_classifier), uff_predicate.types,
                        notuff_classifier)
                    assert notuff_predicate.arity == 1
                    yield (notuff_predicate, cost + 2)  # add arity + 1 to cost


################################################################################
#                              Score Functions                                 #
################################################################################

def _create_score_function(
        score_function_name: str,
        initial_predicates: Set[Predicate],
        atom_dataset: List[GroundAtomTrajectory],
        train_tasks: List[Task],
        candidates: Dict[Predicate, float]
        ) -> _PredicateSearchScoreFunction:
    if score_function_name == "prediction_error":
        return _PredictionErrorScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates)
    if score_function_name == "branching_factor":
        return _BranchingFactorScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates)
    if score_function_name == "hadd_match":
        return _RelaxationHeuristicMatchBasedScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates, "hadd")
    if score_function_name == "hadd_lookahead":
        return _RelaxationHeuristicLookaheadBasedScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates, "hadd")
    if score_function_name == "hadd_lookahead_depth1":
        return _RelaxationHeuristicLookaheadBasedScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates, "hadd",
            lookahead_depth=1)
    if score_function_name == "hadd_lookahead_depth2":
        return _RelaxationHeuristicLookaheadBasedScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates, "hadd",
            lookahead_depth=2)
    if score_function_name == "hmax_lookahead":
        return _RelaxationHeuristicLookaheadBasedScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates, "hmax")
    if score_function_name == "hff_lookahead":
        return _RelaxationHeuristicLookaheadBasedScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates, "hff")
    if score_function_name == "exact_lookahead":
        return _ExactHeuristicLookaheadBasedScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates)
    if score_function_name == "task_planning":
        return _TaskPlanningScoreFunction(
            initial_predicates, atom_dataset, train_tasks, candidates)
    raise NotImplementedError(
        f"Unknown score function: {score_function_name}.")


@dataclass(frozen=True, eq=False, repr=False)
class _PredicateSearchScoreFunction:
    """A score function for guiding search over predicate sets.
    """
    _initial_predicates: Set[Predicate]  # predicates given by the environment
    _atom_dataset: List[GroundAtomTrajectory]  # data with all candidates
    _train_tasks: List[Task]  # training tasks that this data was generated on
    _candidates: Dict[Predicate, float]  # candidate predicates to costs

    def evaluate(self, predicates: FrozenSet[Predicate]) -> float:
        """Get the score for the set of predicates.

        Lower is better.
        """
        raise NotImplementedError("Override me!")

    def _get_predicate_penalty(self, predicates: FrozenSet[Predicate]) -> float:
        """Get a score penalty based on the predicate complexities.
        """
        total_pred_cost = sum(self._candidates[p] for p in predicates)
        return CFG.grammar_search_pred_complexity_weight * total_pred_cost


@dataclass(frozen=True, eq=False, repr=False)
class _OperatorLearningBasedScoreFunction(_PredicateSearchScoreFunction):
    """A score function that learns operators given the set of predicates.
    """

    def evaluate(self, predicates: FrozenSet[Predicate]) -> float:
        total_cost = sum(self._candidates[pred] for pred in predicates)
        print(f"Evaluating predicates: {predicates}, with total cost "
              f"{total_cost}")
        start_time = time.time()
        pruned_atom_data = utils.prune_ground_atom_dataset(
            self._atom_dataset, predicates | self._initial_predicates)
        segments = [seg for traj in pruned_atom_data
                    for seg in segment_trajectory(traj)]
        strips_ops, partitions = learn_strips_operators(segments,
                                                        verbose=False)
        option_specs = [p.option_spec for p in partitions]
        op_score = self._evaluate_with_operators(predicates,
            pruned_atom_data, segments, strips_ops, option_specs)
        pred_penalty = self._get_predicate_penalty(predicates)
        op_penalty = self._get_operator_penalty(strips_ops)
        total_score = op_score + pred_penalty + op_penalty
        print(f"\tTotal score: {total_score} computed in "
              f"{time.time()-start_time:.3f} seconds")
        return total_score

    def _evaluate_with_operators(self, predicates: FrozenSet[Predicate],
                                 pruned_atom_data: List[GroundAtomTrajectory],
                                 segments: List[Segment],
                                 strips_ops: List[STRIPSOperator],
                                 option_specs: List[OptionSpec]) -> float:
        """Use learned operators to compute a score for the predicates.
        """
        raise NotImplementedError("Override me!")

    @staticmethod
    def _get_operator_penalty(strips_ops: Collection[STRIPSOperator]
                             ) -> float:
        """Get a score penalty based on the operator complexities.
        """
        size = 0
        for op in strips_ops:
            size += len(op.parameters) + len(op.preconditions) + \
                    len(op.add_effects) + len(op.delete_effects)
        return CFG.grammar_search_size_weight * size


@dataclass(frozen=True, eq=False, repr=False)
class _PredictionErrorScoreFunction(_OperatorLearningBasedScoreFunction):
    """Score a predicate set by learning operators and counting false positives.
    """

    def _evaluate_with_operators(self, predicates: FrozenSet[Predicate],
                                 pruned_atom_data: List[GroundAtomTrajectory],
                                 segments: List[Segment],
                                 strips_ops: List[STRIPSOperator],
                                 option_specs: List[OptionSpec]) -> float:
        del predicates, pruned_atom_data  # unused
        num_true_positives, num_false_positives, _, _ = \
            _count_positives_for_ops(strips_ops, option_specs, segments)
        return CFG.grammar_search_false_pos_weight * num_false_positives + \
               CFG.grammar_search_true_pos_weight * (-num_true_positives)


@dataclass(frozen=True, eq=False, repr=False)
class _BranchingFactorScoreFunction(_OperatorLearningBasedScoreFunction):
    """Score a predicate set by learning operators and counting the number of
    ground operators that are applicable at each state in the data.
    """

    def _evaluate_with_operators(self, predicates: FrozenSet[Predicate],
                                 pruned_atom_data: List[GroundAtomTrajectory],
                                 segments: List[Segment],
                                 strips_ops: List[STRIPSOperator],
                                 option_specs: List[OptionSpec]) -> float:
        del predicates, pruned_atom_data, option_specs  # unused
        total_branching_factor = _count_branching_factor(strips_ops, segments)
        return CFG.grammar_search_bf_weight * total_branching_factor


@dataclass(frozen=True, eq=False, repr=False)
class _TaskPlanningScoreFunction(_OperatorLearningBasedScoreFunction):
    """Score a predicate set by learning operators and planning in the training
    tasks. The score corresponds to the total number of nodes expanded across
    all training problems. If no plan is found, a large penalty is added, which
    is meant to be an upper bound on the number of nodes that could be expanded.
    """

    def _evaluate_with_operators(self, predicates: FrozenSet[Predicate],
                                 pruned_atom_data: List[GroundAtomTrajectory],
                                 segments: List[Segment],
                                 strips_ops: List[STRIPSOperator],
                                 option_specs: List[OptionSpec]) -> float:
        del pruned_atom_data, segments  # unused
        score = 0.0
        node_expansion_upper_bound = 1e7
        for task in self._train_tasks:
            init_atoms = utils.abstract(
                task.init, predicates | self._initial_predicates)
            objects = set(task.init)
            try:
                _, _, metrics = task_plan(
                    init_atoms, objects, task.goal, strips_ops, option_specs,
                    CFG.seed, CFG.grammar_search_task_planning_timeout)
                node_expansions = metrics["num_nodes_expanded"]
                assert node_expansions < node_expansion_upper_bound
                score += node_expansions
            except (ApproachFailure, ApproachTimeout):
                score += node_expansion_upper_bound
        return score



@dataclass(frozen=True, eq=False, repr=False)
class _HeuristicBasedScoreFunction(_OperatorLearningBasedScoreFunction):
    """Score a predicate set by learning operators and comparing some
    heuristic against the demonstrations. Subclasses must choose the
    heuristic function and how to evaluate against the demonstrations.
    """
    def _evaluate_with_operators(self, predicates: FrozenSet[Predicate],
                                 pruned_atom_data: List[GroundAtomTrajectory],
                                 segments: List[Segment],
                                 strips_ops: List[STRIPSOperator],
                                 option_specs: List[OptionSpec]) -> float:
        score = 0.0  # lower is better
        seen_demos = 0
        for traj, atoms_sequence in pruned_atom_data:
            if not traj.is_demo:  # we only care about demonstrations
                continue
            if seen_demos == CFG.grammar_search_heuristic_based_max_demos:
                break
            seen_demos += 1
            init_atoms = atoms_sequence[0]
            objects = set(traj.states[0])
            goal = traj.goal
            ground_ops = {op for strips_op in strips_ops for op
                          in utils.all_ground_operators(strips_op, objects)}
            heuristic_fn = self._generate_heuristic(
                init_atoms, objects, goal, strips_ops, option_specs, ground_ops)
            score += self._evaluate_atom_trajectory(
                atoms_sequence, heuristic_fn, ground_ops)
        return CFG.grammar_search_heuristic_based_weight * score

    def _generate_heuristic(self, init_atoms: Set[GroundAtom],
                            objects: Set[Object],
                            goal: Set[GroundAtom],
                            strips_ops: Sequence[STRIPSOperator],
                            option_specs: Sequence[OptionSpec],
                            ground_ops: Set[_GroundSTRIPSOperator]
                            ) -> Callable[[Set[GroundAtom]], float]:
        raise NotImplementedError("Override me!")

    def _evaluate_atom_trajectory(
            self, atoms_sequence: List[Set[GroundAtom]],
            heuristic_fn: Callable[[Set[GroundAtom]], float],
            ground_ops: Set[_GroundSTRIPSOperator]) -> float:
        raise NotImplementedError("Override me!")


@dataclass(frozen=True, eq=False, repr=False)
class _HeuristicMatchBasedScoreFunction(_HeuristicBasedScoreFunction):  # pylint:disable=abstract-method
    """Implement _evaluate_atom_trajectory() by expecting the heuristic
    to match the exact costs-to-go of the states in the demonstrations.
    """
    def _evaluate_atom_trajectory(
            self, atoms_sequence: List[Set[GroundAtom]],
            heuristic_fn: Callable[[Set[GroundAtom]], float],
            ground_ops: Set[_GroundSTRIPSOperator]) -> float:
        score = 0.0
        for i, atoms in enumerate(atoms_sequence):
            ideal_h = len(atoms_sequence) - i - 1
            h = heuristic_fn(atoms)
            score += abs(h - ideal_h)
        return score


@dataclass(frozen=True, eq=False, repr=False)
class _HeuristicLookaheadBasedScoreFunction(_HeuristicBasedScoreFunction):  # pylint:disable=abstract-method
    """Implement _evaluate_atom_trajectory() by using the induced operators
    to compute an energy-based policy, and comparing that policy to demos.

    Overview of the idea:
    1. Predicates induce operators. Denote this ops(preds).
    2. Operators induce a heuristic. Denote this h(state, ops(preds)).
    3. The heuristic induces a greedy one-step lookahead energy-based policy.
       Denote this pi(a | s) propto exp(-k * h(succ(s, a), ops(preds)) where
       k is CFG.grammar_search_lookahead_based_temperature.
    4. The objective for predicate learning is to maximize prod pi(a | s)
       where the product is over demonstrations.
    """
    def _evaluate_atom_trajectory(
            self, atoms_sequence: List[Set[GroundAtom]],
            heuristic_fn: Callable[[Set[GroundAtom]], float],
            ground_ops: Set[_GroundSTRIPSOperator]) -> float:
        score = 0.0
        for i in range(len(atoms_sequence)-1):
            atoms, next_atoms = atoms_sequence[i], atoms_sequence[i+1]
            ground_op_demo_lpm = -np.inf  # total log prob mass for demo actions
            ground_op_total_lpm = -np.inf  # total log prob mass for all actions
            for predicted_next_atoms in utils.get_successors_from_ground_ops(
                atoms, ground_ops, unique=False):
                # Compute the heuristic for the successor atoms.
                h = heuristic_fn(predicted_next_atoms)
                # Compute the probability that the correct next atoms would be
                # output under an energy-based policy.
                k = CFG.grammar_search_lookahead_based_temperature
                log_p = -k * h
                ground_op_total_lpm = np.logaddexp(log_p, ground_op_total_lpm)
                # Check whether the successor atoms match the demonstration.
                if predicted_next_atoms == next_atoms:
                    ground_op_demo_lpm = np.logaddexp(log_p, ground_op_demo_lpm)
            # If there is a demonstration state that is a dead-end under the
            # operators, immediately return a very bad score, because planning
            # with these operators would never be able to recover the demo.
            if ground_op_demo_lpm == -np.inf:
                return float("inf")
            # Accumulate the log probability of each (state, action) in this
            # demonstrated trajectory.
            trans_log_prob = ground_op_demo_lpm - ground_op_total_lpm
            score += -trans_log_prob  # remember that lower is better
        return score


@dataclass(frozen=True, eq=False, repr=False)
class _RelaxationHeuristicBasedScoreFunction(_HeuristicBasedScoreFunction):  # pylint:disable=abstract-method
    """Implement _generate_heuristic() with a delete relaxation heuristic like
    hadd, hmax, or hff.
    """
    heuristic_name: str
    lookahead_depth: int = field(default=0)

    def _generate_heuristic(self, init_atoms: Set[GroundAtom],
                            objects: Set[Object],
                            goal: Set[GroundAtom],
                            strips_ops: Sequence[STRIPSOperator],
                            option_specs: Sequence[OptionSpec],
                            ground_ops: Set[_GroundSTRIPSOperator]
                            ) -> Callable[[Set[GroundAtom]], float]:
        h_fn = utils.create_heuristic(self.heuristic_name, init_atoms, goal,
                                      ground_ops)
        del init_atoms  # unused after this
        cache: Dict[Tuple[FrozenSet[GroundAtom], int], float] = {}
        def _relaxation_h(atoms: Set[GroundAtom], depth: int=0) -> float:
            cache_key = (frozenset(atoms), depth)
            if cache_key in cache:
                return cache[cache_key]
            if goal.issubset(atoms):
                result = 0.0
            elif depth == self.lookahead_depth:
                result = h_fn(utils.atoms_to_tuples(atoms))
            else:
                successor_hs = [_relaxation_h(next_atoms, depth+1)
                    for next_atoms in utils.get_successors_from_ground_ops(
                    atoms, ground_ops)]
                if not successor_hs:
                    return float("inf")
                result = 1.0 + min(successor_hs)
            cache[cache_key] = result
            return result
        return _relaxation_h


@dataclass(frozen=True, eq=False, repr=False)
class _ExactHeuristicBasedScoreFunction(_HeuristicBasedScoreFunction):  # pylint:disable=abstract-method
    """Implement _generate_heuristic() with task planning.
    """
    def _generate_heuristic(self, init_atoms: Set[GroundAtom],
                            objects: Set[Object],
                            goal: Set[GroundAtom],
                            strips_ops: Sequence[STRIPSOperator],
                            option_specs: Sequence[OptionSpec],
                            ground_ops: Set[_GroundSTRIPSOperator]
                            ) -> Callable[[Set[GroundAtom]], float]:
        del init_atoms  # unused
        cache: Dict[FrozenSet[GroundAtom], float] = {}
        def _task_planning_h(atoms: Set[GroundAtom]) -> float:
            """Run task planning and return the length of the skeleton,
            or inf if no skeleton is found.
            """
            if frozenset(atoms) in cache:
                return cache[frozenset(atoms)]
            try:
                skeleton, atoms_sequence, _ = task_plan(
                    atoms, objects, goal, strips_ops, option_specs,
                    CFG.seed, CFG.grammar_search_task_planning_timeout)
            except (ApproachFailure, ApproachTimeout):
                return float("inf")
            assert atoms_sequence[0] == atoms
            for i, actual_atoms in enumerate(atoms_sequence):
                cache[frozenset(actual_atoms)] = float(len(skeleton) - i)
            return cache[frozenset(atoms)]
        return _task_planning_h


@dataclass(frozen=True, eq=False, repr=False)
class _RelaxationHeuristicMatchBasedScoreFunction(
        _RelaxationHeuristicBasedScoreFunction,
        _HeuristicMatchBasedScoreFunction):
    """Implement _generate_heuristic() with a delete relaxation heuristic and
    _evaluate_atom_trajectory() with matching.
    """


@dataclass(frozen=True, eq=False, repr=False)
class _RelaxationHeuristicLookaheadBasedScoreFunction(
        _RelaxationHeuristicBasedScoreFunction,
        _HeuristicLookaheadBasedScoreFunction):
    """Implement _generate_heuristic() with a delete relaxation heuristic and
    _evaluate_atom_trajectory() with one-step lookahead.
    """


@dataclass(frozen=True, eq=False, repr=False)
class _ExactHeuristicLookaheadBasedScoreFunction(
        _ExactHeuristicBasedScoreFunction,
        _HeuristicLookaheadBasedScoreFunction):
    """Implement _generate_heuristic() with task planning and
    _evaluate_atom_trajectory() with a lookahead-based policy.
    """


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
        grammar = _create_grammar(self._dataset, self._initial_predicates)
        candidates = grammar.generate(max_num=CFG.grammar_search_max_predicates)
        print(f"Done: created {len(candidates)} candidates:")
        for predicate, cost in candidates.items():
            print(predicate, cost)
        # Apply the candidate predicates to the data.
        print("Applying predicates to data...")
        atom_dataset = utils.create_ground_atom_dataset(
            self._dataset, set(candidates) | self._initial_predicates)
        print("Done.")
        # Create the score function that will be used to guide search.
        score_function = _create_score_function(
            CFG.grammar_search_score_function, self._initial_predicates,
            atom_dataset, train_tasks, candidates)
        # Select a subset of the candidates to keep.
        print("Selecting a subset...")
        self._learned_predicates = _select_predicates_to_keep(
            candidates, score_function)
        print("Done.")
        # Finally, learn NSRTs via superclass, using all the kept predicates.
        self._learn_nsrts()


def _select_predicates_to_keep(
        candidates: Dict[Predicate, float],
        score_function: _PredicateSearchScoreFunction
        ) -> Set[Predicate]:
    """Perform a greedy search over predicate sets.
    """

    # There are no goal states for this search; run until exhausted.
    def _check_goal(s: FrozenSet[Predicate]) -> bool:
        del s  # unused
        return False

    # Successively consider larger predicate sets.
    def _get_successors(s: FrozenSet[Predicate]
            ) -> Iterator[Tuple[None, FrozenSet[Predicate], float]]:
        for predicate in sorted(set(candidates) - s):  # determinism
            # Actions not needed. Frozensets for hashing.
            # The cost of 1.0 is irrelevant because we're doing GBFS
            # and not A* (because we don't care about the path).
            yield (None, frozenset(s | {predicate}), 1.0)

    # Start the search with no candidates.
    init : FrozenSet[Predicate] = frozenset()

    # Greedy best first search.
    path, _ = utils.run_gbfs(
        init, _check_goal, _get_successors, score_function.evaluate,
        max_evals=CFG.grammar_search_max_evals)
    kept_predicates = path[-1]

    print(f"\nSelected {len(kept_predicates)} predicates out of "
          f"{len(candidates)} candidates:")
    for pred in kept_predicates:
        print("\t", pred)
    score_function.evaluate(kept_predicates)  # print out useful numbers

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
    # The following two lists are just useful for debugging.
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


def _count_branching_factor(strips_ops: List[STRIPSOperator],
                            segments: List[Segment]
                            ) -> int:
    """Returns the total branching factor for all states in the segments.
    """
    total_branching_factor = 0
    for segment in segments:
        atoms = segment.init_atoms
        objects = set(segment.states[0])
        ground_ops = {ground_op for op in strips_ops
                      for ground_op in utils.all_ground_operators(op, objects)}
        for _ in utils.get_applicable_operators(ground_ops, atoms):
            total_branching_factor += 1
    return total_branching_factor
