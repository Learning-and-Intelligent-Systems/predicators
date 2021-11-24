"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar.
"""

from dataclasses import dataclass
import itertools
from functools import cached_property
from operator import ge, le
from typing import Set, Callable, List, Sequence, FrozenSet, Iterator, Tuple, \
    Dict
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach
from predicators.src.nsrt_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, Object, GroundAtomTrajectory, STRIPSOperator
from predicators.src.settings import CFG


@dataclass(frozen=True, eq=False, repr=False)
class _PredicateGrammar:
    """A grammar for generating predicate candidates.
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

    def generate(self, max_num: int) -> Dict[Predicate, float]:
        """Generate candidate predicates from the grammar.
        The dict values are costs, e.g., negative log prior probability for the
        predicate in a PCFG.
        """
        candidates = {}
        for i, (candidate, cost) in enumerate(self._generate()):
            if i >= max_num:
                break
            candidates[candidate] = cost
        return candidates

    def _generate(self) -> Iterator[Tuple[Predicate, float]]:
        raise NotImplementedError("Override me!")


@dataclass(frozen=True, eq=False, repr=False)
class _SingleAttributeCompareClassifier:
    """Compare a single feature value with a constant value.
    """
    object_index: int
    attribute_name: str
    constant: float
    compare: Callable[[float, float], bool]
    compare_str: str

    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        obj = o[self.object_index]
        return self.compare(s.get(obj, self.attribute_name), self.constant)

    def __str__(self) -> str:
        return f"({self.object_index}.{self.attribute_name}" + \
               f"{self.compare_str}{self.constant:.3})"


@dataclass(frozen=True, eq=False, repr=False)
class _HoldingDummyPredicateGrammar(_PredicateGrammar):
    """A hardcoded cover-specific grammar.

    Good for testing with:
        python src/main.py --env cover --approach grammar_search_invention \
            --seed 0 --excluded_predicates Holding
    """
    def _generate(self) -> Iterator[Tuple[Predicate, float]]:
        # A necessary predicate
        block_type = [t for t in self.types if t.name == "block"][0]
        types = [block_type]
        classifier = _SingleAttributeCompareClassifier(0, "grasp", -0.9,
                                                       ge, ">=")
        yield (Predicate(str(classifier), types, classifier), 1.)

        # An unnecessary predicate (because it's redundant)
        classifier = _SingleAttributeCompareClassifier(0, "is_block", 0.5,
                                                       ge, ">=")
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
class _SingleFeatureInequalitiesPredicateGrammar(_PredicateGrammar):
    """Generates features of the form "?x.feature >= c" or "?x.feature <= c".
    """
    def _generate(self) -> Iterator[Tuple[Predicate, float]]:
        # Get ranges of feature values from data.
        feature_ranges = self._get_feature_ranges()
        # 0., 1., 0.5, 0.25, 0.75, 0.125, 0.375, ...
        constant_generator = itertools.chain([0., 1.],
            _halving_constant_generator(0., 1.))
        for c in constant_generator:
            for t in sorted(self.types):
                for f in t.feature_names:
                    lb, ub = feature_ranges[t][f]
                    # Optimization: if lb == ub, there is no variation
                    # among this feature, so there's no point in trying to
                    # learn a classifier with it. So, skip the feature.
                    if lb == ub:
                        continue
                    # Scale the constant by the feature range.
                    k = (c + lb) / (ub - lb)
                    for (comp, comp_str) in [(ge, ">="), (le, "<=")]:
                        classifier = _SingleAttributeCompareClassifier(
                            0, f, k, comp, comp_str)
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


def _create_grammar(grammar_name: str, dataset: Dataset) -> _PredicateGrammar:
    if grammar_name == "holding_dummy":
        return _HoldingDummyPredicateGrammar(dataset)
    if grammar_name == "single_feat_ineqs":
        return _SingleFeatureInequalitiesPredicateGrammar(dataset)
    raise NotImplementedError(f"Unknown grammar name: {grammar_name}.")



class GrammarSearchInventionApproach(NSRTLearningApproach):
    """An approach that invents predicates by searching over candidate sets,
    with the candidates proposed from a grammar.
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
        # Generate a candidate set of predicates.
        print("Generating candidate predicates...")
        grammar = _create_grammar(CFG.grammar_search_grammar_name, dataset)
        candidates = grammar.generate(max_num=CFG.grammar_search_max_predicates)
        print(f"Done: created {len(candidates)} candidates.")
        # Apply the candidate predicates to the data.
        print("Applying predicates to data...")
        atom_dataset = utils.create_ground_atom_dataset(dataset,
            set(candidates) | self._initial_predicates)
        print("Done.")
        # Select a subset of the candidates to keep.
        print("Selecting a subset...")
        self._learned_predicates = self._select_predicates_to_keep(candidates,
            atom_dataset)
        print("Done.")
        # Finally, learn NSRTs via superclass, using all the kept predicates.
        self._learn_nsrts(dataset)

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
            strips_ops, _ = learn_strips_operators(segments, verbose=False)
            # Score based on how well the operators fit the data.
            num_true_positives, num_false_positives = \
                _count_positives_for_ops(strips_ops, pruned_atom_data)
            # Also add a size penalty.
            op_size = _get_operators_size(strips_ops)
            # Also add a penalty based on predicate complexity.
            pred_complexity = sum(candidates[p] for p in s)
            # Lower is better.
            return CFG.grammar_search_false_pos_weight * num_false_positives + \
                CFG.grammar_search_true_pos_weight * (-num_true_positives) + \
                CFG.grammar_search_size_weight * op_size + \
                CFG.grammar_search_pred_complexity_weight * pred_complexity

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
                             pruned_atom_data: List[GroundAtomTrajectory]
                             ) -> Tuple[int, int]:
    """Returns num true positives, num false positives.
    """
    num_true_positives = 0
    num_false_positives = 0
    for (states, _, atom_sequence) in pruned_atom_data:
        objects = set(states[0])
        ground_ops = [o for op in strips_ops
                      for o in utils.all_ground_operators(op, objects)]
        for i in range(len(atom_sequence)-1):
            s = atom_sequence[i]
            ns = atom_sequence[i+1]
            for ground_op in ground_ops:
                if not ground_op.preconditions.issubset(s):
                    continue
                if ground_op.add_effects == ns - s and \
                   ground_op.delete_effects == s - ns:
                    num_true_positives += 1
                else:
                    num_false_positives += 1
    return num_true_positives, num_false_positives


def _get_operators_size(strips_ops: List[STRIPSOperator]) -> int:
    size = 0
    for op in strips_ops:
        size += len(op.parameters) + len(op.preconditions) + \
                len(op.add_effects) + len(op.delete_effects)
    return size
