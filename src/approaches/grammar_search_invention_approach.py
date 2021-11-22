"""An approach that invents predicates by searching over candidate sets, with
the candidates proposed from a grammar.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Callable, List, Optional, DefaultDict, Dict, Sequence, \
    Any, FrozenSet, Iterator
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach
from predicators.src.nsrt_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, GroundAtom, Transition, LiftedAtom, \
    Array, Object, GroundAtomTrajectory
from predicators.src.torch_models import LearnedPredicateClassifier, \
    MLPClassifier
from predicators.src.settings import CFG


@dataclass(frozen=True, eq=False, repr=False)
class _SingleAttributeGEClassifier:
    """Check whether a single attribute value on an object is >= some value.
    """
    object_index: int
    attribute_name: str
    value: float

    def __call__(self, s: State, o: Sequence[Object]) -> bool:
        obj = o[self.object_index]
        return s.get(obj, self.attribute_name) >= self.value


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
        # Generate a large candidate set of predicates.
        candidates = self._generate_candidate_predicates()
        # Apply the candidate predicates to the data.
        atom_dataset = utils.create_ground_atom_dataset(dataset, candidates)
        # Select a subset of the candidates to keep.
        self._learned_predicates = self._select_predicates_to_keep(candidates,
            atom_dataset)
        # Finally, learn NSRTs via superclass, using all the kept predicates.
        self._learn_nsrts(dataset)

    def _generate_candidate_predicates(self) -> Set[Predicate]:
        # TODO
        # Testing: python src/main.py --env cover --approach grammar_search_invention --seed 0 --excluded_predicates Holding

        candidates = set()

        # A necessary predicate
        name = "InventedHolding"
        block_type = [t for t in self._types if t.name == "block"][0]
        types = [block_type]
        classifier = _SingleAttributeGEClassifier(0, "grasp", -0.9)
        predicate = Predicate(name, types, classifier)
        candidates.add(predicate)

        # An unnecessary predicate (because it's redundant)
        name = "InventedDummy"
        block_type = [t for t in self._types if t.name == "block"][0]
        types = [block_type]
        classifier = _SingleAttributeGEClassifier(0, "is_block", 0.5)
        predicate = Predicate(name, types, classifier)
        candidates.add(predicate)

        return candidates

    def _select_predicates_to_keep(self, candidates: Set[Predicate],
                                   atom_dataset: List[GroundAtomTrajectory]
                                   ) -> Set[Predicate]:
        # Perform a greedy search over predicate sets.
        # Successively consider small predicate sets.
        def _get_successors(s: FrozenSet[Predicate]
                            ) -> Iterator[FrozenSet[Predicate]]:
            for predicate in sorted(s):  # sorting for determinism
                yield frozenset(s - {predicate})  # frozenset for hashing

        # The heuristic is where the action happens...
        def _heuristic(s: FrozenSet[Predicate]) -> float:
            # Relearn operators with the current predicates.
            pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset, s)
            segments = [seg for traj in pruned_atom_data
                        for seg in segment_trajectory(traj)]
            strips_ops, _ = learn_strips_operators(segments)
            # Score based on how well the operators fit the data.
            import ipdb; ipdb.set_trace()
            # Also add a size penalty.

        # There are no goal states for this search; run until exhausted.
        def _check_goal(s : FrozenSet[Predicate]) -> bool:
            return False

        # Start the search with all of the candidates.
        init = frozenset(candidates)

        # Greedy best first search.
        path, _ = utils.run_gbfs(
            init, _check_goal, _get_successors, _heuristic,
            max_expansions=CFG.grammar_search_max_expansions)
        kept_predicates = path[-1]

        print(f"Selected {len(kept_predicates)} predicates out of "
              f"{len(candidates)} candidates:")
        for pred in kept_predicates:
            print(pred)

        return kept_predicates
