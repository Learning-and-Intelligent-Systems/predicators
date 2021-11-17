"""An approach that invents predicates using a technique that is inspired by
Bonet, Frances & Geffner (2018), hence the name BFG.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Set, Callable, List, Optional, DefaultDict, Dict, Sequence, \
    Any
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import OperatorLearningApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Action, Dataset, GroundAtom, Transition, Operator, LiftedAtom, \
    Array, Object
from predicators.src.models import LearnedPredicateClassifier, MLPClassifier
from predicators.src.operator_learning import generate_transitions, \
    learn_operators_for_option
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


class BFGApproach(OperatorLearningApproach):
    """An approach that invents predicates using BFG.
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
        # First, generate a large candidate set of predicates.
        candidates = self._generate_candidate_predicates()
        # Apply the candidate predicates to the data.
        transitions_by_option = generate_transitions(
            dataset, candidates | self._initial_predicates)
        # Select a subset of the candidates to keep.
        self._learned_predicates = self._select_predicates_to_keep(candidates,
            transitions_by_option)
        # Finally, learn operators via superclass, using all the predicates.
        self._learn_operators(dataset)

    def _generate_candidate_predicates(self) -> Set[Predicate]:
        # TODO

        # Testing: python src/main.py --env cover --approach bfg --seed 0 --excluded_predicates Holding
        name = "InventedHolding"
        block_type = [t for t in self._types if t.name == "block"][0]
        types = [block_type]

        classifier = _SingleAttributeGEClassifier(0, "grasp", -0.9)
        
        predicate = Predicate(name, types, classifier)

        return {predicate}

    def _select_predicates_to_keep(self, candidates: Set[Predicate],
            transitions_by_option: DefaultDict[
                ParameterizedOption, List[Transition]]) -> Set[Predicate]:
        # TODO
        return candidates
