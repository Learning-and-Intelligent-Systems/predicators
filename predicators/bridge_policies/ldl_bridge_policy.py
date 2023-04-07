"""A bridge policy that uses a lifted decision list with failure conditions."""

import abc
import logging
from typing import Callable, List, Set

from predicators import utils
from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.structs import NSRT, GroundAtom, LiftedDecisionList, Object, \
    ParameterizedOption, Predicate, State, Type, _Option, BridgePolicyDoneNSRT


class LDLBridgePolicy(BaseBridgePolicy):
    """A lifted decision list bridge policy with failure conditions."""

    def __init__(self, types: Set[Type], predicates: Set[Predicate],
                 options: Set[ParameterizedOption], nsrts: Set[NSRT]) -> None:
        super().__init__(types, predicates, options, nsrts)
        self._failure_predicates = utils.get_all_failure_predicates(options)
        self._offending_object_predicates = utils.get_all_offending_object_predicates(
            types)

    @abc.abstractmethod
    def _get_current_ldl(self) -> LiftedDecisionList:
        """Return the current lifted decision list policy."""

    def _bridge_policy(self, state: State, atoms: Set[GroundAtom],
                       failed_options: List[_Option],
                       offending_objects: Set[Object]) -> _Option:
        ldl = self._get_current_ldl()
        # Add failure atoms based on failed_options.
        atoms |= utils.get_failure_atoms(failed_options)
        # Add offending object atoms based on offending_objects.
        atoms |= utils.get_offending_object_atoms(offending_objects)
        objects = set(state)
        goal: Set[GroundAtom] = set()  # task goal not used
        next_nsrt = utils.query_ldl(ldl, atoms, objects, goal)
        if next_nsrt.parent == BridgePolicyDoneNSRT:
            raise BridgePolicyDone()
        return next_nsrt.sample_option(state, goal, self._rng)

    def get_option_policy(self) -> Callable[[State], _Option]:

        def _option_policy(state: State) -> _Option:
            atoms = utils.abstract(state, self._predicates)
            option = self._bridge_policy(state, atoms, self._failed_options,
                                         self._offending_objects)
            logging.debug(f"Using option {option.name}{option.objects} "
                          "from bridge policy.")
            return option

        return _option_policy
