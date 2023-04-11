"""A bridge policy that uses a lifted decision list with failure conditions."""

import abc
import logging
from typing import Callable, List, Set, Tuple

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.structs import NSRT, BridgePolicyDoneNSRT, \
    BridgePolicyFailure, GroundAtom, LiftedDecisionList, Object, \
    ParameterizedOption, Predicate, State, Type, _Option


class LDLBridgePolicy(BaseBridgePolicy):
    """A lifted decision list bridge policy with failure conditions."""

    def __init__(self, types: Set[Type], predicates: Set[Predicate],
                 options: Set[ParameterizedOption], nsrts: Set[NSRT]) -> None:
        super().__init__(types, predicates, options, nsrts)
        self._failure_predicates = utils.get_all_failure_predicates(options)
        self._offending_object_predicates = \
            utils.get_all_offending_object_predicates(types)

    @abc.abstractmethod
    def _get_current_ldl(self) -> LiftedDecisionList:
        """Return the current lifted decision list policy."""

    def _bridge_policy(self, state: State, atoms: Set[GroundAtom]) -> _Option:
        ldl = self._get_current_ldl()
        objects = set(state)
        goal: Set[GroundAtom] = set()  # task goal not used
        next_nsrt = utils.query_ldl(ldl, atoms, objects, goal)
        if next_nsrt is None:
            raise ApproachFailure("LDL bridge policy not applicable.")
        if next_nsrt.parent == BridgePolicyDoneNSRT:
            raise BridgePolicyDone()
        return next_nsrt.sample_option(state, goal, self._rng)

    def get_option_policy(self) -> Callable[[State], _Option]:

        def _option_policy(state: State) -> _Option:
            # Process history into set of predicates.
            state_history = self._state_history + [state]
            new_atoms = utils.abstract(state, self._predicates)
            atoms_history = self._atoms_history + [new_atoms]
            atoms = self._history_to_atoms(state_history, atoms_history,
                                           self._option_history,
                                           self._failure_history)
            option = self._bridge_policy(state, atoms)
            logging.debug(f"Using option {option.name}{option.objects} "
                          "from bridge policy.")
            return option

        return _option_policy

    def _history_to_atoms(
        self, state_history: List[State], atoms_history: List[Set[GroundAtom]],
        option_history: List[_Option],
        failure_history: List[Tuple[int,
                                    BridgePolicyFailure]]) -> Set[GroundAtom]:
        assert len(state_history) == len(atoms_history)
        assert len(state_history) == len(option_history) + 1
        all_failed_options = [o for _, (o, _) in failure_history]
        failure_atoms = utils.get_failure_atoms(all_failed_options)
        last_atoms = atoms_history[-1]
        all_offending_objects = {o for _, (_, (o, )) in failure_history}
        all_offending_atoms = utils.get_offending_object_atoms(
            all_offending_objects)
        last_offending_objects = failure_history[-1][1][1]
        last_offending_atoms = utils.get_offending_object_atoms(
            last_offending_objects, last_failure=True)
        atoms = last_atoms | failure_atoms | all_offending_atoms | \
            last_offending_atoms
        return atoms
