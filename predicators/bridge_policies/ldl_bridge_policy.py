"""A bridge policy that uses a lifted decision list with failure conditions."""

import abc
import functools
import logging
from typing import Callable, List, Set, Tuple

from predicators import utils
from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.structs import GroundAtom, LiftedDecisionList, \
    ParameterizedOption, Predicate, State, _Option


@functools.lru_cache(maxsize=None)
def get_failure_predicate(option: ParameterizedOption,
                          idxs: Tuple[int]) -> Predicate:
    """Create a Failure predicate for a parameterized option."""
    idx_str = ",".join(map(str, idxs))
    arg_types = [option.types[i] for i in idxs]
    return Predicate(f"{option.name}Failed-arg{idx_str}",
                     arg_types,
                     _classifier=lambda s, o: False)


class LDLBridgePolicy(BaseBridgePolicy):
    """A lifted decision list bridge policy with failure conditions."""

    @abc.abstractmethod
    def _get_current_ldl(self) -> LiftedDecisionList:
        """Return the current lifted decision list policy."""

    def _bridge_policy(self, state: State, atoms: Set[GroundAtom],
                       failed_options: List[_Option]) -> _Option:

        ldl = self._get_current_ldl()

        # Add failure atoms based on failed_options.
        atoms_with_failures = set(atoms)
        failed_option_specs = {(o.parent, tuple(o.objects))
                               for o in failed_options}
        for (param_opt, objs) in failed_option_specs:
            for i, obj in enumerate(objs):
                # Just unary for now.
                idxs = (i, )
                pred = get_failure_predicate(param_opt, idxs)
                failure_atom = GroundAtom(pred, [obj])
                atoms_with_failures.add(failure_atom)

        objects = set(state)
        goal: Set[GroundAtom] = set()  # task goal not used
        next_nsrt = utils.query_ldl(ldl, atoms_with_failures, objects, goal)
        if next_nsrt is None:
            raise BridgePolicyDone()

        return next_nsrt.sample_option(state, goal, self._rng)

    def get_option_policy(self) -> Callable[[State], _Option]:

        def _option_policy(state: State) -> _Option:
            atoms = utils.abstract(state, self._predicates)
            option = self._bridge_policy(state, atoms, self._failed_options)
            logging.debug(f"Using option {option.name}{option.objects} "
                          "from bridge policy.")
            return option

        return _option_policy