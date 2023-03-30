"""Learn a macro-based bridge policy."""

import logging
from typing import Callable, Dict, Set

import numpy as np

from predicators import utils
from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.settings import CFG
from predicators.structs import NSRT, Action, BridgeDataset, BridgePolicy, \
    DummyOption, GroundAtom, GroundMacro, Macro, Object, Predicate, State, \
    _GroundNSRT, _Option


class MacroLearningBridgePolicy(BaseBridgePolicy):
    """A macro-learning bridge policy."""

    def __init__(self, predicates: Set[Predicate], nsrts: Set[NSRT]) -> None:
        super().__init__(predicates, nsrts)
        self._macros: Dict[NSRT, Set[Macro]] = {}  # failed NSRT to macros

    @classmethod
    def get_name(cls) -> str:
        return "macro_learning"

    @property
    def is_learning_based(self) -> bool:
        return True

    def get_policy(self,
                   failed_nsrt: _GroundNSRT) -> Callable[[State], Action]:

        ground_macro = None
        cur_option = DummyOption

        def _policy(state: State) -> Action:
            nonlocal cur_option, ground_macro
            # First time calling, select a macro.
            if ground_macro is None:
                ground_macro = self._get_ground_macro(failed_nsrt, state)
            if cur_option is DummyOption or cur_option.terminal(state):
                # Macro is finished.
                if len(ground_macro) == 0:
                    raise BridgePolicyDone()
                # Pop the next ground NSRT.
                ground_nsrt, ground_macro = ground_macro.pop()
                logging.debug(
                    f"Using NSRT {ground_nsrt.name}{ground_nsrt.objects} "
                    "from bridge policy.")
                # Sample randomly, assuming goal not used by sampler.
                goal: Set[GroundAtom] = set()
                cur_option = ground_nsrt.sample_option(state, goal, self._rng)
                if not cur_option.initiable(state):
                    raise utils.OptionExecutionFailure(
                        "Bridge option not initiable.")
            act = cur_option.policy(state)
            return act

        return _policy

    def _get_ground_macro(self, failed_nsrt: _GroundNSRT,
                          state: State) -> GroundMacro:
        failed_nsrt_macros = self._macros.get(failed_nsrt.parent, set())
        objects = set(state)
        atoms = utils.abstract(state, self._predicates)
        # Sort arbitrarily for reproducibility.
        for macro in sorted(failed_nsrt_macros):
            types = [p.type for p in macro.parameters]
            for choice in utils.get_object_combinations(objects, types):
                ground_macro = macro.ground(choice)
                if ground_macro.preconditions.issubset(atoms):
                    return ground_macro

        raise utils.OptionExecutionFailure("No bridge macro found.")

    def learn_from_demos(self, dataset: BridgeDataset) -> None:
        """For learning-based approaches, learn whatever is needed from the
        given dataset."""
        for failed_ground_nsrt, ground_nsrts, _, _ in dataset:
            failed_nsrt = failed_ground_nsrt.parent
            macro = GroundMacro.from_ground_nsrts(ground_nsrts).parent
            if failed_nsrt not in self._macros:
                self._macros[failed_nsrt] = set()
            self._macros[failed_nsrt].add(macro)
