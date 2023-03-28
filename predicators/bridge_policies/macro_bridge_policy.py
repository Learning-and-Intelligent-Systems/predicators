"""Learn a macro-based bridge policy."""

import logging
from typing import Callable, Dict, Set

import numpy as np

from predicators import utils
from predicators.bridge_policies import BaseBridgePolicy, BridgePolicyDone
from predicators.settings import CFG
from predicators.structs import NSRT, Action, BridgePolicy, DummyOption, \
    GroundAtom, Predicate, State, _GroundNSRT, _Option, GroundMacro, NSRT, _Macro, Object


class MacroLearningBridgePolicy(BaseBridgePolicy):
    """A macro-learning bridge policy."""

    def __init__(self, predicates: Set[Predicate], nsrts: Set[NSRT]) -> None:
        super().__init__(predicates, nsrts)
        self._macros: Dict[NSRT, Set[_Macro]] = {}  # failed NSRT to macros

        # TODO: remove and learn instead!
        nsrt_name_to_nsrt = {n.name: n for n in nsrts}
        PlaceInBox = nsrt_name_to_nsrt["PlaceInBox"]
        PlaceOnTable = nsrt_name_to_nsrt["PlaceOnTable"]
        OpenLid = nsrt_name_to_nsrt["OpenLid"]
        obj_var, robot_var = PlaceOnTable.parameters
        lid_var, _ = OpenLid.parameters
        held_obj = Object("held_obj", obj_var.type)
        robot = Object("robot", robot_var.type)
        lid = Object("lid", lid_var.type)
        oracle_ground_macro = GroundMacro([
            PlaceOnTable.ground([held_obj, robot]),
            OpenLid.ground([lid, robot]),
        ])
        oracle_macro = oracle_ground_macro.parent
        self._macros[PlaceInBox] = {oracle_macro}

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
                # Sample randomly, assuming goal not used by sampler.
                goal: Set[GroundAtom] = set()
                cur_option = ground_nsrt.sample_option(state, goal, self._rng)
                if not cur_option.initiable(state):
                    raise utils.OptionExecutionFailure(
                        "Bridge option not initiable.")
            act = cur_option.policy(state)
            return act

        return _policy

    def _get_ground_macro(self, failed_nsrt: _GroundNSRT, state: State) -> GroundMacro:
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

