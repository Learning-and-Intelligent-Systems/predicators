"""Ground-truth NSRTs for the noisy button environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class NoisyButtonGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the touch point environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"noisy_button"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        button_type = types["button"]

        # Predicates
        Pressed = predicates["Pressed"]

        # Options
        Click = options["Click"]

        nsrts = set()

        # MoveTo
        button = Variable("?button", button_type)
        parameters = [button]
        option_vars = [button]
        option = Click
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(Pressed, [button])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects: Set[Predicate] = set()

        def moveto_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del goal  # unused
            # If the position is unknown, move randomly.
            button = objs[0]
            if state.get(button, "position_known") < 0.5:  # pragma: no cover
                lb = 0.0
                ub = 1.0
            # If the position is known, move nearby.
            else:
                center = state.get(button, "position")
                lb = center - 2e-2
                ub = center + 2e-2
            pos = rng.uniform(lb, ub)
            return np.array([pos, 0.0], dtype=np.float32)

        move_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         moveto_sampler)
        nsrts.add(move_nsrt)

        return nsrts
