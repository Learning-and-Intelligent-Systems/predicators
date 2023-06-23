"""Ground-truth NSRTs for the cover environment."""

from typing import Dict, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, Sequence, State, Type, Variable


class KitchenGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the Kitchen environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        gripper_type = types["gripper"]
        object_type = types["obj"]

        # Objects
        gripper = Variable("?gripper", gripper_type)
        obj = Variable("?obj", object_type)

        # Predicates
        At = predicates["At"]

        # Options
        MoveTo = options["Move_delta_ee_pose"]

        nsrts = set()

        def moveto_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
            del rng, goal  # unused
            _, target = objs
            assert target.is_instance(object_type)
            tx = state.get(target, "x")
            ty = state.get(target, "y")
            tz = state.get(target, "z")
            return np.array([tx, ty, tz])

        # MoveTo
        # Player, from_loc, to_loc
        parameters = [gripper, obj]
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(At, [gripper, obj])}
        delete_effects: Set[LiftedAtom] = set()
        option = MoveTo
        option_vars = [gripper, obj]
        move_to_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            moveto_sampler)
        nsrts.add(move_to_nsrt)

        return nsrts
