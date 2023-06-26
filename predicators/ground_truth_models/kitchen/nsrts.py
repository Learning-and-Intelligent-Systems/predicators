"""Ground-truth NSRTs for the cover environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


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
                            null_sampler)
        nsrts.add(move_to_nsrt)

        return nsrts
