"""Ground-truth NSRTs for the cover environment."""

from typing import Dict, List, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable, Sequence
from predicators.utils import null_sampler
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
import numpy as np

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
            gripper, target, = objs
            assert gripper.is_instance(gripper_type)
            assert target.is_instance(object_type)
            rx = state.get(gripper, "x")
            ry = state.get(gripper, "y")
            rz = state.get(gripper, "z")
            tx = state.get(target, "x")
            ty = state.get(target, "y")
            tz = state.get(target, "z")
            dx = tx - rx
            dy = ty - ry
            dz = tz - rz
            return np.array([dx, dy, dz], dtype=np.float32)

        # MoveTo
        # Player, from_loc, to_loc
        parameters = [gripper, obj]
        preconditions = set()
        add_effects = {LiftedAtom(At, [gripper, obj])}
        delete_effects = set()
        option = MoveTo
        option_vars: List[Variable] = []  # dummy - not used
        move_to_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            moveto_sampler)
        nsrts.add(move_to_nsrt)

        return nsrts
