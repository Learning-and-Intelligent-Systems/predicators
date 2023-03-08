"""Ground-truth NSRTs for the pybullet shelf environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class PyBulletShelfGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the pybullet shelf environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_shelf"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        block_type = types["block"]
        robot_type = types["robot"]
        shelf_type = types["shelf"]

        # Predicates
        OnTable = predicates["OnTable"]
        InShelf = predicates["InShelf"]

        # Options
        PickPlace = options["PickPlace"]

        nsrts = set()

        robot = Variable("?robot", robot_type)
        block = Variable("?block", block_type)
        shelf = Variable("?shelf", shelf_type)
        parameters = [robot, block, shelf]
        option_vars = [robot, block]
        option = PickPlace
        preconditions = {
            LiftedAtom(OnTable, [block])
        }
        add_effects = {
            LiftedAtom(InShelf, [block, shelf])
        }
        delete_effects = {
            LiftedAtom(OnTable, [block])
        }
        ignore_effects = set()
        pick_place_nsrt = NSRT("PickPlace", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, null_sampler)
        nsrts.add(pick_place_nsrt)

        return nsrts
