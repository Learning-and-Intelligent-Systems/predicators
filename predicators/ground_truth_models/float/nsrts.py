"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, DummyParameterizedOption, LiftedAtom, \
    ParameterizedOption, Predicate, Type, Variable
from predicators.utils import null_sampler


class PyBulletFloatGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the float environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_float"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["block"]
        robot_type = types["robot"]
        vessel_type = types["vessel"]

        # Predicates
        InWater = predicates["InWater"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]

        # Options
        Pick = options["PickBlock"]
        Drop = options["Drop"]

        nsrts = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, block]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(HandEmpty, [robot])
        }
        add_effects = {
            LiftedAtom(Holding, [robot, block])
            }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot])
        }

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars, null_sampler)
        nsrts.add(pickfromtable_nsrt)

        # DropInWater
        robot = Variable("?robot", robot_type)
        block = Variable("?block", block_type)
        vessel = Variable("?vessel", vessel_type)
        parameters = [robot, vessel, block]
        option_vars = [robot, vessel]
        option = Drop
        preconditions = {
            LiftedAtom(Holding, [robot, block]),
        }
        add_effects = {
            LiftedAtom(InWater, [block]),
            LiftedAtom(HandEmpty, [robot])
            }
        delete_effects = {
            LiftedAtom(Holding, [robot, block]),
        }
        drop_in_water_nsrt = NSRT("DropInWater", parameters,
                                    preconditions, add_effects, delete_effects,
                                    set(), option, option_vars, null_sampler)
        nsrts.add(drop_in_water_nsrt)

        return nsrts
