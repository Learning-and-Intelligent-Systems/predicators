"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, DummyParameterizedOption, LiftedAtom, \
    ParameterizedOption, Predicate, Type, Variable
from predicators.utils import null_sampler


class PyBulletLaserGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the laser environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_laser"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        mirror_type = types["mirror"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]

        # Options
        Pick = options["PickMirror"]
        Place = options["Place"]

        nsrts = set()

        # PickMirror
        robot = Variable("?robot", robot_type)
        mirror = Variable("?mirror", mirror_type)
        parameters = [robot, mirror]
        option_vars = [robot, mirror]
        option = Pick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, mirror]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        pick_mirror_nsrt = NSRT("PickMirror", parameters,
                                   preconditions, add_effects, delete_effects,
                                   set(), option, option_vars, null_sampler)
        nsrts.add(pick_mirror_nsrt)

        # PlaceFirstMirror. Place first mirror to light and battery.
        robot = Variable("?robot", robot_type)
        parameters = [robot, mirror]
        option_vars = [robot]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [robot, mirror]),
            # Should add one that says the distance between the terminals are
            # close enough
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, mirror]),
        }
        connect_first_mirror_nsrt = NSRT("PlaceMirror", parameters,
                                            preconditions, add_effects,
                                            delete_effects, set(), option,
                                            option_vars, null_sampler)
        nsrts.add(connect_first_mirror_nsrt)


        return nsrts
