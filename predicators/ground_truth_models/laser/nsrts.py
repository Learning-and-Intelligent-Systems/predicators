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
        target_type = types["target"]
        station_type = types["station"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        TargetHit = predicates["TargetHit"]
        IsSplitMirror = predicates["IsSplitMirror"]
        SwitchedOn = predicates["StationOn"]

        # Options
        Pick = options["PickMirror"]
        Place = options["Place"]
        SwitchOn = options["SwitchOn"]

        nsrts = set()

        # PickMirror
        robot = Variable("?robot", robot_type)
        mirror = Variable("?mirror", mirror_type)
        parameters = [robot, mirror]
        option_vars = [robot, mirror]
        option = Pick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(IsSplitMirror, [mirror]),
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

        # PlaceFirstMirror. Place first mirror to light and station.
        robot = Variable("?robot", robot_type)
        mirror = Variable("?mirror", mirror_type)
        target1 = Variable("?target1", target_type)
        target2 = Variable("?target2", target_type)
        parameters = [robot, mirror, target1, target2]
        option_vars = [robot]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [robot, mirror]),
            # Should add one that says the distance between the terminals are
            # close enough
        }
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(TargetHit, [target1]),
            LiftedAtom(TargetHit, [target2]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, mirror]),
        }
        connect_first_mirror_nsrt = NSRT("PlaceMirror", parameters,
                                         preconditions, add_effects,
                                         delete_effects, set(), option,
                                         option_vars, null_sampler)
        nsrts.add(connect_first_mirror_nsrt)

        # SwitchOn
        robot = Variable("?robot", robot_type)
        station = Variable("?station", station_type)
        parameters = [robot, station]
        option_vars = [robot, station]
        option = SwitchOn
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {
            LiftedAtom(SwitchedOn, [station]),
        }
        delete_effects = set()
        switch_on_nsrt = NSRT("SwitchOn", parameters, preconditions,
                              add_effects, delete_effects, set(), option,
                              option_vars, null_sampler)
        nsrts.add(switch_on_nsrt)

        return nsrts
