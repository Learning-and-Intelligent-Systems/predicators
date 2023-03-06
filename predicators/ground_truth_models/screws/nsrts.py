"""Ground-truth NSRTs for the screws environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class ScrewsGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the screws environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"screws"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        screw_type = types["screw"]
        gripper_type = types["gripper"]
        receptacle_type = types["receptacle"]

        # Predicates
        GripperCanPickScrew = predicates["GripperCanPickScrew"]
        AboveReceptacle = predicates["AboveReceptacle"]
        HoldingScrew = predicates["HoldingScrew"]
        ScrewInReceptacle = predicates["ScrewInReceptacle"]

        # Options
        MoveToScrew = options["MoveToScrew"]
        MoveToReceptacle = options["MoveToReceptacle"]
        MagnetizeGripper = options["MagnetizeGripper"]
        DemagnetizeGripper = options["DemagnetizeGripper"]

        nsrts = set()

        # MoveToScrew
        robot = Variable("?robot", gripper_type)
        screw = Variable("?screw", screw_type)
        parameters = [robot, screw]
        option_vars = [robot, screw]
        option = MoveToScrew
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(GripperCanPickScrew, [robot, screw])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {GripperCanPickScrew}
        move_to_screw_nsrt = NSRT("MoveToScrew", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, null_sampler)
        nsrts.add(move_to_screw_nsrt)

        # MoveToReceptacle
        robot = Variable("?robot", gripper_type)
        receptacle = Variable("?receptacle", receptacle_type)
        screw = Variable("?screw", screw_type)
        parameters = [robot, receptacle, screw]
        option_vars = [robot, receptacle, screw]
        option = MoveToReceptacle
        preconditions = {LiftedAtom(HoldingScrew, [robot, screw])}
        add_effects = {LiftedAtom(AboveReceptacle, [robot, receptacle])}
        ignore_effects = {GripperCanPickScrew}
        move_to_receptacle_nsrt = NSRT("MoveToReceptacle", parameters,
                                       preconditions, add_effects,
                                       delete_effects, ignore_effects, option,
                                       option_vars, null_sampler)
        nsrts.add(move_to_receptacle_nsrt)

        # MagnetizeGripper
        robot = Variable("?robot", gripper_type)
        screw = Variable("?screw", screw_type)
        parameters = [robot, screw]
        option_vars = [robot]
        option = MagnetizeGripper
        preconditions = {LiftedAtom(GripperCanPickScrew, [robot, screw])}
        add_effects = {LiftedAtom(HoldingScrew, [robot, screw])}
        ignore_effects = {HoldingScrew}
        magnetize_gripper_nsrt = NSRT("MagnetizeGripper", parameters,
                                      preconditions, add_effects,
                                      delete_effects, ignore_effects, option,
                                      option_vars, null_sampler)
        nsrts.add(magnetize_gripper_nsrt)

        # DemagnetizeGripper
        robot = Variable("?robot", gripper_type)
        screw = Variable("?screw", screw_type)
        receptacle = Variable("?receptacle", receptacle_type)
        parameters = [robot, screw, receptacle]
        option_vars = [robot]
        option = DemagnetizeGripper
        preconditions = {
            LiftedAtom(HoldingScrew, [robot, screw]),
            LiftedAtom(AboveReceptacle, [robot, receptacle])
        }
        add_effects = {LiftedAtom(ScrewInReceptacle, [screw, receptacle])}
        delete_effects = {LiftedAtom(HoldingScrew, [robot, screw])}
        ignore_effects = {HoldingScrew}
        demagnetize_gripper_nsrt = NSRT("DemagnetizeGripper", parameters,
                                        preconditions, add_effects,
                                        delete_effects, ignore_effects, option,
                                        option_vars, null_sampler)
        nsrts.add(demagnetize_gripper_nsrt)

        return nsrts
