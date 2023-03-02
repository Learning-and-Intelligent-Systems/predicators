"""Ground-truth NSRTs for the tools environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.tools import ToolsEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class ToolsGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the tools environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"tools"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        screw_type = types["screw"]
        screwdriver_type = types["screwdriver"]
        nail_type = types["nail"]
        hammer_type = types["hammer"]
        bolt_type = types["bolt"]
        wrench_type = types["wrench"]
        contraption_type = types["contraption"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        HoldingScrew = predicates["HoldingScrew"]
        HoldingScrewdriver = predicates["HoldingScrewdriver"]
        HoldingNail = predicates["HoldingNail"]
        HoldingHammer = predicates["HoldingHammer"]
        HoldingBolt = predicates["HoldingBolt"]
        HoldingWrench = predicates["HoldingWrench"]
        ScrewPlaced = predicates["ScrewPlaced"]
        NailPlaced = predicates["NailPlaced"]
        BoltPlaced = predicates["BoltPlaced"]
        ScrewFastened = predicates["ScrewFastened"]
        NailFastened = predicates["NailFastened"]
        BoltFastened = predicates["BoltFastened"]
        ScrewdriverGraspable = predicates["ScrewdriverGraspable"]
        HammerGraspable = predicates["HammerGraspable"]

        # Options
        PickScrew = options["PickScrew"]
        PickScrewdriver = options["PickScrewdriver"]
        PickNail = options["PickNail"]
        PickHammer = options["PickHammer"]
        PickBolt = options["PickBolt"]
        PickWrench = options["PickWrench"]
        Place = options["Place"]
        FastenScrewWithScrewdriver = options["FastenScrewWithScrewdriver"]
        FastenScrewByHand = options["FastenScrewByHand"]
        FastenNailWithHammer = options["FastenNailWithHammer"]
        FastenBoltWithWrench = options["FastenBoltWithWrench"]

        def placeback_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
            # Sampler for placing an item back in its initial spot.
            del goal, rng  # unused
            _, item = objs
            pose_x = state.get(item, "pose_x")
            pose_y = state.get(item, "pose_y")
            return np.array([pose_x, pose_y], dtype=np.float32)

        def placeoncontraption_sampler(state: State, goal: Set[GroundAtom],
                                       rng: np.random.Generator,
                                       objs: Sequence[Object]) -> Array:
            # Sampler for placing an item on a contraption.
            del goal  # unused
            _, _, contraption = objs
            pose_lx = state.get(contraption, "pose_lx")
            pose_ly = state.get(contraption, "pose_ly")
            pose_ux = pose_lx + ToolsEnv.contraption_size
            pose_uy = pose_ly + ToolsEnv.contraption_size
            # Note: Here we just use the average (plus noise), to make sampler
            # learning easier. We found that it's harder to learn to imitate the
            # more preferable sampler, which uses rng.uniform over the bounds.
            pose_x = pose_lx + (pose_ux - pose_lx) / 2.0 + rng.uniform() * 0.01
            pose_y = pose_ly + (pose_uy - pose_ly) / 2.0 + rng.uniform() * 0.01
            return np.array([pose_x, pose_y], dtype=np.float32)

        nsrts = set()

        # PickScrew
        robot = Variable("?robot", robot_type)
        screw = Variable("?screw", screw_type)
        parameters = [robot, screw]
        option_vars = [robot, screw]
        option = PickScrew
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {LiftedAtom(HoldingScrew, [screw])}
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        nsrts.add(
            NSRT("PickScrew", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars, null_sampler))

        # PickScrewdriver
        robot = Variable("?robot", robot_type)
        screwdriver = Variable("?screwdriver", screwdriver_type)
        parameters = [robot, screwdriver]
        option_vars = [robot, screwdriver]
        option = PickScrewdriver
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(ScrewdriverGraspable, [screwdriver])
        }
        add_effects = {LiftedAtom(HoldingScrewdriver, [screwdriver])}
        delete_effects = {LiftedAtom(HandEmpty, [robot])}
        nsrts.add(
            NSRT("PickScrewDriver", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars, null_sampler))

        # PickNail
        robot = Variable("?robot", robot_type)
        nail = Variable("?nail", nail_type)
        parameters = [robot, nail]
        option_vars = [robot, nail]
        option = PickNail
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {LiftedAtom(HoldingNail, [nail])}
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        nsrts.add(
            NSRT("PickNail", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars, null_sampler))

        # PickHammer
        robot = Variable("?robot", robot_type)
        hammer = Variable("?hammer", hammer_type)
        parameters = [robot, hammer]
        option_vars = [robot, hammer]
        option = PickHammer
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(HammerGraspable, [hammer])
        }
        add_effects = {LiftedAtom(HoldingHammer, [hammer])}
        delete_effects = {LiftedAtom(HandEmpty, [robot])}
        nsrts.add(
            NSRT("PickHammer", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars, null_sampler))

        # PickBolt
        robot = Variable("?robot", robot_type)
        bolt = Variable("?bolt", bolt_type)
        parameters = [robot, bolt]
        option_vars = [robot, bolt]
        option = PickBolt
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
        }
        add_effects = {LiftedAtom(HoldingBolt, [bolt])}
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
        }
        nsrts.add(
            NSRT("PickBolt", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars, null_sampler))

        # PickWrench
        robot = Variable("?robot", robot_type)
        wrench = Variable("?wrench", wrench_type)
        parameters = [robot, wrench]
        option_vars = [robot, wrench]
        option = PickWrench
        preconditions = {LiftedAtom(HandEmpty, [robot])}
        add_effects = {LiftedAtom(HoldingWrench, [wrench])}
        delete_effects = {LiftedAtom(HandEmpty, [robot])}
        nsrts.add(
            NSRT("PickWrench", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars, null_sampler))

        # PlaceScrewdriverBack
        robot = Variable("?robot", robot_type)
        screwdriver = Variable("?screwdriver", screwdriver_type)
        parameters = [robot, screwdriver]
        option_vars = [robot]
        option = Place
        preconditions = {
            LiftedAtom(HoldingScrewdriver, [screwdriver]),
            LiftedAtom(ScrewdriverGraspable, [screwdriver])
        }
        add_effects = {LiftedAtom(HandEmpty, [robot])}
        delete_effects = {LiftedAtom(HoldingScrewdriver, [screwdriver])}
        nsrts.add(
            NSRT("PlaceScrewdriverBack", parameters, preconditions,
                 add_effects, delete_effects, set(), option, option_vars,
                 placeback_sampler))

        # PlaceHammerBack
        robot = Variable("?robot", robot_type)
        hammer = Variable("?hammer", hammer_type)
        parameters = [robot, hammer]
        option_vars = [robot]
        option = Place
        preconditions = {
            LiftedAtom(HoldingHammer, [hammer]),
            LiftedAtom(HammerGraspable, [hammer])
        }
        add_effects = {LiftedAtom(HandEmpty, [robot])}
        delete_effects = {LiftedAtom(HoldingHammer, [hammer])}
        nsrts.add(
            NSRT("PlaceHammerBack", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars,
                 placeback_sampler))

        # PlaceWrenchBack
        robot = Variable("?robot", robot_type)
        wrench = Variable("?wrench", wrench_type)
        parameters = [robot, wrench]
        option_vars = [robot]
        option = Place
        preconditions = {LiftedAtom(HoldingWrench, [wrench])}
        add_effects = {LiftedAtom(HandEmpty, [robot])}
        delete_effects = {LiftedAtom(HoldingWrench, [wrench])}
        nsrts.add(
            NSRT("PlaceWrenchBack", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars,
                 placeback_sampler))

        # PlaceScrewOnContraption
        robot = Variable("?robot", robot_type)
        screw = Variable("?screw", screw_type)
        contraption = Variable("?contraption", contraption_type)
        parameters = [robot, screw, contraption]
        option_vars = [robot]
        option = Place
        preconditions = {LiftedAtom(HoldingScrew, [screw])}
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(ScrewPlaced, [screw, contraption])
        }
        delete_effects = {LiftedAtom(HoldingScrew, [screw])}
        nsrts.add(
            NSRT("PlaceScrewOnContraption", parameters, preconditions,
                 add_effects, delete_effects, set(), option, option_vars,
                 placeoncontraption_sampler))

        # PlaceNailOnContraption
        robot = Variable("?robot", robot_type)
        nail = Variable("?nail", nail_type)
        contraption = Variable("?contraption", contraption_type)
        parameters = [robot, nail, contraption]
        option_vars = [robot]
        option = Place
        preconditions = {LiftedAtom(HoldingNail, [nail])}
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(NailPlaced, [nail, contraption])
        }
        delete_effects = {LiftedAtom(HoldingNail, [nail])}
        nsrts.add(
            NSRT("PlaceNailOnContraption", parameters, preconditions,
                 add_effects, delete_effects, set(), option, option_vars,
                 placeoncontraption_sampler))

        # PlaceBoltOnContraption
        robot = Variable("?robot", robot_type)
        bolt = Variable("?bolt", bolt_type)
        contraption = Variable("?contraption", contraption_type)
        parameters = [robot, bolt, contraption]
        option_vars = [robot]
        option = Place
        preconditions = {LiftedAtom(HoldingBolt, [bolt])}
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(BoltPlaced, [bolt, contraption])
        }
        delete_effects = {LiftedAtom(HoldingBolt, [bolt])}
        nsrts.add(
            NSRT("PlaceBoltOnContraption", parameters, preconditions,
                 add_effects, delete_effects, set(), option, option_vars,
                 placeoncontraption_sampler))

        # FastenScrewWithScrewdriver
        robot = Variable("?robot", robot_type)
        screw = Variable("?screw", screw_type)
        screwdriver = Variable("?screwdriver", screwdriver_type)
        contraption = Variable("?contraption", contraption_type)
        parameters = [robot, screw, screwdriver, contraption]
        option_vars = [robot, screw, screwdriver, contraption]
        option = FastenScrewWithScrewdriver
        preconditions = {
            LiftedAtom(HoldingScrewdriver, [screwdriver]),
            LiftedAtom(ScrewPlaced, [screw, contraption])
        }
        add_effects = {LiftedAtom(ScrewFastened, [screw])}
        delete_effects = set()
        nsrts.add(
            NSRT("FastenScrewWithScrewdriver", parameters, preconditions,
                 add_effects, delete_effects, set(), option, option_vars,
                 null_sampler))

        # FastenScrewByHand
        robot = Variable("?robot", robot_type)
        screw = Variable("?screw", screw_type)
        contraption = Variable("?contraption", contraption_type)
        parameters = [robot, screw, contraption]
        option_vars = [robot, screw, contraption]
        option = FastenScrewByHand
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(ScrewPlaced, [screw, contraption])
        }
        add_effects = {LiftedAtom(ScrewFastened, [screw])}
        delete_effects = set()
        nsrts.add(
            NSRT("FastenScrewByHand", parameters, preconditions, add_effects,
                 delete_effects, set(), option, option_vars, null_sampler))

        # FastenNailWithHammer
        robot = Variable("?robot", robot_type)
        nail = Variable("?nail", nail_type)
        hammer = Variable("?hammer", hammer_type)
        contraption = Variable("?contraption", contraption_type)
        parameters = [robot, nail, hammer, contraption]
        option_vars = [robot, nail, hammer, contraption]
        option = FastenNailWithHammer
        preconditions = {
            LiftedAtom(HoldingHammer, [hammer]),
            LiftedAtom(NailPlaced, [nail, contraption])
        }
        add_effects = {LiftedAtom(NailFastened, [nail])}
        delete_effects = set()
        nsrts.add(
            NSRT("FastenNailWithHammer", parameters, preconditions,
                 add_effects, delete_effects, set(), option, option_vars,
                 null_sampler))

        # FastenBoltWithWrench
        robot = Variable("?robot", robot_type)
        bolt = Variable("?bolt", bolt_type)
        wrench = Variable("?wrench", wrench_type)
        contraption = Variable("?contraption", contraption_type)
        parameters = [robot, bolt, wrench, contraption]
        option_vars = [robot, bolt, wrench, contraption]
        option = FastenBoltWithWrench
        preconditions = {
            LiftedAtom(HoldingWrench, [wrench]),
            LiftedAtom(BoltPlaced, [bolt, contraption])
        }
        add_effects = {LiftedAtom(BoltFastened, [bolt])}
        delete_effects = set()
        nsrts.add(
            NSRT("FastenBoltWithWrench", parameters, preconditions,
                 add_effects, delete_effects, set(), option, option_vars,
                 null_sampler))

        return nsrts
