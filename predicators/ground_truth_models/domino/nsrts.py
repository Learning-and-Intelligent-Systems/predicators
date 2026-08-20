"""Ground-truth NSRTs for the domino environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class PyBulletDominoGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the domino environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino_grid"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        domino_type = types["domino"]
        position_type = types["loc"]
        rotation_type = types["angle"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        _ = predicates["InFront"]
        Upright = predicates["Upright"]
        StartBlock = predicates["InitialBlock"]
        _ = predicates["Toppled"]
        Tilting = predicates["Tilting"]
        DominoAtPos = predicates["DominoAtPos"]
        DominoAtRot = predicates["DominoAtRot"]
        MovableBlock = predicates["MovableBlock"]
        PosClear = predicates["PosClear"]
        if CFG.domino_include_connected_predicate:
            Connected = predicates["Connected"]
        else:
            AdjacentTo = predicates["AdjacentTo"]
        if CFG.domino_has_glued_dominos:
            _ = predicates["DominoNotGlued"]

        # Options
        Pick = options["Pick"]
        Place = options["Place"]
        Wait = options["Wait"]

        nsrts = set()

        # PushStartBlock: Push the start block to initiate the domino chain
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        parameters = [robot, domino]
        if CFG.domino_restricted_push:
            option = options["PushRestricted"]
            option_vars = [robot]
        else:
            option = options["Push"]
            option_vars = [robot, domino]
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(StartBlock, [domino]),
            LiftedAtom(Upright, [domino]),
        }
        add_effects = {
            LiftedAtom(Tilting, [domino]),
        }
        delete_effects = {
            LiftedAtom(Upright, [domino]),
        }
        push_start_block_nsrt = NSRT("PushStartBlock", parameters,
                                     preconditions, add_effects,
                                     delete_effects, set(), option,
                                     option_vars, null_sampler)
        nsrts.add(push_start_block_nsrt)

        # PickDomino: Position-based pick process
        robot = Variable("?robot", robot_type)
        domino = Variable("?domino", domino_type)
        position = Variable("?pos", position_type)
        rotation = Variable("?rot", rotation_type)
        parameters = [robot, domino, position, rotation]
        option_vars = [robot, domino]
        option = Pick
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino, position]),
            LiftedAtom(DominoAtRot, [domino, rotation]),
            LiftedAtom(MovableBlock, [domino]),
            LiftedAtom(Upright, [domino]),
        }
        add_effects = {
            LiftedAtom(Holding, [robot, domino]),
            LiftedAtom(PosClear, [position]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino, position]),
            LiftedAtom(DominoAtRot, [domino, rotation]),
        }
        pick_domino_nsrt = NSRT("PickDomino", parameters,
                                preconditions, add_effects, delete_effects,
                                set(), option, option_vars, null_sampler)
        nsrts.add(pick_domino_nsrt)

        # PlaceDomino: Place domino at specific position and rotation
        robot = Variable("?robot", robot_type)
        domino1 = Variable("?domino1", domino_type)
        domino2 = Variable("?domino2", domino_type)
        target_pos = Variable("?pos1", position_type)
        rotation = Variable("?rot", rotation_type)
        if CFG.domino_include_connected_predicate:
            d2_pos = Variable("?pos2", position_type)
            parameters = [
                robot, domino1, domino2, target_pos, d2_pos, rotation
            ]
        else:
            parameters = [robot, domino1, domino2, target_pos, rotation]
        option_vars = [robot, domino1, domino2, target_pos, rotation]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [robot, domino1]),
            LiftedAtom(PosClear, [target_pos]),
            LiftedAtom(Upright, [domino2]),
        }
        if CFG.domino_include_connected_predicate:
            preconditions.update({
                LiftedAtom(DominoAtPos, [domino2, d2_pos]),
                LiftedAtom(Connected, [target_pos, d2_pos]),
            })
        else:
            preconditions.update({
                LiftedAtom(AdjacentTo, [target_pos, domino2]),
            })
        add_effects = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(DominoAtPos, [domino1, target_pos]),
            LiftedAtom(DominoAtRot, [domino1, rotation]),
        }
        delete_effects = {
            LiftedAtom(Holding, [robot, domino1]),
            LiftedAtom(PosClear, [target_pos]),
        }
        place_domino_nsrt = NSRT("PlaceDomino", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars, null_sampler)
        nsrts.add(place_domino_nsrt)

        # Wait
        robot = Variable("?robot", robot_type)
        parameters = [robot]
        option_vars = [robot]
        option = Wait
        preconditions = set()
        add_effects = set()
        delete_effects = set()
        wait_nsrt = NSRT("Wait", parameters, preconditions, add_effects,
                         delete_effects, set(), option, option_vars,
                         null_sampler)
        nsrts.add(wait_nsrt)

        return nsrts
