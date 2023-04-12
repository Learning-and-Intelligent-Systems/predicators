"""Ground-truth NSRTs for the sandwich environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, LiftedAtom, ParameterizedOption, \
    Predicate, Type, Variable
from predicators.utils import null_sampler


class SandwichGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the sandwich environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"sandwich", "sandwich_clear"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        ingredient_type = types["ingredient"]
        board_type = types["board"]
        holder_type = types["holder"]

        # Predicates
        On = predicates["On"]
        OnBoard = predicates["OnBoard"]
        InHolder = predicates["InHolder"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]
        BoardClear = predicates["BoardClear"]

        # Options
        Pick = options["Pick"]
        Stack = options["Stack"]
        PutOnBoard = options["PutOnBoard"]

        nsrts = set()

        # PickFromHolder
        ing = Variable("?ing", ingredient_type)
        robot = Variable("?robot", robot_type)
        holder = Variable("?holder", holder_type)
        parameters = [ing, robot, holder]
        option_vars = [robot, ing]
        option = Pick
        preconditions = {
            LiftedAtom(InHolder, [ing, holder]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(Holding, [ing, robot])}
        delete_effects = {
            LiftedAtom(InHolder, [ing, holder]),
            LiftedAtom(GripperOpen, [robot])
        }

        pickfromholder_nsrt = NSRT("PickFromHolder", parameters,
                                   preconditions, add_effects, delete_effects,
                                   set(), option, option_vars, null_sampler)
        nsrts.add(pickfromholder_nsrt)

        # Stack
        ing = Variable("?ing", ingredient_type)
        othering = Variable("?othering", ingredient_type)
        robot = Variable("?robot", robot_type)
        parameters = [ing, othering, robot]
        option_vars = [robot, othering]
        option = Stack
        preconditions = {
            LiftedAtom(Holding, [ing, robot]),
            LiftedAtom(Clear, [othering])
        }
        add_effects = {
            LiftedAtom(On, [ing, othering]),
            LiftedAtom(Clear, [ing]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [ing, robot]),
            LiftedAtom(Clear, [othering])
        }

        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          null_sampler)
        nsrts.add(stack_nsrt)

        # PutOnBoard
        ing = Variable("?ing", ingredient_type)
        robot = Variable("?robot", robot_type)
        board = Variable("?board", board_type)
        parameters = [ing, robot, board]
        option_vars = [robot, board]
        option = PutOnBoard
        preconditions = {
            LiftedAtom(Holding, [ing, robot]),
            LiftedAtom(BoardClear, [board]),
        }
        add_effects = {
            LiftedAtom(OnBoard, [ing, board]),
            LiftedAtom(Clear, [ing]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [ing, robot]),
            LiftedAtom(BoardClear, [board]),
        }

        putonboard_nsrt = NSRT("PutOnBoard", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, null_sampler)
        nsrts.add(putonboard_nsrt)

        return nsrts
