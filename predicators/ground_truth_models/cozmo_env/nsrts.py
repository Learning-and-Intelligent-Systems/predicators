"""Ground-truth NSRTs for the touch point environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class CozmoGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the touch point environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cozmo"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        cube_type = types["cube"]

        # Predicates
        NextTo = predicates["NextTo"]

        # Options
        MoveTo = options["MoveTo"]

        nsrts = set()

        # MoveTo
        robot = Variable("?robot", robot_type)
        target = Variable("?target", cube_type)
        parameters = [robot, target]
        option_vars = [robot, target]
        option = MoveTo
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(NextTo, [robot, target])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects: Set[Predicate] = set()

        moveto_sampler = null_sampler

        move_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         moveto_sampler)
        nsrts.add(move_nsrt)

        return nsrts
