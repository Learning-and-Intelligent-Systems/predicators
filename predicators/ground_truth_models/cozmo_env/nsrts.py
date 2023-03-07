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
        dock_type = types["dock"]

        # Predicates
        Reachable = predicates["Reachable"]
        NextTo = predicates["NextTo"]
        Touched = predicates["Touched"]
        IsRed = predicates["IsRed"]
        IsBlue = predicates["IsBlue"]
        IsGreen = predicates["IsGreen"]
        OnTop = predicates["OnTop"]
        Under = predicates["Under"]

        # Options
        MoveTo = options["MoveTo"]
        Touch = options["Touch"]
        Paint = options["Paint"]
        PlaceOntop = options["PlaceOntop"]
        # Roll = options["Roll"]

        nsrts = set()

        # MoveTo(?robot, ?obj)
        robot = Variable("?robot", robot_type)
        target = Variable("?target", cube_type)
        parameters = [robot, target]
        option_vars = [robot, target]
        option = MoveTo
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(Reachable, [robot, target])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {Reachable, NextTo}

        moveto_sampler = null_sampler

        move_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         moveto_sampler)
        nsrts.add(move_nsrt)

        ### Touch(?obj)
        target = Variable("?target", cube_type)
        parameters = [target]
        option_vars = [target]
        option = Touch
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(Touched, [target])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {Reachable, NextTo}

        touch_sampler = null_sampler

        touch_nsrt = NSRT("Touch", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         touch_sampler)
        nsrts.add(touch_nsrt)

        ### Paint(?obj, [color])
        robot = Variable("?robot", robot_type)
        target = Variable("?target", cube_type)
        parameters = [robot, target]
        option_vars = [robot, target]
        option = Paint
        preconditions = {LiftedAtom(Reachable, [robot, target])}
        add_effects = {LiftedAtom(IsRed, [target])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {Reachable, NextTo}

        paint_red_sampler = (lambda state, goal, rng, objs: 1)

        paint_red_nsrt = NSRT("PaintRed", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         paint_red_sampler)
        nsrts.add(paint_red_nsrt)

        # Blue
        add_effects = {LiftedAtom(IsBlue, [target])}
        paint_blue_sampler = (lambda state, goal, rng, objs: 2)

        paint_blue_nsrt = NSRT("PaintBlue", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         paint_blue_sampler)
        nsrts.add(paint_blue_nsrt)

        # Green
        add_effects = {LiftedAtom(IsGreen, [target])}
        paint_green_sampler = (lambda state, goal, rng, objs: 3)

        paint_green_nsrt = NSRT("PaintGreen", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         paint_green_sampler)
        nsrts.add(paint_green_nsrt)

        ### PlaceOntop(?obj1, ?obj2)
        obj = Variable("?obj", cube_type)
        target = Variable("?target", cube_type)
        parameters = [obj, target]
        option_vars = [obj, target]
        option = PlaceOntop
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(OnTop, [obj, target]), LiftedAtom(Under, [target, obj])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects = {Reachable, NextTo}

        place_sampler = null_sampler

        place_nsrt = NSRT("PlaceOntop", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         place_sampler)
        nsrts.add(place_nsrt)

        return nsrts
