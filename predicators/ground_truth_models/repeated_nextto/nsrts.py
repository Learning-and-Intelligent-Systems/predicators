"""Ground-truth NSRTs for the repeated nextto environment."""

from typing import Dict, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, DummyOption, LiftedAtom, \
    ParameterizedOption, Predicate, Type, Variable
from predicators.utils import null_sampler


class RepeatedNextToGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the repeated nextto environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "repeated_nextto", "repeated_nextto_ambiguous",
            "repeated_nextto_simple"
        }

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        dot_type = types["dot"]

        # Predicates
        NextTo = predicates["NextTo"]
        NextToNothing = predicates["NextToNothing"]
        Grasped = predicates["Grasped"]

        # Options
        Move = options["Move"]
        Grasp = options["Grasp"]

        nsrts = set()

        # Move
        robot = Variable("?robot", robot_type)
        targetdot = Variable("?targetdot", dot_type)
        parameters = [robot, targetdot]
        option_vars = [robot, targetdot]
        option = Move
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(NextTo, [robot, targetdot])}
        delete_effects: Set[LiftedAtom] = set()
        # Moving could have us end up NextTo other objects. It could also
        # include NextToNothing as a delete effect.
        ignore_effects = {NextTo, NextToNothing}
        move_nsrt = NSRT("Move", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         lambda s, g, rng, o: np.zeros(1, dtype=np.float32))
        nsrts.add(move_nsrt)

        # Grasp
        robot = Variable("?robot", robot_type)
        targetdot = Variable("?targetdot", dot_type)
        parameters = [robot, targetdot]
        option_vars = [robot, targetdot]
        option = Grasp
        preconditions = {LiftedAtom(NextTo, [robot, targetdot])}
        add_effects = {LiftedAtom(Grasped, [robot, targetdot])}
        delete_effects = {LiftedAtom(NextTo, [robot, targetdot])}
        # After grasping, it's possible that you could end up NextToNothing,
        # but it's also possible that you remain next to something else.
        # Note that NextTo isn't an ignore effect here because it's not
        # something we'd be unsure about for any object. For every object we
        # are NextTo but did not grasp, we will stay NextTo it.
        ignore_effects = {NextToNothing}
        grasp_nsrt = NSRT("Grasp", parameters, preconditions, add_effects,
                          delete_effects, ignore_effects, option, option_vars,
                          null_sampler)
        nsrts.add(grasp_nsrt)

        return nsrts


class RNTSingleOptGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for RepeatedNextToSingleOptionEnv."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"repeated_nextto_single_option"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Use dummy options that we will swap out below.
        parent_options = {
            "Move": DummyOption.parent,
            "Grasp": DummyOption.parent
        }
        parent_nsrts = RepeatedNextToGroundTruthNSRTFactory.get_nsrts(
            "repeated_nextto", types, predicates, parent_options)
        rn_grasp_nsrt, rn_move_nsrt = sorted(parent_nsrts)
        assert rn_grasp_nsrt.name == "Grasp"
        assert rn_move_nsrt.name == "Move"

        MoveGrasp = options["MoveGrasp"]

        nsrts = set()

        # Move
        move_nsrt = NSRT(
            rn_move_nsrt.name, rn_move_nsrt.parameters,
            rn_move_nsrt.preconditions, rn_move_nsrt.add_effects,
            rn_move_nsrt.delete_effects, rn_move_nsrt.ignore_effects,
            MoveGrasp, rn_move_nsrt.option_vars,
            lambda s, g, rng, o: np.array([-1.0, 0.0], dtype=np.float32))
        nsrts.add(move_nsrt)

        # Grasp
        grasp_nsrt = NSRT(
            rn_grasp_nsrt.name, rn_grasp_nsrt.parameters,
            rn_grasp_nsrt.preconditions, rn_grasp_nsrt.add_effects,
            rn_grasp_nsrt.delete_effects, rn_grasp_nsrt.ignore_effects,
            MoveGrasp, rn_grasp_nsrt.option_vars,
            lambda s, g, rng, o: np.array([1.0, 0.0], dtype=np.float32))
        nsrts.add(grasp_nsrt)

        return nsrts
