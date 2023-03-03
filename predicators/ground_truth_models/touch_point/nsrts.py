"""Ground-truth NSRTs for the touch point environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class TouchPointGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the touch point environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"touch_point", "touch_point_param"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        robot_type = types["robot"]
        target_type = types["target"]

        # Predicates
        Touched = predicates["Touched"]

        # Options
        MoveTo = options["MoveTo"]

        nsrts = set()

        # MoveTo
        robot = Variable("?robot", robot_type)
        target = Variable("?target", target_type)
        parameters = [robot, target]
        option_vars = [robot, target]
        option = MoveTo
        preconditions: Set[LiftedAtom] = set()
        add_effects = {LiftedAtom(Touched, [robot, target])}
        delete_effects: Set[LiftedAtom] = set()
        ignore_effects: Set[Predicate] = set()

        if CFG.env == "touch_point_param":

            def moveto_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
                del rng, goal  # unused
                robot, target, = objs
                assert robot.is_instance(robot_type)
                assert target.is_instance(target_type)
                rx = state.get(robot, "x")
                ry = state.get(robot, "y")
                tx = state.get(target, "x")
                ty = state.get(target, "y")
                dx = tx - rx
                dy = ty - ry
                return np.array([dx, dy], dtype=np.float32)

        elif CFG.env == "touch_point":
            moveto_sampler = null_sampler

        move_nsrt = NSRT("MoveTo", parameters, preconditions, add_effects,
                         delete_effects, ignore_effects, option, option_vars,
                         moveto_sampler)
        nsrts.add(move_nsrt)

        return nsrts
