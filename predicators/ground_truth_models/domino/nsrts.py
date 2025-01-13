"""Ground-truth NSRTs for the coffee environment."""

from typing import Dict, Set

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, DummyParameterizedOption, LiftedAtom, \
    ParameterizedOption, Predicate, Type, Variable
from predicators.utils import null_sampler


class PyBulletDominoGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the domino environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"pybullet_domino"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["domino"]
        robot_type = types["robot"]
        target_type = types["target"]

        # Predicates
        StartBlock = predicates["StartBlock"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        Toppled = predicates["Toppled"]

        # Options
        Push = options["Push"]

        nsrts = set()

        # Push
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        target = Variable("?target", target_type)
        parameters = [robot, block, target]
        option_vars = [robot, block]
        option = Push
        preconditions = {
            LiftedAtom(HandEmpty, [robot]),
            LiftedAtom(StartBlock, [block])
        }
        add_effects = {LiftedAtom(Toppled, [target])}
        delete_effects = {}

        push_nsrt = NSRT("Push", parameters, preconditions, add_effects,
                         delete_effects, set(), option, option_vars,
                         null_sampler)
        nsrts.add(push_nsrt)

        return nsrts
