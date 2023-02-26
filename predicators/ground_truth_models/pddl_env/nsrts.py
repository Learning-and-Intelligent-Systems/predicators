"""Ground-truth NSRTs for the PDDLEnv."""

from typing import Dict, Set

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.pddl_env import _PDDLEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, ParameterizedOption, Predicate, Type
from predicators.utils import null_sampler


class PDDLEnvGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the PDDLEnv."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        pddl_env_classes = utils.get_all_subclasses(_PDDLEnv)
        return {
            c.get_name()
            for c in pddl_env_classes if not c.__abstractmethods__
        }

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        env = get_or_create_env(env_name)
        assert isinstance(env, _PDDLEnv)

        nsrts = set()

        for strips_op in env.strips_operators:
            option = options[strips_op.name]
            nsrt = strips_op.make_nsrt(
                option=option,
                option_vars=strips_op.parameters,
                sampler=null_sampler,
            )
            nsrts.add(nsrt)

        return nsrts
