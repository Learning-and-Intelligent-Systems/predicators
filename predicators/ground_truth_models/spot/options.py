"""Ground-truth options for PDDL environments."""

from typing import Dict, Sequence, Set

from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, STRIPSOperator, Type


class SpotEnvsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for PDDL environments."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"spot_grocery_env", "spot_bike_env"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # Note that these are 1:1 with the operators.
        env = get_or_create_env(env_name)
        assert isinstance(env, SpotEnv)
        options = {
            cls._strips_operator_to_parameterized_option(op, env)
            for op in env.strips_operators
        }
        return options

    @staticmethod
    def _strips_operator_to_parameterized_option(
            op: STRIPSOperator, env: SpotEnv) -> ParameterizedOption:

        def policy(s: State, m: Dict, o: Sequence[Object], p: Array) -> Action:
            del m  # unused
            return env.build_action(s, op, o, p)

        controller_name = env.operator_to_controller_name(op)
        params_space = env.controller_name_to_param_space(controller_name)
        types = [p.type for p in op.parameters]

        return utils.SingletonParameterizedOption(controller_name,
                                                  policy,
                                                  types,
                                                  params_space=params_space)
