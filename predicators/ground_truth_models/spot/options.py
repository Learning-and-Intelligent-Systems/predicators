"""Ground-truth options for PDDL environments."""

from typing import Dict, List, Sequence, Set, Tuple

from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, STRIPSOperator, Type


class _SpotEnvOption(utils.SingletonParameterizedOption):
    """An option defined by an operator in the spot environment.

    Defined in this way to avoid pickling anything bosdyn related. The
    key thing to note is that the args to __init__ are just strings, and
    the magic happens in __getnewargs__().
    """

    def __init__(self, operator_name: str, env_name: str) -> None:
        self._operator_name = operator_name
        self._env_name = env_name
        types = [p.type for p in self._get_operator().parameters]
        policy = self._policy_from_operator
        params_space = self._create_params_space()
        super().__init__(self._operator_name,
                         policy,
                         types,
                         params_space=params_space)

    def __getnewargs__(self) -> Tuple:
        return (self._operator_name, self._env_name)

    def _get_env(self) -> SpotEnv:
        env = get_or_create_env(self._env_name)
        assert isinstance(env, SpotEnv)
        return env

    def _get_operator(self) -> STRIPSOperator:
        matches = [
            op for op in self._get_env().strips_operators
            if op.name == self._operator_name
        ]
        assert len(matches) == 1
        return matches[0]

    def _policy_from_operator(self, s: State, m: Dict, o: Sequence[Object],
                              p: Array) -> Action:
        del s, m  # unused
        operator = self._get_operator()
        return self._get_env().build_action(operator, o, p)

    def _types_from_operator(self) -> List[Type]:
        return [p.type for p in self._get_operator().parameters]

    def _create_params_space(self) -> Box:
        env = self._get_env()
        operator = self._get_operator()
        controller_name = env.operator_to_controller_name(operator)
        return env.controller_name_to_param_space(controller_name)


class SpotEnvsGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for PDDL environments."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"spot_bike_env"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        # Note that these are 1:1 with the operators.
        env = get_or_create_env(env_name)
        assert isinstance(env, SpotEnv)
        options: Set[ParameterizedOption] = {
            _SpotEnvOption(op.name, env_name)
            for op in env.strips_operators
        }
        return options
