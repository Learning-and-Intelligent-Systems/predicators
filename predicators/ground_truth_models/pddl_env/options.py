"""Ground-truth options for PDDL environments."""

from typing import Dict, List, Sequence, Set
from typing import Type as TypingType

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs.pddl_env import _parse_pddl_domain, _PDDLEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    Predicate, State, STRIPSOperator, Type


class PDDLEnvGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for PDDL environments."""

    @classmethod
    def _get_env_name_to_env_class(cls) -> Dict[str, TypingType[_PDDLEnv]]:
        pddl_env_classes = utils.get_all_subclasses(_PDDLEnv)
        return {
            c.get_name(): c
            for c in pddl_env_classes if not c.__abstractmethods__
        }

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return set(cls._get_env_name_to_env_class())

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        env_cls = cls._get_env_name_to_env_class()[env_name]
        domain_str = env_cls.get_domain_str()
        _, _, strips_operators = _parse_pddl_domain(domain_str)
        ordered_strips_operators = sorted(strips_operators)
        return {
            cls._strips_operator_to_parameterized_option(
                op, ordered_strips_operators, action_space.shape[0])
            for op in strips_operators
        }

    @staticmethod
    def _strips_operator_to_parameterized_option(
            op: STRIPSOperator, ordered_operators: List[STRIPSOperator],
            action_dims: int) -> ParameterizedOption:
        name = op.name
        types = [p.type for p in op.parameters]
        op_idx = ordered_operators.index(op)

        def policy(s: State, m: Dict, o: Sequence[Object], p: Array) -> Action:
            del m, p  # unused
            ordered_objs = list(s)
            # The first dimension of an action encodes the operator.
            # The second dimension of an action encodes the first object
            # argument to the ground operator. The third dimension encodes the
            # second object and so on. Actions are always padded so that their
            # length is equal to the max number of arguments for any operator.
            obj_idxs = [ordered_objs.index(obj) for obj in o]
            act_arr = np.zeros(action_dims, dtype=np.float32)
            act_arr[0] = op_idx
            act_arr[1:(len(obj_idxs) + 1)] = obj_idxs
            return Action(act_arr)

        # Note: the initiable is deliberately always True. This only makes a
        # difference for exploration. If the initiable took into account the
        # ground-truth preconditions, that would make exploration too easy,
        # because the options would only ever get used in states where their
        # preconditions hold. Instead, with always-True initiable, there is a
        # difficult exploration problem because most options will have trivial
        # effects on the environment.
        return utils.SingletonParameterizedOption(name, policy, types)
