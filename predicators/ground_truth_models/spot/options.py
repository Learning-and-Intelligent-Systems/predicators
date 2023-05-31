"""Ground-truth options for PDDL environments."""

import logging
from typing import Dict, List, Sequence, Set

import bosdyn
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import get_or_create_env
from predicators.envs.spot_env import SpotEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.spot_utils.spot_utils import get_spot_controllers
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

        env = get_or_create_env(env_name)
        assert isinstance(env, SpotEnv)
        ordered_strips_operators = sorted(env.strips_operators)

        # Options (aka Controllers)
        # Note that these are 1:1 with the operators; in the future,
        # we will actually implement these with robot-specific API calls.
        options = {
            cls._strips_operator_to_parameterized_option(
                op, ordered_strips_operators, action_space.shape[0],
                list(predicates.values()))
            for op in ordered_strips_operators
        }
        return options

    @staticmethod
    def _strips_operator_to_parameterized_option(
            op: STRIPSOperator, ordered_operators: List[STRIPSOperator],
            action_dims: int,
            predicates: Sequence[Predicate]) -> ParameterizedOption:
        name = op.name
        types = [p.type for p in op.parameters]
        op_idx = ordered_operators.index(op)

        def policy(s: State, m: Dict, o: Sequence[Object], p: Array) -> Action:
            nonlocal name
            nonlocal predicates

            try:  # pragma: no cover
                spot_controllers = get_spot_controllers()
                if 'MoveToBag' in name:
                    spot_controllers.navigateToController(
                        utils.abstract(s, predicates), o, list(p))
                elif 'MoveTo' in name:
                    spot_controllers.navigateToController(
                        m['start_state'].simulator_state, o, list(p))
                elif 'Grasp' in name:
                    spot_controllers.graspController(o, list(p))
                elif 'Place' in name:
                    spot_controllers.placeOntopController(o, list(p))
                else:
                    raise NotImplementedError(
                        f"Spot controller not implemented for operator {name}")
            except (bosdyn.client.exceptions.ProxyConnectionError,
                    bosdyn.client.exceptions.UnableToConnectToRobotError,
                    RuntimeError):
                logging.info("Could not connect to Spot!")

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

        params_space = Box(0, 1, (0, ))
        if "MoveTo" in op.name:
            params_space = Box(-5.0, 5.0, (3, ))
        elif "Grasp" in op.name:
            params_space = Box(-1.0, 1.0, (4, ))
        elif "Place" in op.name:
            params_space = Box(-5.0, 5.0, (3, ))

        # Note: the initiable is deliberately always True. This only makes a
        # difference for exploration. If the initiable took into account the
        # ground-truth preconditions, that would make exploration too easy,
        # because the options would only ever get used in states where their
        # preconditions hold. Instead, with always-True initiable, there is a
        # difficult exploration problem because most options will have trivial
        # effects on the environment.
        return utils.SingletonParameterizedOption(name,
                                                  policy,
                                                  types,
                                                  params_space=params_space)
