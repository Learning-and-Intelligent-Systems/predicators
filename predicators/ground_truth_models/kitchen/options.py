"""Ground-truth options for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

try:
    from mujoco_kitchen.utils import \
        primitive_and_params_to_primitive_action, \
        primitive_name_to_action_idx
    _MJKITCHEN_IMPORTED = True
except ImportError:
    _MJKITCHEN_IMPORTED = False
from predicators import utils
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.ground_truth_models.kitchen.operators import \
    KitchenGroundTruthOperatorFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, ParameterizedTerminal, Predicate, State, \
    STRIPSOperator, Type


class KitchenGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the Kitchen environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"kitchen"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:
        assert _MJKITCHEN_IMPORTED
        # Operators
        operators = KitchenGroundTruthOperatorFactory.get_operators(
            env_name, types, predicates)

        # Reformat names for consistency with other option naming.
        def _format_name(name: str) -> str:
            return "".join([n.capitalize() for n in name.split(" ")])

        options: Set[ParameterizedOption] = set()
        for op in operators:
            assert "MoveTo" in op.name or "Push" in op.name
            val = "move_delta_ee_pose"
            if isinstance(primitive_name_to_action_idx[val], int):
                params_space = Box(-np.ones(1) * 5, np.ones(1) * 5)
            else:
                n = len(primitive_name_to_action_idx[val])
                params_space = Box(-np.ones(n) * 5, np.ones(n) * 5)
            obj_types = [param.type for param in op.parameters]
            options.add(
                utils.ParameterizedOption(
                    name=op.name.lower() + "_option",
                    types=obj_types,
                    params_space=params_space,
                    policy=cls._create_policy(name=val),
                    initiable=lambda _1, _2, _3, _4: True,
                    terminal=cls._create_terminal(name=val, operator=op)))
        return options

    @classmethod
    def _create_policy(cls, name: str) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del memory  # unused.
            if name.lower() == "move_delta_ee_pose":
                assert len(params) == 3
                if len(objects) == 2:
                    gripper, _ = objects
                else:
                    assert len(objects) == 3
                    gripper, _, _ = objects
                assert gripper.name == "gripper"
                gx = state.get(gripper, "x")
                gy = state.get(gripper, "y")
                gz = state.get(gripper, "z")
                dx = params[0] - gx
                dy = params[1] - gy
                dz = params[2] - gz
                primitive_params = np.array([dx, dy, dz],
                                            dtype=np.float32).clip(-0.1, 0.1)
            else:
                primitive_params = params
            arr = primitive_and_params_to_primitive_action(
                name, primitive_params)
            return Action(arr)

        return policy

    @classmethod
    def _create_terminal(cls, name: str,
                         operator: STRIPSOperator) -> ParameterizedTerminal:
        max_step_count = 25

        def terminal(state: State, memory: Dict, objects: Sequence[Object],
                     params: Array) -> bool:
            """Terminate when the option's corresponding operator's effects
            have been reached."""

            # NOTE: MoveTo terminates when the target pose is reached, unlike
            # the other options, which terminate when the effects are reached.
            if "step_count" in memory:
                step_count = memory["step_count"]
            else:
                step_count = 0

            if "MoveTo" in operator.name:
                if len(objects) == 2:
                    gripper, _ = objects
                else:
                    assert len(objects) == 3
                    gripper, _, _ = objects

                gx = state.get(gripper, "x")
                gy = state.get(gripper, "y")
                gz = state.get(gripper, "z")

                assert name.lower() == "move_delta_ee_pose"
                if "target_pose" not in memory:
                    memory["target_pose"] = np.array(
                        [params[0], params[1], params[2]])

                if np.allclose(np.array([gx, gy, gz]),
                               memory["target_pose"],
                               atol=0.2):
                    return True
            else:
                grounded_op = operator.ground(tuple(objects))
                if all(e.holds(state) for e in grounded_op.add_effects):
                    step_count = 0
                    return True
            if step_count > max_step_count:
                step_count = 0
                return True
            step_count += 1
            memory["step_count"] = step_count
            return False

        return terminal
