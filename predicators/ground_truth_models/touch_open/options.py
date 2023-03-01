"""Ground-truth options for the touch open environment."""

from typing import Dict, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.envs.touch_point import TouchOpenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    ParameterizedPolicy, Predicate, State, Type


class TouchOpenGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the touch open environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"touch_open"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        robot_type = types["robot"]
        door_type = types["door"]
        TouchingDoor = predicates["TouchingDoor"]
        DoorIsOpen = predicates["DoorIsOpen"]

        # Note that the parameter spaces are designed to match what would be
        # learned by the neural option learners, in that they correspond to
        # only the dimensions that change when the option is run.

        # MoveToDoor
        def _MoveToDoor_terminal(state: State, memory: Dict,
                                 objects: Sequence[Object],
                                 params: Array) -> bool:
            del memory, params  # unused
            return TouchingDoor.holds(state, objects)

        MoveToDoor = ParameterizedOption(
            "MoveToDoor",
            types=[robot_type, door_type],
            params_space=Box(TouchOpenEnv.action_limits[0],
                             TouchOpenEnv.action_limits[1], (2, )),
            policy=cls._create_move_to_door_policy(),
            initiable=lambda s, m, o, p: True,
            terminal=_MoveToDoor_terminal)

        # OpenDoor
        def _OpenDoor_initiable(state: State, memory: Dict,
                                objects: Sequence[Object],
                                params: Array) -> bool:
            del memory, params  # unused
            # You can only open the door if touching it.
            door, robot = objects
            return TouchingDoor.holds(state, [robot, door])

        def _OpenDoor_terminal(state: State, memory: Dict,
                               objects: Sequence[Object],
                               params: Array) -> bool:
            del memory, params  # unused
            door, _ = objects
            return DoorIsOpen.holds(state, [door])

        OpenDoor = ParameterizedOption("OpenDoor",
                                       types=[door_type, robot_type],
                                       params_space=Box(
                                           -np.inf, np.inf, (2, )),
                                       policy=cls._create_open_door_policy(),
                                       initiable=_OpenDoor_initiable,
                                       terminal=_OpenDoor_terminal)

        return {MoveToDoor, OpenDoor}

    @classmethod
    def _create_move_to_door_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            dx, dy = params
            return Action(np.array([dx, dy, 0.0], dtype=np.float32))

        return policy

    @classmethod
    def _create_open_door_policy(cls) -> ParameterizedPolicy:

        def policy(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> Action:
            del state, memory, objects  # unused
            delta_rot, _ = params
            return Action(np.array([0.0, 0.0, delta_rot], dtype=np.float32))

        return policy
