"""A perceiver specific to the spot bike env."""

import logging
from typing import Dict, Optional, Set, Tuple

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.envs.spot_env import SpotBikeEnv, _PartialPerceptionState, \
    _SpotObservation
from predicators.perception.base_perceiver import BasePerceiver
from predicators.settings import CFG
from predicators.spot_utils.spot_utils import obj_name_to_apriltag_id
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Observation, Predicate, State, Task


class SpotBikePerceiver(BasePerceiver):
    """A perceiver specific to the spot bike env."""

    def __init__(self) -> None:
        super().__init__()
        self._known_object_poses: Dict[Object, Tuple[float, float, float]] = {}
        self._robot: Optional[Object] = None
        self._nonpercept_atoms: Set[GroundAtom] = set()
        self._nonpercept_predicates: Set[Predicate] = set()
        self._holding_item_id_feature = 0.0
        self._gripper_open_percentage = 0.0
        assert CFG.env == "spot_bike_env"
        self._curr_env: Optional[BaseEnv] = None

    @classmethod
    def get_name(cls) -> str:
        return "spot_bike_env"

    def reset(self, env_task: EnvironmentTask) -> Task:
        self._update_state_from_observation(env_task.init_obs)
        self._curr_env = get_or_create_env("spot_bike_env")
        assert isinstance(self._curr_env, SpotBikeEnv)
        init_state = self._create_state()
        return Task(init_state, env_task.goal)

    def update_perceiver_with_action(self, action: Action) -> None:
        # Update the curr held item when applicable.
        assert self._curr_env is not None and isinstance(
            self._curr_env, SpotBikeEnv)
        controller_name, objects, _ = self._curr_env.parse_action(action)
        # The robot is always the 0th argument of an
        # operator!
        if "grasp" in controller_name.lower():
            assert self._holding_item_id_feature == 0.0
            # We know that the object that we attempted to grasp was
            # the second argument to the controller.
            object_attempted_to_grasp = objects[1].name
            grasp_obj_id = obj_name_to_apriltag_id[object_attempted_to_grasp]
            self._holding_item_id_feature = grasp_obj_id
        elif "place" in controller_name.lower():
            self._holding_item_id_feature = 0.0

    def step(self, observation: Observation) -> State:
        self._update_state_from_observation(observation)
        return self._create_state()

    def _update_state_from_observation(self, observation: Observation) -> None:
        assert isinstance(observation, _SpotObservation)
        self._robot = observation.robot
        self._known_object_poses.update(observation.objects_in_view)
        self._nonpercept_atoms = observation.nonpercept_atoms
        self._nonpercept_predicates = observation.nonpercept_predicates
        self._gripper_open_percentage = observation.gripper_open_percentage

    def _create_state(self) -> _PartialPerceptionState:
        # Build the continuous part of the state.
        assert self._robot is not None
        state_dict = {
            self._robot: {
                "gripper_open_percentage": self._gripper_open_percentage,
                "curr_held_item_id": self._holding_item_id_feature,
            },
        }
        for obj, (x, y, z) in self._known_object_poses.items():
            state_dict[obj] = {
                "x": x,
                "y": y,
                "z": z,
            }
        # Construct a regular state before adding atoms.
        percept_state = utils.create_state_from_dict(state_dict)
        logging.debug("Percept state:")
        logging.debug(percept_state.pretty_str())
        # Prepare the simulator state.
        simulator_state = {
            "predicates": self._nonpercept_predicates,
            "atoms": self._nonpercept_atoms,
        }
        logging.debug("Simulator state:")
        logging.debug(simulator_state)
        # Now finish the state.
        state = _PartialPerceptionState(percept_state.data,
                                        simulator_state=simulator_state)
        return state
