"""A perceiver specific to the spot bike env."""

import logging
from typing import Dict, Optional, Set, Tuple

import numpy as np

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.envs.spot_env import HANDEMPTY_GRIPPER_THRESHOLD, \
    SpotBikeEnv, _PartialPerceptionState, _SpotObservation
from predicators.perception.base_perceiver import BasePerceiver
from predicators.settings import CFG
from predicators.spot_utils.spot_utils import obj_name_to_apriltag_id
from predicators.structs import Action, DefaultState, EnvironmentTask, \
    GroundAtom, Object, Observation, Predicate, State, Task


class SpotBikePerceiver(BasePerceiver):
    """A perceiver specific to the spot bike env."""

    def __init__(self) -> None:
        super().__init__()
        self._known_object_poses: Dict[Object, Tuple[float, float, float]] = {}
        self._known_objects_in_hand_view: Set[Object] = set()
        self._robot: Optional[Object] = None
        self._nonpercept_atoms: Set[GroundAtom] = set()
        self._nonpercept_predicates: Set[Predicate] = set()
        self._percept_predicates: Set[Predicate] = set()
        self._prev_action: Optional[Action] = None
        self._holding_item_id_feature = 0.0
        self._gripper_open_percentage = 0.0
        self._robot_pos = (0.0, 0.0, 0.0, 0.0)
        self._lost_objects: Set[Object] = set()
        assert CFG.env == "spot_bike_env"
        self._curr_env: Optional[BaseEnv] = None
        self._waiting_for_observation = True
        # Keep track of objects that are contained (out of view) in another
        # object, like a bag or bucket. This is important not only for gremlins
        # but also for small changes in the container's perceived pose.
        self._container_to_contained_objects: Dict[Object, Set[Object]] = {}

    @classmethod
    def get_name(cls) -> str:
        return "spot_bike_env"

    def reset(self, env_task: EnvironmentTask) -> Task:
        self._waiting_for_observation = True
        self._curr_env = get_or_create_env("spot_bike_env")
        assert isinstance(self._curr_env, SpotBikeEnv)
        self._known_object_poses = {}
        self._known_objects_in_hand_view = set()
        self._robot = None
        self._nonpercept_atoms = set()
        self._nonpercept_predicates = set()
        self._percept_predicates = self._curr_env.percept_predicates
        self._prev_action = None
        self._holding_item_id_feature = 0.0
        self._gripper_open_percentage = 0.0
        self._robot_pos = (0.0, 0.0, 0.0, 0.0)
        self._lost_objects = set()
        self._container_to_contained_objects = {}
        init_state = self._create_state()
        return Task(init_state, env_task.goal)

    def update_perceiver_with_action(self, action: Action) -> None:
        # NOTE: we need to keep track of the previous action
        # because the step function (where we need knowledge
        # of the previous action) occurs *after* the action
        # has already been taken.
        self._prev_action = action

    def step(self, observation: Observation) -> State:
        self._update_state_from_observation(observation)
        # Update the curr held item when applicable.
        assert self._curr_env is not None and isinstance(
            self._curr_env, SpotBikeEnv)
        if self._prev_action is not None:
            controller_name, objects, _ = self._curr_env.parse_action(
                self._prev_action)
            logging.info(f"[Perceiver] Previous action was {controller_name}.")
            # The robot is always the 0th argument of an
            # operator!
            if "grasp" in controller_name.lower():
                assert self._holding_item_id_feature == 0.0
                # We know that the object that we attempted to grasp was
                # the second argument to the controller.
                object_attempted_to_grasp = objects[1]
                # Remove from contained objects.
                for contained in self._container_to_contained_objects.values():
                    contained.discard(object_attempted_to_grasp)
                grasp_obj_id = obj_name_to_apriltag_id[
                    object_attempted_to_grasp.name]
                # We only want to update the holding item id feature
                # if we successfully picked something.
                if self._gripper_open_percentage > HANDEMPTY_GRIPPER_THRESHOLD:
                    self._holding_item_id_feature = grasp_obj_id
                else:
                    # We lost the object!
                    logging.info("[Perceiver] Object was lost!")
                    self._lost_objects.add(object_attempted_to_grasp)
            elif "place" in controller_name.lower():
                self._holding_item_id_feature = 0.0
                # Check if the item we just placed is in view. It needs to
                # be in view to assess whether it was placed correctly.
                robot, obj, surface = objects
                state = self._create_state()
                in_view_classifier = self._curr_env._tool_in_view_classifier  # pylint: disable=protected-access
                in_bag_classifier = self._curr_env._inbag_classifier  # pylint: disable=protected-access
                is_in_view = in_view_classifier(state, [robot, obj])
                if not is_in_view:
                    # We lost the object!
                    logging.info("[Perceiver] Object was lost!")
                    self._lost_objects.add(obj)
                elif surface.type.name == "bag" and in_bag_classifier(
                        state, [obj, surface]):
                    # The object is now contained.
                    if surface not in self._container_to_contained_objects:
                        self._container_to_contained_objects[surface] = set()
                    self._container_to_contained_objects[surface].add(obj)
            else:
                # We ensure the holding item feature is set
                # back to 0.0 if the hand is ever empty.
                prev_holding_item_id = self._holding_item_id_feature
                if self._gripper_open_percentage <= HANDEMPTY_GRIPPER_THRESHOLD:
                    self._holding_item_id_feature = 0.0
                    # This can only happen if the item was dropped during
                    # something other than a place.
                    if prev_holding_item_id != 0.0:
                        tag_id = int(np.round(prev_holding_item_id))
                        # We lost the object that we were holding!
                        apriltag_id_to_obj_name = {
                            v: k
                            for k, v in obj_name_to_apriltag_id.items()
                        }
                        obj_name = apriltag_id_to_obj_name[tag_id]
                        obj = [
                            o for o in self._known_object_poses
                            if o.name == obj_name
                        ][0]
                        # We lost the object!
                        logging.info("[Perceiver] Object was lost!")
                        self._lost_objects.add(obj)

        return self._create_state()

    def _update_state_from_observation(self, observation: Observation) -> None:
        assert isinstance(observation, _SpotObservation)
        # If a container is being updated, change the poses for contained
        # objects.
        for container in observation.objects_in_view:
            if container not in self._container_to_contained_objects:
                continue
            if container not in self._known_object_poses:
                continue
            last_container_pose = self._known_object_poses[container]
            new_container_pose = observation.objects_in_view[container]
            dx, dy, dz = np.subtract(new_container_pose, last_container_pose)
            for obj in self._container_to_contained_objects[container]:
                x, y, z = self._known_object_poses[obj]
                new_obj_pose = (x + dx, y + dy, z + dz)
                self._known_object_poses[obj] = new_obj_pose
        self._waiting_for_observation = False
        self._robot = observation.robot
        self._known_object_poses.update(observation.objects_in_view)
        self._known_objects_in_hand_view = observation.objects_in_hand_view
        self._nonpercept_atoms = observation.nonpercept_atoms
        self._nonpercept_predicates = observation.nonpercept_predicates
        self._gripper_open_percentage = observation.gripper_open_percentage
        self._robot_pos = observation.robot_pos
        for obj in observation.objects_in_view:
            self._lost_objects.discard(obj)

    def _create_state(self) -> State:
        if self._waiting_for_observation:
            return DefaultState
        # Build the continuous part of the state.
        assert self._robot is not None
        state_dict = {
            self._robot: {
                "gripper_open_percentage": self._gripper_open_percentage,
                "curr_held_item_id": self._holding_item_id_feature,
                "x": self._robot_pos[0],
                "y": self._robot_pos[1],
                "z": self._robot_pos[2],
                "yaw": self._robot_pos[3],
            },
        }
        for obj, (x, y, z) in self._known_object_poses.items():
            state_dict[obj] = {
                "x": x,
                "y": y,
                "z": z,
            }
            if obj.type.name in ("tool", "platform"):
                # Detect if the object is in view currently.
                if obj in self._known_objects_in_hand_view:
                    in_view_val = 1.0
                else:
                    in_view_val = 0.0
                state_dict[obj]["in_view"] = in_view_val
                # Detect if we have lost the tool.
                if obj in self._lost_objects:
                    lost_val = 1.0
                else:
                    lost_val = 0.0
                state_dict[obj]["lost"] = lost_val
        # Construct a regular state before adding atoms.
        percept_state = utils.create_state_from_dict(state_dict)
        logging.info("Percept state:")
        logging.info(percept_state.pretty_str())
        logging.info("Percept atoms:")
        atom_str = "\n".join(
            map(
                str,
                sorted(utils.abstract(percept_state,
                                      self._percept_predicates))))
        logging.info(atom_str)
        # Prepare the simulator state.
        simulator_state = {
            "predicates": self._nonpercept_predicates,
            "atoms": self._nonpercept_atoms,
        }
        logging.info("Simulator state:")
        logging.info(simulator_state)
        # Now finish the state.
        state = _PartialPerceptionState(percept_state.data,
                                        simulator_state=simulator_state)
        return state
