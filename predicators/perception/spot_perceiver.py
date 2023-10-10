"""A perceiver specific to spot envs."""

import logging
from typing import Dict, Optional, Set

import numpy as np
from bosdyn.client import math_helpers

from predicators import utils
from predicators.envs import BaseEnv, get_or_create_env
from predicators.envs.spot_env import HANDEMPTY_GRIPPER_THRESHOLD, \
    SpotCubeEnv, SpotEnv, _PartialPerceptionState, _SpotObservation, \
    tool_in_view_classifier
from predicators.perception.base_perceiver import BasePerceiver
from predicators.settings import CFG
from predicators.structs import Action, DefaultState, EnvironmentTask, \
    GoalDescription, GroundAtom, Object, Observation, Predicate, State, Task


class SpotPerceiver(BasePerceiver):
    """A perceiver specific to spot envs."""

    def __init__(self) -> None:
        super().__init__()
        self._known_object_poses: Dict[Object, math_helpers.SE3Pose] = {}
        self._known_objects_in_hand_view: Set[Object] = set()
        self._robot: Optional[Object] = None
        self._nonpercept_atoms: Set[GroundAtom] = set()
        self._nonpercept_predicates: Set[Predicate] = set()
        self._percept_predicates: Set[Predicate] = set()
        self._prev_action: Optional[Action] = None
        self._held_object: Optional[Object] = None
        self._gripper_open_percentage = 0.0
        self._robot_pos: math_helpers.SE3Pose = math_helpers.SE3Pose(
            0, 0, 0, math_helpers.Quat())
        self._lost_objects: Set[Object] = set()
        self._curr_env: Optional[BaseEnv] = None
        self._waiting_for_observation = True
        # Keep track of objects that are contained (out of view) in another
        # object, like a bag or bucket. This is important not only for gremlins
        # but also for small changes in the container's perceived pose.
        self._container_to_contained_objects: Dict[Object, Set[Object]] = {}

    @classmethod
    def get_name(cls) -> str:
        return "spot_perceiver"

    def reset(self, env_task: EnvironmentTask) -> Task:
        self._waiting_for_observation = True
        self._curr_env = get_or_create_env(CFG.env)
        assert isinstance(self._curr_env, SpotEnv)
        self._known_object_poses = {}
        self._known_objects_in_hand_view = set()
        self._robot = None
        self._nonpercept_atoms = set()
        self._nonpercept_predicates = set()
        self._percept_predicates = self._curr_env.percept_predicates
        self._prev_action = None
        self._held_object = None
        self._gripper_open_percentage = 0.0
        self._robot_pos = math_helpers.SE3Pose(0, 0, 0, math_helpers.Quat())
        self._lost_objects = set()
        self._container_to_contained_objects = {}
        init_state = self._create_state()
        goal = self._create_goal(init_state, env_task.goal_description)
        return Task(init_state, goal)

    def update_perceiver_with_action(self, action: Action) -> None:
        # NOTE: we need to keep track of the previous action
        # because the step function (where we need knowledge
        # of the previous action) occurs *after* the action
        # has already been taken.
        self._prev_action = action

    def step(self, observation: Observation) -> State:
        self._update_state_from_observation(observation)
        # Update the curr held item when applicable.
        assert self._curr_env is not None
        if self._prev_action is not None:
            assert isinstance(self._prev_action.extra_info, (list, tuple))
            controller_name, objects, _, _ = self._prev_action.extra_info
            logging.info(f"[Perceiver] Previous action was {controller_name}.")
            # The robot is always the 0th argument of an
            # operator!
            if "grasp" in controller_name.lower():
                assert self._held_object is None
                # We know that the object that we attempted to grasp was
                # the second argument to the controller.
                object_attempted_to_grasp = objects[1]
                # Remove from contained objects.
                for contained in self._container_to_contained_objects.values():
                    contained.discard(object_attempted_to_grasp)
                # We only want to update the holding item id feature
                # if we successfully picked something.
                if self._gripper_open_percentage > HANDEMPTY_GRIPPER_THRESHOLD:
                    self._held_object = object_attempted_to_grasp
                else:
                    # We lost the object!
                    logging.info("[Perceiver] Object was lost!")
                    self._lost_objects.add(object_attempted_to_grasp)
            elif "place" in controller_name.lower():
                self._held_object = None
                # Check if the item we just placed is in view. It needs to
                # be in view to assess whether it was placed correctly.
                robot, obj = objects[:2]
                state = self._create_state()
                is_in_view = tool_in_view_classifier(state, [robot, obj])
                if not is_in_view:
                    # We lost the object!
                    logging.info("[Perceiver] Object was lost!")
                    self._lost_objects.add(obj)
            else:
                # Ensure the held object is reset if the hand is empty.
                prev_held_object = self._held_object
                if self._gripper_open_percentage <= HANDEMPTY_GRIPPER_THRESHOLD:
                    self._held_object = None
                    # This can only happen if the item was dropped during
                    # something other than a place.
                    if prev_held_object is not None:
                        # We lost the object!
                        logging.info("[Perceiver] An object was lost: "
                                     f"{prev_held_object} was lost!")
                        self._lost_objects.add(prev_held_object)

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
                x, y, z = self._known_object_poses[
                    obj].x, self._known_object_poses[
                        obj].y, self._known_object_poses[obj].z
                new_obj_pose = (x + dx, y + dy, z + dz)
                self._known_object_poses[obj] = math_helpers.SE3Pose(
                    new_obj_pose[0], new_obj_pose[1], new_obj_pose[2],
                    self._known_object_poses[obj].rot)
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
                "x": self._robot_pos.x,
                "y": self._robot_pos.y,
                "z": self._robot_pos.z,
                "W_quat": self._robot_pos.rot.w,
                "X_quat": self._robot_pos.rot.x,
                "Y_quat": self._robot_pos.rot.y,
                "Z_quat": self._robot_pos.rot.z,
            },
        }
        for obj, pose in self._known_object_poses.items():
            state_dict[obj] = {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "W_quat": pose.rot.w,
                "X_quat": pose.rot.x,
                "Y_quat": pose.rot.y,
                "Z_quat": pose.rot.z,
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
                if obj == self._held_object:
                    held_val = 1.0
                else:
                    held_val = 0.0
                state_dict[obj]["held"] = held_val
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

    def _create_goal(self, state: State,
                     goal_description: GoalDescription) -> Set[GroundAtom]:
        del state  # not used
        # Unfortunate hack to deal with the fact that the state is actually
        # not yet set. Hopefully one day other cleanups will enable cleaning.
        assert self._curr_env is not None
        type_name_to_type = {t.name: t for t in self._curr_env.types}
        pred_name_to_pred = {p.name: p for p in self._curr_env.predicates}
        assert goal_description == "put the cube on the sticky table"
        assert isinstance(self._curr_env, SpotCubeEnv)
        cube = Object("cube", type_name_to_type["tool"])
        target = Object("extra_room_table", type_name_to_type["flat_surface"])
        On = pred_name_to_pred["On"]
        return {GroundAtom(On, [cube, target])}
