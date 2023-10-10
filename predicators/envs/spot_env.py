"""Basic environment for the Boston Dynamics Spot Robot."""
import abc
import functools
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import numpy as np
from bosdyn.client import create_standard_sdk, math_helpers
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.sdk import Robot
from bosdyn.client.util import authenticate, setup_logging
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.spot_utils.perception.object_detection import \
    AprilTagObjectDetectionID, ObjectDetectionID, detect_objects, \
    visualize_all_artifacts
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_find_objects import \
    init_search_for_objects
from predicators.spot_utils.skills.spot_navigation import go_home, \
    navigate_to_absolute_pose
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.utils import get_graph_nav_dir, \
    get_robot_gripper_open_percentage, verify_estop
from predicators.structs import Action, EnvironmentTask, GoalDescription, \
    GroundAtom, LiftedAtom, Object, Observation, Predicate, State, \
    STRIPSOperator, Type, Variable

###############################################################################
#                                Base Class                                   #
###############################################################################


@dataclass(frozen=True)
class _SpotObservation:
    """An observation for a SpotEnv."""
    # Camera name to image
    images: Dict[str, RGBDImageWithContext]
    # Objects that are seen in the current image and their positions in world
    objects_in_view: Dict[Object, math_helpers.SE3Pose]
    # Objects seen only by the hand camera
    objects_in_hand_view: Set[Object]
    # Expose the robot object.
    robot: Object
    # Status of the robot gripper.
    gripper_open_percentage: float
    # Robot SE3 Pose
    robot_pos: math_helpers.SE3Pose
    # Ground atoms without ground-truth classifiers
    # A placeholder until all predicates have classifiers
    nonpercept_atoms: Set[GroundAtom]
    nonpercept_predicates: Set[Predicate]


class _PartialPerceptionState(State):
    """Some continuous object features, and ground atoms in simulator_state.

    The main idea here is that we have some predicates with actual
    classifiers implemented, but not all.

    NOTE: these states are only created in the perceiver, but they are used
    in the classifier definitions for the dummy predicates
    """

    @property
    def _simulator_state_predicates(self) -> Set[Predicate]:
        assert isinstance(self.simulator_state, Dict)
        return self.simulator_state["predicates"]

    @property
    def _simulator_state_atoms(self) -> Set[GroundAtom]:
        assert isinstance(self.simulator_state, Dict)
        return self.simulator_state["atoms"]

    def simulator_state_atom_holds(self, atom: GroundAtom) -> bool:
        """Check whether an atom holds in the simulator state."""
        assert atom.predicate in self._simulator_state_predicates
        return atom in self._simulator_state_atoms

    def allclose(self, other: State) -> bool:
        if self.simulator_state != other.simulator_state:
            return False
        return self._allclose(other)

    def copy(self) -> State:
        state_copy = {o: self._copy_state_value(self.data[o]) for o in self}
        sim_state_copy = {
            "predicates": self._simulator_state_predicates.copy(),
            "atoms": self._simulator_state_atoms.copy()
        }
        return _PartialPerceptionState(state_copy,
                                       simulator_state=sim_state_copy)


def _create_dummy_predicate_classifier(
        pred: Predicate) -> Callable[[State, Sequence[Object]], bool]:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        assert isinstance(s, _PartialPerceptionState)
        atom = GroundAtom(pred, objs)
        return s.simulator_state_atom_holds(atom)

    return _classifier


@functools.lru_cache(maxsize=None)
def get_robot() -> Tuple[Robot, SpotLocalizer, LeaseClient]:
    """Create the robot only once."""
    setup_logging(False)
    hostname = CFG.spot_robot_ip
    path = get_graph_nav_dir()
    sdk = create_standard_sdk("PredicatorsClient-")
    robot = sdk.create_robot(hostname)
    authenticate(robot)
    verify_estop(robot)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    lease_client.take()
    lease_keepalive = LeaseKeepAlive(lease_client,
                                     must_acquire=True,
                                     return_at_exit=True)
    assert path.exists()
    localizer = SpotLocalizer(robot, path, lease_client, lease_keepalive)
    return robot, localizer, lease_client


@functools.lru_cache(maxsize=None)
def get_detection_id_for_object(obj: Object) -> ObjectDetectionID:
    """Exposed for wrapper and options."""
    # Avoid circular import issues.
    from predicators.envs import \
        get_or_create_env  # pylint: disable=import-outside-toplevel
    env = get_or_create_env(CFG.env)
    assert isinstance(env, SpotEnv)
    detection_id_to_obj = env._detection_id_to_obj  # pylint: disable=protected-access
    obj_to_detection_id = {o: d for d, o in detection_id_to_obj.items()}
    return obj_to_detection_id[obj]


class SpotEnv(BaseEnv):
    """An environment containing tasks for a real Spot robot to execute.

    This is a base class that specific sub-classes that define actual
    tasks should inherit from.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        assert "spot_wrapper" in CFG.approach, \
            "Must use spot wrapper in spot envs!"
        robot, localizer, lease_client = get_robot()
        self._robot = robot
        self._localizer = localizer
        self._lease_client = lease_client
        # Note that we need to include the operators in this
        # class because they're used to update the symbolic
        # parts of the state during execution.
        self._strips_operators: Set[STRIPSOperator] = set()
        self._current_task_goal_reached = False

        # Create constant objects.
        self._spot_object = Object("robot", _robot_type)

    @property
    def strips_operators(self) -> Set[STRIPSOperator]:
        """Expose the STRIPSOperators for use by oracles."""
        return self._strips_operators

    @property
    def percept_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return set()

    @property
    def action_space(self) -> Box:
        # The action space is effectively empty because only the extra info
        # part of actions are used.
        return Box(0, 1, (0, ))

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        # NOTE: task_idx and train_or_test ignored unless loading from JSON!
        if CFG.test_task_json_dir is not None and train_or_test == "test":
            self._current_task = self._test_tasks[task_idx]
        else:
            prompt = f"Please set up {train_or_test} task {task_idx}!"
            utils.prompt_user(prompt)
            self._lease_client.take()
            self._current_task = self._actively_construct_env_task()
        self._current_observation = self._current_task.init_obs
        self._current_task_goal_reached = False
        return self._current_task.init_obs

    def step(self, action: Action) -> Observation:
        """Override step() because simulate() is not implemented."""
        assert isinstance(action.extra_info, (list, tuple))
        action_name, _, action_fn, action_fn_args = action.extra_info
        # The extra info is (action name, objects, function, function args).
        # The action name is either an operator name (for use with nonpercept
        # predicates) or a special name. See below for the special names.

        obs = self._current_observation
        assert isinstance(obs, _SpotObservation)
        assert self.action_space.contains(action.arr)

        # Special case: the action is "done", indicating that the robot
        # believes it has finished the task. Used for goal checking.
        if action_name == "done":
            while True:
                goal_description = self._current_task.goal_description
                logging.info(f"The goal is: {goal_description}")
                prompt = "Is the goal accomplished? Answer y or n. "
                response = utils.prompt_user(prompt).strip()
                if response == "y":
                    self._current_task_goal_reached = True
                    break
                if response == "n":
                    self._current_task_goal_reached = False
                    break
                logging.info("Invalid input, must be either 'y' or 'n'")
            return self._current_observation

        # Otherwise, the action is either an operator to execute or a special
        # action. The only difference between the two is that operators update
        # the non-perfect states.

        operator_names = {o.name for o in self._strips_operators}

        # The action corresponds to an operator finishing.
        if action_name in operator_names:
            # Update the non-percepts.
            operator_names = {o.name for o in self._strips_operators}
            next_nonpercept = self._get_next_nonpercept_atoms(obs, action)
        else:
            next_nonpercept = obs.nonpercept_atoms

        # Execute the action in the real environment.
        action_fn(*action_fn_args)  # type: ignore

        # Get the new observation.
        self._current_observation = self._build_observation(next_nonpercept)

        return self._current_observation

    def get_observation(self) -> Observation:
        return self._current_observation

    def goal_reached(self) -> bool:
        return self._current_task_goal_reached

    def _build_observation(self,
                           ground_atoms: Set[GroundAtom]) -> _SpotObservation:
        """Helper for building a new _SpotObservation().

        This is an environment method because the nonpercept predicates
        may vary per environment.
        """
        # Make sure the robot pose is up to date.
        self._localizer.localize()
        # Get the universe of all object detections.
        all_object_detection_ids = set(self._detection_id_to_obj)
        # Get the camera images.
        rgbds = capture_images(self._robot, self._localizer)
        all_detections, all_artifacts = detect_objects(
            all_object_detection_ids, rgbds)

        root_dir = Path(__file__).parent.parent.parent
        outdir = root_dir / CFG.spot_perception_outdir
        time_str = time.strftime("%Y%m%d-%H%M%S")
        detections_outfile = outdir / f"detections_{time_str}.png"
        no_detections_outfile = outdir / f"no_detections_{time_str}.png"

        visualize_all_artifacts(all_artifacts, detections_outfile,
                                no_detections_outfile)

        # Separately, get detections for the hand in particular.
        hand_rgbd = {
            k: v
            for (k, v) in rgbds.items() if k == "hand_color_image"
        }
        hand_detections, hand_artifacts = detect_objects(
            all_object_detection_ids, hand_rgbd)

        hand_detections_outfile = outdir / f"hand_detections_{time_str}.png"
        hand_no_detect_outfile = outdir / f"hand_no_detections_{time_str}.png"

        visualize_all_artifacts(hand_artifacts, hand_detections_outfile,
                                hand_no_detect_outfile)

        # Now construct a dict of all objects in view, as well as a set
        # of objects that the hand can see.
        objects_in_view = {
            self._detection_id_to_obj[det_id]: val
            for (det_id, val) in all_detections.items()
        }
        objects_in_hand_view = set(self._detection_id_to_obj[det_id]
                                   for det_id in hand_detections)
        gripper_open_percentage = get_robot_gripper_open_percentage(
            self._robot)
        robot_pos = self._localizer.get_last_robot_pose()
        # Prepare the non-percepts.
        nonpercept_preds = self.predicates - self.percept_predicates
        assert all(a.predicate in nonpercept_preds for a in ground_atoms)
        obs = _SpotObservation(rgbds, objects_in_view, objects_in_hand_view,
                               self._spot_object, gripper_open_percentage,
                               robot_pos, ground_atoms, nonpercept_preds)

        return obs

    def _get_next_nonpercept_atoms(self, obs: _SpotObservation,
                                   action: Action) -> Set[GroundAtom]:
        """Helper for step().

        This should be deprecated eventually.
        """
        assert isinstance(action.extra_info, (list, tuple))
        op_name, op_objects, _, _ = action.extra_info
        op_name_to_op = {o.name: o for o in self._strips_operators}
        op = op_name_to_op[op_name]
        ground_op = op.ground(tuple(op_objects))
        # Update the atoms using the operator.
        next_ground_atoms = utils.apply_operator(ground_op,
                                                 obs.nonpercept_atoms)
        # Return only the atoms for the non-percept predicates.
        return {
            a
            for a in next_ground_atoms
            if a.predicate not in self.percept_predicates
        }

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for SpotEnv.")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        goal = self._generate_goal_description()  # currently just one goal
        return [
            EnvironmentTask(None, goal) for _ in range(CFG.num_train_tasks)
        ]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        goal = self._generate_goal_description()  # currently just one goal
        return [EnvironmentTask(None, goal) for _ in range(CFG.num_test_tasks)]

    def _actively_construct_env_task(self) -> EnvironmentTask:
        # Have the spot walk around the environment once to construct
        # an initial observation.
        objects_in_view = self._actively_construct_initial_object_views()
        rgb_images = capture_images(self._robot, self._localizer)
        gripper_open_percentage = get_robot_gripper_open_percentage(
            self._robot)
        self._localizer.localize()
        robot_pos = self._localizer.get_last_robot_pose()
        nonpercept_atoms = self._get_initial_nonpercept_atoms()
        nonpercept_preds = self.predicates - self.percept_predicates
        assert all(a.predicate in nonpercept_preds for a in nonpercept_atoms)
        obs = _SpotObservation(rgb_images, objects_in_view, set(),
                               self._spot_object, gripper_open_percentage,
                               robot_pos, nonpercept_atoms, nonpercept_preds)
        goal_description = self._generate_goal_description()
        task = EnvironmentTask(obs, goal_description)
        # Save the task for future use.
        json_objects = {o.name: o.type.name for o in objects_in_view}
        json_objects[self._spot_object.name] = self._spot_object.type.name
        init_json_dict = {
            o.name: {
                "x": pose.x,
                "y": pose.y,
                "z": pose.z,
                "W_quat": pose.rot.w,
                "X_quat": pose.rot.x,
                "Y_quat": pose.rot.y,
                "Z_quat": pose.rot.z,
            }
            for o, pose in objects_in_view.items()
        }
        for obj in objects_in_view:
            if "lost" in obj.type.feature_names:
                init_json_dict[obj.name]["lost"] = 0.0
            if "in_view" in obj.type.feature_names:
                init_json_dict[obj.name]["in_view"] = 1.0
            if "held" in obj.type.feature_names:
                init_json_dict[obj.name]["held"] = 0.0
        init_json_dict[self._spot_object.name] = {
            "gripper_open_percentage": gripper_open_percentage,
            "x": robot_pos.x,
            "y": robot_pos.y,
            "z": robot_pos.z,
            "W_quat": robot_pos.rot.w,
            "X_quat": robot_pos.rot.x,
            "Y_quat": robot_pos.rot.y,
            "Z_quat": robot_pos.rot.z,
        }
        json_dict = {
            "objects": json_objects,
            "init": init_json_dict,
            "goal_description": goal_description,
        }
        outfile = utils.get_env_asset_path("task_jsons/spot/last.json",
                                           assert_exists=False)
        outpath = Path(outfile)
        outpath.parent.mkdir(parents=True, exist_ok=True)
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)
        logging.info(f"Dumped task to {outfile}. Rename it to save it.")
        return task

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        # Use the BaseEnv default code for loading from JSON, which will
        # create a State as an observation. We'll then convert that State
        # into a _SpotObservation instead.
        base_env_task = super()._load_task_from_json(json_file)
        init = base_env_task.init
        # Images not currently saved or used.
        images: Dict[str, RGBDImageWithContext] = {}
        objects_in_view: Dict[Object, math_helpers.SE3Pose] = {}
        known_objects = set(
            self._detection_id_to_obj.values()) | {self._spot_object}
        robot: Optional[Object] = None
        for obj in init:
            assert obj in known_objects
            if obj.is_instance(_robot_type):
                robot = obj
                continue
            pos = math_helpers.SE3Pose(
                init.get(obj, "x"), init.get(obj, "y"), init.get(obj, "z"),
                math_helpers.Quat(init.get(obj, "W_quat"),
                                  init.get(obj, "X_quat"),
                                  init.get(obj, "Y_quat"),
                                  init.get(obj, "Z_quat")))
            objects_in_view[obj] = pos
        assert robot is not None
        assert robot == self._spot_object
        gripper_open_percentage = init.get(robot, "gripper_open_percentage")
        robot_pos = math_helpers.SE3Pose(
            init.get(robot, "x"), init.get(robot, "y"), init.get(robot, "z"),
            math_helpers.Quat(init.get(robot, "W_quat"),
                              init.get(robot, "X_quat"),
                              init.get(robot, "Y_quat"),
                              init.get(robot, "Z_quat")))

        # Reset the robot to the given position.
        self._localizer.localize()
        navigate_to_absolute_pose(self._robot, self._localizer,
                                  robot_pos.get_closest_se2_transform())

        # Prepare the non-percepts.
        nonpercept_atoms = self._get_initial_nonpercept_atoms()
        nonpercept_preds = self.predicates - self.percept_predicates
        init_obs = _SpotObservation(
            images,
            objects_in_view,
            set(),
            robot,
            gripper_open_percentage,
            robot_pos,
            nonpercept_atoms,
            nonpercept_preds,
        )
        # The goal can remain the same.
        goal = base_env_task.goal_description
        return EnvironmentTask(init_obs, goal)

    def _actively_construct_initial_object_views(
            self) -> Dict[Object, math_helpers.SE3Pose]:
        stow_arm(self._robot)
        go_home(self._robot, self._localizer)
        self._localizer.localize()
        detection_ids = self._detection_id_to_obj.keys()
        detections, _ = init_search_for_objects(self._robot, self._localizer,
                                                detection_ids)
        obj_to_se3_pose = {
            self._detection_id_to_obj[det_id]: val
            for (det_id, val) in detections.items()
        }
        return obj_to_se3_pose

    @property
    @abc.abstractmethod
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:
        """Get an object from a perception detection ID."""

    @abc.abstractmethod
    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        """Get the initial atoms for nonpercept predicates."""

    @abc.abstractmethod
    def _generate_goal_description(self) -> GoalDescription:
        """For now, we assume that there's only one goal per environment."""


###############################################################################
#                   Shared Types, Predicates, Operators                       #
###############################################################################

## Constants
HANDEMPTY_GRIPPER_THRESHOLD = 5.0  # made public for use in perceiver
_ONTOP_THRESHOLD = 0.55
_ONTOP_MAX_HEIGHT_THRESHOLD = 0.25
_REACHABLE_THRESHOLD = 1.7
_REACHABLE_YAW_THRESHOLD = 0.95  # higher better

## Types
_robot_type = Type("robot", [
    "gripper_open_percentage", "x", "y", "z", "W_quat", "X_quat", "Y_quat",
    "Z_quat"
])
_tool_type = Type("tool", [
    "x", "y", "z", "W_quat", "X_quat", "Y_quat", "Z_quat", "held", "lost",
    "in_view"
])
_surface_type = Type("flat_surface",
                     ["x", "y", "z", "W_quat", "X_quat", "Y_quat", "Z_quat"])


## Predicates
def _handempty_classifier(state: State, objects: Sequence[Object]) -> bool:
    spot = objects[0]
    gripper_open_percentage = state.get(spot, "gripper_open_percentage")
    return gripper_open_percentage <= HANDEMPTY_GRIPPER_THRESHOLD


def _nothandempty_classifier(state: State, objects: Sequence[Object]) -> bool:
    return not _handempty_classifier(state, objects)


def _holding_tool_classifier(state: State, objects: Sequence[Object]) -> bool:
    spot, obj = objects
    if _handempty_classifier(state, [spot]):
        return False
    return state.get(obj, "held") > 0.5


def _ontop_classifier(state: State, objects: Sequence[Object]) -> bool:
    obj_on, obj_surface = objects

    spot = [o for o in state if o.type.name == "robot"][0]
    if _holding_tool_classifier(state, [spot, obj_on]):
        return False

    obj_on_pose = [
        state.get(obj_on, "x"),
        state.get(obj_on, "y"),
        state.get(obj_on, "z")
    ]
    obj_surface_pose = [
        state.get(obj_surface, "x"),
        state.get(obj_surface, "y"),
        state.get(obj_surface, "z")
    ]
    is_x_same = np.sqrt(
        (obj_on_pose[0] - obj_surface_pose[0])**2) <= _ONTOP_THRESHOLD
    is_y_same = np.sqrt(
        (obj_on_pose[1] - obj_surface_pose[1])**2) <= _ONTOP_THRESHOLD
    is_above_z = 0.0 < (obj_on_pose[2] -
                        obj_surface_pose[2]) < _ONTOP_MAX_HEIGHT_THRESHOLD
    return is_x_same and is_y_same and is_above_z


def _onfloor_classifier(state: State, objects: Sequence[Object]) -> bool:
    obj_on, = objects
    return state.get(obj_on, "z") < 0.0


def tool_in_view_classifier(state: State, objects: Sequence[Object]) -> bool:
    """Made public for perceiver."""
    _, tool = objects
    return state.get(tool, "in_view") > 0.5


def _reachable_classifier(state: State, objects: Sequence[Object]) -> bool:
    spot, obj = objects
    spot_pose = [
        state.get(spot, "x"),
        state.get(spot, "y"),
        state.get(spot, "z"),
        state.get(spot, "W_quat"),
        state.get(spot, "X_quat"),
        state.get(spot, "Y_quat"),
        state.get(spot, "Z_quat")
    ]
    obj_pose = [state.get(obj, "x"), state.get(obj, "y"), state.get(obj, "z")]
    is_xy_near = np.sqrt(
        (spot_pose[0] - obj_pose[0])**2 +
        (spot_pose[1] - obj_pose[1])**2) <= _REACHABLE_THRESHOLD

    # Compute angle between spot's forward direction and the line from
    # spot to the object.
    spot_yaw = math_helpers.SE3Pose(
        spot_pose[0], spot_pose[1], spot_pose[2],
        math_helpers.Quat(spot_pose[3], spot_pose[4], spot_pose[5],
                          spot_pose[6])).get_closest_se2_transform().angle
    forward_unit = [np.cos(spot_yaw), np.sin(spot_yaw)]
    spot_to_obj = np.subtract(obj_pose[:2], spot_pose[:2])
    spot_to_obj_unit = spot_to_obj / np.linalg.norm(spot_to_obj)
    angle_between_robot_and_obj = np.arccos(
        np.dot(forward_unit, spot_to_obj_unit))
    is_yaw_near = abs(angle_between_robot_and_obj) < _REACHABLE_YAW_THRESHOLD

    return is_xy_near and is_yaw_near


_On = Predicate("On", [_tool_type, _surface_type], _ontop_classifier)
_OnFloor = Predicate("OnFloor", [_tool_type], _onfloor_classifier)
_HandEmpty = Predicate("HandEmpty", [_robot_type], _handempty_classifier)
_notHandEmpty = Predicate("Not-HandEmpty", [_robot_type],
                          _nothandempty_classifier)
_HoldingTool = Predicate("HoldingTool", [_robot_type, _tool_type],
                         _holding_tool_classifier)
_InViewTool = Predicate("InViewTool", [_robot_type, _tool_type],
                        tool_in_view_classifier)
_ReachableSurface = Predicate("ReachableSurface", [_robot_type, _surface_type],
                              _reachable_classifier)


## Operators (needed in the environment for non-percept atom hack)
def _create_operators() -> Iterator[STRIPSOperator]:
    """Inside a function to avoid scoping issues."""

    # MoveToToolOnSurface
    spot = Variable("?robot", _robot_type)
    tool = Variable("?tool", _tool_type)
    surface = Variable("?surface", _surface_type)
    preconditions = {
        LiftedAtom(_On, [tool, surface]),
    }
    add_effs = {LiftedAtom(_InViewTool, [spot, tool])}
    ignore_effs = {_ReachableSurface, _InViewTool}
    yield STRIPSOperator("MoveToToolOnSurface", [spot, tool, surface],
                         preconditions, add_effs, set(), ignore_effs)
    # MoveToToolOnFloor
    spot = Variable("?robot", _robot_type)
    tool = Variable("?tool", _tool_type)
    preconditions = {LiftedAtom(_OnFloor, [tool])}
    add_effs = {LiftedAtom(_InViewTool, [spot, tool])}
    ignore_effs = {_ReachableSurface, _InViewTool}
    yield STRIPSOperator("MoveToToolOnFloor", [spot, tool], preconditions,
                         add_effs, set(), ignore_effs)
    # MoveToSurface
    spot = Variable("?robot", _robot_type)
    surface = Variable("?surface", _surface_type)
    preconditions = set()
    add_effs = {LiftedAtom(_ReachableSurface, [spot, surface])}
    ignore_effs = {_ReachableSurface, _InViewTool}
    yield STRIPSOperator("MoveToSurface", [spot, surface], preconditions,
                         add_effs, set(), ignore_effs)
    # GraspToolFromSurface
    spot = Variable("?robot", _robot_type)
    tool = Variable("?tool", _tool_type)
    surface = Variable("?surface", _surface_type)
    preconds = {
        LiftedAtom(_On, [tool, surface]),
        LiftedAtom(_HandEmpty, [spot]),
        LiftedAtom(_InViewTool, [spot, tool])
    }
    add_effs = {
        LiftedAtom(_HoldingTool, [spot, tool]),
        LiftedAtom(_notHandEmpty, [spot])
    }
    del_effs = {
        LiftedAtom(_On, [tool, surface]),
        LiftedAtom(_HandEmpty, [spot]),
        LiftedAtom(_InViewTool, [spot, tool])
    }
    yield STRIPSOperator("GraspToolFromSurface", [spot, tool, surface],
                         preconds, add_effs, del_effs, set())
    # GraspToolFromFloor
    spot = Variable("?robot", _robot_type)
    tool = Variable("?tool", _tool_type)
    preconds = {
        LiftedAtom(_OnFloor, [tool]),
        LiftedAtom(_HandEmpty, [spot]),
        LiftedAtom(_InViewTool, [spot, tool])
    }
    add_effs = {
        LiftedAtom(_HoldingTool, [spot, tool]),
        LiftedAtom(_notHandEmpty, [spot])
    }
    del_effs = {
        LiftedAtom(_OnFloor, [tool]),
        LiftedAtom(_HandEmpty, [spot]),
        LiftedAtom(_InViewTool, [spot, tool])
    }
    yield STRIPSOperator("GraspToolFromFloor", [spot, tool], preconds,
                         add_effs, del_effs, set())
    # PlaceToolOnSurface
    spot = Variable("?robot", _robot_type)
    tool = Variable("?tool", _tool_type)
    surface = Variable("?surface", _surface_type)
    preconds = {
        LiftedAtom(_ReachableSurface, [spot, surface]),
        LiftedAtom(_HoldingTool, [spot, tool]),
        LiftedAtom(_notHandEmpty, [spot])
    }
    add_effs = {
        LiftedAtom(_On, [tool, surface]),
        LiftedAtom(_HandEmpty, [spot])
    }
    del_effs = {
        LiftedAtom(_HoldingTool, [spot, tool]),
        LiftedAtom(_notHandEmpty, [spot]),
        LiftedAtom(_ReachableSurface, [spot, surface]),
    }
    yield STRIPSOperator("PlaceToolOnSurface", [spot, tool, surface], preconds,
                         add_effs, del_effs, set())
    # PlaceToolOnFloor
    spot = Variable("?robot", _robot_type)
    tool = Variable("?tool", _tool_type)
    preconds = {
        LiftedAtom(_HoldingTool, [spot, tool]),
        LiftedAtom(_notHandEmpty, [spot]),
    }
    add_effs = {
        LiftedAtom(_OnFloor, [tool]),
    }
    del_effs = {
        LiftedAtom(_HoldingTool, [spot, tool]),
        LiftedAtom(_notHandEmpty, [spot]),
    }
    yield STRIPSOperator("PlaceToolOnFloor", [spot, tool], preconds, add_effs,
                         del_effs, set())


###############################################################################
#                                Cube Table Env                               #
###############################################################################


class SpotCubeEnv(SpotEnv):
    """An environment corresponding to the spot cube task where the robot
    attempts to place an April Tag cube onto a particular table."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        op_to_name = {o.name: o for o in _create_operators()}
        op_names_to_keep = {
            "MoveToToolOnSurface",
            "MoveToToolOnFloor",
            "MoveToSurface",
            "GraspToolFromSurface",
            "GraspToolFromFloor",
            "PlaceToolOnSurface",
            "PlaceToolOnFloor",
        }
        self._strips_operators = {op_to_name[o] for o in op_names_to_keep}

    @classmethod
    def get_name(cls) -> str:
        return "spot_cube_env"

    @property
    def types(self) -> Set[Type]:
        return {
            _robot_type,
            _tool_type,
            _surface_type,
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            _On, _HandEmpty, _HoldingTool, _ReachableSurface, _notHandEmpty,
            _InViewTool, _OnFloor
        }

    @property
    def percept_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return {
            _HandEmpty, _notHandEmpty, _HoldingTool, _On, _InViewTool, _OnFloor
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.predicates

    @property
    def _detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:

        cube = Object("cube", _tool_type)
        cube_detection = AprilTagObjectDetectionID(410)

        tool_room_table = Object("tool_room_table", _surface_type)
        tool_room_table_detection = AprilTagObjectDetectionID(408)

        extra_room_table = Object("extra_room_table", _surface_type)
        extra_room_table_detection = AprilTagObjectDetectionID(409)

        return {
            cube_detection: cube,
            tool_room_table_detection: tool_room_table,
            extra_room_table_detection: extra_room_table
        }

    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        return set()

    def _generate_goal_description(self) -> GoalDescription:
        return "put the cube on the sticky table"
