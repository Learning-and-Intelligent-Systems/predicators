"""Basic environment for the Boston Dynamics Spot Robot.

Example usage with apriltag grasping:
    python predicators/main.py --env spot_bike_env --approach oracle --seed 0
     --num_train_tasks 0 --num_test_tasks 1 --spot_robot_ip $SPOT_IP
     --bilevel_plan_without_sim True --spot_grasp_use_apriltag True
     --perceiver spot_bike_env
"""
import abc
import functools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Set, \
    Tuple, Union

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
    AprilTagObjectDetectionID, LanguageObjectDetectionID, ObjectDetectionID, \
    detect_objects, get_object_center_pixel_from_artifacts
from predicators.spot_utils.perception.perception_structs import \
    RGBDImageWithContext
from predicators.spot_utils.perception.spot_cameras import capture_images
from predicators.spot_utils.skills.spot_find_objects import \
    init_search_for_objects
from predicators.spot_utils.skills.spot_navigation import go_home, \
    navigate_to_relative_pose
from predicators.spot_utils.skills.spot_stow_arm import stow_arm
from predicators.spot_utils.spot_localization import SpotLocalizer
from predicators.spot_utils.spot_utils import CAMERA_NAMES, \
    get_spot_interface, obj_name_to_apriltag_id
from predicators.spot_utils.utils import DEFAULT_HAND_LOOK_DOWN_POSE, \
    get_relative_se2_from_se3, get_robot_gripper_open_percentage, \
    verify_estop
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    LiftedAtom, Object, Observation, Predicate, State, STRIPSOperator, Type, \
    Variable

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


# Special actions are ones that are not exposed to the planner. Used by the
# approach wrappper for finding objects.
_SPECIAL_ACTIONS = {
    "find": 0,
    "stow": 1,
    "done": 2,
}


@functools.lru_cache(maxsize=None)
def get_robot() -> Tuple[Robot, SpotLocalizer, LeaseClient]:
    """Create the robot only once."""
    setup_logging(False)
    hostname = CFG.spot_robot_ip
    upload_dir = Path(__file__).parent.parent / "spot_utils" / "graph_nav_maps"
    path = upload_dir / CFG.spot_graph_nav_map
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
        # Special counter variable useful for the special
        # 'find' action.
        self._find_controller_move_queue_idx = 0

    @property
    def _num_operators(self) -> int:
        return len(self._strips_operators)

    @property
    def _max_operator_arity(self) -> int:
        return max(len(o.parameters) for o in self._strips_operators)

    @property
    def _max_controller_params(self) -> int:
        return max(p.shape[0] for p in self.params_spaces.values())

    @property
    def strips_operators(self) -> Set[STRIPSOperator]:
        """Expose the STRIPSOperators for use by oracles."""
        return self._strips_operators

    @property
    def percept_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return set()

    @abc.abstractmethod
    def _get_object_detection_ids(self) -> Set[ObjectDetectionID]:
        raise NotImplementedError

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
        assert len(action.extra_info) == 4
        # The extra info is (action name, objects, function, function args).
        # The action name is either an operator name (for use with nonpercept
        # predicates) or a special name. See below for the special names.

        obs = self._current_observation
        assert isinstance(obs, _SpotObservation)
        assert self.action_space.contains(action.arr)

        action_name = action.extra_info[0]
        operator_names = {o.name for o in self._strips_operators}

        # The action corresponds to an operator finishing.
        if action_name in operator_names:
            # Update the non-percepts.
            operator_names = {o.name for o in self._strips_operators}
            next_nonpercept = self._get_next_nonpercept_atoms(obs, action)
            # NOTE: the observation is only updated after an operator finishes!
            # This assumes options don't really need to be closed-loop. We do
            # this for significant speed-up purposes.
            self._current_observation = self._build_observation(next_nonpercept)
        # The action corresponds to the task finishing.
        elif action_name == "done":
            while True:
                logging.info(f"The goal is: {self._current_task.goal}")
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
        # The action is a real action to be executed.
        # TODO handle finding.
        else:
            assert action_name == "execute"
            assert isinstance(action.extra_info[2], Callable)
            # Execute!
            action.extra_info[2](*action.extra_info[3])
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
        # Get the camera images.
        rgbds = capture_images(self._robot, self._localizer)
        all_detections, _ = detect_objects(self._get_object_detection_ids(),
                                           rgbds)
        # Separately, get detections for the hand in particular.
        hand_rgbd = {
            k: v
            for (k, v) in rgbds.items() if k == "hand_color_image"
        }
        # TODO refactor to avoid extra call to detect!
        hand_detections, _ = detect_objects(self._get_object_detection_ids(),
                                            hand_rgbd)
        # Now construct a dict of all objects in view, as well as a set
        # of objects that the hand can see.
        objects_in_view = {
            self._detection_id_to_obj(det_id): val
            for (det_id, val) in all_detections.items()
        }
        objects_in_hand_view = set(
            self._detection_id_to_obj(det_id)
            for det_id in hand_detections.keys())
        gripper_open_percentage = get_robot_gripper_open_percentage(
            self._robot)
        robot_pos = self._localizer.get_last_robot_pose()
        robot = self._obj_name_to_obj("spot")
        # Prepare the non-percepts.
        nonpercept_preds = self.predicates - self.percept_predicates
        assert all(a.predicate in nonpercept_preds for a in ground_atoms)
        obs = _SpotObservation(all_detections, objects_in_view,
                               objects_in_hand_view, robot,
                               gripper_open_percentage, robot_pos,
                               ground_atoms, nonpercept_preds)

        return obs

    def _get_next_nonpercept_atoms(self, obs: _SpotObservation,
                                   action: Action) -> Set[GroundAtom]:
        """Helper for step().

        This should be deprecated eventually.
        """
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

    def _actively_construct_initial_object_views(
            self) -> Dict[str, math_helpers.SE3Pose]:
        raise NotImplementedError("Subclass must override!")

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for SpotEnv.")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        goal = self._generate_task_goal()  # currently just one goal
        return [
            EnvironmentTask(None, goal) for _ in range(CFG.num_train_tasks)
        ]

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        goal = self._generate_task_goal()  # currently just one goal
        return [EnvironmentTask(None, goal) for _ in range(CFG.num_test_tasks)]

    def _actively_construct_env_task(self) -> EnvironmentTask:
        # Have the spot walk around the environment once to construct
        # an initial observation.
        object_names_in_view = self._actively_construct_initial_object_views()
        objects_in_view = {
            self._obj_name_to_obj(n): v
            for n, v in object_names_in_view.items()
        }
        robot_type = next(t for t in self.types if t.name == "robot")
        robot = Object("spot", robot_type)
        rgb_images = capture_images(self._robot, self._localizer)
        gripper_open_percentage = get_robot_gripper_open_percentage(
            self._robot)
        self._localizer.localize()
        robot_pos = self._localizer.get_last_robot_pose()
        nonpercept_atoms = self._get_initial_nonpercept_atoms()
        nonpercept_preds = self.predicates - self.percept_predicates
        assert all(a.predicate in nonpercept_preds for a in nonpercept_atoms)
        obs = _SpotObservation(rgb_images, objects_in_view, set(), robot,
                               gripper_open_percentage, robot_pos,
                               nonpercept_atoms, nonpercept_preds)
        goal = self._generate_task_goal()
        task = EnvironmentTask(obs, goal)
        # Save the task for future use.
        json_objects = {o.name: o.type.name for o in objects_in_view}
        json_objects[robot.name] = robot.type.name
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
        init_json_dict[robot.name] = {
            "gripper_open_percentage": gripper_open_percentage,
            "curr_held_item_id": 0,
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
            "goal": utils.create_json_dict_from_ground_atoms(goal),
        }
        outfile = utils.get_env_asset_path(
            "task_jsons/spot_bike_env/last.json", assert_exists=False)
        with open(outfile, "w", encoding="utf-8") as f:
            json.dump(json_dict, f, indent=4)
        logging.info(f"Dumped task to {outfile}. Rename it to save it.")
        return task

    @abc.abstractmethod
    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_task_goal(self) -> Set[GroundAtom]:
        raise NotImplementedError

    @abc.abstractmethod
    def _make_object_name_to_obj_and_detectionid_dict(
            self) -> Dict[str, Tuple[Object, ObjectDetectionID]]:
        raise NotImplementedError

    @abc.abstractmethod
    def _obj_name_to_obj(self, obj_name: str) -> Object:
        raise NotImplementedError

    @functools.lru_cache(maxsize=None)
    def _make_detection_id_to_obj_name(self) -> Dict[ObjectDetectionID, str]:
        return {
            vals[1]: obj_name
            for (obj_name, vals) in
            self._make_object_name_to_obj_and_detectionid_dict().items()
        }

    @functools.lru_cache(maxsize=None)
    def _make_detection_id_to_obj(self) -> Dict[ObjectDetectionID, Object]:
        return {
            vals[1]: vals[0]
            for (_, vals) in
            self._make_object_name_to_obj_and_detectionid_dict().items()
        }

    def _make_obj_to_detection_id(self) -> Dict[Object, ObjectDetectionID]:
        return {v: k for (k, v) in self._make_detection_id_to_obj().items()}

    def _detection_id_to_obj_name(self, det_id: ObjectDetectionID) -> str:
        return self._make_detection_id_to_obj_name()[det_id]

    def _detection_id_to_obj(self, det_id: ObjectDetectionID) -> Object:
        return self._make_detection_id_to_obj()[det_id]

    def obj_to_detection_id(self, obj: Object) -> ObjectDetectionID:
        return self._make_obj_to_detection_id()[obj]

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
        images: Dict[str, Image] = {}
        objects_in_view: Dict[Object, Tuple[float, float, float]] = {}
        known_objects = set(self._make_object_name_to_obj_dict().values())
        robot: Optional[Object] = None
        for obj in init:
            assert obj in known_objects
            if obj.name == "spot":
                robot = obj
                continue
            pos = (init.get(obj, "x"), init.get(obj, "y"), init.get(obj, "z"))
            objects_in_view[obj] = pos
        assert robot is not None
        gripper_open_percentage = init.get(robot, "gripper_open_percentage")
        robot_pos = (init.get(robot, "x"), init.get(robot, "y"),
                     init.get(robot, "z"), init.get(robot, "yaw"))
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
        goal = base_env_task.goal
        return EnvironmentTask(init_obs, goal)


###############################################################################
#                                Cube Table Env                               #
###############################################################################
HANDEMPTY_GRIPPER_THRESHOLD = 2.5


class SpotCubeEnv(SpotEnv):
    """An environment corresponding to the spot cube task where the robot
    attempts to place an April Tag cube onto a particular table."""

    _ontop_threshold: ClassVar[float] = 0.55
    _reachable_threshold: ClassVar[float] = 1.7
    _bucket_center_offset_x: ClassVar[float] = 0.0
    _bucket_center_offset_y: ClassVar[float] = -0.15
    _inbag_threshold: ClassVar[float] = 0.25
    _reachable_yaw_threshold: ClassVar[float] = 0.95  # higher better
    _handempty_gripper_threshold: ClassVar[float] = HANDEMPTY_GRIPPER_THRESHOLD
    _robot_on_platform_threshold: ClassVar[float] = 0.18
    _surface_too_high_threshold: ClassVar[float] = 0.7
    _ontop_max_height_threshold: ClassVar[float] = 0.25

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", [
            "gripper_open_percentage", "curr_held_item_id", "x", "y", "z",
            "W_quat", "X_quat", "Y_quat", "Z_quat"
        ])
        self._tool_type = Type("tool", [
            "x", "y", "z", "W_quat", "X_quat", "Y_quat", "Z_quat", "lost",
            "in_view"
        ])
        self._surface_type = Type(
            "flat_surface",
            ["x", "y", "z", "W_quat", "X_quat", "Y_quat", "Z_quat"])
        self._bag_type = Type(
            "bag", ["x", "y", "z", "W_quat", "X_quat", "Y_quat", "Z_quat"])
        self._floor_type = Type(
            "floor", ["x", "y", "z", "W_quat", "X_quat", "Y_quat", "Z_quat"])

        # Predicates
        # Note that all classifiers assigned here just directly use
        # the ground atoms from the low-level simulator state.
        self._On = Predicate("On", [self._tool_type, self._surface_type],
                             self._ontop_classifier)
        self._OnFloor = Predicate("OnFloor",
                                  [self._tool_type, self._floor_type],
                                  self._onfloor_classifier)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._handempty_classifier)
        self._notHandEmpty = Predicate("Not-HandEmpty", [self._robot_type],
                                       self._nothandempty_classifier)
        self._HoldingTool = Predicate("HoldingTool",
                                      [self._robot_type, self._tool_type],
                                      self._holding_tool_classifier)
        self._InViewTool = Predicate("InViewTool",
                                     [self._robot_type, self._tool_type],
                                     self._tool_in_view_classifier)
        self._ReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            self._reachable_classifier)
        self._InBag = Predicate("InBag", [self._tool_type, self._bag_type],
                                self._inbag_classifier)

        # STRIPS Operators (needed for option creation)
        # MoveToToolOnSurface
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        surface = Variable("?surface", self._surface_type)
        preconditions = {
            LiftedAtom(self._On, [tool, surface]),
        }
        add_effs = {LiftedAtom(self._InViewTool, [spot, tool])}
        ignore_effs = {self._ReachableSurface, self._InViewTool}
        self._MoveToToolOnSurfaceOp = STRIPSOperator("MoveToToolOnSurface",
                                                     [spot, tool, surface],
                                                     preconditions, add_effs,
                                                     set(), ignore_effs)
        # MoveToToolOnFloor
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        floor = Variable("?floor", self._floor_type)
        preconditions = {LiftedAtom(self._OnFloor, [tool, floor])}
        add_effs = {LiftedAtom(self._InViewTool, [spot, tool])}
        ignore_effs = {self._ReachableSurface, self._InViewTool}
        self._MoveToToolOnFloorOp = STRIPSOperator("MoveToToolOnFloor",
                                                   [spot, tool, floor],
                                                   preconditions, add_effs,
                                                   set(), ignore_effs)
        # MoveToSurface
        spot = Variable("?robot", self._robot_type)
        surface = Variable("?surface", self._surface_type)
        preconditions = set()
        add_effs = {LiftedAtom(self._ReachableSurface, [spot, surface])}
        ignore_effs = {self._ReachableSurface, self._InViewTool}
        self._MoveToSurfaceOp = STRIPSOperator("MoveToSurface",
                                               [spot, surface], preconditions,
                                               add_effs, set(), ignore_effs)
        # GraspToolFromSurface
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        add_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        self._GraspToolFromSurfaceOp = STRIPSOperator("GraspToolFromSurface",
                                                      [spot, tool, surface],
                                                      preconds, add_effs,
                                                      del_effs, set())
        # GraspToolFromFloor
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        floor = Variable("?floor", self._floor_type)
        preconds = {
            LiftedAtom(self._OnFloor, [tool, floor]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        add_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._OnFloor, [tool, floor]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        self._GraspToolFromFloorOp = STRIPSOperator("GraspToolFromFloor",
                                                    [spot, tool, floor],
                                                    preconds, add_effs,
                                                    del_effs, set())
        # Rpplae
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._ReachableSurface, [spot, surface]),
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        add_effs = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot]),
            LiftedAtom(self._ReachableSurface, [spot, surface]),
        }
        self._PlaceToolOnSurfaceOp = STRIPSOperator("PlaceToolOnSurface",
                                                    [spot, tool, surface],
                                                    preconds, add_effs,
                                                    del_effs, set())
        # PlaceToolOnFloor
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        floor = Variable("?floor", self._floor_type)
        preconds = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot]),
        }
        add_effs = {
            LiftedAtom(self._OnFloor, [tool, floor]),
        }
        del_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot]),
        }
        self._PlaceToolOnFloorOp = STRIPSOperator("PlaceToolOnFloor",
                                                  [spot, tool, floor],
                                                  preconds, add_effs, del_effs,
                                                  set())

        self._strips_operators = {
            self._MoveToToolOnSurfaceOp,
            self._MoveToToolOnFloorOp,
            self._MoveToSurfaceOp,
            self._GraspToolFromSurfaceOp,
            self._GraspToolFromFloorOp,
            self._PlaceToolOnSurfaceOp,
            self._PlaceToolOnFloorOp,
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._tool_type, self._surface_type,
            self._floor_type
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._HandEmpty, self._HoldingTool,
            self._ReachableSurface, self._notHandEmpty, self._InViewTool,
            self._OnFloor
        }

    @classmethod
    def _handempty_classifier(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        spot = objects[0]
        gripper_open_percentage = state.get(spot, "gripper_open_percentage")
        return gripper_open_percentage <= cls._handempty_gripper_threshold

    @classmethod
    def _nothandempty_classifier(cls, state: State,
                                 objects: Sequence[Object]) -> bool:
        return not cls._handempty_classifier(state, objects)

    @classmethod
    def _holding_tool_classifier(cls, state: State,
                                 objects: Sequence[Object]) -> bool:
        spot, obj_to_grasp = objects
        assert obj_name_to_apriltag_id.get(obj_to_grasp.name) is not None
        spot_holding_obj_id = state.get(spot, "curr_held_item_id")
        return int(spot_holding_obj_id) == obj_name_to_apriltag_id[
            obj_to_grasp.name] and cls._nothandempty_classifier(state, [spot])

    @classmethod
    def _ontop_classifier(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        obj_on, obj_surface = objects
        assert obj_name_to_apriltag_id.get(obj_on.name) is not None
        assert obj_name_to_apriltag_id.get(obj_surface.name) is not None

        spot = [o for o in state if o.type.name == "robot"][0]
        if cls._holding_tool_classifier(state, [spot, obj_on]):
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
            (obj_on_pose[0] - obj_surface_pose[0])**2) <= cls._ontop_threshold
        is_y_same = np.sqrt(
            (obj_on_pose[1] - obj_surface_pose[1])**2) <= cls._ontop_threshold
        is_above_z = 0.0 < (obj_on_pose[2] - obj_surface_pose[2]
                            ) < cls._ontop_max_height_threshold
        return is_x_same and is_y_same and is_above_z

    @staticmethod
    def _onfloor_classifier(state: State, objects: Sequence[Object]) -> bool:
        obj_on, _ = objects
        return state.get(obj_on, "z") < 0.0

    @classmethod
    def _inbag_classifier(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        obj, bag = objects
        obj_x = state.get(obj, "x")
        obj_y = state.get(obj, "y")
        bag_x = state.get(bag, "x") + cls._bucket_center_offset_x
        bag_y = state.get(bag, "y") + cls._bucket_center_offset_y
        dist = np.sqrt((obj_x - bag_x)**2 + (obj_y - bag_y)**2)
        return dist <= cls._inbag_threshold

    @classmethod
    def _reachable_classifier(cls, state: State,
                              objects: Sequence[Object]) -> bool:
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
        obj_pose = [
            state.get(obj, "x"),
            state.get(obj, "y"),
            state.get(obj, "z")
        ]
        is_xy_near = np.sqrt(
            (spot_pose[0] - obj_pose[0])**2 +
            (spot_pose[1] - obj_pose[1])**2) <= cls._reachable_threshold

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
        is_yaw_near = abs(
            angle_between_robot_and_obj) < cls._reachable_yaw_threshold

        return is_xy_near and is_yaw_near

    @staticmethod
    def _tool_in_view_classifier(state: State,
                                 objects: Sequence[Object]) -> bool:
        _, tool = objects
        return state.get(tool, "in_view") > 0.5

    @classmethod
    def get_name(cls) -> str:
        return "spot_cube_env"

    @property
    def percept_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return {
            self._HandEmpty, self._notHandEmpty, self._HoldingTool, self._On,
            self._InViewTool, self._OnFloor
        }

    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        return set()

    def _generate_task_goal(self) -> Set[GroundAtom]:
        cube = self._obj_name_to_obj("cube")
        extra_table = self._obj_name_to_obj("extra_room_table")
        return {GroundAtom(self._On, [cube, extra_table])}

    @functools.lru_cache(maxsize=None)
    def _make_object_name_to_obj_and_detectionid_dict(
            self) -> Dict[str, Tuple[Object, ObjectDetectionID]]:

        objects_and_detections: List[Tuple[Object, ObjectDetectionID]] = []
        # Initialize all objects to detections with default
        # values.
        cube = Object("cube", self._tool_type)
        cube_detection = AprilTagObjectDetectionID(
            410, math_helpers.SE3Pose(0.0, 0.0, 0.0, math_helpers.Quat()))
        objects_and_detections.append((cube, cube_detection))
        spot = Object("spot", self._robot_type)
        tool_room_table = Object("tool_room_table", self._surface_type)
        tool_room_table_detection = AprilTagObjectDetectionID(
            408, math_helpers.SE3Pose(0.0, 0.25, 0.0, math_helpers.Quat()))
        extra_room_table = Object("extra_room_table", self._surface_type)
        extra_room_table_detection = AprilTagObjectDetectionID(
            409, math_helpers.SE3Pose(0.0, 0.25, 0.0, math_helpers.Quat()))
        floor = Object("floor", self._floor_type)
        # Spot and the floor get initialized with non-existent detection IDs.
        spot_detection = AprilTagObjectDetectionID(
            -1, math_helpers.SE3Pose(0.0, 0.0, 0.0, math_helpers.Quat()))
        floor_detection = AprilTagObjectDetectionID(
            -2, math_helpers.SE3Pose(0.0, 0.0, 0.0, math_helpers.Quat()))
        objects_and_detections.extend([
            (spot, spot_detection),
            (tool_room_table, tool_room_table_detection),
            (extra_room_table, extra_room_table_detection),
            (floor, floor_detection)
        ])
        return {o.name: (o, d) for o, d in objects_and_detections}

    def _obj_name_to_obj(self, obj_name: str) -> Object:
        return self._make_object_name_to_obj_and_detectionid_dict(
        )[obj_name][0]

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.predicates

    @functools.lru_cache(maxsize=None)
    def _get_object_detection_ids(self) -> Set[ObjectDetectionID]:
        """Useful helper function for getting the object detection ids we might
        want to detect to pass to our vision pipeline."""
        obj_names = set(
            self._make_object_name_to_obj_and_detectionid_dict().keys())
        obj_names.remove("spot")
        obj_names.remove("floor")
        return set(self._make_object_name_to_obj_and_detectionid_dict()[o][1]
                   for o in obj_names)

    def _actively_construct_initial_object_views(
            self) -> Dict[str, math_helpers.SE3Pose]:
        stow_arm(self._robot)
        go_home(self._robot, self._localizer)
        self._localizer.localize()
        detections, _ = init_search_for_objects(
            self._robot, self._localizer, self._get_object_detection_ids())
        obj_name_to_se3_pose = {
            self._detection_id_to_obj_name(det_id): val
            for (det_id, val) in detections.items()
        }
        return obj_name_to_se3_pose


###############################################################################
#                                Bike Repair Env                              #
###############################################################################


class SpotBikeEnv(SpotEnv):
    """An environment containing bike-repair related tasks for a real Spot
    robot to execute."""

    _ontop_threshold: ClassVar[float] = 0.55
    _reachable_threshold: ClassVar[float] = 1.7
    _bucket_center_offset_x: ClassVar[float] = 0.0
    _bucket_center_offset_y: ClassVar[float] = -0.15
    _inbag_threshold: ClassVar[float] = 0.25
    _reachable_yaw_threshold: ClassVar[float] = 0.95  # higher better
    _handempty_gripper_threshold: ClassVar[float] = HANDEMPTY_GRIPPER_THRESHOLD
    _robot_on_platform_threshold: ClassVar[float] = 0.18
    _surface_too_high_threshold: ClassVar[float] = 0.7
    _ontop_max_height_threshold: ClassVar[float] = 0.25

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", [
            "gripper_open_percentage", "curr_held_item_id", "x", "y", "z",
            "yaw"
        ])
        self._tool_type = Type("tool", ["x", "y", "z", "lost", "in_view"])
        self._surface_type = Type("flat_surface", ["x", "y", "z"])
        self._bag_type = Type("bag", ["x", "y", "z"])
        self._platform_type = Type("platform",
                                   ["x", "y", "z", "lost", "in_view"])
        self._floor_type = Type("floor", ["x", "y", "z"])

        # Predicates
        # Note that all classifiers assigned here just directly use
        # the ground atoms from the low-level simulator state.
        self._On = Predicate("On", [self._tool_type, self._surface_type],
                             self._ontop_classifier)
        self._OnFloor = Predicate("OnFloor",
                                  [self._tool_type, self._floor_type],
                                  self._onfloor_classifier)
        self._InBag = Predicate("InBag", [self._tool_type, self._bag_type],
                                self._inbag_classifier)
        self._HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                    self._handempty_classifier)
        self._notHandEmpty = Predicate("Not-HandEmpty", [self._robot_type],
                                       self._nothandempty_classifier)
        self._HoldingTool = Predicate("HoldingTool",
                                      [self._robot_type, self._tool_type],
                                      self._holding_tool_classifier)
        self._temp_HoldingBag = Predicate("HoldingBag",
                                          [self._robot_type, self._bag_type],
                                          lambda s, o: False)
        self._HoldingBag = Predicate(
            "HoldingBag", [self._robot_type, self._bag_type],
            _create_dummy_predicate_classifier(self._temp_HoldingBag))
        self._temp_HoldingPlatformLeash = Predicate(
            "HoldingPlatformLeash", [self._robot_type, self._platform_type],
            lambda s, o: False)
        self._HoldingPlatformLeash = Predicate(
            "HoldingPlatformLeash", [self._robot_type, self._platform_type],
            _create_dummy_predicate_classifier(
                self._temp_HoldingPlatformLeash))
        self._InViewTool = Predicate("InViewTool",
                                     [self._robot_type, self._tool_type],
                                     self._tool_in_view_classifier)
        self._InViewPlatform = Predicate(
            "InViewPlatform", [self._robot_type, self._platform_type],
            self._platform_in_view_classifier)
        self._ReachableBag = Predicate("ReachableBag",
                                       [self._robot_type, self._bag_type],
                                       self._reachable_classifier)
        self._ReachablePlatform = Predicate(
            "ReachablePlatform", [self._robot_type, self._platform_type],
            self._reachable_classifier)
        self._ReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            self._reachable_classifier)
        self._SurfaceTooHigh = Predicate(
            "SurfaceTooHigh", [self._robot_type, self._surface_type],
            self._surface_too_high_classifier)
        self._SurfaceNotTooHigh = Predicate(
            "SurfaceNotTooHigh", [self._robot_type, self._surface_type],
            self._surface_not_too_high_classifier)
        self._PlatformNear = Predicate(
            "PlatformNear", [self._platform_type, self._surface_type],
            self._platform_is_near)
        self._RobotStandingOnPlatform = Predicate(
            "RobotOnPlatform", [self._robot_type, self._platform_type],
            self._robot_on_platform_classifier)

        # STRIPS Operators (needed for option creation)
        # MoveToToolOnSurfaceNotHigh
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        surface = Variable("?surface", self._surface_type)
        preconditions = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._SurfaceNotTooHigh, [spot, surface])
        }
        add_effs = {LiftedAtom(self._InViewTool, [spot, tool])}
        ignore_effs = {
            self._ReachableBag, self._ReachableSurface,
            self._ReachablePlatform, self._InViewTool, self._InViewPlatform
        }
        self._MoveToToolOnSurfaceNotHighOp = STRIPSOperator(
            "MoveToToolOnSurfaceNotHigh", [spot, tool, surface], preconditions,
            add_effs, set(), ignore_effs)
        # MoveToToolOnSurfaceTooHigh
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        surface = Variable("?surface", self._surface_type)
        platform = Variable("?platform", self._platform_type)
        preconditions = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._PlatformNear, [platform, surface]),
            LiftedAtom(self._SurfaceTooHigh, [spot, surface])
        }
        add_effs = {
            LiftedAtom(self._InViewTool, [spot, tool]),
            LiftedAtom(self._RobotStandingOnPlatform, [spot, platform])
        }
        ignore_effs = {
            self._ReachableBag, self._ReachableSurface,
            self._ReachablePlatform, self._InViewTool, self._InViewPlatform
        }
        self._MoveToToolOnSurfaceTooHighOp = STRIPSOperator(
            "MoveToToolOnSurfaceTooHigh", [spot, tool, surface, platform],
            preconditions, add_effs, set(), ignore_effs)
        # MoveToToolOnFloor
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        floor = Variable("?floor", self._floor_type)
        preconditions = {LiftedAtom(self._OnFloor, [tool, floor])}
        add_effs = {LiftedAtom(self._InViewTool, [spot, tool])}
        ignore_effs = {
            self._ReachableBag, self._ReachableSurface,
            self._ReachablePlatform, self._InViewTool, self._InViewPlatform
        }
        self._MoveToToolOnFloorOp = STRIPSOperator("MoveToToolOnFloor",
                                                   [spot, tool, floor],
                                                   preconditions, add_effs,
                                                   set(), ignore_effs)
        # MoveToSurfaceNotHigh
        spot = Variable("?robot", self._robot_type)
        surface = Variable("?surface", self._surface_type)
        preconditions = {LiftedAtom(self._SurfaceNotTooHigh, [spot, surface])}
        add_effs = {LiftedAtom(self._ReachableSurface, [spot, surface])}
        ignore_effs = {
            self._ReachableBag, self._ReachableSurface,
            self._ReachablePlatform, self._InViewTool, self._InViewPlatform
        }
        self._MoveToSurfaceNotHighOp = STRIPSOperator("MoveToSurfaceNotHigh",
                                                      [spot, surface],
                                                      preconditions, add_effs,
                                                      set(), ignore_effs)
        # MoveToSurfaceTooHigh
        spot = Variable("?robot", self._robot_type)
        surface = Variable("?surface", self._surface_type)
        platform = Variable("?platform", self._platform_type)
        preconditions = {
            LiftedAtom(self._PlatformNear, [platform, surface]),
            LiftedAtom(self._SurfaceTooHigh, [spot, surface])
        }
        add_effs = {
            LiftedAtom(self._ReachableSurface, [spot, surface]),
            LiftedAtom(self._RobotStandingOnPlatform, [spot, platform])
        }
        ignore_effs = {
            self._ReachableBag, self._ReachableSurface,
            self._ReachablePlatform, self._InViewTool, self._InViewPlatform
        }
        self._MoveToSurfaceTooHighOp = STRIPSOperator(
            "MoveToSurfaceTooHigh", [spot, surface, platform], preconditions,
            add_effs, set(), ignore_effs)
        # MoveToPlatform
        spot = Variable("?robot", self._robot_type)
        platform = Variable("?platform", self._platform_type)
        add_effs = {
            LiftedAtom(self._ReachablePlatform, [spot, platform]),
            LiftedAtom(self._InViewPlatform, [spot, platform])
        }
        ignore_effs = {
            self._ReachableBag, self._ReachableSurface,
            self._ReachablePlatform, self._InViewTool, self._InViewPlatform
        }
        self._MoveToPlatformOp = STRIPSOperator("MoveToPlatform",
                                                [spot, platform], set(),
                                                add_effs, set(), ignore_effs)
        # MoveToBag
        spot = Variable("?robot", self._robot_type)
        bag = Variable("?platform", self._bag_type)
        add_effs = {LiftedAtom(self._ReachableBag, [spot, bag])}
        ignore_effs = {
            self._ReachableBag, self._ReachableSurface,
            self._ReachablePlatform, self._InViewTool, self._InViewPlatform
        }
        self._MoveToBagOp = STRIPSOperator("MoveToBag", [spot, bag], set(),
                                           add_effs, set(), ignore_effs)
        # GraspToolFromNotHigh
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._SurfaceNotTooHigh, [spot, surface]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        add_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        self._GraspToolFromNotHighOp = STRIPSOperator("GraspToolFromNotHigh",
                                                      [spot, tool, surface],
                                                      preconds, add_effs,
                                                      del_effs, set())
        # GraspToolFromFloor
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        floor = Variable("?floor", self._floor_type)
        preconds = {
            LiftedAtom(self._OnFloor, [tool, floor]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        add_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._OnFloor, [tool, floor]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        self._GraspToolFromFloorOp = STRIPSOperator("GraspToolFromFloor",
                                                    [spot, tool, floor],
                                                    preconds, add_effs,
                                                    del_effs, set())
        # GrabPlatformLeash
        spot = Variable("?robot", self._robot_type)
        platform = Variable("?platform", self._platform_type)
        preconds = {
            LiftedAtom(self._ReachablePlatform, [spot, platform]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewPlatform, [spot, platform])
        }
        add_effs = {
            LiftedAtom(self._HoldingPlatformLeash, [spot, platform]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._ReachablePlatform, [spot, platform]),
            LiftedAtom(self._InViewPlatform, [spot, platform])
        }
        self._GraspPlatformLeashOp = STRIPSOperator("GraspPlatformLeash",
                                                    [spot, platform], preconds,
                                                    add_effs, del_effs, set())
        # DragPlatform
        spot = Variable("?robot", self._robot_type)
        platform = Variable("?platform", self._platform_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._HoldingPlatformLeash, [spot, platform]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        add_effs = {
            LiftedAtom(self._PlatformNear, [platform, surface]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._HoldingPlatformLeash, [spot, platform]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        self._DragPlatformOp = STRIPSOperator("DragPlatform",
                                              [spot, platform, surface],
                                              preconds, add_effs, del_effs,
                                              set())
        # GraspToolFromHigh
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        platform = Variable("?platform", self._platform_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._SurfaceTooHigh, [spot, surface]),
            LiftedAtom(self._PlatformNear, [platform, surface]),
            LiftedAtom(self._InViewTool, [spot, tool]),
            LiftedAtom(self._RobotStandingOnPlatform, [spot, platform])
        }
        add_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._InViewTool, [spot, tool])
        }
        self._GraspToolFromHighOp = STRIPSOperator(
            "GraspToolFromHigh", [spot, tool, surface, platform], preconds,
            add_effs, del_effs, set())
        # GraspBag
        spot = Variable("?robot", self._robot_type)
        bag = Variable("?bag", self._bag_type)
        preconds = {
            LiftedAtom(self._ReachableBag, [spot, bag]),
            LiftedAtom(self._HandEmpty, [spot]),
        }
        add_effs = {
            LiftedAtom(self._HoldingBag, [spot, bag]),
            LiftedAtom(self._notHandEmpty, [spot]),
        }
        del_effs = {
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._ReachableBag, [spot, bag]),
        }
        self._GraspBagOp = STRIPSOperator("GraspBag", [spot, bag], preconds,
                                          add_effs, del_effs, set())
        # PlaceToolOnNotHighSurface
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._ReachableSurface, [spot, surface]),
            LiftedAtom(self._SurfaceNotTooHigh, [spot, surface]),
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        add_effs = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot]),
            LiftedAtom(self._ReachableSurface, [spot, surface]),
        }
        self._PlaceToolNotHighOp = STRIPSOperator("PlaceToolNotHigh",
                                                  [spot, tool, surface],
                                                  preconds, add_effs, del_effs,
                                                  set())
        # PlaceToolOnFloor
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        floor = Variable("?floor", self._floor_type)
        preconds = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot]),
        }
        add_effs = {
            LiftedAtom(self._OnFloor, [tool, floor]),
        }
        del_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot]),
        }
        self._PlaceToolOnFloorOp = STRIPSOperator("PlaceToolOnFloor",
                                                  [spot, tool, floor],
                                                  preconds, add_effs, del_effs,
                                                  set())

        # PlaceIntoBag
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        bag = Variable("?bag", self._bag_type)
        preconds = {
            LiftedAtom(self._ReachableBag, [spot, bag]),
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        add_effs = {
            LiftedAtom(self._InBag, [tool, bag]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot]),
            LiftedAtom(self._ReachableBag, [spot, bag]),
        }
        self._PlaceIntoBagOp = STRIPSOperator("PlaceIntoBag",
                                              [spot, tool, bag], preconds,
                                              add_effs, del_effs, set())

        self._strips_operators = {
            self._MoveToToolOnSurfaceNotHighOp,
            self._MoveToToolOnSurfaceTooHighOp,
            self._MoveToSurfaceNotHighOp,
            self._MoveToSurfaceTooHighOp,
            self._MoveToPlatformOp,
            self._MoveToBagOp,
            self._GraspToolFromNotHighOp,
            self._GraspPlatformLeashOp,
            self._DragPlatformOp,
            self._GraspToolFromHighOp,
            self._GraspBagOp,
            self._PlaceToolNotHighOp,
            self._PlaceToolOnFloorOp,
            self._PlaceIntoBagOp,
            self._MoveToToolOnFloorOp,
            self._GraspToolFromFloorOp,
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._tool_type, self._surface_type,
            self._bag_type, self._platform_type, self._floor_type
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._InBag, self._HandEmpty, self._HoldingTool,
            self._HoldingBag, self._HoldingPlatformLeash, self._ReachableBag,
            self._ReachablePlatform, self._ReachableSurface,
            self._SurfaceTooHigh, self._SurfaceNotTooHigh, self._PlatformNear,
            self._notHandEmpty, self._InViewTool, self._InViewPlatform,
            self._OnFloor, self._RobotStandingOnPlatform
        }

    @classmethod
    def _handempty_classifier(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        spot = objects[0]
        gripper_open_percentage = state.get(spot, "gripper_open_percentage")
        return gripper_open_percentage <= cls._handempty_gripper_threshold

    @classmethod
    def _nothandempty_classifier(cls, state: State,
                                 objects: Sequence[Object]) -> bool:
        return not cls._handempty_classifier(state, objects)

    @classmethod
    def _holding_tool_classifier(cls, state: State,
                                 objects: Sequence[Object]) -> bool:
        spot, obj_to_grasp = objects
        assert obj_name_to_apriltag_id.get(obj_to_grasp.name) is not None
        spot_holding_obj_id = state.get(spot, "curr_held_item_id")
        return int(spot_holding_obj_id) == obj_name_to_apriltag_id[
            obj_to_grasp.name] and cls._nothandempty_classifier(state, [spot])

    @classmethod
    def _ontop_classifier(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        obj_on, obj_surface = objects
        assert obj_name_to_apriltag_id.get(obj_on.name) is not None
        assert obj_name_to_apriltag_id.get(obj_surface.name) is not None

        spot = [o for o in state if o.type.name == "robot"][0]
        if cls._holding_tool_classifier(state, [spot, obj_on]):
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
            (obj_on_pose[0] - obj_surface_pose[0])**2) <= cls._ontop_threshold
        is_y_same = np.sqrt(
            (obj_on_pose[1] - obj_surface_pose[1])**2) <= cls._ontop_threshold
        is_above_z = 0.0 < (obj_on_pose[2] - obj_surface_pose[2]
                            ) < cls._ontop_max_height_threshold
        return is_x_same and is_y_same and is_above_z

    @staticmethod
    def _onfloor_classifier(state: State, objects: Sequence[Object]) -> bool:
        obj_on, _ = objects
        return state.get(obj_on, "z") < 0.0

    @classmethod
    def _inbag_classifier(cls, state: State,
                          objects: Sequence[Object]) -> bool:
        obj, bag = objects
        obj_x = state.get(obj, "x")
        obj_y = state.get(obj, "y")
        bag_x = state.get(bag, "x") + cls._bucket_center_offset_x
        bag_y = state.get(bag, "y") + cls._bucket_center_offset_y
        dist = np.sqrt((obj_x - bag_x)**2 + (obj_y - bag_y)**2)
        return dist <= cls._inbag_threshold

    @classmethod
    def _reachable_classifier(cls, state: State,
                              objects: Sequence[Object]) -> bool:
        spot, obj = objects
        spot_pose = [
            state.get(spot, "x"),
            state.get(spot, "y"),
            state.get(spot, "z"),
            state.get(spot, "yaw")
        ]
        obj_pose = [
            state.get(obj, "x"),
            state.get(obj, "y"),
            state.get(obj, "z")
        ]
        is_xy_near = np.sqrt(
            (spot_pose[0] - obj_pose[0])**2 +
            (spot_pose[1] - obj_pose[1])**2) <= cls._reachable_threshold

        # Compute angle between spot's forward direction and the line from
        # spot to the object.
        forward_unit = [np.cos(spot_pose[3]), np.sin(spot_pose[3])]
        spot_to_obj = np.subtract(obj_pose[:2], spot_pose[:2])
        spot_to_obj_unit = spot_to_obj / np.linalg.norm(spot_to_obj)
        angle_between_robot_and_obj = np.arccos(
            np.dot(forward_unit, spot_to_obj_unit))
        is_yaw_near = abs(
            angle_between_robot_and_obj) < cls._reachable_yaw_threshold

        return is_xy_near and is_yaw_near

    @classmethod
    def _surface_too_high_classifier(cls, state: State,
                                     objects: Sequence[Object]) -> bool:
        _, surface = objects
        return state.get(surface, "z") > cls._surface_too_high_threshold

    @classmethod
    def _surface_not_too_high_classifier(cls, state: State,
                                         objects: Sequence[Object]) -> bool:
        return not cls._surface_too_high_classifier(state, objects)

    @staticmethod
    def _tool_in_view_classifier(state: State,
                                 objects: Sequence[Object]) -> bool:
        _, tool = objects
        return state.get(tool, "in_view") > 0.5

    @staticmethod
    def _platform_in_view_classifier(state: State,
                                     objects: Sequence[Object]) -> bool:
        _, platform = objects
        return state.get(platform, "in_view") > 0.5

    @staticmethod
    def _platform_is_near(state: State, objects: Sequence[Object]) -> bool:
        platform, surface = objects
        px = state.get(platform, "x")
        py = state.get(platform, "y")
        sx = state.get(surface, "x")
        sy = state.get(surface, "y")
        return abs(px - sx) < 1.25 and abs(py - sy) < 0.85

    @classmethod
    def _robot_on_platform_classifier(cls, state: State,
                                      objects: Sequence[Object]) -> bool:
        robot, _ = objects
        rz = state.get(robot, "z")
        return rz > cls._robot_on_platform_threshold

    @classmethod
    def get_name(cls) -> str:
        return "spot_bike_env"

    @property
    def percept_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return {
            self._HandEmpty, self._notHandEmpty, self._HoldingTool, self._On,
            self._SurfaceTooHigh, self._SurfaceNotTooHigh, self._ReachableBag,
            self._ReachablePlatform, self._InViewTool, self._InViewPlatform,
            self._RobotStandingOnPlatform, self._InBag, self._OnFloor
        }

    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        return set()

    def _generate_task_goal(self) -> Set[GroundAtom]:
        if CFG.spot_cube_only:
            cube = self._obj_name_to_obj("cube")
            extra_table = self._obj_name_to_obj("extra_room_table")
            return {GroundAtom(self._On, [cube, extra_table])}
        if CFG.spot_platform_only:
            platform = self._obj_name_to_obj("platform")
            high_wall_rack = self._obj_name_to_obj("high_wall_rack")
            hammer = self._obj_name_to_obj("hammer")
            bucket = self._obj_name_to_obj("bucket")
            return {
                GroundAtom(self._PlatformNear, [platform, high_wall_rack]),
                GroundAtom(self._InBag, [hammer, bucket])
            }
        hammer = self._obj_name_to_obj("hammer")
        measuring_tape = self._obj_name_to_obj("measuring_tape")
        brush = self._obj_name_to_obj("brush")
        bag = self._obj_name_to_obj("bucket")
        return {
            GroundAtom(self._InBag, [hammer, bag]),
            GroundAtom(self._InBag, [brush, bag]),
            GroundAtom(self._InBag, [measuring_tape, bag]),
        }

    @functools.lru_cache(maxsize=None)
    def _make_object_name_to_obj_dict(self) -> Dict[str, Object]:
        objects: List[Object] = []
        if CFG.spot_cube_only:
            cube = Object("cube", self._tool_type)
            objects.append(cube)
        if CFG.spot_platform_only:
            platform = Object("platform", self._platform_type)
            hammer = Object("hammer", self._tool_type)
            objects.extend([platform, hammer])
        else:
            hammer = Object("hammer", self._tool_type)
            measuring_tape = Object("measuring_tape", self._tool_type)
            brush = Object("brush", self._tool_type)
            platform = Object("platform", self._platform_type)
            objects.extend([hammer, measuring_tape, brush, platform])
        spot = Object("spot", self._robot_type)
        tool_room_table = Object("tool_room_table", self._surface_type)
        extra_room_table = Object("extra_room_table", self._surface_type)
        low_wall_rack = Object("low_wall_rack", self._surface_type)
        high_wall_rack = Object("high_wall_rack", self._surface_type)
        bag = Object("bucket", self._bag_type)
        floor = Object("floor", self._floor_type)
        objects.extend([
            spot, tool_room_table, low_wall_rack, high_wall_rack, bag,
            extra_room_table, floor
        ])
        return {o.name: o for o in objects}

    def _obj_name_to_obj(self, obj_name: str) -> Object:
        return self._make_object_name_to_obj_dict()[obj_name]

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.predicates

    def _actively_construct_initial_object_views(
            self) -> Dict[str, math_helpers.SE3Pose]:
        obj_names = set(self._make_object_name_to_obj_dict().keys())
        obj_names.remove("spot")
        return self._spot_interface.actively_construct_initial_object_views(
            obj_names)
