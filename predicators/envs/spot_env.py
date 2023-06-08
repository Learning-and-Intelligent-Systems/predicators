"""Basic environment for the Boston Dynamics Spot Robot."""

import abc
import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Collection, Dict, List, Optional, Sequence, Set, \
    Tuple

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.pddl_env import _action_to_ground_strips_op
from predicators.settings import CFG
from predicators.spot_utils.spot_utils import apriltag_id_to_obj_poses, \
    get_spot_interface, obj_name_to_apriltag_id
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    Image, LiftedAtom, Object, Observation, Predicate, State, STRIPSOperator, \
    Type, Variable

###############################################################################
#                                Base Class                                   #
###############################################################################


@dataclass(frozen=True)
class _SpotObservation:
    """An observation for a SpotEnv."""
    # Camera name to image
    images: Dict[str, Image]
    # Objects that are seen in the current image and their positions in world
    objects_in_view: Dict[Object, Tuple[float, float, float]]
    # Status of the robot gripper.
    gripper_open_percentage: float
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


class SpotEnv(BaseEnv):
    """An environment containing tasks for a real Spot robot to execute.

    This is a base class that specific sub-classes that define actual
    tasks should inherit from.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._spot_interface = get_spot_interface()
        # Note that we need to include the operators in this
        # class because they're used to update the symbolic
        # parts of the state during execution.
        self._strips_operators: Set[STRIPSOperator] = set()

    @property
    def _ordered_strips_operators(self) -> List[STRIPSOperator]:
        return sorted(self._strips_operators)

    @property
    def _num_operators(self) -> int:
        return len(self._strips_operators)

    @property
    def _max_operator_arity(self) -> int:
        return max(len(o.parameters) for o in self._strips_operators)

    @property
    def _max_controller_params(self) -> int:
        return max(p.shape[0]
                   for p in self._spot_interface.params_spaces.values())

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
        # The first entry is the controller identity.
        lb = [0.0]
        ub = [self._num_operators - 1.0]
        # The next max_arity entries are the object identities.
        for _ in range(self._max_operator_arity):
            lb.append(0.0)
            ub.append(np.inf)
        # The next max_params entries are the parameters.
        for _ in range(self._max_controller_params):
            lb.append(-np.inf)
            ub.append(np.inf)
        lb_arr = np.array(lb, dtype=np.float32)
        ub_arr = np.array(ub, dtype=np.float32)
        return Box(lb_arr, ub_arr, dtype=np.float32)

    def parse_action(self, action: Action, ordered_objs: List[Object]) -> Tuple[str, List[Object], Array]:
        """(Only for this environment) A convenience method that converts low-
        level actions into more interpretable high-level actions by exploiting
        knowledge of how we encode actions."""
        # Convert the first action part into a _GroundSTRIPSOperator.
        first_action_part_len = self._max_operator_arity + 1
        op_action = Action(action.arr[:first_action_part_len])
        ground_op = _action_to_ground_strips_op(op_action, ordered_objs,
                                                self._ordered_strips_operators)
        assert ground_op is not None
        # Convert the operator into a controller (name).
        controller_name = self.operator_to_controller_name(ground_op.parent)
        # Extract the objects.
        objects = list(ground_op.objects)
        # Extract the parameters.
        n = self.controller_name_to_param_space(controller_name).shape[0]
        params = action.arr[first_action_part_len:first_action_part_len + n]
        return controller_name, objects, params

    def operator_to_controller_name(self, operator: STRIPSOperator) -> str:
        """Helper to convert operators to controllers.

        Exposed for use by oracle options.
        """
        if "MoveTo" in operator.name:
            return "navigate"
        if "Grasp" in operator.name:
            return "grasp"
        if "Place" in operator.name:
            return "placeOnTop"
        # Forthcoming controllers.
        return "noop"

    def controller_name_to_param_space(self, name: str) -> Box:
        """Helper for defining the controller param spaces.

        Exposed for use by oracle options.
        """
        return self._spot_interface.params_spaces[name]

    def build_action(self, all_objects: Collection[Object], op: STRIPSOperator,
                     objects: Sequence[Object], params: Array) -> Action:
        """Helper function exposed for use by oracle options."""
        # Initialize the action array.
        action_arr = np.zeros(self.action_space.shape[0], dtype=np.float32)
        # Add the operator index.
        op_idx = self._ordered_strips_operators.index(op)
        action_arr[0] = op_idx
        # Add the object indices.
        ordered_objects = sorted(all_objects)
        for i, o in enumerate(objects):
            obj_idx = ordered_objects.index(o)
            action_arr[i + 1] = obj_idx
        # Add the parameters.
        first_action_part_len = self._max_operator_arity + 1
        n = len(params)
        action_arr[first_action_part_len:first_action_part_len + n] = params
        # Finalize action.
        return Action(action_arr)

    def step(self, action: Action) -> Observation:  # pragma: no cover
        """Override step() because simulate() is not implemented."""
        obs = self._current_observation
        assert isinstance(obs, _SpotObservation)
        assert self.action_space.contains(action.arr)
        # Parse the action into the components needed for a controller.
        all_objects = set(self._make_object_name_to_obj_dict().values())
        ordered_objects = sorted(all_objects)
        name, objects, params = self.parse_action(action, ordered_objects)
        # Execute the controller in the real environment.
        self._spot_interface.execute(name, objects, params)
        # Now update the part of the state that is cheated based on the
        # ground-truth STRIPS operators.
        next_nonpercept = self._get_next_nonpercept_atoms(obs, action)
        self._current_observation = self._build_observation(next_nonpercept)
        return self._current_observation

    def _build_observation(self,
                           ground_atoms: Set[GroundAtom]) -> _SpotObservation:
        """Helper for building a new _SpotObservation().

        This is an environment method because the nonpercept predicates
        may vary per environment.
        """
        # Get the camera images.
        images = self._spot_interface.get_camera_images()

        # Detect objects.
        object_names_in_view = self._spot_interface.get_objects_in_view()

        objects_in_view = {
            self._obj_name_to_obj(n): v
            for n, v in object_names_in_view.items()
        }

        # Get the robot status.
        gripper_open_percentage = self._spot_interface.get_gripper_obs()

        # Prepare the non-percepts.
        nonpercept_preds = self.predicates - self.percept_predicates
        assert all(a.predicate in nonpercept_preds for a in ground_atoms)
        obs = _SpotObservation(images, objects_in_view,
                               gripper_open_percentage, ground_atoms,
                               nonpercept_preds)

        return obs

    def _get_next_nonpercept_atoms(
            self, obs: _SpotObservation,
            action: Action) -> Set[GroundAtom]:
        """Helper for step().

        This should be deprecated eventually.
        """
        # Get the ground operator.
        all_objects = set(self._make_object_name_to_obj_dict().values())
        ordered_objs = sorted(all_objects)
        ground_op = _action_to_ground_strips_op(action, ordered_objs,
                                                self._ordered_strips_operators)
        assert ground_op is not None
        # Update the atoms using the operator.
        next_ground_atoms = set(obs.nonpercept_atoms)
        # Add add effects.
        next_ground_atoms.update(ground_op.add_effects)
        # Remove delete effects.
        next_ground_atoms -= ground_op.delete_effects
        # Return only the atoms for the non-percept predicates.
        return {
            a
            for a in next_ground_atoms
            if a.predicate not in self.percept_predicates
        }

    def _actively_construct_initial_object_views(
            self) -> Dict[str, Tuple[float, float, float]]:
        raise NotImplementedError("Subclass must override!")

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for SpotEnv.")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        assert CFG.num_train_tasks == 0, "Use JSON loading instead"
        return []

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        assert CFG.num_test_tasks == 1, "Use JSON loading instead"
        return self._generate_tasks(CFG.num_test_tasks)

    def _generate_tasks(self, num_tasks: int) -> List[EnvironmentTask]:
        assert num_tasks == 1, "Use JSON loading instead"
        # Have the spot walk around the environment once to construct
        # an initial observation.
        object_names_in_view = self._actively_construct_initial_object_views()
        objects_in_view = {
            self._obj_name_to_obj(n): v
            for n, v in object_names_in_view.items()
        }
        images = self._spot_interface.get_camera_images()
        gripper_open_percentage = self._spot_interface.get_gripper_obs()
        nonpercept_atoms = self._get_initial_nonpercept_atoms()
        nonpercept_preds = self.predicates - self.percept_predicates
        # assert all(a.predicate in nonpercept_preds for a in ground_atoms)
        obs = _SpotObservation(images, objects_in_view,
                               gripper_open_percentage, nonpercept_atoms,
                               nonpercept_preds)
        goal = self._generate_task_goal()
        return [EnvironmentTask(obs, goal)]

    @abc.abstractmethod
    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        raise NotImplementedError

    @abc.abstractmethod
    def _generate_task_goal(self) -> Set[GroundAtom]:
        raise NotImplementedError

    @abc.abstractmethod
    def _make_object_name_to_obj_dict(self) -> Dict[str, Object]:
        raise NotImplementedError

    @abc.abstractmethod
    def _obj_name_to_obj(self, obj_name: str) -> Object:
        raise NotImplementedError

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")



###############################################################################
#                                Grocery Env                                  #
###############################################################################


class SpotGroceryEnv(SpotEnv):
    """An environment containing tasks for a real Spot robot to execute.

    Currently, the robot can move to specific 'surfaces' (e.g. tables),
    pick objects from on top these surfaces, and then place them
    elsewhere.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", [])
        self._can_type = Type("soda_can", [])
        self._surface_type = Type("flat_surface", [])

        # Predicates
        # Note that all classifiers assigned here just directly use
        # the ground atoms from the low-level simulator state.
        self._temp_On = Predicate("On", [self._can_type, self._surface_type],
                                  lambda s, o: False)
        self._On = Predicate("On", [self._can_type, self._surface_type],
                             _create_dummy_predicate_classifier(self._temp_On))
        self._temp_HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                         lambda s, o: False)
        self._HandEmpty = Predicate(
            "HandEmpty", [self._robot_type],
            _create_dummy_predicate_classifier(self._temp_HandEmpty))
        self._temp_HoldingCan = Predicate("HoldingCan",
                                          [self._robot_type, self._can_type],
                                          lambda s, o: False)
        self._HoldingCan = Predicate(
            "HoldingCan", [self._robot_type, self._can_type],
            _create_dummy_predicate_classifier(self._temp_HoldingCan))
        self._temp_ReachableCan = Predicate("ReachableCan",
                                            [self._robot_type, self._can_type],
                                            lambda s, o: False)
        self._ReachableCan = Predicate(
            "ReachableCan", [self._robot_type, self._can_type],
            _create_dummy_predicate_classifier(self._temp_ReachableCan))
        self._temp_ReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            lambda s, o: False)
        self._ReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            _create_dummy_predicate_classifier(self._temp_ReachableSurface))

        # STRIPS Operators (needed for option creation)
        # MoveToCan
        spot = Variable("?robot", self._robot_type)
        can = Variable("?can", self._can_type)
        add_effs = {LiftedAtom(self._ReachableCan, [spot, can])}
        ignore_effs = {self._ReachableCan, self._ReachableSurface}
        self._MoveToCanOp = STRIPSOperator("MoveToCan", [spot, can], set(),
                                           add_effs, set(), ignore_effs)
        # MoveToSurface
        spot = Variable("?robot", self._robot_type)
        surface = Variable("?surface", self._surface_type)
        add_effs = {LiftedAtom(self._ReachableSurface, [spot, surface])}
        ignore_effs = {self._ReachableCan, self._ReachableSurface}
        self._MoveToSurfaceOp = STRIPSOperator("MoveToSurface",
                                               [spot, surface], set(),
                                               add_effs, set(), ignore_effs)
        # GraspCan
        spot = Variable("?robot", self._robot_type)
        can = Variable("?can", self._can_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._On, [can, surface]),
            LiftedAtom(self._ReachableCan, [spot, can]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        add_effs = {LiftedAtom(self._HoldingCan, [spot, can])}
        del_effs = {
            LiftedAtom(self._On, [can, surface]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        self._GraspCanOp = STRIPSOperator("GraspCan", [spot, can, surface],
                                          preconds, add_effs, del_effs, set())
        # Place Can
        spot = Variable("?robot", self._robot_type)
        can = Variable("?can", self._can_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._ReachableSurface, [spot, surface]),
            LiftedAtom(self._HoldingCan, [spot, can])
        }
        add_effs = {
            LiftedAtom(self._On, [can, surface]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {LiftedAtom(self._HoldingCan, [spot, can])}
        self._PlaceCanOp = STRIPSOperator("PlaceCanOntop",
                                          [spot, can, surface], preconds,
                                          add_effs, del_effs, set())

        self._strips_operators = {
            self._MoveToCanOp, self._MoveToSurfaceOp, self._GraspCanOp,
            self._PlaceCanOp
        }

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._can_type, self._surface_type}

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._HandEmpty, self._HoldingCan, self._ReachableCan,
            self._ReachableSurface
        }

    @classmethod
    def get_name(cls) -> str:
        return "spot_grocery_env"

    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        spot = self._obj_name_to_obj("spot")
        kitchen_counter = self._obj_name_to_obj("kitchen_counter")
        soda_can = self._obj_name_to_obj("soda_can")
        return {
            GroundAtom(self._HandEmpty, [spot]),
            GroundAtom(self._On, [soda_can, kitchen_counter])
        }

    def _generate_task_goal(self) -> Set[GroundAtom]:
        snack_table = self._obj_name_to_obj("snack_table")
        soda_can = self._obj_name_to_obj("soda_can")
        return {GroundAtom(self._On, [soda_can, snack_table])}

    @functools.lru_cache(maxsize=None)
    def _make_object_name_to_obj_dict(self) -> Dict[str, Object]:
        spot = Object("spot", self._robot_type)
        kitchen_counter = Object("counter", self._surface_type)
        snack_table = Object("snack_table", self._surface_type)
        soda_can = Object("soda_can", self._can_type)
        objects = [spot, kitchen_counter, snack_table, soda_can]
        return {o.name: o for o in objects}

    def _obj_name_to_obj(self, obj_name: str) -> Object:
        return self._make_object_name_to_obj_dict()[obj_name]

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On}


###############################################################################
#                                Bike Repair Env                              #
###############################################################################


class SpotBikeEnv(SpotEnv):
    """An environment containing bike-repair related tasks for a real Spot
    robot to execute."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type(
            "robot",
            ["gripper_open_percentage", "curr_held_item_id", "x", "y", "z"])
        self._tool_type = Type("tool", ["x", "y", "z"])
        self._surface_type = Type("flat_surface", ["x", "y", "z"])
        self._bag_type = Type("bag", ["x", "y", "z"])
        self._platform_type = Type("platform", ["x", "y", "z"])

        # Predicates
        # Note that all classifiers assigned here just directly use
        # the ground atoms from the low-level simulator state.
        self._temp_On = Predicate("On", [self._tool_type, self._surface_type],
                                  lambda s, o: False)
        self._On = Predicate("On", [self._tool_type, self._surface_type],
                             _create_dummy_predicate_classifier(self._temp_On))
        self._temp_InBag = Predicate("InBag",
                                     [self._tool_type, self._bag_type],
                                     lambda s, o: False)
        self._InBag = Predicate(
            "InBag", [self._tool_type, self._bag_type],
            _create_dummy_predicate_classifier(self._temp_InBag))
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
        self._temp_ReachableTool = Predicate(
            "ReachableTool", [self._robot_type, self._tool_type],
            lambda s, o: False)
        self._ReachableTool = Predicate(
            "ReachableTool", [self._robot_type, self._tool_type],
            _create_dummy_predicate_classifier(self._temp_ReachableTool))
        self._temp_ReachableBag = Predicate("ReachableBag",
                                            [self._robot_type, self._bag_type],
                                            lambda s, o: False)
        self._ReachableBag = Predicate(
            "ReachableBag", [self._robot_type, self._bag_type],
            _create_dummy_predicate_classifier(self._temp_ReachableBag))
        self._temp_ReachablePlatform = Predicate(
            "ReachablePlatform", [self._robot_type, self._platform_type],
            lambda s, o: False)
        self._ReachablePlatform = Predicate(
            "ReachablePlatform", [self._robot_type, self._platform_type],
            _create_dummy_predicate_classifier(self._temp_ReachablePlatform))
        self._temp_XYReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            lambda s, o: False)
        self._XYReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            _create_dummy_predicate_classifier(self._temp_XYReachableSurface))
        self._temp_SurfaceTooHigh = Predicate(
            "SurfaceTooHigh", [self._robot_type, self._surface_type],
            lambda s, o: False)
        self._SurfaceTooHigh = Predicate(
            "SurfaceTooHigh", [self._robot_type, self._surface_type],
            _create_dummy_predicate_classifier(self._temp_SurfaceTooHigh))
        self._temp_SurfaceNotTooHigh = Predicate(
            "SurfaceNotTooHigh", [self._robot_type, self._surface_type],
            lambda s, o: False)
        self._SurfaceNotTooHigh = Predicate(
            "SurfaceNotTooHigh", [self._robot_type, self._surface_type],
            _create_dummy_predicate_classifier(self._temp_SurfaceNotTooHigh))
        self._temp_PlatformNear = Predicate(
            "PlatformNear", [self._platform_type, self._surface_type],
            lambda s, o: False)
        self._PlatformNear = Predicate(
            "PlatformNear", [self._platform_type, self._surface_type],
            _create_dummy_predicate_classifier(self._temp_PlatformNear))

        # STRIPS Operators (needed for option creation)
        # MoveToTool
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        add_effs = {LiftedAtom(self._ReachableTool, [spot, tool])}
        ignore_effs = {
            self._ReachableTool, self._ReachableBag, self._XYReachableSurface,
            self._ReachablePlatform
        }
        self._MoveToToolOp = STRIPSOperator("MoveToTool", [spot, tool], set(),
                                            add_effs, set(), ignore_effs)
        # MoveToSurface
        spot = Variable("?robot", self._robot_type)
        surface = Variable("?surface", self._surface_type)
        add_effs = {LiftedAtom(self._XYReachableSurface, [spot, surface])}
        ignore_effs = {
            self._ReachableTool, self._ReachableBag, self._XYReachableSurface,
            self._ReachablePlatform
        }
        self._MoveToSurfaceOp = STRIPSOperator("MoveToSurface",
                                               [spot, surface], set(),
                                               add_effs, set(), ignore_effs)
        # MoveToPlatform
        spot = Variable("?robot", self._robot_type)
        platform = Variable("?platform", self._platform_type)
        add_effs = {LiftedAtom(self._ReachablePlatform, [spot, platform])}
        ignore_effs = {
            self._ReachableTool, self._ReachableBag, self._XYReachableSurface,
            self._ReachablePlatform
        }
        self._MoveToPlatformOp = STRIPSOperator("MoveToPlatform",
                                                [spot, platform], set(),
                                                add_effs, set(), ignore_effs)
        # MoveToBag
        spot = Variable("?robot", self._robot_type)
        bag = Variable("?platform", self._bag_type)
        add_effs = {LiftedAtom(self._ReachableBag, [spot, bag])}
        ignore_effs = {
            self._ReachableTool, self._ReachableBag, self._XYReachableSurface,
            self._ReachablePlatform
        }
        self._MoveToBagOp = STRIPSOperator("MoveToBag", [spot, bag], set(),
                                           add_effs, set(), ignore_effs)
        # GraspToolFromNotHigh
        spot = Variable("?robot", self._robot_type)
        tool = Variable("?tool", self._tool_type)
        surface = Variable("?surface", self._surface_type)
        preconds = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._ReachableTool, [spot, tool]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._SurfaceNotTooHigh, [spot, surface])
        }
        add_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._ReachableTool, [spot, tool]),
        }
        self._GraspToolFromNotHighOp = STRIPSOperator("GraspToolFromNotHigh",
                                                      [spot, tool, surface],
                                                      preconds, add_effs,
                                                      del_effs, set())
        # GrabPlatformLeash
        spot = Variable("?robot", self._robot_type)
        platform = Variable("?platform", self._platform_type)
        preconds = {
            LiftedAtom(self._ReachablePlatform, [spot, platform]),
            LiftedAtom(self._HandEmpty, [spot]),
        }
        add_effs = {
            LiftedAtom(self._HoldingPlatformLeash, [spot, platform]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._ReachablePlatform, [spot, platform]),
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
            LiftedAtom(self._ReachableTool, [spot, tool]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._SurfaceTooHigh, [spot, surface]),
            LiftedAtom(self._PlatformNear, [platform, surface])
        }
        add_effs = {
            LiftedAtom(self._HoldingTool, [spot, tool]),
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot]),
            LiftedAtom(self._ReachableTool, [spot, tool]),
        }
        self._GraspToolFromHighOp = STRIPSOperator(
            "GraspToolFromHigh", [spot, tool, platform, surface], preconds,
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
            LiftedAtom(self._XYReachableSurface, [spot, surface]),
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
            LiftedAtom(self._XYReachableSurface, [spot, surface]),
        }
        self._PlaceToolNotHighOp = STRIPSOperator("PlaceToolNotHigh",
                                                  [spot, tool, surface],
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
            self._MoveToToolOp,
            self._MoveToSurfaceOp,
            self._MoveToPlatformOp,
            self._MoveToBagOp,
            self._GraspToolFromNotHighOp,
            self._GraspPlatformLeashOp,
            self._DragPlatformOp,
            self._GraspToolFromHighOp,
            self._GraspBagOp,
            self._PlaceToolNotHighOp,
            self._PlaceIntoBagOp,
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._tool_type, self._surface_type,
            self._bag_type, self._platform_type
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On, self._InBag, self._HandEmpty, self._HoldingTool,
            self._HoldingBag, self._HoldingPlatformLeash, self._ReachableTool,
            self._ReachableBag, self._ReachablePlatform,
            self._XYReachableSurface, self._SurfaceTooHigh,
            self._SurfaceNotTooHigh, self._PlatformNear, self._notHandEmpty
        }

    def _handempty_classifier(self, state: State,
                              objects: Sequence[Object]) -> bool:
        spot = objects[0]
        gripper_open_percentage = state.get(spot, "gripper_open_percentage")
        return gripper_open_percentage <= 1.5

    def _nothandempty_classifier(self, state: State,
                                 objects: Sequence[Object]) -> bool:
        return not self._handempty_classifier(state, objects)

    def _holding_tool_classifier(self, state: State,
                                 objects: Sequence[Object]) -> bool:
        spot, obj_to_grasp = objects
        assert obj_name_to_apriltag_id.get(obj_to_grasp.name) is not None
        spot_holding_obj_id = state.get(spot, "curr_held_item_id")
        return int(spot_holding_obj_id) == obj_name_to_apriltag_id[
            obj_to_grasp.name] and self._nothandempty_classifier(
                state, [spot])

    @classmethod
    def get_name(cls) -> str:
        return "spot_bike_env"

    @property
    def percept_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return {self._HandEmpty, self._notHandEmpty, self._HoldingTool}

    def _get_initial_nonpercept_atoms(self) -> Set[GroundAtom]:
        spot = self._obj_name_to_obj("spot")
        hammer = self._obj_name_to_obj("hammer")
        hex_key = self._obj_name_to_obj("hex_key")
        low_wall_rack = self._obj_name_to_obj("low_wall_rack")
        high_wall_rack = self._obj_name_to_obj("high_wall_rack")
        brush = self._obj_name_to_obj("brush")
        tool_room_table = self._obj_name_to_obj("tool_room_table")
        hex_screwdriver = self._obj_name_to_obj("hex_screwdriver")
        return {
            GroundAtom(self._On, [hammer, low_wall_rack]),
            GroundAtom(self._On, [hex_key, low_wall_rack]),
            GroundAtom(self._On, [brush, tool_room_table]),
            GroundAtom(self._On, [hex_screwdriver, tool_room_table]),
            GroundAtom(self._SurfaceNotTooHigh, [spot, low_wall_rack]),
            GroundAtom(self._SurfaceNotTooHigh, [spot, tool_room_table]),
            GroundAtom(self._SurfaceTooHigh, [spot, high_wall_rack]),
        }

    def _generate_task_goal(self) -> Set[GroundAtom]:
        hammer = self._obj_name_to_obj("hammer")
        hex_key = self._obj_name_to_obj("hex_key")
        brush = self._obj_name_to_obj("brush")
        hex_screwdriver = self._obj_name_to_obj("hex_screwdriver")
        bag = self._obj_name_to_obj("bag")
        return {
            GroundAtom(self._InBag, [hammer, bag]),
            GroundAtom(self._InBag, [brush, bag]),
            GroundAtom(self._InBag, [hex_key, bag]),
            GroundAtom(self._InBag, [hex_screwdriver, bag]),
        }

    @functools.lru_cache(maxsize=None)
    def _make_object_name_to_obj_dict(self) -> Dict[str, Object]:
        spot = Object("spot", self._robot_type)
        hammer = Object("hammer", self._tool_type)
        hex_key = Object("hex_key", self._tool_type)
        hex_screwdriver = Object("hex_screwdriver", self._tool_type)
        brush = Object("brush", self._tool_type)
        tool_room_table = Object("tool_room_table", self._surface_type)
        low_wall_rack = Object("low_wall_rack", self._surface_type)
        high_wall_rack = Object("high_wall_rack", self._surface_type)
        bag = Object("toolbag", self._bag_type)
        movable_platform = Object("movable_platform", self._platform_type)
        objects = [
            spot, hammer, hex_key, hex_screwdriver, brush, tool_room_table,
            low_wall_rack, high_wall_rack, bag, movable_platform
        ]
        return {o.name: o for o in objects}

    def _obj_name_to_obj(self, obj_name: str) -> Object:
        return self._make_object_name_to_obj_dict()[obj_name]

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.predicates

    def _actively_construct_initial_object_views(
            self) -> Dict[str, Tuple[float, float, float]]:
        obj_names = list(self._make_object_name_to_obj_dict().keys())
        return self._spot_interface.actively_construct_initial_object_views(
            obj_names)

