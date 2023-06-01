"""Basic environment for the Boston Dynamics Spot Robot."""

import abc
import json
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
from predicators.spot_utils.spot_utils import get_spot_interface
from predicators.structs import Action, Array, EnvironmentTask, GroundAtom, \
    LiftedAtom, Object, Observation, Predicate, State, STRIPSOperator, Type, \
    Variable

###############################################################################
#                                Base Class                                   #
###############################################################################


class _PartialPerceptionState(State):
    """Some continuous object features, and ground atoms in simulator_state.

    The main idea here is that we have some predicates with actual
    classifiers implemented, but not all.
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
    def continuous_feature_predicates(self) -> Set[Predicate]:
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

    def _parse_action(self, state: State,
                      action: Action) -> Tuple[str, List[Object], Array]:
        # Convert the first action part into a _GroundSTRIPSOperator.
        first_action_part_len = self._max_operator_arity + 1
        op_action = Action(action.arr[:first_action_part_len])
        ordered_objs = list(state)
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

    def build_action(self, state: State, op: STRIPSOperator,
                     objects: Sequence[Object], params: Array) -> Action:
        """Helper function exposed for use by oracle options."""
        # Initialize the action array.
        action_arr = np.zeros(self.action_space.shape[0], dtype=np.float32)
        # Add the operator index.
        op_idx = self._ordered_strips_operators.index(op)
        action_arr[0] = op_idx
        # Add the object indices.
        ordered_objects = list(state)
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
        state = self._current_observation
        assert isinstance(state, _PartialPerceptionState)
        assert self.action_space.contains(action.arr)
        # Parse the action into the components needed for a controller.
        name, objects, params = self._parse_action(state, action)
        # Execute the controller in the real environment.
        current_atoms = utils.abstract(state, self.predicates)
        self._spot_interface.execute(name, current_atoms, objects, params)
        # Get the part of the new state that is determined based on
        # continuous feature values.
        next_state = self._get_continuous_observation()
        # Now update the part of the state that is cheated based on the
        # ground-truth STRIPS operators.
        next_sim_state_ground_atoms = self._get_next_simulator_state(
            state, action)
        if next_sim_state_ground_atoms is None:  # inapplicable action
            return state.copy()
        # Combine the two to get the new _PartialPerceptionState.
        self._current_observation = self._build_partial_perception_state(
            next_state.data, next_sim_state_ground_atoms)
        return self._current_observation.copy()

    def _build_partial_perception_state(
            self, state_data: Dict[Object, Array],
            ground_atoms: Set[GroundAtom]) -> _PartialPerceptionState:
        """Helper for building a new _PartialPerceptionState().

        This is an environment method because the predicates stored in
        the simulator_state may vary per environment.
        """
        sim_state_preds = self.predicates - self.continuous_feature_predicates
        assert all(a.predicate in sim_state_preds for a in ground_atoms)
        simulator_state = {
            "predicates": sim_state_preds,
            "atoms": ground_atoms
        }
        return _PartialPerceptionState(state_data.copy(),
                                       simulator_state=simulator_state)

    def _get_next_simulator_state(self, state: _PartialPerceptionState,
                                  action: Action) -> Optional[Set[GroundAtom]]:
        """Helper for step().

        Returns None if the action is not applicable. This should be
        deprecated soon when we move to regular State instances in this
        class.
        """
        ordered_objs = list(state)
        # Get the high-level state (i.e. set of GroundAtoms) by abstracting
        # the low-level state. Note that this will automatically take
        # care of using the hardcoded predicates vs. actually running
        # classifier functions where appropriate.
        ground_atoms = utils.abstract(state, self.predicates)
        # Convert the action into a _GroundSTRIPSOperator.
        ground_op = _action_to_ground_strips_op(action, ordered_objs,
                                                self._ordered_strips_operators)
        # If the operator is not applicable in this state, noop.
        if ground_op is None or not ground_op.preconditions.issubset(
                ground_atoms):
            return None
        # Apply the operator.
        next_ground_atoms = utils.apply_operator(ground_op, ground_atoms)
        # Return only the atoms for the non-continuous-feature predicates.
        return {
            a
            for a in next_ground_atoms
            if a.predicate not in self.continuous_feature_predicates
        }

    def _get_continuous_observation(self) -> State:
        """Helper for step()."""
        return State(self._current_observation.data.copy())

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for SpotEnv.")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._generate_tasks(CFG.num_train_tasks)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._generate_tasks(CFG.num_test_tasks)

    @abc.abstractmethod
    def _generate_tasks(self, num_tasks: int) -> List[EnvironmentTask]:
        raise NotImplementedError

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    @abc.abstractmethod
    def _get_language_goal_prompt_prefix(self,
                                         object_names: Collection[str]) -> str:
        raise NotImplementedError

    def _parse_init_preds_from_json(
            self, spec: Dict[str, List[List[str]]],
            id_to_obj: Dict[str, Object]) -> Set[GroundAtom]:
        """Helper for parsing init preds from JSON task specifications."""
        pred_names = {p.name for p in self.predicates}
        assert set(spec.keys()).issubset(pred_names)
        pred_to_args = {p: spec.get(p.name, []) for p in self.predicates}
        init_preds: Set[GroundAtom] = set()
        for pred, args in pred_to_args.items():
            for id_args in args:
                obj_args = [id_to_obj[a] for a in id_args]
                init_atom = GroundAtom(pred, obj_args)
                init_preds.add(init_atom)
        return init_preds

    def _load_task_from_json(self, json_file: Path) -> EnvironmentTask:
        """Create a task from a JSON file.

        By default, we assume JSON files are in the following format:

        {
            "objects": {
                <object name>: <type name>
            }
            "init": {
                <object name>: {
                    <feature name>: <value>
                }
            }
            "goal": {
                <predicate name> : [
                    [<object name>]
                ]
            }
        }

        Instead of "goal", "language_goal" can also be used.

        Environments can override this method to handle different formats.
        """
        with open(json_file, "r", encoding="utf-8") as f:
            json_dict = json.load(f)
        object_name_to_object: Dict[str, Object] = {}
        # Parse objects.
        type_name_to_type = {t.name: t for t in self.types}
        for obj_name, type_name in json_dict["objects"].items():
            obj_type = type_name_to_type[type_name]
            obj = Object(obj_name, obj_type)
            object_name_to_object[obj_name] = obj
        assert set(object_name_to_object).\
            issubset(set(json_dict["init"])), \
            "The init state can only include objects in `objects`."
        assert set(object_name_to_object).\
            issuperset(set(json_dict["init"])), \
            "The init state must include every object in `objects`."
        # Parse initial state.
        # NOTE: this is currently ignored; we will update after we add
        # predicates that are defined in terms of the continuous state.
        init_dict: Dict[Object, Array] = {
            o: np.zeros(0, dtype=np.float32)
            for o in object_name_to_object.values()
        }
        # NOTE: We need to parse out init preds to create a simulator state.
        init_atoms = self._parse_init_preds_from_json(json_dict["init_preds"],
                                                      object_name_to_object)
        # Remove any atoms that are defined via classifiers.
        init_atoms = {
            a
            for a in init_atoms
            if a.predicate not in self.continuous_feature_predicates
        }
        init_state = self._build_partial_perception_state(
            init_dict, init_atoms)
        # Parse goal.
        if "goal" in json_dict:
            goal = self._parse_goal_from_json(json_dict["goal"],
                                              object_name_to_object)
        else:  # pragma: no cover
            if CFG.override_json_with_input:
                goal = self._parse_goal_from_input_to_json(
                    init_state, json_dict, object_name_to_object)
            else:
                assert "language_goal" in json_dict
                goal = self._parse_language_goal_from_json(
                    json_dict["language_goal"], object_name_to_object)
        return EnvironmentTask(init_state, goal)


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

    def _generate_tasks(self, num_tasks: int) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []
        spot = Object("spot", self._robot_type)
        kitchen_counter = Object("counter", self._surface_type)
        snack_table = Object("snack_table", self._surface_type)
        soda_can = Object("soda_can", self._can_type)
        objects = [spot, kitchen_counter, snack_table, soda_can]
        for _ in range(num_tasks):
            init_dict: Dict[Object, Array] = {
                o: np.zeros(0, dtype=np.float32)
                for o in objects
            }
            init_atoms = {
                GroundAtom(self._HandEmpty, [spot]),
                GroundAtom(self._On, [soda_can, kitchen_counter])
            }
            init_state = self._build_partial_perception_state(
                init_dict, init_atoms)
            goal = {GroundAtom(self._On, [soda_can, snack_table])}
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On}

    def _get_language_goal_prompt_prefix(self,
                                         object_names: Collection[str]) -> str:
        # pylint:disable=line-too-long
        available_predicates = ", ".join(
            [p.name for p in sorted(self.goal_predicates)])
        available_objects = ", ".join(sorted(object_names))
        # We could extract the object names, but this is simpler.
        assert {"spot", "counter", "snack_table",
                "soda_can"}.issubset(object_names)
        prompt = f"""# The available predicates are: {available_predicates}
# The available objects are: {available_objects}
# Use the available predicates and objects to convert natural language goals into JSON goals.
        
# Hey spot, can you move the soda can to the snack table?
{{"On": [["soda_can", "snack_table"]]}}
"""
        return prompt


###############################################################################
#                                Bike Repair Env                              #
###############################################################################


class SpotBikeEnv(SpotEnv):
    """An environment containing bike-repair related tasks for a real Spot
    robot to execute."""

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", ["gripper_open_percentage"])
        self._tool_type = Type("tool", [])
        self._surface_type = Type("flat_surface", [])
        self._bag_type = Type("bag", [])
        self._platform_type = Type("platform", [])

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

        self._temp_HoldingTool = Predicate("HoldingTool",
                                           [self._robot_type, self._tool_type],
                                           lambda s, o: False)
        self._HoldingTool = Predicate(
            "HoldingTool", [self._robot_type, self._tool_type],
            _create_dummy_predicate_classifier(self._temp_HoldingTool))
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
            LiftedAtom(self._HandEmpty, [spot])
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
        del_effs = {LiftedAtom(self._HandEmpty, [spot])}
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
            LiftedAtom(self._HandEmpty, [spot])
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
            LiftedAtom(self._notHandEmpty, [spot])
        }
        del_effs = {LiftedAtom(self._HandEmpty, [spot])}
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
            LiftedAtom(self._notHandEmpty, [spot])
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
            LiftedAtom(self._notHandEmpty, [spot])
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

    @classmethod
    def get_name(cls) -> str:
        return "spot_bike_env"

    @property
    def continuous_feature_predicates(self) -> Set[Predicate]:
        """The predicates that are NOT stored in the simulator state."""
        return {self._HandEmpty}

    def _get_continuous_observation(self) -> State:  # pragma: no cover
        """Helper for step()."""
        curr_state = State(
            {k: v.copy()
             for k, v in self._current_observation.data.items()})
        new_gripper_open_perc = self._spot_interface.get_gripper_obs()
        spot = curr_state.get_objects(self._robot_type)[0]
        curr_state.set(spot, "gripper_open_percentage", new_gripper_open_perc)
        return curr_state

    def _generate_tasks(self, num_tasks: int) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []
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
        for _ in range(num_tasks):
            init_dict = {spot: np.array([0.0])}

            for obj in objects:
                if obj != spot:
                    init_dict[obj] = np.array([])

            init_atoms = {
                GroundAtom(self._On, [hammer, low_wall_rack]),
                GroundAtom(self._On, [hex_key, low_wall_rack]),
                GroundAtom(self._On, [brush, tool_room_table]),
                GroundAtom(self._On, [hex_screwdriver, tool_room_table]),
                GroundAtom(self._SurfaceNotTooHigh, [spot, low_wall_rack]),
                GroundAtom(self._SurfaceNotTooHigh, [spot, tool_room_table]),
                GroundAtom(self._SurfaceTooHigh, [spot, high_wall_rack]),
            }
            init_state = self._build_partial_perception_state(
                init_dict, init_atoms)
            goal = {
                GroundAtom(self._InBag, [hammer, bag]),
                GroundAtom(self._InBag, [brush, bag]),
                GroundAtom(self._InBag, [hex_key, bag]),
                GroundAtom(self._InBag, [hex_screwdriver, bag]),
            }
            tasks.append(EnvironmentTask(init_state, goal))
        return tasks

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self.predicates

    def _get_language_goal_prompt_prefix(self,
                                         object_names: Collection[str]) -> str:
        # pylint:disable=line-too-long
        available_predicates = ", ".join([
            str((p.name, p.types, p.arity))
            for p in sorted(self.goal_predicates)
        ])
        available_objects = ", ".join(sorted(object_names))
        # We could extract the object names, but this is simpler.
        assert {"spot", "hammer", "toolbag",
                "low_wall_rack"}.issubset(object_names)
        prompt = f"""# The available predicates are: {available_predicates}
# The available objects are: {available_objects}
# Use the available predicates and objects to convert natural language goals into JSON goals.

# Hey spot, can you put the hammer into the bag?
{{"InBag": [["hammer", "toolbag"]]}}

# Will you put the bag onto the low rack, please?
{{"On": [["toolbag", "low_wall_rack"]],"HandEmpty": [["spot"]]}}

# Go to the low_wall_rack.
{{"ReachableSurface": [["spot", "low_wall_rack"]]}}
"""
        return prompt
