"""Basic environment for the Boston Dynamics Spot Robot."""

import abc
import json
from pathlib import Path
from typing import Collection, Dict, List, Optional, Set

import matplotlib
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.pddl_env import _action_to_ground_strips_op, \
    _create_predicate_classifier, _PDDLEnvState
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, \
    LiftedAtom, Object, Predicate, State, STRIPSOperator, Type, Variable

###############################################################################
#                                Base Class                                   #
###############################################################################


class SpotEnv(BaseEnv):
    """An environment containing tasks for a real Spot robot to execute.

    This is a base class that specific sub-classes that define actual
    tasks should inherit from.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._strips_operators: Set[STRIPSOperator] = set()
        self._ordered_strips_operators: List[STRIPSOperator] = list(
            self._strips_operators)

    # @classmethod
    # def get_name(cls) -> str:
    #     return "spot_base_env"

    @property
    def strips_operators(self) -> Set[STRIPSOperator]:
        """Expose the STRIPSOperators for use by oracles."""
        return self._strips_operators

    @property
    def action_space(self) -> Box:
        # See class docstring for explanation.
        num_ops = len(self._strips_operators)
        max_arity = max(len(op.parameters) for op in self._strips_operators)
        lb = np.array([0.0 for _ in range(max_arity + 1)], dtype=np.float32)
        ub = np.array([num_ops - 1.0] + [np.inf for _ in range(max_arity)],
                      dtype=np.float32)
        return Box(lb, ub, dtype=np.float32)

    def simulate(self, state: State, action: Action) -> State:
        assert isinstance(state, _PDDLEnvState)
        assert self.action_space.contains(action.arr)
        ordered_objs = list(state)
        # Convert the state into a Set[GroundAtom].
        ground_atoms = state.get_ground_atoms()
        # Convert the action into a _GroundSTRIPSOperator.
        ground_op = _action_to_ground_strips_op(action, ordered_objs,
                                                self._ordered_strips_operators)
        # If the operator is not applicable in this state, noop.
        if ground_op is None or not ground_op.preconditions.issubset(
                ground_atoms):
            return state.copy()
        # Apply the operator.
        next_ground_atoms = utils.apply_operator(ground_op, ground_atoms)
        # Convert back into a State.
        next_state = _PDDLEnvState.from_ground_atoms(next_ground_atoms,
                                                     ordered_objs)
        return next_state

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
        init_dict: Dict[Object, Dict[str, float]] = {}
        for obj_name, obj_dict in json_dict["init"].items():
            obj = object_name_to_object[obj_name]
            init_dict[obj] = obj_dict.copy()
        # NOTE: We need to parse out init preds to create a simulator state.
        init_preds = self._parse_init_preds_from_json(json_dict["init_preds"],
                                                      object_name_to_object)
        # NOTE: mypy gets mad at this usage here because we're putting
        # predicates into the PDDLEnvState when the signature actually
        # expects Arrays.
        init_state = _PDDLEnvState(init_dict, init_preds)  # type: ignore

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
                             _create_predicate_classifier(self._temp_On))
        self._temp_HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                         lambda s, o: False)
        self._HandEmpty = Predicate(
            "HandEmpty", [self._robot_type],
            _create_predicate_classifier(self._temp_HandEmpty))
        self._temp_HoldingCan = Predicate("HoldingCan",
                                          [self._robot_type, self._can_type],
                                          lambda s, o: False)
        self._HoldingCan = Predicate(
            "HoldingCan", [self._robot_type, self._can_type],
            _create_predicate_classifier(self._temp_HoldingCan))
        self._temp_ReachableCan = Predicate("ReachableCan",
                                            [self._robot_type, self._can_type],
                                            lambda s, o: False)
        self._ReachableCan = Predicate(
            "ReachableCan", [self._robot_type, self._can_type],
            _create_predicate_classifier(self._temp_ReachableCan))
        self._temp_ReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            lambda s, o: False)
        self._ReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            _create_predicate_classifier(self._temp_ReachableSurface))

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
        self._ordered_strips_operators = sorted(self._strips_operators)

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
        for _ in range(num_tasks):
            init_state = _PDDLEnvState.from_ground_atoms(
                {
                    GroundAtom(self._HandEmpty, [spot]),
                    GroundAtom(self._On, [soda_can, kitchen_counter])
                }, [spot, kitchen_counter, snack_table, soda_can])
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
    robot to execute.

    TODO:
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", [])
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
                             _create_predicate_classifier(self._temp_On))
        self._temp_InBag = Predicate("InBag",
                                     [self._tool_type, self._bag_type],
                                     lambda s, o: False)
        self._InBag = Predicate("InBag", [self._tool_type, self._bag_type],
                                _create_predicate_classifier(self._temp_InBag))
        self._temp_HandEmpty = Predicate("HandEmpty", [self._robot_type],
                                         lambda s, o: False)
        self._HandEmpty = Predicate(
            "HandEmpty", [self._robot_type],
            _create_predicate_classifier(self._temp_HandEmpty))
        self._temp_HoldingTool = Predicate("HoldingTool",
                                           [self._robot_type, self._tool_type],
                                           lambda s, o: False)
        self._HoldingTool = Predicate(
            "HoldingTool", [self._robot_type, self._tool_type],
            _create_predicate_classifier(self._temp_HoldingTool))
        self._temp_HoldingBag = Predicate("HoldingBag",
                                          [self._robot_type, self._bag_type],
                                          lambda s, o: False)
        self._HoldingBag = Predicate(
            "HoldingBag", [self._robot_type, self._bag_type],
            _create_predicate_classifier(self._temp_HoldingBag))
        self._temp_HoldingPlatformLeash = Predicate(
            "HoldingPlatformLeash", [self._robot_type, self._platform_type],
            lambda s, o: False)
        self._HoldingPlatformLeash = Predicate(
            "HoldingPlatformLeash", [self._robot_type, self._platform_type],
            _create_predicate_classifier(self._temp_HoldingPlatformLeash))
        self._temp_ReachableTool = Predicate(
            "ReachableTool", [self._robot_type, self._tool_type],
            lambda s, o: False)
        self._ReachableTool = Predicate(
            "ReachableTool", [self._robot_type, self._tool_type],
            _create_predicate_classifier(self._temp_ReachableTool))
        self._temp_ReachableBag = Predicate("ReachableBag",
                                            [self._robot_type, self._bag_type],
                                            lambda s, o: False)
        self._ReachableBag = Predicate(
            "ReachableBag", [self._robot_type, self._bag_type],
            _create_predicate_classifier(self._temp_ReachableBag))
        self._temp_ReachablePlatform = Predicate(
            "ReachablePlatform", [self._robot_type, self._platform_type],
            lambda s, o: False)
        self._ReachablePlatform = Predicate(
            "ReachablePlatform", [self._robot_type, self._platform_type],
            _create_predicate_classifier(self._temp_ReachablePlatform))
        self._temp_XYReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            lambda s, o: False)
        self._XYReachableSurface = Predicate(
            "ReachableSurface", [self._robot_type, self._surface_type],
            _create_predicate_classifier(self._temp_XYReachableSurface))
        self._temp_SurfaceTooHigh = Predicate(
            "SurfaceTooHigh", [self._robot_type, self._surface_type],
            lambda s, o: False)
        self._SurfaceTooHigh = Predicate(
            "SurfaceTooHigh", [self._robot_type, self._surface_type],
            _create_predicate_classifier(self._temp_SurfaceTooHigh))
        self._temp_SurfaceNotTooHigh = Predicate(
            "SurfaceNotTooHigh", [self._robot_type, self._surface_type],
            lambda s, o: False)
        self._SurfaceNotTooHigh = Predicate(
            "SurfaceNotTooHigh", [self._robot_type, self._surface_type],
            _create_predicate_classifier(self._temp_SurfaceNotTooHigh))
        self._temp_PlatformNear = Predicate(
            "PlatformNear", [self._platform_type, self._surface_type],
            lambda s, o: False)
        self._PlatformNear = Predicate(
            "PlatformNear", [self._platform_type, self._surface_type],
            _create_predicate_classifier(self._temp_PlatformNear))

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
        add_effs = {LiftedAtom(self._HoldingTool, [spot, tool])}
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
        add_effs = {LiftedAtom(self._HoldingPlatformLeash, [spot, platform])}
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
        }
        add_effs = {
            LiftedAtom(self._PlatformNear, [platform, surface]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {LiftedAtom(self._HoldingPlatformLeash, [spot, platform])}
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
        add_effs = {LiftedAtom(self._HoldingTool, [spot, tool])}
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
        add_effs = {LiftedAtom(self._HoldingBag, [spot, bag])}
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
            LiftedAtom(self._HoldingTool, [spot, tool])
        }
        add_effs = {
            LiftedAtom(self._On, [tool, surface]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {LiftedAtom(self._HoldingTool, [spot, tool])}
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
            LiftedAtom(self._HoldingTool, [spot, tool])
        }
        add_effs = {
            LiftedAtom(self._InBag, [tool, bag]),
            LiftedAtom(self._HandEmpty, [spot])
        }
        del_effs = {LiftedAtom(self._HoldingTool, [spot, tool])}
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
        self._ordered_strips_operators = sorted(self._strips_operators)

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._tool_type, self._surface_type,
            self._bag_type, self._platform_type
        }

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._On,
            self._InBag,
            self._HandEmpty,
            self._HoldingTool,
            self._HoldingBag,
            self._HoldingPlatformLeash,
            self._ReachableTool,
            self._ReachableBag,
            self._ReachablePlatform,
            self._XYReachableSurface,
            self._SurfaceTooHigh,
            self._SurfaceNotTooHigh,
            self._PlatformNear,
        }

    @classmethod
    def get_name(cls) -> str:
        return "spot_bike_env"

    def _generate_tasks(self, num_tasks: int) -> List[EnvironmentTask]:
        tasks: List[EnvironmentTask] = []
        spot = Object("spot", self._robot_type)
        hammer = Object("hammer", self._tool_type)
        low_wall_rack = Object("low_wall_rack", self._surface_type)
        bag = Object("toolbag", self._bag_type)
        movable_platform = Object("movable_platform", self._platform_type)

        for _ in range(num_tasks):
            init_state = _PDDLEnvState.from_ground_atoms(
                {
                    GroundAtom(self._HandEmpty, [spot]),
                    GroundAtom(self._On, [hammer, low_wall_rack]),
                    GroundAtom(self._SurfaceNotTooHigh, [spot, low_wall_rack]),
                }, [spot, hammer, low_wall_rack, bag, movable_platform])
            goal = {GroundAtom(self._InBag, [hammer, bag])}
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
