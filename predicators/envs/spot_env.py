"""Simple pick-place-move environment for the Boston Dynamics Spot Robot."""

from typing import Collection, List, Optional, Set, Dict
import json

import matplotlib
import numpy as np
from gym.spaces import Box
from pathlib import Path

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.pddl_env import _action_to_ground_strips_op, \
    _create_predicate_classifier, _PDDLEnvState, \
    _strips_operator_to_parameterized_option
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, STRIPSOperator, Task, Type, \
    Variable


class SpotEnv(BaseEnv):
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

        # Options (aka Controllers)
        # Note that these are 1:1 with the operators; in the future,
        # we will actually implement these with robot-specific API calls.
        self._options = {
            _strips_operator_to_parameterized_option(
                op, self._ordered_strips_operators, self.action_space.shape[0])
            for op in self._strips_operators
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

    @property
    def options(self) -> Set[ParameterizedOption]:
        return self._options

    @property
    def action_space(self) -> Box:
        # See class docstring for explanation.
        num_ops = len(self._strips_operators)
        max_arity = max(len(op.parameters) for op in self._strips_operators)
        lb = np.array([0.0 for _ in range(max_arity + 1)], dtype=np.float32)
        ub = np.array([num_ops - 1.0] + [np.inf for _ in range(max_arity)],
                      dtype=np.float32)
        return Box(lb, ub, dtype=np.float32)

    @classmethod
    def get_name(cls) -> str:
        return "realworld_spot"

    @property
    def strips_operators(self) -> Set[STRIPSOperator]:
        """Expose the STRIPSOperators for use by oracles."""
        return self._strips_operators

    def simulate(self, state: State, action: Action) -> State:
        assert isinstance(state, _PDDLEnvState)
        assert self.action_space.contains(action.arr)
        ordered_objs = list(state)
        # Convert the state into a Set[GroundAtom].
        ground_atoms = state.get_ground_atoms()
        # Convert the action into a _GroundSTRIPSOperator.
        ground_op = _action_to_ground_strips_op(action, ordered_objs,
                                                self._ordered_strips_operators)
        # If we couldn't turn this action into a ground operator, noop.
        if ground_op is None:
            return state.copy()
        # If the operator is not applicable in this state, noop.
        if not ground_op.preconditions.issubset(ground_atoms):
            return state.copy()
        # Apply the operator.
        next_ground_atoms = utils.apply_operator(ground_op, ground_atoms)
        # Convert back into a State.
        next_state = _PDDLEnvState.from_ground_atoms(next_ground_atoms,
                                                     ordered_objs)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._generate_tasks(CFG.num_train_tasks)

    def _generate_test_tasks(self) -> List[Task]:
        return self._generate_tasks(CFG.num_test_tasks)

    def _generate_tasks(self, num_tasks: int) -> List[Task]:
        tasks: List[Task] = []
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
            tasks.append(Task(init_state, goal))
        return tasks

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._On}

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

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

    def _load_task_from_json(self, json_file: Path) -> Task:
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
        # Parse objects.
        type_name_to_type = {t.name: t for t in self.types}
        object_name_to_object: Dict[str, Object] = {}
        for obj_name, type_name in json_dict["objects"].items():
            obj_type = type_name_to_type[type_name]
            obj = Object(obj_name, obj_type)
            object_name_to_object[obj_name] = obj
        assert set(object_name_to_object).issubset(set(json_dict["init"])), \
            "The init state can only include objects in `objects`."
        assert set(object_name_to_object).issuperset(set(json_dict["init"])), \
            "The init state must include every object in `objects`."
        # Parse initial state.
        init_dict: Dict[Object, Dict[str, float]] = {}
        for obj_name, obj_dict in json_dict["init"].items():
            obj = object_name_to_object[obj_name]
            init_dict[obj] = obj_dict.copy()
        
        # TODO: Parse out the predicates and yeet them into the simulator state
        # to correctly construct!
        init_state = _PDDLEnvState(init_dict, )
        
        
        # Parse goal.
        if "goal" in json_dict:
            goal = self._parse_goal_from_json(json_dict["goal"],
                                              object_name_to_object)
        else:
            assert "language_goal" in json_dict
            goal = self._parse_language_goal_from_json(
                json_dict["language_goal"], object_name_to_object)
        return Task(init_state, goal)


