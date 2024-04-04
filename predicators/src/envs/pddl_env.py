"""Environments that are derived from PDDL.

There are no continuous aspects of the state or action space. These
environments are similar to PDDLGym.
"""
from __future__ import annotations

import abc
import functools
from typing import Callable, Collection, Dict, List, Optional, Sequence, Set, \
    Tuple, cast

import matplotlib
import numpy as np
from gym.spaces import Box
from pyperplan.pddl.parser import TraversePDDLDomain, TraversePDDLProblem, \
    parse_domain_def, parse_lisp_iterator, parse_problem_def
from pyperplan.pddl.pddl import Domain as PyperplanDomain

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.envs.pddl_procedural_generation import \
    create_blocks_pddl_generator
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, LiftedAtom, \
    Object, ParameterizedOption, PDDLProblemGenerator, Predicate, State, \
    STRIPSOperator, Task, Type, Variable, Video, _GroundSTRIPSOperator

###############################################################################
#                                Base Classes                                 #
###############################################################################


class _PDDLEnvState(State):
    """No continuous object features, and ground atoms in simulator_state."""

    def get_ground_atoms(self) -> Set[GroundAtom]:
        """Expose the ground atoms in the simulator_state."""
        return cast(Set[GroundAtom], self.simulator_state)

    @classmethod
    def from_ground_atoms(cls, ground_atoms: Set[GroundAtom],
                          objects: Collection[Object]) -> _PDDLEnvState:
        """Create a state from ground atoms and objects."""
        # Keep a dummy state dict so we know what objects are in the state.
        dummy_state_dict = {o: np.zeros(0, dtype=np.float32) for o in objects}
        return _PDDLEnvState(dummy_state_dict, simulator_state=ground_atoms)

    def allclose(self, other: State) -> bool:
        return self.simulator_state == other.simulator_state

    def copy(self) -> State:
        # The important part is that copy needs to return a _PDDLEnvState.
        # For extra peace of mind, we also copy the ground atom set.
        ground_atoms = self.get_ground_atoms()
        objects = set(self)
        return self.from_ground_atoms(ground_atoms.copy(), objects)


class _PDDLEnv(BaseEnv):
    """An environment induced by PDDL.

    The state space is mostly unused. The continuous vectors are dummies. What
    is actually used is state.simulator_state, which holds the current ground
    atoms. Note that we need to use this pattern, as opposed to just
    maintaining the ground atoms internally in the env, because the predicate
    classifiers need access to the ground atoms.

    The action space is hacked to conform to our convention that actions
    are fixed-dimensional vectors. Users of this class should not need
    to worry about the action space because it would never make sense to
    learn anything using this action space. The dimensionality is 1 +
    max operator arity. The first dimension encodes the operator. The
    next dimensions encode the object used to ground the operator. The
    encoding assumes a fixed ordering over operators and objects in the
    state.

    The parameterized options are 1:1 with the STRIPS operators. They have the
    same object parameters and no continuous parameters.
    """

    def __init__(self) -> None:
        super().__init__()
        # Parse the domain str.
        self._types, self._predicates, self._strips_operators = \
            _parse_pddl_domain(self._domain_str)
        # The order is used for constructing actions; see class docstring.
        self._ordered_strips_operators = sorted(self._strips_operators)
        # Compute the options. Note that they are 1:1 with the operators.
        self._options = {
            _strips_operator_to_parameterized_option(
                op, self._ordered_strips_operators, self.action_space.shape[0])
            for op in self._strips_operators
        }
        # Compute the train and test tasks.
        self._pregenerated_train_tasks = self._generate_tasks(
            CFG.num_train_tasks, self._pddl_train_problem_generator,
            self._train_rng)
        self._pregenerated_test_tasks = self._generate_tasks(
            CFG.num_test_tasks, self._pddl_test_problem_generator,
            self._test_rng)
        # Determine the goal predicates from the tasks.
        tasks = self._pregenerated_train_tasks + self._pregenerated_test_tasks
        self._goal_predicates = {a.predicate for t in tasks for a in t.goal}

    @property
    @abc.abstractmethod
    def _domain_str(self) -> str:
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _pddl_train_problem_generator(self) -> PDDLProblemGenerator:
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _pddl_test_problem_generator(self) -> PDDLProblemGenerator:
        raise NotImplementedError("Override me!")

    @property
    def strips_operators(self) -> Set[STRIPSOperator]:
        """Expose the STRIPSOperators for use by oracles."""
        return self._strips_operators

    def simulate(self, state: State, action: Action) -> State:
        assert isinstance(state, _PDDLEnvState)
        ordered_objs = list(state)
        # Convert the state into a Set[GroundAtom].
        ground_atoms = state.get_ground_atoms()
        # Convert the action into a _GroundSTRIPSOperator.
        ground_op = _action_to_ground_strips_op(action, ordered_objs,
                                                self._ordered_strips_operators)
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
        return self._pregenerated_train_tasks

    def _generate_test_tasks(self) -> List[Task]:
        return self._pregenerated_test_tasks

    def _generate_tasks(self, num_tasks: int,
                        problem_gen: PDDLProblemGenerator,
                        rng: np.random.Generator) -> List[Task]:
        tasks = []
        for pddl_problem_str in problem_gen(num_tasks, rng):
            task = _pddl_problem_str_to_task(pddl_problem_str,
                                             self._domain_str, self.types,
                                             self.predicates)
            tasks.append(task)
        return tasks

    @property
    def predicates(self) -> Set[Predicate]:
        return self._predicates

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return self._goal_predicates

    @property
    def types(self) -> Set[Type]:
        return self._types

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

    def render_state_plt(
            self,
            state: State,
            task: Task,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("Render not implemented for PDDLEnv.")


class _FixedTasksPDDLEnv(_PDDLEnv):
    """An environment where tasks are induced by static PDDL problem files."""

    @property
    @abc.abstractmethod
    def _pddl_problem_asset_dir(self) -> str:
        """The name of the problem file directory in assets/pddl."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _train_problem_indices(self) -> List[int]:
        """Return a list of indices corresponding to training problems.

        For each idx in the returned list, it is expected that there
        will be a problem file named task{idx}.pddl.
        """
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _test_problem_indices(self) -> List[int]:
        """Return a list of indices corresponding to test problems.

        For each idx in the returned list, it is expected that there
        will be a problem file named task{idx}.pddl.
        """
        raise NotImplementedError("Override me!")

    @property
    def _pddl_train_problem_generator(self) -> PDDLProblemGenerator:
        assert len(self._train_problem_indices) >= CFG.num_train_tasks
        return _file_problem_generator(self._pddl_problem_asset_dir,
                                       self._train_problem_indices)

    @property
    def _pddl_test_problem_generator(self) -> PDDLProblemGenerator:
        assert len(self._test_problem_indices) >= CFG.num_test_tasks
        return _file_problem_generator(self._pddl_problem_asset_dir,
                                       self._test_problem_indices)


###############################################################################
#                              Specific Envs                                  #
###############################################################################


class _BlocksPDDLEnv(_PDDLEnv):
    """The IPC 4-operator blocks world domain."""

    @property
    def _domain_str(self) -> str:
        path = utils.get_env_asset_path("pddl/blocks/domain.pddl")
        with open(path, encoding="utf-8") as f:
            domain_str = f.read()
        return domain_str


class FixedTasksBlocksPDDLEnv(_BlocksPDDLEnv, _FixedTasksPDDLEnv):
    """The IPC 4-operator blocks world domain with a fixed set of tasks."""

    @classmethod
    def get_name(cls) -> str:
        return "pddl_blocks_fixed_tasks"

    @property
    def _pddl_problem_asset_dir(self) -> str:
        return "blocks"

    @property
    def _train_problem_indices(self) -> List[int]:
        return CFG.pddl_blocks_fixed_train_indices

    @property
    def _test_problem_indices(self) -> List[int]:
        return CFG.pddl_blocks_fixed_test_indices


class ProceduralTasksBlocksPDDLEnv(_BlocksPDDLEnv):
    """The IPC 4-operator blocks world domain with procedural generation."""

    @classmethod
    def get_name(cls) -> str:
        return "pddl_blocks_procedural_tasks"

    @property
    def _pddl_train_problem_generator(self) -> PDDLProblemGenerator:
        min_blocks = CFG.pddl_blocks_procedural_train_min_num_blocks
        max_blocks = CFG.pddl_blocks_procedural_train_max_num_blocks
        min_blocks_goal = CFG.pddl_blocks_procedural_train_min_num_blocks_goal
        max_blocks_goal = CFG.pddl_blocks_procedural_train_max_num_blocks_goal
        new_pile_prob = CFG.pddl_blocks_procedural_new_pile_prob
        return create_blocks_pddl_generator(min_blocks, max_blocks,
                                            min_blocks_goal, max_blocks_goal,
                                            new_pile_prob)

    @property
    def _pddl_test_problem_generator(self) -> PDDLProblemGenerator:
        min_blocks = CFG.pddl_blocks_procedural_test_min_num_blocks
        max_blocks = CFG.pddl_blocks_procedural_test_max_num_blocks
        min_blocks_goal = CFG.pddl_blocks_procedural_test_min_num_blocks_goal
        max_blocks_goal = CFG.pddl_blocks_procedural_test_max_num_blocks_goal
        new_pile_prob = CFG.pddl_blocks_procedural_new_pile_prob
        return create_blocks_pddl_generator(min_blocks, max_blocks,
                                            min_blocks_goal, max_blocks_goal,
                                            new_pile_prob)


###############################################################################
#                            Utility functions                                #
###############################################################################


def _action_to_ground_strips_op(
        action: Action, ordered_objects: List[Object],
        ordered_operators: List[STRIPSOperator]) -> _GroundSTRIPSOperator:
    action_arr = action.arr
    op_idx = int(action_arr[0])
    op = ordered_operators[op_idx]
    op_arity = len(op.parameters)
    num_objs = len(ordered_objects)
    obj_idxs = [
        min(int(i), num_objs - 1) for i in action_arr[1:(op_arity + 1)]
    ]
    objs = tuple(ordered_objects[i] for i in obj_idxs)
    return op.ground(objs)


def _strips_operator_to_parameterized_option(
        op: STRIPSOperator, ordered_operators: List[STRIPSOperator],
        action_dims: int) -> ParameterizedOption:
    name = op.name
    types = [p.type for p in op.parameters]
    op_idx = ordered_operators.index(op)

    def policy(s: State, m: Dict, o: Sequence[Object], p: Array) -> Action:
        del m, p  # unused
        ordered_objs = list(s)
        # The first dimension of an action encodes the operator.
        # The second dimension of an action encodes the first object argument
        # to the ground operator. The third dimension encodes the second object
        # and so on. Actions are always padded so that their length is equal
        # to the max number of arguments for any operator.
        obj_idxs = [ordered_objs.index(obj) for obj in o]
        act_arr = np.zeros(action_dims, dtype=np.float32)
        act_arr[0] = op_idx
        act_arr[1:(len(obj_idxs) + 1)] = obj_idxs
        return Action(act_arr)

    def initiable(s: State, m: Dict, o: Sequence[Object], p: Array) -> bool:
        del m, p  # unused
        assert isinstance(s, _PDDLEnvState)
        ground_atoms = s.get_ground_atoms()
        ground_op = op.ground(tuple(o))
        return ground_op.preconditions.issubset(ground_atoms)

    return utils.SingletonParameterizedOption(name,
                                              policy,
                                              types,
                                              initiable=initiable)


@functools.lru_cache(maxsize=None)
def _domain_str_to_pyperplan_domain(domain_str: str) -> PyperplanDomain:
    domain_ast = parse_domain_def(parse_lisp_iterator(domain_str.split("\n")))
    visitor = TraversePDDLDomain()
    domain_ast.accept(visitor)
    return visitor.domain


def _parse_pddl_domain(
    pddl_domain_str: str
) -> Tuple[Set[Type], Set[Predicate], Set[STRIPSOperator]]:
    # Let pyperplan do most of the heavy lifting.
    pyperplan_domain = _domain_str_to_pyperplan_domain(pddl_domain_str)
    pyperplan_types = pyperplan_domain.types
    pyperplan_predicates = pyperplan_domain.predicates
    pyperplan_operators = pyperplan_domain.actions
    # Convert the pyperplan domain into our structs.
    pyperplan_type_to_type = {
        pyperplan_types[t]: Type(t, [])
        for t in pyperplan_types
    }
    # Convert the predicates.
    predicate_name_to_predicate = {}
    for pyper_pred in pyperplan_predicates.values():
        name = pyper_pred.name
        pred_types = [
            pyperplan_type_to_type[t] for _, (t, ) in pyper_pred.signature
        ]
        # This is incredibly hacky, but necessary, because the classifier needs
        # to refer to the predicate instance, so the predicate instance needs
        # to be created before the classifier. Note that this relies heavily
        # on the Predicate __eq__ method.
        temp_classifier = lambda s, o: False
        temp_pred = Predicate(name, pred_types, temp_classifier)
        classifier = _create_predicate_classifier(temp_pred)
        pred = Predicate(name, pred_types, classifier)
        predicate_name_to_predicate[name] = pred
    # Convert the operators.
    operators = set()
    for pyper_op in pyperplan_operators.values():
        name = pyper_op.name
        parameters = [
            Variable(n, pyperplan_type_to_type[t])
            for n, (t, ) in pyper_op.signature
        ]
        param_name_to_param = {p.name: p for p in parameters}
        preconditions = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [param_name_to_param[n] for n, _ in a.signature])
            for a in pyper_op.precondition
        }
        add_effects = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [param_name_to_param[n] for n, _ in a.signature])
            for a in pyper_op.effect.addlist
        }
        delete_effects = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [param_name_to_param[n] for n, _ in a.signature])
            for a in pyper_op.effect.dellist
        }
        strips_op = STRIPSOperator(name,
                                   parameters,
                                   preconditions,
                                   add_effects,
                                   delete_effects,
                                   side_predicates=set())
        operators.add(strips_op)
    # Collect the final outputs.
    types = set(pyperplan_type_to_type.values())
    predicates = set(predicate_name_to_predicate.values())
    return types, predicates, operators


def _pddl_problem_str_to_task(pddl_problem_str: str, pddl_domain_str: str,
                              types: Set[Type],
                              predicates: Set[Predicate]) -> Task:
    # Let pyperplan do most of the heavy lifting.
    # Pyperplan needs the domain to parse the problem. Note that this is
    # cached by lru_cache.
    pyperplan_domain = _domain_str_to_pyperplan_domain(pddl_domain_str)
    # Now that we have the domain, parse the problem.
    lisp_iterator = parse_lisp_iterator(pddl_problem_str.split("\n"))
    problem_ast = parse_problem_def(lisp_iterator)
    visitor = TraversePDDLProblem(pyperplan_domain)
    problem_ast.accept(visitor)
    pyperplan_problem = visitor.get_problem()
    # Create the objects.
    type_name_to_type = {t.name: t for t in types}
    object_name_to_obj = {
        o: Object(o, type_name_to_type[t.name])
        for o, t in pyperplan_problem.objects.items()
    }
    objects = set(object_name_to_obj.values())
    # Create the initial state.
    predicate_name_to_predicate = {p.name: p for p in predicates}
    init_ground_atoms = {
        GroundAtom(predicate_name_to_predicate[a.name],
                   [object_name_to_obj[n] for n, _ in a.signature])
        for a in pyperplan_problem.initial_state
    }
    init = _PDDLEnvState.from_ground_atoms(init_ground_atoms, objects)
    # Create the goal.
    goal = {
        GroundAtom(predicate_name_to_predicate[a.name],
                   [object_name_to_obj[n] for n, _ in a.signature])
        for a in pyperplan_problem.goal
    }
    # Finalize the task.
    task = Task(init, goal)
    return task


def _create_predicate_classifier(
        pred: Predicate) -> Callable[[State, Sequence[Object]], bool]:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        assert isinstance(s, _PDDLEnvState)
        return GroundAtom(pred, objs) in s.get_ground_atoms()

    return _classifier


def _file_problem_generator(dir_name: str,
                            indices: Sequence[int]) -> PDDLProblemGenerator:
    # Load all of the PDDL problem strings from files.
    problems = []
    for idx in indices:
        path = utils.get_env_asset_path(f"pddl/{dir_name}/task{idx}.pddl")
        with open(path, encoding="utf-8") as f:
            problem_str = f.read()
        problems.append(problem_str)

    def _problem_gen(num: int, rng: np.random.Generator) -> List[str]:
        del rng  # unused
        assert len(problems) >= num
        return problems[:num]

    return _problem_gen
