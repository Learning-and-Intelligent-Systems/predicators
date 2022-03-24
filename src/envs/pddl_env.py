"""Environments that are derived from PDDL.

There are no continuous aspects of the state or action space. These
environments are similar to PDDLGym.
"""

import abc
from typing import Callable, Collection, Dict, List, Optional, Sequence, Set, \
    Tuple, cast

import numpy as np
from gym.spaces import Box
from pyperplan.pddl.parser import TraversePDDLDomain, TraversePDDLProblem, \
    parse_domain_def, parse_lisp_iterator, parse_problem_def

from predicators.src import utils
from predicators.src.envs import BaseEnv
from predicators.src.settings import CFG
from predicators.src.structs import Action, Array, GroundAtom, Image, \
    LiftedAtom, Object, ParameterizedOption, Predicate, State, \
    STRIPSOperator, Task, Type, Variable, _GroundSTRIPSOperator

###############################################################################
#                                    ABCs                                     #
###############################################################################

# Given a desired number of problems and an rng, returns a list of that many
# PDDL problem strs.
_PDDLProblemGenerator = Callable[[int, np.random.Generator], List[str]]
# The type of state.simulator_state in these enviroments.
_PDDLEnvSimulatorState = Set[GroundAtom]


class _PDDLEnv(BaseEnv):
    """An environment induced by PDDL.

    The state space is mostly unused. The continuous vectors are dummies. What
    is actually used is state.simulator_state, which holds the current ground
    atoms (see _PDDLEnvSimulatorState). Note that we need to use this pattern,
    as opposed to just maintaining the ground atoms internally in the env,
    because the predicate classifiers need access to the ground atoms.

    The action space is hacked to conform to our convention that actions
    are fixed dimensional vectors. Users of this class should not need
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

    @property
    @abc.abstractmethod
    def _domain_str(self) -> str:
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _pddl_train_problem_generator(self) -> _PDDLProblemGenerator:
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _pddl_test_problem_generator(self) -> _PDDLProblemGenerator:
        raise NotImplementedError("Override me!")

    @property
    def strips_operators(self) -> Set[STRIPSOperator]:
        """Expose the STRIPSOperators for use by oracles."""
        return self._strips_operators

    def simulate(self, state: State, action: Action) -> State:
        objs = list(state)
        # Convert the state into a Set[GroundAtom].
        ground_atoms = _state_to_ground_atoms(state)
        # Convert the action into a _GroundSTRIPSOperator.
        ground_op = self._action_to_ground_strips_op(action, objs)
        # Apply the operator.
        next_ground_atoms = utils.apply_operator(ground_op, ground_atoms)
        # Convert back into a State.
        next_state = _ground_atoms_to_state(next_ground_atoms, objs)
        return next_state

    def _generate_train_tasks(self) -> List[Task]:
        return self._generate_tasks(CFG.num_train_tasks,
                                    self._pddl_train_problem_generator,
                                    self._train_rng)

    def _generate_test_tasks(self) -> List[Task]:
        return self._generate_tasks(CFG.num_test_tasks,
                                    self._pddl_test_problem_generator,
                                    self._test_rng)

    def _generate_tasks(self, num_tasks: int,
                        problem_gen: _PDDLProblemGenerator,
                        rng: np.random.Generator) -> List[Task]:
        tasks = []
        for pddl_problem_str in problem_gen(num_tasks, rng):
            task = self._pddl_problem_str_to_task(pddl_problem_str)
            tasks.append(task)
        return tasks

    @property
    def predicates(self) -> Set[Predicate]:
        return self._predicates

    @property
    def goal_predicates(self) -> Set[Predicate]:
        # For now, we assume that all predicates may be included as part of the
        # goals. This is not important because these environments are not
        # currently used for predicate invention. If we do want to use these
        # for predicate invention in the future, we can revisit this, and
        # try to automatically detect which predicates appear in problem goals.
        return self._predicates

    @property
    def types(self) -> Set[Type]:
        return self._types

    @property
    def options(self) -> Set[ParameterizedOption]:
        # The parameterized options are 1:1 with the STRIPS operators.
        return {
            self._strips_operator_to_parameterized_option(op)
            for op in self._strips_operators
        }

    @property
    def action_space(self) -> Box:
        # See class docstring for explanation.
        num_ops = len(self._strips_operators)
        max_arity = max(len(op.parameters) for op in self._strips_operators)
        lb = np.array([0.0 for _ in range(max_arity + 1)], dtype=np.float32)
        ub = np.array([num_ops - 1.0] + [np.inf for _ in range(max_arity)],
                      dtype=np.float32)
        return Box(lb, ub, dtype=np.float32)

    def _action_to_ground_strips_op(
            self, action: Action,
            ordered_objects: List[Object]) -> _GroundSTRIPSOperator:
        action_arr = action.arr
        assert all(float(a).is_integer() for a in action_arr)
        op_idx = int(action_arr[0])
        op = self._ordered_strips_operators[op_idx]
        op_arity = len(op.parameters)
        obj_idxs = [int(i) for i in action_arr[1:op_arity + 1]]
        objs = tuple(ordered_objects[i] for i in obj_idxs)
        return op.ground(objs)

    def render_state(self,
                     state: State,
                     task: Task,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> List[Image]:
        raise NotImplementedError("Render not implemented for PDDLEnv.")

    def _pddl_problem_str_to_task(self, pddl_problem_str: str) -> Task:
        # Let pyperplan do most of the heavy lifting.
        # Pyperplan needs the domain to parse the problem.
        lisp_iterator = parse_lisp_iterator(self._domain_str.split("\n"))
        domain_ast = parse_domain_def(lisp_iterator)
        visitor = TraversePDDLDomain()
        domain_ast.accept(visitor)
        pyperplan_domain = visitor.domain
        # Now that we have the domain, parse the problem.
        lisp_iterator = parse_lisp_iterator(pddl_problem_str.split("\n"))
        problem_ast = parse_problem_def(lisp_iterator)
        visitor = TraversePDDLProblem(pyperplan_domain)
        problem_ast.accept(visitor)
        pyperplan_problem = visitor.get_problem()
        # Create the objects.
        type_name_to_type = {t.name: t for t in self.types}
        object_name_to_obj = {
            o: Object(o, type_name_to_type[t.name])
            for o, t in pyperplan_problem.objects.items()
        }
        objects = set(object_name_to_obj.values())
        # Create the initial state.
        predicate_name_to_predicate = {p.name: p for p in self.predicates}
        init_ground_atoms = {
            GroundAtom(predicate_name_to_predicate[a.name],
                       [object_name_to_obj[n] for n, _ in a.signature])
            for a in pyperplan_problem.initial_state
        }
        init = _ground_atoms_to_state(init_ground_atoms, objects)
        # Create the goal.
        goal = {
            GroundAtom(predicate_name_to_predicate[a.name],
                       [object_name_to_obj[n] for n, _ in a.signature])
            for a in pyperplan_problem.goal
        }
        # Finalize the task.
        task = Task(init, goal)
        return task

    def _strips_operator_to_parameterized_option(
            self, op: STRIPSOperator) -> ParameterizedOption:
        name = op.name
        types = [p.type for p in op.parameters]
        op_idx = self._ordered_strips_operators.index(op)
        act_dims = self.action_space.shape[0]

        def policy(s: State, m: Dict, o: Sequence[Object], p: Array) -> Action:
            del m, p  # unused
            ordered_objs = list(s)
            obj_idxs = [ordered_objs.index(obj) for obj in o]
            act_arr = np.zeros(act_dims, dtype=np.float32)
            act_arr[0] = op_idx
            act_arr[1:len(obj_idxs) + 1] = obj_idxs
            return Action(act_arr)

        def initiable(s: State, m: Dict, o: Sequence[Object],
                      p: Array) -> bool:
            del m, p  # unused
            ground_atoms = _state_to_ground_atoms(s)
            ground_op = op.ground(tuple(o))
            return ground_op.preconditions.issubset(ground_atoms)

        return utils.SingletonParameterizedOption(name,
                                                  policy,
                                                  types,
                                                  initiable=initiable)


class _FixedTasksPDDLEnv(_PDDLEnv):
    """An env where tasks are induced by static PDDL problem files."""

    @property
    @abc.abstractmethod
    def _pddl_problem_asset_dir(self) -> str:
        """The name of the problem file directory in assets/pddl."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _train_problem_indices(self) -> List[int]:
        """The problem file names should be task{idx}.pddl."""
        raise NotImplementedError("Override me!")

    @property
    @abc.abstractmethod
    def _test_problem_indices(self) -> List[int]:
        """The problem file names should be task{idx}.pddl."""
        raise NotImplementedError("Override me!")

    @property
    def _pddl_train_problem_generator(self) -> _PDDLProblemGenerator:
        assert len(self._train_problem_indices) >= CFG.num_train_tasks
        return _file_problem_generator(self._pddl_problem_asset_dir,
                                       self._train_problem_indices)

    @property
    def _pddl_test_problem_generator(self) -> _PDDLProblemGenerator:
        assert len(self._test_problem_indices) >= CFG.num_test_tasks
        return _file_problem_generator(self._pddl_problem_asset_dir,
                                       self._test_problem_indices)


###############################################################################
#                              Concrete Envs                                  #
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
        return list(range(1, 6))

    @property
    def _test_problem_indices(self) -> List[int]:
        return list(range(6, 11))


###############################################################################
#                            Utility functions                                #
###############################################################################


def _state_to_ground_atoms(state: State) -> Set[GroundAtom]:
    return cast(_PDDLEnvSimulatorState, state.simulator_state)


def _ground_atoms_to_state(ground_atoms: Set[GroundAtom],
                           objects: Collection[Object]) -> State:
    dummy_state_dict = {o: np.zeros(1, dtype=np.float32) for o in objects}
    return State(dummy_state_dict, simulator_state=ground_atoms)


def _parse_pddl_domain(
        domain_str: str
) -> Tuple[Set[Type], Set[Predicate], Set[STRIPSOperator]]:
    # Let pyperplan do most of the heavy lifting.
    domain_ast = parse_domain_def(parse_lisp_iterator(domain_str.split("\n")))
    visitor = TraversePDDLDomain()
    domain_ast.accept(visitor)
    pyperplan_domain = visitor.domain
    pyperplan_types = pyperplan_domain.types
    pyperplan_predicates = pyperplan_domain.predicates
    pyperplan_operators = pyperplan_domain.actions
    # Convert the pyperplan domain into our structs.
    pyperplan_type_to_type = {
        pyperplan_types[t]: Type(t, ["dummy_feat"])
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
        name_to_param = {p.name: p for p in parameters}
        preconditions = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [name_to_param[n] for n, _ in a.signature])
            for a in pyper_op.precondition
        }
        add_effects = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [name_to_param[n] for n, _ in a.signature])
            for a in pyper_op.effect.addlist
        }
        delete_effects = {
            LiftedAtom(predicate_name_to_predicate[a.name],
                       [name_to_param[n] for n, _ in a.signature])
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


def _create_predicate_classifier(
        pred: Predicate) -> Callable[[State, Sequence[Object]], bool]:

    def _classifier(s: State, objs: Sequence[Object]) -> bool:
        sim = cast(_PDDLEnvSimulatorState, s.simulator_state)
        return GroundAtom(pred, objs) in sim

    return _classifier


def _file_problem_generator(dir_name: str,
                            indices: Sequence[int]) -> _PDDLProblemGenerator:
    # Load all of the PDDL problem strs from files.
    problems = []
    for idx in indices:
        path = utils.get_env_asset_path(f"pddl/{dir_name}/task{idx}.pddl")
        with open(path, encoding="utf-8") as f:
            problem_str = f.read()
        problems.append(problem_str)

    def _problem_gen(num: int, rng: np.random.Generator) -> List[str]:
        del rng  # unused
        return problems[:num]

    return _problem_gen
