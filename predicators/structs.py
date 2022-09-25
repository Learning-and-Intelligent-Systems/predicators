"""Structs used throughout the codebase."""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from functools import cached_property, lru_cache
from typing import Any, Callable, Collection, DefaultDict, Dict, Iterator, \
    List, Optional, Sequence, Set, Tuple, TypeVar, Union, cast

import numpy as np
from gym.spaces import Box
from numpy.typing import NDArray
from tabulate import tabulate

from predicators.settings import CFG


@dataclass(frozen=True, order=True)
class Type:
    """Struct defining a type."""
    name: str
    feature_names: Sequence[str] = field(repr=False)
    parent: Optional[Type] = field(default=None, repr=False)

    @property
    def dim(self) -> int:
        """Dimensionality of the feature vector of this object type."""
        return len(self.feature_names)

    def __call__(self, name: str) -> _TypedEntity:
        """Convenience method for generating _TypedEntities."""
        if name.startswith("?"):
            return Variable(name, self)
        return Object(name, self)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.feature_names)))


@dataclass(frozen=True, order=True, repr=False)
class _TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block).

    Should not be instantiated externally.
    """
    name: str
    type: Type

    @cached_property
    def _str(self) -> str:
        return f"{self.name}:{self.type.name}"

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return self._str

    def is_instance(self, t: Type) -> bool:
        """Return whether this entity is an instance of the given type, taking
        hierarchical typing into account."""
        cur_type: Optional[Type] = self.type
        while cur_type is not None:
            if cur_type == t:
                return True
            cur_type = cur_type.parent
        return False


@dataclass(frozen=True, order=True, repr=False)
class Object(_TypedEntity):
    """Struct defining an Object, which is just a _TypedEntity whose name does
    not start with "?"."""

    def __post_init__(self) -> None:
        assert not self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass(frozen=True, order=True, repr=False)
class Variable(_TypedEntity):
    """Struct defining a Variable, which is just a _TypedEntity whose name
    starts with "?"."""

    def __post_init__(self) -> None:
        assert self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass
class State:
    """Struct defining the low-level state of the world."""
    data: Dict[Object, Array]
    # Some environments will need to store additional simulator state, so
    # this field is provided.
    simulator_state: Optional[Any] = None

    def __post_init__(self) -> None:
        # Check feature vector dimensions.
        for obj in self:
            assert len(self[obj]) == obj.type.dim

    def __iter__(self) -> Iterator[Object]:
        """An iterator over the state's objects, in sorted order."""
        return iter(sorted(self.data))

    def __getitem__(self, key: Object) -> Array:
        return self.data[key]

    def get(self, obj: Object, feature_name: str) -> Any:
        """Look up an object feature by name."""
        idx = obj.type.feature_names.index(feature_name)
        return self.data[obj][idx]

    def set(self, obj: Object, feature_name: str, feature_val: Any) -> None:
        """Set the value of an object feature by name."""
        idx = obj.type.feature_names.index(feature_name)
        self.data[obj][idx] = feature_val

    def get_objects(self, object_type: Type) -> List[Object]:
        """Return objects of the given type in the order of __iter__()."""
        return [o for o in self if o.type == object_type]

    def vec(self, objects: Sequence[Object]) -> Array:
        """Concatenated vector of features for each of the objects in the given
        ordered list."""
        feats: List[Array] = []
        if len(objects) == 0:
            return np.zeros(0, dtype=np.float32)
        for obj in objects:
            feats.append(self[obj])
        return np.hstack(feats)

    def copy(self) -> State:
        """Return a copy of this state.

        The simulator state is assumed to be immutable.
        """
        new_data = {}
        for obj in self:
            new_data[obj] = self._copy_state_value(self.data[obj])
        return State(new_data, simulator_state=self.simulator_state)

    def _copy_state_value(self, val: Any) -> Any:
        if val is None or isinstance(val, (float, bool, int, str)):
            return val
        if isinstance(val, (list, tuple, set)):
            return type(val)(self._copy_state_value(v) for v in val)
        assert hasattr(val, "copy")
        return val.copy()

    def allclose(self, other: State) -> bool:
        """Return whether this state is close enough to another one, i.e., its
        objects are the same, and the features are close."""
        if self.simulator_state is not None or \
           other.simulator_state is not None:
            raise NotImplementedError("Cannot use allclose when "
                                      "simulator_state is not None.")
        if not sorted(self.data) == sorted(other.data):
            return False
        for obj in self.data:
            if not np.allclose(self.data[obj], other.data[obj], atol=1e-3):
                return False
        return True

    def pretty_str(self) -> str:
        """Display the state in a nice human-readable format."""
        type_to_table: Dict[Type, List[List[str]]] = {}
        for obj in self:
            if obj.type not in type_to_table:
                type_to_table[obj.type] = []
            type_to_table[obj.type].append([obj.name] + \
                                            list(map(str, self[obj])))
        table_strs = []
        for t in sorted(type_to_table):
            headers = ["type: " + t.name] + list(t.feature_names)
            table_strs.append(tabulate(type_to_table[t], headers=headers))
        ll = max(
            len(line) for table in table_strs for line in table.split("\n"))
        prefix = "#" * (ll // 2 - 3) + " STATE " + "#" * (ll - ll // 2 -
                                                          4) + "\n"
        suffix = "\n" + "#" * ll + "\n"
        return prefix + "\n\n".join(table_strs) + suffix


DefaultState = State({})


@dataclass(frozen=True, order=True, repr=False)
class Predicate:
    """Struct defining a predicate (a lifted classifier over states)."""
    name: str
    types: Sequence[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[State, Sequence[Object]],
                          bool] = field(compare=False)

    def __call__(self, entities: Sequence[_TypedEntity]) -> _Atom:
        """Convenience method for generating Atoms."""
        if self.arity == 0:
            raise ValueError("Cannot use __call__ on a 0-arity predicate, "
                             "since we can't determine whether it becomes a "
                             "LiftedAtom or a GroundAtom. Use the LiftedAtom "
                             "or GroundAtom constructors directly instead")
        if all(isinstance(ent, Variable) for ent in entities):
            return LiftedAtom(self, entities)
        if all(isinstance(ent, Object) for ent in entities):
            return GroundAtom(self, entities)
        raise ValueError("Cannot instantiate Atom with mix of "
                         "variables and objects")

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __hash__(self) -> int:
        return self._hash

    @cached_property
    def arity(self) -> int:
        """The arity of this predicate (number of arguments)."""
        return len(self.types)

    def holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Public method for calling the classifier.

        Performs type checking first.
        """
        assert len(objects) == self.arity
        for obj, pred_type in zip(objects, self.types):
            assert isinstance(obj, Object)
            assert obj.is_instance(pred_type)
        return self._classifier(state, objects)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)

    def pretty_str(self) -> Tuple[str, str]:
        """Display the predicate in a nice human-readable format.

        Returns a tuple of (variables string, body string).
        """
        if hasattr(self._classifier, "pretty_str"):
            # This is an invented predicate, from the predicate grammar.
            pretty_str_f = getattr(self._classifier, "pretty_str")
            return pretty_str_f()
        # This is a known predicate, not from the predicate grammar.
        vars_str = ", ".join(
            f"{CFG.grammar_search_classifier_pretty_str_names[i]}:{t.name}"
            for i, t in enumerate(self.types))
        vars_str_no_types = ", ".join(
            f"{CFG.grammar_search_classifier_pretty_str_names[i]}"
            for i in range(self.arity))
        body_str = f"{self.name}({vars_str_no_types})"
        return vars_str, body_str

    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to a PDDL
        file."""
        if self.arity == 0:
            return f"({self.name})"
        vars_str = " ".join(f"?x{i} - {t.name}"
                            for i, t in enumerate(self.types))
        return f"({self.name} {vars_str})"

    def get_negation(self) -> Predicate:
        """Return a negated version of this predicate."""
        return Predicate("NOT-" + self.name, self.types,
                         self._negated_classifier)

    def _negated_classifier(self, state: State,
                            objects: Sequence[Object]) -> bool:
        # Separate this into a named function for pickling reasons.
        return not self._classifier(state, objects)


@dataclass(frozen=True, repr=False, eq=False)
class _Atom:
    """Struct defining an atom (a predicate applied to either variables or
    objects).

    Should not be instantiated externally.
    """
    predicate: Predicate
    entities: Sequence[_TypedEntity]

    def __post_init__(self) -> None:
        if isinstance(self.entities, _TypedEntity):
            raise ValueError("Atoms expect a sequence of entities, not a "
                             "single entity.")
        assert len(self.entities) == self.predicate.arity
        for ent, pred_type in zip(self.entities, self.predicate.types):
            assert ent.is_instance(pred_type)

    @property
    def _str(self) -> str:
        raise NotImplementedError("Override me")

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to a PDDL
        file."""
        if not self.entities:
            return f"({self.predicate.name})"
        entities_str = " ".join(e.name for e in self.entities)
        return f"({self.predicate.name} {entities_str})"

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) < str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LiftedAtom(_Atom):
    """Struct defining a lifted atom (a predicate applied to variables)."""

    @cached_property
    def variables(self) -> List[Variable]:
        """Arguments for this lifted atom.

        A list of "Variable"s.
        """
        return list(cast(Variable, ent) for ent in self.entities)

    @cached_property
    def _str(self) -> str:
        return (str(self.predicate) + "(" +
                ", ".join(map(str, self.variables)) + ")")

    def ground(self, sub: VarToObjSub) -> GroundAtom:
        """Create a GroundAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return GroundAtom(self.predicate, [sub[v] for v in self.variables])


@dataclass(frozen=True, repr=False, eq=False)
class GroundAtom(_Atom):
    """Struct defining a ground atom (a predicate applied to objects)."""

    @cached_property
    def objects(self) -> List[Object]:
        """Arguments for this ground atom.

        A list of "Object"s.
        """
        return list(cast(Object, ent) for ent in self.entities)

    @cached_property
    def _str(self) -> str:
        return (str(self.predicate) + "(" + ", ".join(map(str, self.objects)) +
                ")")

    def lift(self, sub: ObjToVarSub) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.objects).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[o] for o in self.objects])

    def holds(self, state: State) -> bool:
        """Check whether this ground atom holds in the given state."""
        return self.predicate.holds(state, self.objects)


@dataclass(frozen=True, eq=False)
class Task:
    """Struct defining a task, which is a pair of initial state and goal."""
    init: State
    goal: Set[GroundAtom]

    def __post_init__(self) -> None:
        # Verify types.
        assert isinstance(self.init, State)
        for atom in self.goal:
            assert isinstance(atom, GroundAtom)

    def goal_holds(self, state: State) -> bool:
        """Return whether the goal of this task holds in the given state."""
        return all(goal_atom.holds(state) for goal_atom in self.goal)


DefaultTask = Task(DefaultState, set())


@dataclass(frozen=True, eq=False)
class ParameterizedOption:
    """Struct defining a parameterized option, which has a parameter space and
    can be ground into an Option, given parameter values.

    An option is composed of a policy, an initiation classifier, and a
    termination condition. We will stick with deterministic termination
    conditions. For a parameterized option, all of these are conditioned
    on parameters.
    """
    name: str
    types: Sequence[Type]
    params_space: Box = field(repr=False)
    # A policy maps a state, memory dict, objects, and parameters to an action.
    # The objects' types will match those in self.types. The parameters
    # will be contained in params_space.
    policy: Callable[[State, Dict, Sequence[Object], Array],
                     Action] = field(repr=False)
    # An initiation classifier maps a state, memory dict, objects, and
    # parameters to a bool, which is True iff the option can start
    # now. The objects' types will match those in self.types. The
    # parameters will be contained in params_space.
    initiable: Callable[[State, Dict, Sequence[Object], Array],
                        bool] = field(repr=False)
    # A termination condition maps a state, memory dict, objects, and
    # parameters to a bool, which is True iff the option should
    # terminate now. The objects' types will match those in
    # self.types. The parameters will be contained in params_space.
    terminal: Callable[[State, Dict, Sequence[Object], Array],
                       bool] = field(repr=False)

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, ParameterizedOption)
        return self.name == other.name

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, ParameterizedOption)
        return self.name < other.name

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, ParameterizedOption)
        return self.name > other.name

    def __hash__(self) -> int:
        return self._hash

    def ground(self, objects: Sequence[Object], params: Array) -> _Option:
        """Ground into an Option, given objects and parameter values."""
        assert len(objects) == len(self.types)
        for obj, t in zip(objects, self.types):
            assert obj.is_instance(t)
        params = np.array(params, dtype=self.params_space.dtype)
        assert self.params_space.contains(params)
        memory: Dict = {}  # each option has its own memory dict
        return _Option(
            self.name,
            lambda s: self.policy(s, memory, objects, params),
            initiable=lambda s: self.initiable(s, memory, objects, params),
            terminal=lambda s: self.terminal(s, memory, objects, params),
            parent=self,
            objects=objects,
            params=params,
            memory=memory)


@dataclass(eq=False)
class _Option:
    """Struct defining an option, which is like a parameterized option except
    that its components are not conditioned on objects/parameters.

    Should not be instantiated externally.
    """
    name: str
    # A policy maps a state to an action.
    _policy: Callable[[State], Action] = field(repr=False)
    # An initiation classifier maps a state to a bool, which is True
    # iff the option can start now.
    initiable: Callable[[State], bool] = field(repr=False)
    # A termination condition maps a state to a bool, which is True
    # iff the option should terminate now.
    terminal: Callable[[State], bool] = field(repr=False)
    # The parameterized option that generated this option.
    parent: ParameterizedOption = field(repr=False)
    # The objects that were used to ground this option.
    objects: Sequence[Object]
    # The parameters that were used to ground this option.
    params: Array
    # The memory dictionary for this option.
    memory: Dict = field(repr=False)

    def policy(self, state: State) -> Action:
        """Call the policy and set the action's option."""
        action = self._policy(state)
        action.set_option(self)
        return action


DummyOption: _Option = ParameterizedOption(
    "DummyOption", [], Box(0, 1,
                           (1, )), lambda s, m, o, p: Action(np.array([0.0])),
    lambda s, m, o, p: False, lambda s, m, o, p: True).ground([],
                                                              np.array([0.0]))
DummyOption.parent.params_space.seed(0)  # for reproducibility


@dataclass(frozen=True, repr=False, eq=False)
class STRIPSOperator:
    """Struct defining a symbolic operator (as in STRIPS).

    Lifted! Note here that the ignore_effects - unlike the
    add_effects and delete_effects - are universally
    quantified over all possible groundings.
    """
    name: str
    parameters: Sequence[Variable]
    preconditions: Set[LiftedAtom]
    add_effects: Set[LiftedAtom]
    delete_effects: Set[LiftedAtom]
    ignore_effects: Set[Predicate]

    def make_nsrt(
        self,
        option: ParameterizedOption,
        option_vars: Sequence[Variable],
        sampler: NSRTSampler = field(repr=False)
    ) -> NSRT:
        """Make an NSRT out of this STRIPSOperator object, given the necessary
        additional fields."""
        return NSRT(self.name, self.parameters, self.preconditions,
                    self.add_effects, self.delete_effects, self.ignore_effects,
                    option, option_vars, sampler)

    @lru_cache(maxsize=None)
    def ground(self, objects: Tuple[Object]) -> _GroundSTRIPSOperator:
        """Ground into a _GroundSTRIPSOperator, given objects.

        Insist that objects are tuple for hashing in cache.
        """
        assert isinstance(objects, tuple)
        assert len(objects) == len(self.parameters)
        assert all(
            o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        preconditions = {atom.ground(sub) for atom in self.preconditions}
        add_effects = {atom.ground(sub) for atom in self.add_effects}
        delete_effects = {atom.ground(sub) for atom in self.delete_effects}
        return _GroundSTRIPSOperator(self, list(objects), preconditions,
                                     add_effects, delete_effects)

    @cached_property
    def _str(self) -> str:
        return f"""STRIPS-{self.name}:
    Parameters: {self.parameters}
    Preconditions: {sorted(self.preconditions, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}
    Ignore Effects: {sorted(self.ignore_effects, key=str)}"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to a PDDL
        file."""
        params_str = " ".join(f"{p.name} - {p.type.name}"
                              for p in self.parameters)
        preconds_str = "\n        ".join(
            atom.pddl_str() for atom in sorted(self.preconditions))
        effects_str = "\n        ".join(atom.pddl_str()
                                        for atom in sorted(self.add_effects))
        if self.delete_effects:
            effects_str += "\n        "
            effects_str += "\n        ".join(
                f"(not {atom.pddl_str()})"
                for atom in sorted(self.delete_effects))
        if self.ignore_effects:
            if len(effects_str) != 0:
                effects_str += "\n        "
            for pred in sorted(self.ignore_effects):
                pred_types_str = " ".join(f"?x{i} - {t.name}"
                                          for i, t in enumerate(pred.types))
                pred_eff_variables_str = " ".join(f"?x{i}"
                                                  for i in range(pred.arity))
                effects_str += f"(forall ({pred_types_str})" +\
                    f" (not ({pred.name} {pred_eff_variables_str})))"
                effects_str += "\n        "
        return f"""(:action {self.name}
    :parameters ({params_str})
    :precondition (and {preconds_str})
    :effect (and {effects_str})
  )"""

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, STRIPSOperator)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, STRIPSOperator)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, STRIPSOperator)
        return str(self) > str(other)

    def copy_with(self, **kwargs: Any) -> STRIPSOperator:
        """Create a copy of the operator, optionally while replacing any of the
        arguments."""
        default_kwargs = dict(name=self.name,
                              parameters=self.parameters,
                              preconditions=self.preconditions,
                              add_effects=self.add_effects,
                              delete_effects=self.delete_effects,
                              ignore_effects=self.ignore_effects)
        assert set(kwargs.keys()).issubset(default_kwargs.keys())
        default_kwargs.update(kwargs)
        # mypy is known to have issues with this pattern:
        # https://github.com/python/mypy/issues/5382
        return STRIPSOperator(**default_kwargs)  # type: ignore

    def effect_to_ignore_effect(self, effect: LiftedAtom,
                                option_vars: Sequence[Variable],
                                add_or_delete: str) -> STRIPSOperator:
        """Return a new STRIPS operator resulting from turning the given effect
        (either add or delete) into an ignore effect."""
        assert add_or_delete in ("add", "delete")
        if add_or_delete == "add":
            assert effect in self.add_effects
            new_add_effects = self.add_effects - {effect}
            new_delete_effects = self.delete_effects
        else:
            new_add_effects = self.add_effects
            assert effect in self.delete_effects
            new_delete_effects = self.delete_effects - {effect}
        # Since we are removing an effect, it could be the case
        # that parameters need to be removed from the operator.
        remaining_params = {
            p
            for atom in self.preconditions | new_add_effects
            | new_delete_effects for p in atom.variables
        } | set(option_vars)
        new_params = [p for p in self.parameters if p in remaining_params]
        return STRIPSOperator(self.name, new_params, self.preconditions,
                              new_add_effects, new_delete_effects,
                              self.ignore_effects | {effect.predicate})

    def get_complexity(self) -> float:
        """Get the complexity of this operator.

        We only care about the arity of the operator, since that is what
        affects grounding. We'll use 2^arity as a measure of grounding
        effort.
        """
        return float(2**len(self.parameters))


@dataclass(frozen=True, repr=False, eq=False)
class _GroundSTRIPSOperator:
    """A STRIPSOperator + objects.

    Should not be instantiated externally.
    """
    parent: STRIPSOperator
    objects: Sequence[Object]
    preconditions: Set[GroundAtom]
    add_effects: Set[GroundAtom]
    delete_effects: Set[GroundAtom]

    @cached_property
    def _str(self) -> str:
        return f"""GroundSTRIPS-{self.name}:
    Parameters: {self.objects}
    Preconditions: {sorted(self.preconditions, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}
    Ignore Effects: {sorted(self.ignore_effects, key=str)}"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @property
    def name(self) -> str:
        """Name of this ground STRIPSOperator."""
        return self.parent.name

    @property
    def ignore_effects(self) -> Set[Predicate]:
        """Ignore effects from the parent."""
        return self.parent.ignore_effects

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _GroundSTRIPSOperator)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _GroundSTRIPSOperator)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, _GroundSTRIPSOperator)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class NSRT:
    """Struct defining an NSRT, which contains the components of a STRIPS
    operator, a parameterized option, and a sampler function.

    "NSRT" stands for "Neuro-Symbolic Relational Transition Model".
    Paper: https://arxiv.org/abs/2105.14074
    """
    name: str
    parameters: Sequence[Variable]
    preconditions: Set[LiftedAtom]
    add_effects: Set[LiftedAtom]
    delete_effects: Set[LiftedAtom]
    ignore_effects: Set[Predicate]
    option: ParameterizedOption
    # A subset of parameters corresponding to the (lifted) arguments of the
    # option that this NSRT contains.
    option_vars: Sequence[Variable]
    # A sampler maps a state, RNG, and objects to option parameters.
    _sampler: NSRTSampler = field(repr=False)

    @cached_property
    def _str(self) -> str:
        option_var_str = ", ".join([str(v) for v in self.option_vars])
        return f"""NSRT-{self.name}:
    Parameters: {self.parameters}
    Preconditions: {sorted(self.preconditions, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}
    Ignore Effects: {sorted(self.ignore_effects, key=str)}
    Option Spec: {self.option.name}({option_var_str})"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @property
    def op(self) -> STRIPSOperator:
        """Return the STRIPSOperator associated with this NSRT."""
        return STRIPSOperator(self.name, self.parameters, self.preconditions,
                              self.add_effects, self.delete_effects,
                              self.ignore_effects)

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def pddl_str(self) -> str:
        """Get a string representation suitable for writing out to a PDDL
        file."""
        return self.op.pddl_str()

    def pretty_str(self, name_map: Dict[str, str]) -> str:
        """Display the NSRT in a nice human-readable format, given a mapping to
        new predicate names for any invented predicates."""
        out = ""
        out += f"{self.name}:\n\tParameters: {self.parameters}"
        for name, atoms in [("Preconditions", self.preconditions),
                            ("Add Effects", self.add_effects),
                            ("Delete Effects", self.delete_effects)]:
            out += f"\n\t{name}:"
            for atom in atoms:
                pretty_pred = atom.predicate.pretty_str()[1]
                new_name = (name_map[pretty_pred] if pretty_pred in name_map
                            else str(atom.predicate))
                var_str = ", ".join(map(str, atom.variables))
                out += f"\n\t\t{new_name}({var_str})"
        option_var_strs = [str(v) for v in self.option_vars]
        out += f"\n\tOption Spec: ({self.option.name}, {option_var_strs})"
        return out

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, NSRT)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, NSRT)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, NSRT)
        return str(self) > str(other)

    @property
    def sampler(self) -> NSRTSampler:
        """This NSRT's sampler."""
        return self._sampler

    def ground(self, objects: Sequence[Object]) -> _GroundNSRT:
        """Ground into a _GroundNSRT, given objects."""
        assert len(objects) == len(self.parameters)
        assert all(
            o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        preconditions = {atom.ground(sub) for atom in self.preconditions}
        add_effects = {atom.ground(sub) for atom in self.add_effects}
        delete_effects = {atom.ground(sub) for atom in self.delete_effects}
        option_objs = [sub[v] for v in self.option_vars]
        return _GroundNSRT(self, objects, preconditions, add_effects,
                           delete_effects, self.option, option_objs,
                           self._sampler)

    def filter_predicates(self, kept: Collection[Predicate]) -> NSRT:
        """Keep only the given predicates in the preconditions, add effects,
        delete effects, and ignore effects.

        Note that the parameters must stay the same for the sake of the
        sampler inputs.
        """
        preconditions = {a for a in self.preconditions if a.predicate in kept}
        add_effects = {a for a in self.add_effects if a.predicate in kept}
        delete_effects = {
            a
            for a in self.delete_effects if a.predicate in kept
        }
        ignore_effects = {a for a in self.ignore_effects if a in kept}
        return NSRT(self.name, self.parameters, preconditions, add_effects,
                    delete_effects, ignore_effects, self.option,
                    self.option_vars, self._sampler)


@dataclass(frozen=True, repr=False, eq=False)
class _GroundNSRT:
    """A ground NSRT is an NSRT + objects.

    Should not be instantiated externally.
    """
    parent: NSRT
    objects: Sequence[Object]
    preconditions: Set[GroundAtom]
    add_effects: Set[GroundAtom]
    delete_effects: Set[GroundAtom]
    option: ParameterizedOption
    option_objs: Sequence[Object]
    _sampler: NSRTSampler = field(repr=False)

    @cached_property
    def _str(self) -> str:
        return f"""GroundNSRT-{self.name}:
    Parameters: {self.objects}
    Preconditions: {sorted(self.preconditions, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}
    Ignore Effects: {sorted(self.ignore_effects, key=str)}
    Option: {self.option}
    Option Objects: {self.option_objs}"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @property
    def name(self) -> str:
        """Name of this ground NSRT."""
        return self.parent.name

    @property
    def ignore_effects(self) -> Set[Predicate]:
        """Ignore effects from the parent."""
        return self.parent.ignore_effects

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _GroundNSRT)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _GroundNSRT)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, _GroundNSRT)
        return str(self) > str(other)

    def sample_option(self, state: State, goal: Set[GroundAtom],
                      rng: np.random.Generator) -> _Option:
        """Sample an _Option for this ground NSRT, by invoking the contained
        sampler.

        On the Option that is returned, one can call, e.g.,
        policy(state).
        """
        # Note that the sampler takes in ALL self.objects, not just the subset
        # self.option_objs of objects that are passed into the option.
        params = self._sampler(state, goal, rng, self.objects)
        # Clip the params into the params_space of self.option, for safety.
        low = self.option.params_space.low
        high = self.option.params_space.high
        params = np.clip(params, low, high)
        return self.option.ground(self.option_objs, params)

    def copy_with(self, **kwargs: Any) -> _GroundNSRT:
        """Create a copy of the ground NSRT, optionally while replacing any of
        the arguments."""
        default_kwargs = dict(parent=self.parent,
                              objects=self.objects,
                              preconditions=self.preconditions,
                              add_effects=self.add_effects,
                              delete_effects=self.delete_effects,
                              option=self.option,
                              option_objs=self.option_objs,
                              _sampler=self._sampler)
        assert set(kwargs.keys()).issubset(default_kwargs.keys())
        default_kwargs.update(kwargs)
        # mypy is known to have issues with this pattern:
        # https://github.com/python/mypy/issues/5382
        return _GroundNSRT(**default_kwargs)  # type: ignore


@dataclass(eq=False)
class Action:
    """An action in an environment.

    This is a light wrapper around a numpy float array that can
    optionally store the option which produced it.
    """
    _arr: Array
    _option: _Option = field(repr=False, default=DummyOption)

    @property
    def arr(self) -> Array:
        """The array representation of this action."""
        return self._arr

    def has_option(self) -> bool:
        """Whether this action has a non-default option attached."""
        return self._option.parent != DummyOption.parent

    def get_option(self) -> _Option:
        """Get the option that produced this action."""
        assert self.has_option()
        return self._option

    def set_option(self, option: _Option) -> None:
        """Set the option that produced this action."""
        self._option = option

    def unset_option(self) -> None:
        """Unset the option that produced this action."""
        self._option = DummyOption
        assert not self.has_option()


@dataclass(frozen=True, repr=False, eq=False)
class LowLevelTrajectory:
    """A structure representing a low-level trajectory, containing a state
    sequence, action sequence, and optional train task id. This trajectory may
    or may not be a demonstration.

    Invariant 1: If this trajectory is a demonstration, it must contain
    a train task idx and achieve the goal in the respective train task. This
    invariant is checked upon creation of the trajectory (in datasets) because
    the trajectory does not have a goal, it only has a train task idx.

    Invariant 2: The length of the state sequence is always one greater than
    the length of the action sequence.
    """
    _states: List[State]
    _actions: List[Action]
    _is_demo: bool = field(default=False)
    _train_task_idx: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        assert len(self._states) == len(self._actions) + 1
        if self._is_demo:
            assert self._train_task_idx is not None

    @property
    def states(self) -> List[State]:
        """States in the trajectory."""
        return self._states

    @property
    def actions(self) -> List[Action]:
        """Actions in the trajectory."""
        return self._actions

    @property
    def is_demo(self) -> bool:
        """Whether this trajectory is a demonstration."""
        return self._is_demo

    @property
    def train_task_idx(self) -> int:
        """The index of the train task."""
        assert self._train_task_idx is not None, \
            "This trajectory doesn't contain a train task idx!"
        return self._train_task_idx


@dataclass(repr=False, eq=False)
class Dataset:
    """A collection of LowLevelTrajectory objects, and optionally, lists of
    annotations, one per trajectory.

    For example, in interactive learning, an annotation for an offline
    learning Dataset would be of type List[Set[GroundAtom]] (with
    predicate classifiers deleted).
    """
    _trajectories: List[LowLevelTrajectory]
    _annotations: Optional[List[Any]] = field(default=None)

    def __post_init__(self) -> None:
        if self._annotations is not None:
            assert len(self._trajectories) == len(self._annotations)

    @property
    def trajectories(self) -> List[LowLevelTrajectory]:
        """The trajectories in the dataset."""
        return self._trajectories

    @property
    def has_annotations(self) -> bool:
        """Whether this dataset has annotations in it."""
        return self._annotations is not None

    @property
    def annotations(self) -> List[Any]:
        """The annotations in the dataset."""
        assert self._annotations is not None
        return self._annotations

    def append(self,
               trajectory: LowLevelTrajectory,
               annotation: Optional[Any] = None) -> None:
        """Append one more trajectory and annotation to the dataset."""
        if annotation is None:
            assert self._annotations is None
        else:
            assert self._annotations is not None
            self._annotations.append(annotation)
        self._trajectories.append(trajectory)


@dataclass(eq=False)
class Segment:
    """A segment represents a low-level trajectory that is the result of
    executing one option. The segment stores the abstract state (ground atoms)
    that held immediately before the option started executing, and the abstract
    state (ground atoms) that held immediately after.

    Segments are used during learning, when we don't necessarily know
    the option associated with the trajectory yet.
    """
    trajectory: LowLevelTrajectory
    init_atoms: Set[GroundAtom]
    final_atoms: Set[GroundAtom]
    _option: _Option = field(repr=False, default=DummyOption)
    _goal: Optional[Set[GroundAtom]] = field(default=None)
    # Field used by the backchaining algorithm (gen_to_spec_learner.py)
    necessary_add_effects: Optional[Set[GroundAtom]] = field(default=None)

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.actions) + 1

    @property
    def states(self) -> List[State]:
        """States in the trajectory."""
        return self.trajectory.states

    @property
    def actions(self) -> List[Action]:
        """Actions in the trajectory."""
        return self.trajectory.actions

    @property
    def add_effects(self) -> Set[GroundAtom]:
        """Atoms in the final atoms but not the init atoms.

        Do not cache; init and final atoms can change.
        """
        return self.final_atoms - self.init_atoms

    @property
    def delete_effects(self) -> Set[GroundAtom]:
        """Atoms in the init atoms but not the final atoms.

        Do not cache; init and final atoms can change.
        """
        return self.init_atoms - self.final_atoms

    def has_option(self) -> bool:
        """Whether this segment has a non-default option attached."""
        return self._option.parent != DummyOption.parent

    def get_option(self) -> _Option:
        """Get the option that produced this segment."""
        assert self.has_option()
        return self._option

    def set_option(self, option: _Option) -> None:
        """Set the option that produced this segment."""
        self._option = option

    def has_goal(self) -> bool:
        """Whether this segment has a non-default goal attached."""
        return self._goal is not None

    def get_goal(self) -> Set[GroundAtom]:
        """Get the goal associated with this segment."""
        assert self._goal is not None
        return self._goal

    def set_goal(self, goal: Set[GroundAtom]) -> None:
        """Set the goal associated with this segment."""
        self._goal = goal


@dataclass(eq=False, repr=False)
class PartialNSRTAndDatastore:
    """PNAD: A helper class for NSRT learning that contains information
    useful to maintain throughout the learning procedure. Each object of
    this class corresponds to a learned NSRT. We use this class because
    we don't want to clutter the NSRT class with a datastore, since data
    is only used for learning and is not part of the representation itself.
    """
    # The symbolic components of the NSRT.
    op: STRIPSOperator
    # The datastore, a list of segments that are covered by the
    # STRIPSOperator self.op. For each such segment, the datastore also
    # maintains a substitution dictionary of type VarToObjSub,
    # under which the ParameterizedOption and effects for all
    # segments in the datastore are equivalent.
    datastore: Datastore
    # The OptionSpec of this NSRT, which is a tuple of (option, option_vars).
    option_spec: OptionSpec
    # The sampler for this NSRT.
    sampler: Optional[NSRTSampler] = field(init=False, default=None)
    # A container for the possible keep effects for this PNAD.
    poss_keep_effects: Set[LiftedAtom] = field(init=False, default_factory=set)
    seg_to_keep_effects_sub: Dict[Segment,
                                  VarToObjSub] = field(init=False,
                                                       default_factory=dict)

    def add_to_datastore(self,
                         member: Tuple[Segment, VarToObjSub],
                         check_effect_equality: bool = True) -> None:
        """Add a new member to self.datastore."""
        seg, var_obj_sub = member
        if len(self.datastore) > 0:
            # All variables should have a corresponding object.
            assert set(var_obj_sub) == set(self.op.parameters)
            # The effects should match.
            if check_effect_equality:
                obj_var_sub = {o: v for (v, o) in var_obj_sub.items()}
                lifted_add_effects = {
                    a.lift(obj_var_sub)
                    for a in seg.add_effects
                }
                lifted_del_effects = {
                    a.lift(obj_var_sub)
                    for a in seg.delete_effects
                }
                assert lifted_add_effects == self.op.add_effects
                assert lifted_del_effects == self.op.delete_effects
            if seg.has_option():
                # The option should match.
                option = seg.get_option()
                part_param_option, part_option_args = self.option_spec
                assert option.parent == part_param_option
                option_args = [var_obj_sub[v] for v in part_option_args]
                assert option.objects == option_args
        # Add to datastore.
        self.datastore.append(member)

    def make_nsrt(self) -> NSRT:
        """Make an NSRT from this PNAD."""
        assert self.sampler is not None
        param_option, option_vars = self.option_spec
        return self.op.make_nsrt(param_option, option_vars, self.sampler)

    def __repr__(self) -> str:
        param_option, option_vars = self.option_spec
        vars_str = ", ".join(str(v) for v in option_vars)
        return f"{self.op}\n    Option Spec: {param_option.name}({vars_str})"

    def __str__(self) -> str:
        return repr(self)


@dataclass(frozen=True, eq=False, repr=False)
class InteractionRequest:
    """A request for interacting with a training task during online learning.
    Contains the index for that training task, an acting policy, a query
    policy, and a termination function. The acting policy may also terminate
    the interaction by raising `utils.RequestActPolicyFailure`.

    Note: the act_policy will not be called on the state where the
    termination_function returns True, but the query_policy will be.
    """
    train_task_idx: int
    act_policy: Callable[[State], Action]
    query_policy: Callable[[State], Optional[Query]]  # query can be None
    termination_function: Callable[[State], bool]


@dataclass(frozen=True, eq=False, repr=False)
class InteractionResult:
    """The result of an InteractionRequest. Contains a list of states, a list
    of actions, and a list of responses to queries if provded.

    Invariant: len(states) == len(responses) == len(actions) + 1
    """
    states: List[State]
    actions: List[Action]
    responses: List[Optional[Response]]

    def __post_init__(self) -> None:
        assert len(self.states) == len(self.responses) == len(self.actions) + 1


@dataclass(frozen=True, eq=False, repr=False)
class Query(abc.ABC):
    """Base class for a Query."""

    @property
    def cost(self) -> float:
        """The cost of making this Query."""
        raise NotImplementedError("Override me")


@dataclass(frozen=True, eq=False, repr=False)
class Response(abc.ABC):
    """Base class for a Response to a query.

    All responses contain the Query object itself, for convenience.
    """
    query: Query


@dataclass(frozen=True, eq=False, repr=False)
class GroundAtomsHoldQuery(Query):
    """A query for whether ground atoms hold in the state."""
    ground_atoms: Collection[GroundAtom]

    @property
    def cost(self) -> float:
        return len(self.ground_atoms)

    def __str__(self) -> str:
        atoms = ", ".join([str(ga) for ga in self.ground_atoms])
        return f"Do these hold? {atoms}"


@dataclass(frozen=True, eq=False, repr=False)
class GroundAtomsHoldResponse(Response):
    """A response to a GroundAtomsHoldQuery, providing boolean answers."""
    holds: Dict[GroundAtom, bool]

    def __str__(self) -> str:
        if not self.holds:
            return "No queries"
        responses = []
        for ga, b in self.holds.items():
            suffix = "holds" if b else "does not hold"
            responses.append(f"{ga} {suffix}")
        return ", ".join(responses)


@dataclass(frozen=True, eq=False, repr=False)
class DemonstrationQuery(Query):
    """A query requesting a demonstration to finish a train task."""
    train_task_idx: int

    @property
    def cost(self) -> float:
        return 1


@dataclass(frozen=True, eq=False, repr=False)
class DemonstrationResponse(Response):
    """A response to a DemonstrationQuery; provides a LowLevelTrajectory if one
    can be found by the teacher, otherwise returns None."""
    teacher_traj: Optional[LowLevelTrajectory]


@dataclass(frozen=True, eq=False, repr=False)
class PathToStateQuery(Query):
    """A query requesting a trajectory that reaches a specific state."""
    goal_state: State

    @property
    def cost(self) -> float:
        return 1


@dataclass(frozen=True, eq=False, repr=False)
class PathToStateResponse(Response):
    """A response to a PathToStateQuery; provides a LowLevelTrajectory if one
    can be found by the teacher, otherwise returns None."""
    teacher_traj: Optional[LowLevelTrajectory]


@dataclass(frozen=True, repr=False, eq=False)
class LDLRule:
    """A lifted decision list rule."""
    name: str
    parameters: Sequence[Variable]  # a superset of the NSRT parameters
    pos_state_preconditions: Set[LiftedAtom]  # a superset of the NSRT preconds
    neg_state_preconditions: Set[LiftedAtom]
    goal_preconditions: Set[LiftedAtom]
    nsrt: NSRT

    def __post_init__(self) -> None:
        assert set(self.parameters).issuperset(self.nsrt.parameters)
        assert self.pos_state_preconditions.issuperset(self.nsrt.preconditions)
        # The preconditions and goal preconditions should only use variables in
        # the rule parameters.
        for atom in self.pos_state_preconditions | \
            self.neg_state_preconditions | self.goal_preconditions:
            assert all(v in self.parameters for v in atom.variables)

    @lru_cache(maxsize=None)
    def ground(self, objects: Tuple[Object]) -> _GroundLDLRule:
        """Ground into a _GroundLDLRule, given objects.

        Insist that objects are tuple for hashing in cache.
        """
        assert isinstance(objects, tuple)
        assert len(objects) == len(self.parameters)
        assert all(
            o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        pos_pre = {atom.ground(sub) for atom in self.pos_state_preconditions}
        neg_pre = {atom.ground(sub) for atom in self.neg_state_preconditions}
        goal_pre = {atom.ground(sub) for atom in self.goal_preconditions}
        nsrt_objects = [sub[v] for v in self.nsrt.parameters]
        ground_nsrt = self.nsrt.ground(nsrt_objects)
        return _GroundLDLRule(self, list(objects), pos_pre, neg_pre, goal_pre,
                              ground_nsrt)

    @cached_property
    def _str(self) -> str:
        nsrt_param_str = ", ".join([str(v) for v in self.nsrt.parameters])
        return f"""LDLRule-{self.name}:
    Parameters: {self.parameters}
    Pos State Pre: {sorted(self.pos_state_preconditions, key=str)}
    Neg State Pre: {sorted(self.neg_state_preconditions, key=str)}
    Goal Pre: {sorted(self.goal_preconditions, key=str)}
    NSRT: {self.nsrt.name}({nsrt_param_str})"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, LDLRule)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class _GroundLDLRule:
    """A ground LDL rule is an LDLRule + objects.

    Should not be instantiated externally.
    """
    parent: LDLRule
    objects: Sequence[Object]
    pos_state_preconditions: Set[GroundAtom]
    neg_state_preconditions: Set[GroundAtom]
    goal_preconditions: Set[GroundAtom]
    ground_nsrt: _GroundNSRT

    @cached_property
    def _str(self) -> str:
        nsrt_obj_str = ", ".join([str(o) for o in self.ground_nsrt.objects])
        return f"""GroundLDLRule-{self.name}:
    Parameters: {self.objects}
    Pos State Pre: {sorted(self.pos_state_preconditions, key=str)}
    Neg State Pre: {sorted(self.neg_state_preconditions, key=str)}
    Goal Pre: {sorted(self.goal_preconditions, key=str)}
    NSRT: {self.ground_nsrt.name}({nsrt_obj_str})"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @property
    def name(self) -> str:
        """Name of this ground LRL rule."""
        return self.parent.name

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, _GroundLDLRule)
        return str(self) > str(other)


@dataclass(frozen=True)
class LiftedDecisionList:
    """A goal-conditioned policy from abstract states to ground NSRTs
    implemented with a lifted decision list.

    The logic described above is implemented in utils.query_ldl().
    """
    rules: Sequence[LDLRule]

    @cached_property
    def _hash(self) -> int:
        return hash(tuple(self.rules))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, LiftedDecisionList)
        if len(self.rules) != len(other.rules):
            return False
        return all(r1 == r2 for r1, r2 in zip(self.rules, other.rules))

    def __str__(self) -> str:
        rule_str = "\n".join(str(r) for r in self.rules)
        return f"LiftedDecisionList[\n{rule_str}\n]"


# Convenience higher-order types useful throughout the code
OptionSpec = Tuple[ParameterizedOption, List[Variable]]
GroundAtomTrajectory = Tuple[LowLevelTrajectory, List[Set[GroundAtom]]]
Image = NDArray[np.uint8]
Video = List[Image]
Array = NDArray[np.float32]
ObjToVarSub = Dict[Object, Variable]
ObjToObjSub = Dict[Object, Object]
VarToObjSub = Dict[Variable, Object]
VarToVarSub = Dict[Variable, Variable]
EntToEntSub = Dict[_TypedEntity, _TypedEntity]
Datastore = List[Tuple[Segment, VarToObjSub]]
NSRTSampler = Callable[
    [State, Set[GroundAtom], np.random.Generator, Sequence[Object]], Array]
Metrics = DefaultDict[str, float]
LiftedOrGroundAtom = TypeVar("LiftedOrGroundAtom", LiftedAtom, GroundAtom,
                             _Atom)
NSRTOrSTRIPSOperator = TypeVar("NSRTOrSTRIPSOperator", NSRT, STRIPSOperator)
GroundNSRTOrSTRIPSOperator = TypeVar("GroundNSRTOrSTRIPSOperator", _GroundNSRT,
                                     _GroundSTRIPSOperator)
ObjectOrVariable = TypeVar("ObjectOrVariable", bound=_TypedEntity)
SamplerDatapoint = Tuple[State, VarToObjSub, _Option,
                         Optional[Set[GroundAtom]]]
# For PDDLEnv environments, given a desired number of problems and an rng,
# returns a list of that many PDDL problem strings.
PDDLProblemGenerator = Callable[[int, np.random.Generator], List[str]]
# Used in ml_models.py. Either the maximum number of training iterations for
# a model, or a function that produces this number given the amount of data.
MaxTrainIters = Union[int, Callable[[int], int]]
ExplorationStrategy = Tuple[Callable[[State], Action], Callable[[State], bool]]
AbstractPolicy = Callable[[Set[GroundAtom], Set[Object], Set[GroundAtom]],
                          Optional[_GroundNSRT]]
