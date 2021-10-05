"""Structs used throughout the codebase.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
from typing import Dict, Iterator, List, Sequence, Callable, Set, Collection, \
    Tuple, Any, cast
import numpy as np
from gym.spaces import Box
from numpy.typing import NDArray

Array = NDArray[np.float32]


@dataclass(frozen=True, order=True)
class Type:
    """Struct defining a type.
    """
    name: str
    feature_names: Sequence[str]

    @property
    def dim(self) -> int:
        """Dimensionality of the feature vector of this object type.
        """
        return len(self.feature_names)

    def __call__(self, name: str) -> _TypedEntity:
        """Convenience method for generating _TypedEntities.
        """
        if name.startswith("?"):
            return Variable(name, self)
        return Object(name, self)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.feature_names)))


@dataclass(frozen=True, order=True, repr=False)
class _TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block). Should not be instantiated
    externally.
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


@dataclass(frozen=True, order=True, repr=False)
class Object(_TypedEntity):
    """Struct defining an Object, which is just a _TypedEntity whose name
    does not start with "?".
    """
    def __post_init__(self) -> None:
        assert not self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass(frozen=True, order=True, repr=False)
class Variable(_TypedEntity):
    """Struct defining a Variable, which is just a _TypedEntity whose name
    starts with "?".
    """
    def __post_init__(self) -> None:
        assert self.name.startswith("?")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


Substitution = Dict[_TypedEntity, _TypedEntity]


@dataclass
class State:
    """Struct defining the low-level state of the world.
    """
    data: Dict[Object, Array]

    def __post_init__(self) -> None:
        # Check feature vector dimensions.
        for obj in self:
            assert len(self[obj]) == obj.type.dim

    def __iter__(self) -> Iterator[Object]:
        """An iterator over the state's objects, in sorted order.
        """
        return iter(sorted(self.data))

    def __getitem__(self, key: Object) -> Array:
        return self.data[key]

    def get(self, obj: Object, feature_name: str) -> np.float32:
        """Look up an object feature by name.
        """
        idx = obj.type.feature_names.index(feature_name)
        return self.data[obj][idx]

    def vec(self, objects: Sequence[Object]) -> Array:
        """Concatenated vector of features for each of the objects in the
        given ordered list.
        """
        feats: List[Array] = []
        for obj in objects:
            feats.append(self[obj])
        return np.hstack(feats)

    def copy(self) -> State:
        """Return a copy of this state.
        """
        new_data = {}
        for obj in self:
            new_data[obj] = self._copy_state_value(self.data[obj])
        return State(new_data)

    def _copy_state_value(self, val: Any) -> Any:
        if val is None or isinstance(val, (float, bool, int, str)):
            return val
        if isinstance(val, (list, tuple, set)):
            return type(val)(self._copy_state_value(v) for v in val)
        assert hasattr(val, "copy")
        return val.copy()


DefaultState = State({})


@dataclass(frozen=True, order=True, repr=False)
class Predicate:
    """Struct defining a predicate (a lifted classifier over states).
    """
    name: str
    types: Sequence[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[State, Sequence[Object]], bool]

    def __call__(self, entities: Sequence[_TypedEntity]) -> _Atom:
        """Convenience method for generating Atoms.
        """
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
        """The arity of this predicate (number of arguments).
        """
        return len(self.types)

    def holds(self, state: State, objects: Sequence[Object]) -> bool:
        """Public method for calling the classifier. Performs
        type checking first.
        """
        assert len(objects) == self.arity
        for obj, pred_type in zip(objects, self.types):
            assert isinstance(obj, Object)
            assert obj.type == pred_type
        return self._classifier(state, objects)

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return str(self)


@dataclass(frozen=True, repr=False, eq=False)
class _Atom:
    """Struct defining an atom (a predicate applied to either variables
    or objects). Should not be instantiated externally.
    """
    predicate: Predicate
    entities: Sequence[_TypedEntity]

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

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _Atom)
        return str(self) == str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LiftedAtom(_Atom):
    """Struct defining a lifted atom (a predicate applied to variables).
    """
    @cached_property
    def variables(self) -> List[Variable]:
        """Arguments for this lifted atom. A list of "Variable"s.
        """
        return list(cast(Variable, ent) for ent in self.entities)

    @cached_property
    def _str(self) -> str:
        return (str(self.predicate) + "(" +
                ", ".join(map(str, self.variables)) + ")")

    def ground(self, sub: dict[Variable, Object]) -> GroundAtom:
        """Create a GroundAtom with a given substitution.
        """
        assert set(self.variables).issubset(set(sub.keys()))
        return GroundAtom(self.predicate, [sub[v] for v in self.variables])


@dataclass(frozen=True, repr=False, eq=False)
class GroundAtom(_Atom):
    """Struct defining a ground atom (a predicate applied to objects).
    """
    @cached_property
    def objects(self) -> List[Object]:
        """Arguments for this ground atom. A list of "Object"s.
        """
        return list(cast(Object, ent) for ent in self.entities)

    @cached_property
    def _str(self) -> str:
        return (str(self.predicate) + "(" +
                ", ".join(map(str, self.objects)) + ")")

    def lift(self, sub: dict[Object, Variable]) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution.
        """
        assert set(self.objects).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[o] for o in self.objects])


@dataclass(frozen=True, eq=False)
class Task:
    """Struct defining a task, which is a pair of initial state and goal.
    """
    init: State
    goal: Set[GroundAtom]

    def __post_init__(self) -> None:
        # Verify types.
        assert isinstance(self.init, State)
        for atom in self.goal:
            assert isinstance(atom, GroundAtom)


@dataclass(frozen=True, eq=False)
class ParameterizedOption:
    """Struct defining a parameterized option, which has a parameter space
    and can be ground into an Option, given parameter values. An option
    is composed of a policy, an initiation classifier, and a termination
    condition. We will stick with deterministic termination conditions.
    For a parameterized option, all of these are conditioned on parameters.
    """
    name: str
    params_space: Box = field(repr=False)
    # A policy maps a state and parameters to an action.
    _policy: Callable[[State, Array], Action] = field(repr=False)
    # An initiation classifier maps a state and parameters to a bool,
    # which is True iff the option can start now.
    _initiable: Callable[[State, Array], bool] = field(repr=False)
    # A termination condition maps a state and parameters to a bool,
    # which is True iff the option should terminate now.
    _terminal: Callable[[State, Array], bool] = field(repr=False)

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, ParameterizedOption)
        return self.name == other.name

    def __hash__(self) -> int:
        return self._hash

    def ground(self, params: Array) -> _Option:
        """Ground into an Option, given parameter values.
        On the Option that is returned, one can call, e.g., policy(state).
        """
        params = np.array(params, dtype=self.params_space.dtype)
        assert self.params_space.contains(params)
        name = self.name + "(" + ", ".join(map(str, params)) + ")"
        return _Option(name, policy=lambda s: self._policy(s, params),
                       initiable=lambda s: self._initiable(s, params),
                       terminal=lambda s: self._terminal(s, params))


@dataclass(frozen=True, eq=False)
class _Option:
    """Struct defining an option, which is like a parameterized option except
    that its components are not conditioned on parameters. Should not be
    instantiated externally.
    """
    name: str
    # A policy maps a state to an action.
    policy: Callable[[State], Action] = field(repr=False)
    # An initiation classifier maps a state to a bool, which is True
    # iff the option can start now.
    initiable: Callable[[State], bool] = field(repr=False)
    # A termination condition maps a state to a bool, which is True
    # iff the option should terminate now.
    terminal: Callable[[State], bool] = field(repr=False)


DefaultOption = _Option("", lambda s: Action(np.array([0.0])),
                        lambda s: False, lambda s: False)


@dataclass(frozen=True, repr=False, eq=False)
class Operator:
    """Struct defining an operator (as in STRIPS). Lifted!
    """
    name: str
    parameters: Sequence[Variable]
    preconditions: Set[LiftedAtom]
    add_effects: Set[LiftedAtom]
    delete_effects: Set[LiftedAtom]
    option: ParameterizedOption
    # A sampler maps a state, RNG, and objects to option parameters.
    _sampler: Callable[[State, np.random.Generator, Sequence[Object]],
                       Array] = field(repr=False)

    @cached_property
    def _str(self) -> str:
        return f"""{self.name}:
    Parameters: {self.parameters}
    Preconditions: {sorted(self.preconditions, key=lambda a: a.predicate)}
    Add Effects: {sorted(self.add_effects, key=lambda a: a.predicate)}
    Delete Effects: {sorted(self.delete_effects, key=lambda a: a.predicate)}
    Option: {self.option}"""

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
        assert isinstance(other, Operator)
        return str(self) == str(other)

    def ground(self, objects: Sequence[Object]) -> _GroundOperator:
        """Ground into a _GroundOperator, given objects.
        """
        assert len(objects) == len(self.parameters)
        assert all(o.type == p.type for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        preconditions = {atom.ground(sub) for atom in self.preconditions}
        add_effects = {atom.ground(sub) for atom in self.add_effects}
        delete_effects = {atom.ground(sub) for atom in self.delete_effects}
        sampler = lambda s, rng: self._sampler(s, rng, objects)
        return _GroundOperator(self, objects, preconditions, add_effects,
                               delete_effects, self.option, sampler)

    def filter_predicates(self, kept: Collection[Predicate]) -> Operator:
        """Keep only the given predicates in the preconditions,
        add effects, and delete effects. Note that the parameters must
        stay the same for the sake of the sampler input arguments.
        """
        preconditions = {a for a in self.preconditions if a.predicate in kept}
        add_effects = {a for a in self.add_effects if a.predicate in kept}
        delete_effects = {a for a in self.delete_effects if a.predicate in kept}
        return Operator(self.name, self.parameters,
                        preconditions, add_effects, delete_effects,
                        self.option, self._sampler)


@dataclass(frozen=True, repr=False, eq=False)
class _GroundOperator:
    """A ground operator is an operator + objects.
    """
    operator: Operator
    objects: Sequence[Object]
    preconditions: Set[GroundAtom]
    add_effects: Set[GroundAtom]
    delete_effects: Set[GroundAtom]
    option: ParameterizedOption
    sampler: Callable[[State, np.random.Generator], Array] = field(repr=False)

    @cached_property
    def _str(self) -> str:
        return f"""{self.name}:
    Parameters: {self.objects}
    Preconditions: {sorted(self.preconditions, key=lambda a: a.predicate)}
    Add Effects: {sorted(self.add_effects, key=lambda a: a.predicate)}
    Delete Effects: {sorted(self.delete_effects, key=lambda a: a.predicate)}
    Option: {self.option}"""

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    @property
    def name(self) -> str:
        """Name of this operator.
        """
        return self.operator.name

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, _GroundOperator)
        return str(self) == str(other)


@dataclass(eq=False)
class Action:
    """An action in an environment. This is a light wrapper around a numpy
    float array that can optionally store the option which produced it.
    The _option field is a tuple of the option itself and a timestep index.
    """
    _arr: Array
    _option: Tuple[_Option, int] = field(
        repr=False, default=(DefaultOption, -1))

    @property
    def arr(self) -> Array:
        """The array representation of this action.
        """
        return self._arr

    def has_option(self) -> bool:
        """Whether this action has a non-default option attached.
        """
        return self._option[0] is not DefaultOption

    def get_option(self) -> Tuple[_Option, int]:
        """Get the option that produced this action.
        """
        assert self.has_option()
        return self._option

    def set_option(self, option: Tuple[_Option, int]) -> None:
        """Set the option that produced this action.
        """
        self._option = option

    def unset_option(self) -> None:
        """Unset the option that produced this action.
        """
        self._option = (DefaultOption, -1)
        assert not self.has_option()


ActionTrajectory = Tuple[List[State], List[Action]]
OptionTrajectory = Tuple[List[State], List[_Option]]
Dataset = List[ActionTrajectory]


Image = NDArray[np.uint8]
Video = List[Image]
