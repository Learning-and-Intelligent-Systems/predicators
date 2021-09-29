"""Structs used throughout the codebase.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from functools import cached_property
from typing import Mapping, Iterable, List, Sequence, Callable, Collection
import numpy as np
from gym.spaces import Box  # type: ignore
from numpy.typing import ArrayLike


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

    def __call__(self, name) -> TypedEntity:
        """Convenience method for generating TypedEntities.
        """
        if name.startswith("?"):
            return Variable(name, self)
        return Object(name, self)

    def __hash__(self):
        return hash((self.name, tuple(self.feature_names)))


@dataclass(frozen=True, order=True, repr=False)
class TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block).
    """
    name: str
    type: Type

    @cached_property
    def _str(self):
        return f"{self.name}:{self.type.name}"

    @cached_property
    def _hash(self):
        return hash(str(self))

    def __str__(self):
        return self._str

    def __repr__(self):
        return self._str


@dataclass(frozen=True, order=True, repr=False)
class Object(TypedEntity):
    """Struct defining an Object, which is just a TypedEntity whose name
    does not start with "?".
    """
    def __post_init__(self):
        assert not self.name.startswith("?")

    def __hash__(self):
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass(frozen=True, order=True, repr=False)
class Variable(TypedEntity):
    """Struct defining a Variable, which is just a TypedEntity whose name
    starts with "?".
    """
    def __post_init__(self):
        assert self.name.startswith("?")

    def __hash__(self):
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash


@dataclass(frozen=True)
class State:
    """Struct defining the low-level state of the world.
    """
    data: Mapping[Object, ArrayLike]

    def __post_init__(self):
        # Check feature vector dimensions.
        for obj in self:
            assert len(self[obj]) == obj.type.dim

    def __iter__(self) -> Iterable[Object]:
        """Iterate over objects in sorted order.
        """
        return iter(sorted(self.data))

    def __getitem__(self, key: Object) -> ArrayLike:
        return self.data[key]

    def vec(self, objects: Sequence[Object]) -> ArrayLike:
        """Concatenated vector of features for each of the objects in the
        given ordered list.
        """
        feats: List[ArrayLike] = []
        for obj in objects:
            feats.append(self[obj])
        return np.hstack(feats)


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

    def __call__(self, entities: Sequence[TypedEntity]) -> _Atom:
        """Convenience method for generating Atoms.
        """
        if all(isinstance(ent, Variable) for ent in entities):
            return LiftedAtom(self, entities)
        if all(isinstance(ent, Object) for ent in entities):
            return GroundAtom(self, entities)
        raise ValueError("Cannot instantiate Atom with mix of "
                         "variables and objects")

    @cached_property
    def _hash(self):
        return hash(str(self))

    def __hash__(self):
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

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


@dataclass(frozen=True, repr=False, eq=False)
class _Atom:
    """Struct defining an atom (a predicate applied to either variables
    or objects). Should not be instantiated externally.
    """
    predicate: Predicate
    entities: Sequence[TypedEntity]

    @property
    def _str(self):
        raise NotImplementedError("Override me")

    @cached_property
    def _hash(self):
        return hash(str(self))

    def __str__(self):
        return self._str

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return str(self) == str(other)


@dataclass(frozen=True, repr=False, eq=False)
class LiftedAtom(_Atom):
    """Struct defining a lifted atom (a predicate applied to variables).
    """
    @cached_property
    def variables(self):
        """Arguments for this lifted atom
        """
        return list(self.entities)

    @cached_property
    def _str(self):
        return (str(self.predicate) + "(" +
                ", ".join(map(str, self.variables)) + ")")


@dataclass(frozen=True, repr=False, eq=False)
class GroundAtom(_Atom):
    """Struct defining a ground atom (a predicate applied to objects).
    """
    @cached_property
    def objects(self):
        """Arguments for this ground atom
        """
        return list(self.entities)

    @cached_property
    def _str(self):
        return (str(self.predicate) + "(" +
                ", ".join(map(str, self.objects)) + ")")


@dataclass(frozen=True, eq=False)
class Task:
    """Struct defining a task, which is a pair of initial state and goal.
    """
    init: State
    goal: Collection[GroundAtom]

    def __post_init__(self):
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
    _policy: Callable[[State, ArrayLike], ArrayLike] = field(repr=False)
    # An initiation classifier maps a state and parameters to a bool,
    # which is True iff the option can start now.
    _initiable: Callable[[State, ArrayLike], bool] = field(repr=False)
    # A termination condition maps a state and parameters to a bool,
    # which is True iff the option should terminate now.
    _terminal: Callable[[State, ArrayLike], bool] = field(repr=False)

    def ground(self, params: ArrayLike) -> _Option:
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
    policy: Callable[[State], ArrayLike] = field(repr=False)
    # An initiation classifier maps a state to a bool, which is True
    # iff the option can start now.
    initiable: Callable[[State], bool] = field(repr=False)
    # A termination condition maps a state to a bool, which is True
    # iff the option should terminate now.
    terminal: Callable[[State], bool] = field(repr=False)
