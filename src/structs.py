"""Structs used throughout the codebase.
"""

from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Mapping, Iterable, List, Sequence, Callable
import numpy as np
from numpy.typing import ArrayLike


@dataclass
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


class TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block).
    """
    def __init__(self, name: str, ent_type: Type):
        self.name = name
        self.type = ent_type
        self._str = f"{self.name}:{self.type.name}"
        self._hash = hash(str(self))

    def __str__(self):
        return self._str

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class Object(TypedEntity):
    """Struct defining an Object, which is just a TypedEntity whose name
    does not start with "?".
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.name.startswith("?")


class Variable(TypedEntity):
    """Struct defining a Variable, which is just a TypedEntity whose name
    starts with "?".
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert self.name.startswith("?")


@dataclass
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


class Predicate:
    """Struct defining a predicate (a lifted classifier over states).
    """
    def __init__(self, name: str, types: Sequence[Type],
                 classifier: Callable[[State, Sequence[Object]], bool]):
        self.name = name
        self.types = types
        # The classifier takes in a complete state and a sequence of objects
        # representing the arguments. These objects should be the only ones
        # treated "specially" by the classifier.
        self._classifier = classifier
        self._hash = hash(str(self))

    def __call__(self, entities: Sequence[TypedEntity]) -> _Atom:
        """Convenience method for generating Atoms.
        """
        if all(isinstance(ent, Variable) for ent in entities):
            return LiftedAtom(self, entities)
        if all(isinstance(ent, Object) for ent in entities):
            return GroundAtom(self, entities)
        raise ValueError("Cannot instantiate Atom with mix of "
                         "variables and objects")

    @property
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

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class Atom:
    """Struct defining an atom (a predicate applied to either variables
    or objects. Should not be used externally.
    """
    def __init__(self, predicate: Predicate, entities: Sequence[TypedEntity]):
        self.predicate = predicate
        self._str = ""
        self._hash = 0
        self._setup(entities)
        assert self._str and self._hash

    @abc.abstractmethod
    def _setup(self, entities):
        raise NotImplementedError("Override me!")

    def __str__(self):
        return self._str

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        return str(self) == str(other)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class LiftedAtom(Atom):
    """Struct defining a lifted atom (a predicate applied to variables).
    """
    def _setup(self, entities: Sequence[Variable]):
        self.variables = list(entities)
        for var in self.variables:
            assert isinstance(var, Variable)
        self._str = (str(self.predicate) + "(" +
                     ", ".join(map(str, self.variables)) + ")")
        self._hash = hash(str(self))


class GroundAtom(Atom):
    """Struct defining a ground atom (a predicate applied to objects).
    """
    def _setup(self, entities: Sequence[Object]):
        self.objects = list(entities)
        for obj in self.objects:
            assert isinstance(obj, Object)
        self._str = (str(self.predicate) + "(" +
                     ", ".join(map(str, self.objects)) + ")")
        self._hash = hash(str(self))
