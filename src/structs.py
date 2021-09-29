"""Structs used throughout the codebase.
"""

from dataclasses import dataclass
from typing import Collection, Mapping, Iterable, List, Sequence, Callable
import numpy as np
from numpy.typing import ArrayLike


@dataclass
class Type:
    """Struct defining a type.
    """
    name: str
    feature_names: Collection[str]

    @property
    def dim(self) -> int:
        """Dimensionality of the feature vector of this object type.
        """
        return len(self.feature_names)


class TypedEntity:
    """Struct defining an entity with some type, either an object (e.g.,
    block3) or a variable (e.g., ?block).
    """
    def __init__(self, name: str, ent_type: Type):
        self.name = name
        self.type = ent_type
        self._str = f"{self.name}:{self.type.name}"

    def __str__(self):
        return self._str

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(str(self))

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

    def vec(self, objects: Collection[Object]) -> ArrayLike:
        """Concatenated vector of features for each of the objects in the
        given list.
        """
        feats: List[ArrayLike] = []
        for obj in objects:
            feats.append(self[obj])
        return np.hstack(feats)


@dataclass
class Predicate:
    """Struct defining a predicate (a lifted classifier over states).
    """
    name: str
    types: Collection[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[State, Sequence[Object]], bool]

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
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

    def __lt__(self, other):
        return str(self) < str(other)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self
