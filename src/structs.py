"""Structs used throughout the codebase.
"""

from dataclasses import dataclass
from typing import Collection, Mapping, Iterable, List
import numpy as np
from numpy.typing import ArrayLike


@dataclass
class ObjectType:
    """Struct defining the type of an object.
    """
    name: str
    feature_names: Collection[str]

    @property
    def dim(self) -> int:
        """Dimensionality of the feature vector of this object type.
        """
        return len(self.feature_names)


class Object:
    """Struct defining an object.
    """
    def __init__(self, name: str, obj_type: ObjectType):
        self.name = name
        self.type = obj_type
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

    def vec(self, objects: Iterable[Object]) -> ArrayLike:
        """Concatenated vector of features for each of the objects in the
        given list.
        """
        feats: List[ArrayLike] = []
        for obj in objects:
            feats.append(self[obj])
        return np.hstack(feats)
