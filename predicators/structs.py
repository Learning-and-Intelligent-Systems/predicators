"""Structs used throughout the codebase."""

from __future__ import annotations

import abc
import copy
import itertools
import random
import textwrap
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache
from inspect import Parameter, getsource, signature
from typing import TYPE_CHECKING, Any, Callable, Collection, DefaultDict, \
    Dict, Iterator, List, Optional, Sequence, Set, Tuple, TypeVar, Union, \
    cast

if TYPE_CHECKING:
    from predicators.utils import VLMQuery, VLMState

# pylint: disable=wrong-import-position
import numpy as np
import PIL.Image
import torch
from gym.spaces import Box
from numpy.typing import NDArray
from tabulate import tabulate
from torch import Tensor

import predicators.pretrained_model_interface
import predicators.utils as utils  # pylint: disable=consider-using-from-import
from predicators.settings import CFG

# pylint: enable=wrong-import-position


@dataclass(frozen=True, order=True)
class Type:
    """Struct defining a type.

    sim_feature_names are features stored in an object, and usually
    won't change throughout and across tasks. An example is the object's
    pybullet id.
    This is convenient for variables that are not easily extractable from the
    sim state -- whether a food block attracts ants, or the joint id for a
    switch -- but are nonetheless for running the simulation.

    Why not store all features here instead of storing in the State object?
    They can only store one value per feature, so if we generate 10 tasks where
    the blocks are at different locations, it won't be able to store all 10
    locations. One might think they could reset any feature at when reset is
    called. But this would require the information is first stored in the State
    object.

    angular_features marks features that are angles in radians (yaw, roll,
    joint angles, ...). Consumers that compare feature values across states
    (e.g. system-identification residuals) wrap differences of these
    features to [-pi, pi] so that equivalent orientations (roll -pi vs +pi)
    do not read as a 2*pi error. Declaring it is optional metadata; the
    default (no angular features) preserves plain arithmetic differences.
    """
    name: str
    feature_names: Sequence[str] = field(repr=False)
    parent: Optional[Type] = field(default=None, repr=False)
    sim_features: Sequence[str] = field(default_factory=lambda: ["id"],
                                        repr=False,
                                        compare=False)
    angular_features: Sequence[str] = field(default_factory=tuple,
                                            repr=False,
                                            compare=False)

    @property
    def dim(self) -> int:
        """Dimensionality of the feature vector of this object type."""
        return len(self.feature_names)

    def get_ancestors(self) -> Set[Type]:
        """Get the set of all types that are ancestors (i.e. parents,
        grandparents, great-grandparents, etc.) of the current type."""
        curr_type: Optional[Type] = self
        ancestors_set = set()
        while curr_type is not None:
            ancestors_set.add(curr_type)
            curr_type = curr_type.parent
        return ancestors_set

    def pretty_str(self) -> str:
        """Display the type in a nice human-readable format."""
        formatted_features = [f"'{name}'" for name in self.feature_names]
        return f"{self.name}: {{{', '.join(formatted_features)}}}"

    def python_definition_str(self) -> str:
        """Display in a format similar to how a type is instantiated."""
        formatted_features = [f"'{name}'" for name in self.feature_names]
        return f"_{self.name}_type = Type('{self.name}', "+\
                f"[{', '.join(formatted_features)}])"

    def __call__(self, name: str) -> _TypedEntity:
        """Convenience method for generating _TypedEntities."""
        if name.startswith("?"):
            return Variable(name, self)
        return Object(name, self)

    def __hash__(self) -> int:
        return hash((self.name, tuple(self.feature_names)))


@dataclass(frozen=False, order=True, repr=False)
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

    def __getstate__(self) -> Dict:
        """Drop cached properties from the pickled state.

        ``_hash`` caches ``hash(str(self))``, and Python string hashes
        are salted per process (PYTHONHASHSEED): an entity pickled in
        one process and loaded in another would carry a stale hash, land
        in the wrong dict bucket, and raise KeyError on every
        ``State.data`` lookup even though ``__eq__`` holds (hit by the
        offline consumers of the persisted ``fit_data`` pickles).
        """
        state = self.__dict__.copy()
        state.pop("_str", None)
        state.pop("_hash", None)
        return state

    def __setstate__(self, state: Dict) -> None:
        """Also scrub on load, so pre-fix pickles are repaired."""
        state.pop("_str", None)
        state.pop("_hash", None)
        self.__dict__.update(state)

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


@dataclass(frozen=False, order=True, repr=False)
class Object(_TypedEntity):
    """Struct defining an Object, which is just a _TypedEntity whose name does
    not start with "?"."""
    sim_data: Dict[str, Any] = field(default_factory=dict,
                                     compare=False,
                                     repr=False)

    def __post_init__(self) -> None:
        assert not self.name.startswith("?")
        # Initialize sim_data from the Type's sim_features
        for sim_feature in self.type.sim_features:
            self.sim_data[sim_feature] = None  # Default to None
        # Keep track of allowed attributes
        object.__setattr__(self, '_allowed_attributes',
                           {"sim_data"}.union(self.sim_data.keys()))

    def __getattr__(self, name: str) -> Any:
        # Bypass custom logic for internal attributes
        # Use object.__getattribute__(...) instead of self.sim_data
        try:
            sim_data = object.__getattribute__(self, "sim_data")
        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            ) from None
        if name in sim_data:
            return sim_data[name]
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value: Any) -> None:
        # Always allow the dataclass fields (e.g., "name", "type", "sim_data").
        if name in {"name", "type", "sim_data", "_allowed_attributes"}:
            super().__setattr__(name, value)
            return

        # For anything else, check _allowed_attributes.
        allowed_attrs = object.__getattribute__(self, "_allowed_attributes") \
            if object.__getattribute__(self, "__dict__").get(
                "_allowed_attributes") else set()
        if name in allowed_attrs:
            sim_data = object.__getattribute__(self, "sim_data")
            if name in sim_data:
                sim_data[name] = value
            else:
                super().__setattr__(name, value)
        else:
            raise AttributeError(f"Cannot set unknown attribute '{name}'")

    def __hash__(self) -> int:
        # By default, the dataclass generates a new __hash__ method when
        # frozen=True and eq=True, so we need to override it.
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Object):
            return False
        return self.name == other.name and self.type == other.type

    @cached_property
    def id_name(self) -> str:
        """Return a name combining the type name and object id."""
        assert self.id is not None, "Object must have an id set to use id_name"
        return f"{self.type.name}{self.id}"


@dataclass(frozen=False, order=True, repr=False)
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
    """Low-level world state.

    Separates the agent's observation (`data`) from two optional hidden
    blocks — the agent's belief (`latent`) and the environment's ground
    truth (`privileged`) — plus opaque simulator bookkeeping
    (`simulator_state`). Only `data` defines state identity (`__hash__`
    and `allclose` ignore the other three).
    """
    # Object-centric *observable* features = the agent's observation.
    # Fully observable: the complete world state. Partially observable:
    # only the exposed features (some causally-relevant features are
    # omitted). The only field that defines state identity (`__hash__`,
    # `allclose`).
    data: Dict[Object, Array]
    # Opaque per-environment simulator bookkeeping (e.g. PyBullet joint
    # positions); env-internal, not agent-facing.
    simulator_state: Optional[Any] = None
    # The agent's *inferred estimate* of the hidden state (its belief),
    # threaded by partially-observable / recurrent approaches; None under
    # full observability. Deep-copied by `copy()`. See
    # `predicators.code_sim_learning.utils.init_latent` for the canonical
    # initial value.
    latent: Optional[Dict[str, Any]] = None
    # The environment's *true* hidden state that the partially-observable
    # observation omits; None under full observability, where those
    # features live in `data` instead. The truth to `latent`'s belief —
    # env-only, never surfaced through any `data`/`feature_names` channel
    # (inspect tools, dict_str, abstraction). Deep-copied by `copy()`.
    privileged: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        # Check feature vector dimensions.
        for obj in self:
            assert len(self[obj]) == obj.type.dim

    def __hash__(self) -> int:
        # Hash object keys and array contents using numpy's built-in hashing
        items = []
        for obj in sorted(self.data.keys()):
            arr = self.data[obj]
            if hasattr(arr, 'tobytes'):
                # For numpy arrays, hash the bytes representation
                items.append((obj, hash(arr.tobytes())))
            else:
                items.append((obj, hash(tuple(arr))))

        data_hash = hash(tuple(items))
        return data_hash

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
        return [o for o in self if o.is_instance(object_type)]

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
        return State(new_data,
                     simulator_state=copy.deepcopy(self.simulator_state),
                     latent=copy.deepcopy(self.latent),
                     privileged=copy.deepcopy(self.privileged))

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
            if not CFG.allow_state_allclose_comparison_despite_simulator_state:
                raise NotImplementedError("Cannot use allclose when "
                                          "simulator_state is not None.")
            if self.simulator_state != other.simulator_state:
                return False
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

    def dict_str(self,
                 indent: int = 0,
                 object_features: bool = True,
                 num_decimal_points: int = 2,
                 use_object_id: bool = False,
                 ignored_features: Optional[List[str]] = None) -> str:
        """Return a dictionary representation of the state."""
        if ignored_features is None:
            ignored_features = ["capacity_liquid", "target_liquid"]
        excluded_objects = []
        if CFG.excluded_objects_in_state_str:
            excluded_objects = CFG.excluded_objects_in_state_str.split(",")
        state_dict = {}

        # Collect all unique types from objects in the state
        object_types = set()
        for obj in self:
            object_types.add(obj.type)

        # Iterate through types and add all objects of each type
        for obj_type in sorted(object_types, key=lambda t: t.name):
            obj_type_name = obj_type.name
            if obj_type_name not in excluded_objects:
                # Get all objects of this type
                objects_of_type = self.get_objects(obj_type)

                # Process each object of this type
                for obj in objects_of_type:
                    obj_dict = {}
                    if obj_type_name == "robot" or object_features:
                        for attribute, value in zip(obj.type.feature_names,
                                                    self[obj]):
                            if attribute not in ignored_features:
                                obj_dict[attribute] = value
                    if use_object_id:
                        obj_name = obj.id_name
                    else:
                        obj_name = obj.name
                    state_dict[f"{obj_name}:{obj.type.name}"] = obj_dict

        # Create a string of n_space spaces
        spaces = " " * indent

        # Create a PrettyPrinter with a large width
        dict_str = spaces + "{"
        n_keys = len(state_dict.keys())
        for i, (key, value) in enumerate(state_dict.items()):
            # Format values in the string representation
            formatted_items = []
            for k, v in value.items():
                if isinstance(v, (float, np.floating)):
                    formatted_items.append(
                        f"'{k}': {v:.{num_decimal_points}f}")
                else:
                    formatted_items.append(f"'{k}': {v}")
            value_str = ', '.join(formatted_items)

            if i == 0:
                dict_str += f"'{key}': {{{value_str}}},\n"
            elif i == n_keys - 1:
                dict_str += spaces + f" '{key}': {{{value_str}}}"
            else:
                dict_str += spaces + f" '{key}': {{{value_str}}},\n"
        dict_str += "}"
        return dict_str


DefaultState = State({})


@lru_cache(maxsize=None)
def _classifier_accepts_latent(classifier: Callable) -> bool:
    """Return True iff `classifier` declares a `latent` parameter or **kwargs.

    Used by `Predicate.holds` to thread the sample's latent state-
    feature block only into classifiers that opted in. Cached because
    predicate classifiers are typically reused across many `.holds()`
    calls and introspecting `inspect.signature` is not free.
    """
    try:
        params = signature(classifier).parameters
    except (TypeError, ValueError):
        # Built-ins, C-extensions, or anything whose signature we can't
        # introspect: assume legacy 2-arg form.
        return False
    if "latent" in params:
        return True
    return any(p.kind == Parameter.VAR_KEYWORD for p in params.values())


@dataclass(frozen=True, order=False, repr=False)
class Predicate:
    """Struct defining a predicate (a lifted classifier over states)."""
    name: str
    types: Sequence[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[State, Sequence[Object]],
                          bool] = field(compare=False)
    natural_language_assertion: Optional[Callable[[List[str]],
                                                  str]] = field(default=None,
                                                                compare=False)

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

    def __eq__(self, other: Predicate) -> bool:  # type: ignore[override]
        # equal by name
        assert isinstance(other, Predicate)
        if self.name != other.name:
            return False
        if len(self.types) != len(other.types):
            return False
        for self_type, other_type in zip(self.types, other.types):
            if self_type != other_type:
                return False
        return True

    @cached_property
    def arity(self) -> int:
        """The arity of this predicate (number of arguments)."""
        return len(self.types)

    def holds(self,
              state: State,
              objects: Sequence[Object],
              latent: Optional[Dict[str, Any]] = None) -> bool:
        """Public method for calling the classifier.

        Performs type checking first. `latent` is the sample's latent
        state-feature block, threaded by approaches that learn over
        partially-observable envs (see
        `agent_po_sim_predicate_invention`). When the caller does not
        pass `latent` explicitly, the block attached to `state.latent`
        is used (so callers like `utils.abstract` do not need to know
        about the recurrent extension). Classifiers that don't accept a
        `latent` kwarg are called with the legacy `(state, objects)`
        signature for backwards compatibility.
        """
        assert len(objects) == self.arity
        for obj, pred_type in zip(objects, self.types):
            assert isinstance(obj, Object)
            assert obj.is_instance(pred_type)
        if _classifier_accepts_latent(self._classifier):
            effective_latent = latent if latent is not None else state.latent
            return self._classifier(
                state, objects,
                latent=effective_latent)  # type: ignore[call-arg]
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

    def pretty_str_with_assertion(self) -> str:
        """Return a pretty string with assertion format."""
        var_names = []
        vars_str = []
        for i, t in enumerate(self.types):
            vars_str.append(
                f"{CFG.grammar_search_classifier_pretty_str_names[i]}:{t.name}"
            )
            var_names.append(
                f"{CFG.grammar_search_classifier_pretty_str_names[i]}")
        vars_str = ", ".join(vars_str)  # type: ignore[assignment]

        body_str = f"{self.name}({vars_str})"
        if hasattr(self, "natural_language_assertion") and\
            self.natural_language_assertion is not None:
            body_str += f": {self.natural_language_assertion(var_names)}"
        return body_str

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

    def __lt__(self, other: Predicate) -> bool:
        return str(self) < str(other)

    def __reduce__(self) -> Tuple:
        """Tell pickle/dill how to re-create a Predicate:

        (constructor, (name, types, classifier))
        """
        # • `tuple(self.types)` ensures the sequence itself is picklable
        # • `_classifier` must be a top-level def or otherwise dill-pickleable
        return (self.__class__, (self.name, tuple(self.types),
                                 self._classifier))


@dataclass(frozen=True, order=False, repr=False)
class DerivedPredicate(Predicate):
    """Struct defining a concept predicate."""
    name: str
    types: Sequence[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[Set[GroundAtom], Sequence[Object]],
                          bool] = field(compare=False)
    untransformed_predicate: Optional[Predicate] = field(default=None,
                                                         compare=False)
    auxiliary_predicates: Optional[Set[Predicate]] = field(default=None,
                                                           compare=False)

    def update_auxiliary_concepts(
            self,
            auxiliary_predicates: Set[DerivedPredicate]) -> DerivedPredicate:
        """Create a new ConceptPredicate with updated auxiliary_concepts."""
        return replace(
            self,
            auxiliary_predicates=auxiliary_predicates  # type: ignore[arg-type]
        )

    @cached_property
    def _hash(self) -> int:
        # Make the hash the same regardless types is a list or tuple.
        return hash(self.name + " ".join(t.name for t in self.types))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Predicate) -> bool:  # type: ignore[override]
        # equal by name
        assert isinstance(other, Predicate)
        if self.name != other.name:
            return False
        if len(self.types) != len(other.types):
            return False
        for self_type, other_type in zip(self.types, other.types):
            if self_type != other_type:
                return False
        return True

    def holds(  # type: ignore[override]  # pylint: disable=arguments-differ
            self, state: Set[GroundAtom], objects: Sequence[Object]) -> bool:
        """Public method for calling the classifier.

        Performs type checking first.
        """
        assert len(objects) == self.arity
        for obj, pred_type in zip(objects, self.types):
            assert isinstance(obj, Object)
            assert obj.is_instance(pred_type)
        return self._classifier(state, objects)

    def _negated_classifier(
            self,
            state: Set[GroundAtom],  # type: ignore[override]
            objects: Sequence[Object]) -> bool:
        # Separate this into a named function for pickling reasons.
        return not self._classifier(state, objects)

    def __reduce__(self) -> Tuple:
        """Tell pickle/dill how to re-create a DerivedPredicate:

        (constructor, (name, types, classifier))
        """
        # • `tuple(self.types)` ensures the sequence itself is picklable
        # • `_classifier` must be a top-level def or otherwise dill-pickleable
        return (self.__class__,
                (self.name, tuple(self.types), self._classifier,
                 self.untransformed_predicate, self.auxiliary_predicates))


@dataclass(frozen=True, order=False, repr=False, eq=False)
class VLMPredicate(Predicate):
    """Struct defining a predicate that calls a VLM as part of returning its
    truth value.

    NOTE: when instantiating a VLMPredicate, we typically pass in a 'Dummy'
    classifier (i.e., one that returns simply raises some kind of error instead
    of actually outputting a value of any kind).
    """
    get_vlm_query_str: Optional[Callable[[Sequence[Object]],
                                         str]] = field(default=None)


class NSPredicate(Predicate):
    """Neuro-Symbolic Predicate."""

    def __init__(
            self, name: str, types: Sequence[Type],
            _classifier: Callable[[VLMState, Sequence[Object]], bool]) -> None:
        self._original_classifier = _classifier
        super().__init__(
            name, types,
            _MemoizedClassifier(_classifier))  # type: ignore[arg-type]

    @cached_property
    def _hash(self) -> int:
        # return hash(str(self))
        return hash(self.name + str(self.types))

    def __hash__(self) -> int:
        return self._hash

    def classifier_str(self) -> str:
        """Get a string representation of the classifier."""
        clf_str = getsource(
            self._original_classifier)  # type: ignore[name-defined]
        clf_str = textwrap.dedent(clf_str)  # type: ignore[name-defined]
        clf_str = clf_str.replace("@staticmethod\n", "")
        return clf_str


@dataclass
class _MemoizedClassifier():
    classifier: Callable[[State, Sequence[Object]],
                         Union[bool, VLMQuery]]  # type: ignore[name-defined]
    cache: Dict = field(default_factory=dict)

    def cache_truth_value(self, state: State, objects: Sequence[Object],
                          truth_value: bool) -> None:
        """Cache the boolean value after querying the VLM and obtaining the
        result."""
        combined_hash = self.hash_state_objs(state, objects)
        self.cache[combined_hash] = truth_value

    def hash_state_objs(self, state: State, objects: Sequence[Object]) -> int:
        """Hash a state and objects pair."""
        objects_tuple_hash = hash(tuple(objects))
        state_hash = hash(state)
        return hash((state_hash, objects_tuple_hash))

    def has_classified(self, state: State, objects: Sequence[Object]) -> bool:
        """Check if the state, object pair has been stored in the cache."""
        combined_hash = self.hash_state_objs(state, objects)
        return combined_hash in self.cache

    def __call__(self, state: State, objects: Sequence[Object]) -> \
        Union[bool, VLMQuery]:  # type: ignore[name-defined]
        """When the classifier is called, return the cached value if it exists
        otherwise call self.classifier."""
        # if state, object exist in cache, return the value
        # else compute the truth value using the classifier
        combined_hash = self.hash_state_objs(state, objects)
        return self.cache.get(combined_hash, self.classifier(state, objects))


@dataclass(frozen=True, order=False, repr=False)
class ConceptPredicate(Predicate):
    """Struct defining a concept predicate."""
    name: str
    types: Sequence[Type]
    # The classifier takes in a complete state and a sequence of objects
    # representing the arguments. These objects should be the only ones
    # treated "specially" by the classifier.
    _classifier: Callable[[Set[GroundAtom], Sequence[Object]],
                          bool] = field(compare=False)
    untransformed_predicate: Optional[Predicate] = field(default=None,
                                                         compare=False)
    auxiliary_concepts: Optional[Set[ConceptPredicate]] = field(default=None,
                                                                compare=False)

    def update_auxiliary_concepts(
            self,
            auxiliary_concepts: Set[ConceptPredicate]) -> ConceptPredicate:
        """Create a new ConceptPredicate with updated auxiliary_concepts."""
        return replace(self, auxiliary_concepts=auxiliary_concepts)

    @cached_property
    def _hash(self) -> int:
        # return hash(str(self))
        return hash(self.name + str(self.types))

    def __hash__(self) -> int:
        return self._hash

    def holds(  # type: ignore[override]  # pylint: disable=arguments-differ
            self, state: Set[GroundAtom], objects: Sequence[Object]) -> bool:
        """Public method for calling the classifier.

        Performs type checking first.
        """
        assert len(objects) == self.arity
        for obj, pred_type in zip(objects, self.types):
            assert isinstance(obj, Object)
            assert obj.is_instance(pred_type)
        return self._classifier(state, objects)

    def _negated_classifier(
            self,
            state: Set[GroundAtom],  # type: ignore[override]
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

    def __reduce__(self) -> Tuple:
        """Return a pickling recipe: call the class with (predicate, entities).

        - This ensures that when the object is unpickled, all dataclass fields
        (predicate, entities) are set before anything like hashing or
        stringification is triggered.
            - This prevents errors where e.g. self.predicate
        does not exist yet at the time __hash__ or __str__ is
        called during deserialization (which is
        exactly what caused crash during parallel pnad learning).
        """
        return (self.__class__, (self.predicate, tuple(self.entities)))


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

    def substitute(self, sub: VarToVarSub) -> LiftedAtom:
        """Create a LiftedAtom with a given substitution."""
        assert set(self.variables).issubset(set(sub.keys()))
        return LiftedAtom(self.predicate, [sub[v] for v in self.variables])


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

    def holds(self,
              state: State,
              latent: Optional[Dict[str, Any]] = None) -> bool:
        """Check whether this ground atom holds in the given state.

        `latent` is forwarded to predicate classifiers that opted in to
        the latent-aware signature; ignored otherwise.
        """
        return self.predicate.holds(state, self.objects, latent=latent)

    def get_vlm_query_str(self) -> str:
        """If this GroundAtom is associated with a VLMPredicate, then get the
        string that will be used to query the VLM."""
        assert isinstance(self.predicate, VLMPredicate)
        # pylint: disable=no-member
        return self.predicate.get_vlm_query_str(  # type: ignore[misc]
            self.objects)
        # pylint: enable=no-member

    def get_negated_atom(self) -> GroundAtom:
        """Get the negated atom of this GroundAtom."""
        from predicators.approaches.grammar_search_invention_approach import \
            _NegationClassifier  # pylint: disable=import-outside-toplevel

        # pylint: disable=protected-access
        if isinstance(self.predicate._classifier, _NegationClassifier):
            return GroundAtom(self.predicate._classifier.body, self.objects)
        # pylint: enable=protected-access
        # classifier = _NegationClassifier(self.predicate)
        # negated_predicate = Predicate(
        #     str(classifier),
        #     self.predicate.types, classifier)
        # return GroundAtom(negated_predicate, self.objects)
        return GroundAtom(self.predicate.get_negation(), self.objects)


@dataclass(frozen=True, eq=False)
class Task:
    """Struct defining a task, which is an initial state and goal."""
    init: State
    goal: Set[GroundAtom]
    # Sometimes we want the task presented to the agent to have goals described
    # in terms of predicates that are different than those describing the goal
    # of the task presented to the demonstrator. In these cases, we will store
    # an "alternative goal" in this field and replace the goal with the
    # alternative goal before giving the task to the agent.
    alt_goal: Optional[Set[GroundAtom]] = field(default_factory=set)
    # Optional natural language description of the goal.  When present,
    # approaches can surface this to an LLM agent so it understands the
    # *intent* behind the goal atoms (e.g. "arrange dominoes so the chain
    # reaction topples the targets" rather than just Toppled(target0)).
    goal_nl: Optional[str] = None
    # Optional per-task ground truth in the standard RL (reward, terminated)
    # shape, propagated from EnvironmentTask.evaluator. Safe to hand to
    # approaches because a TaskEvaluator is by contract a pure,
    # physics-independent function of a state trajectory holding no env
    # handle and no oracle quantities (agent surfaces still expose only
    # VERDICTS, never this object). Dropped by replace_goal_with_alt_goal:
    # its ``goal`` holds the original goal atoms, which alt-goal replacement
    # exists to hide.
    evaluator: Optional[TaskEvaluator] = None

    def __post_init__(self) -> None:
        # Verify types.
        for atom in self.goal:
            assert isinstance(atom, GroundAtom)
        # An attached evaluator must judge THIS task's goal:
        # ``evaluator.terminated`` and ``goal_holds`` are two views of
        # one goal-atom set, and a mismatch (e.g. an evaluator built for
        # the demonstrator goal attached to an alt-goal task) would make
        # them silently disagree. replace_goal_with_alt_goal preserves
        # this by dropping the evaluator together with the goal it holds.
        assert self.evaluator is None or self.evaluator.goal == self.goal

    def goal_holds(
        self,
        state: State,
        vlm: Optional[
            predicators.pretrained_model_interface.VisionLanguageModel] = None
    ) -> bool:
        """Return whether the goal of this task holds in the given state."""
        vlm_atoms = set(atom for atom in self.goal
                        if isinstance(atom.predicate, VLMPredicate))
        for atom in self.goal:
            if atom not in vlm_atoms:
                if not atom.holds(state):
                    return False
        true_vlm_atoms = utils.query_vlm_for_atom_vals(vlm_atoms, state, vlm)
        return len(true_vlm_atoms) == len(vlm_atoms)

    def replace_goal_with_alt_goal(self) -> Task:
        """Return a Task with the goal replaced with the alternative goal if it
        exists."""
        # We may not want the agent to access the goal predicates given to the
        # demonstrator. To prevent leakage of this information, we discard the
        # original goal - and the evaluator, whose ``goal`` holds it.
        if self.alt_goal:
            return Task(self.init, goal=self.alt_goal, goal_nl=self.goal_nl)
        return self


DefaultTask = Task(DefaultState, set())

# A per-step option label: (option name, grounded object names, continuous
# parameters), or None when the action carries no option. Trajectory-level
# evaluator inputs use these instead of raw options so evaluators stay
# picklable and comparison-friendly; the parameters let physics-replaying
# certificates (the domino counterfactual push probe) re-run a step with the
# plan's own continuous values. Consumers must tolerate legacy
# (name, objects) 2-tuples, which some tests and agent-authored label lists
# still produce.
StepOption = Optional[Tuple[str, Tuple[str, ...], Tuple[float, ...]]]


def step_option_labels(actions: Sequence[Action]) -> List[StepOption]:
    """Label each action with its producing option as a ``StepOption``."""
    labels: List[StepOption] = []
    for act in actions:
        if act.has_option():
            option = act.get_option()
            labels.append((option.name, tuple(o.name for o in option.objects),
                           tuple(float(p) for p in option.params)))
        else:
            labels.append(None)
    return labels


class TaskEvaluator:
    """Per-task success/legitimacy/reward: the environment-side ground truth
    for an ``EnvironmentTask``, in the standard RL (reward, terminated) shape.

    ``terminated`` is purely physical: the goal atoms hold in the given
    state, however that came about (an illegitimate topple still
    terminates). Legitimacy (``_certify``) gates only the success bonus
    inside ``reward``, so a rule-violating episode terminates with no
    bonus rather than "not counting" as terminal. ``_certify`` is
    private by design: the agent contract is the public
    (solved, reward) pair - both of which the real environment
    genuinely reveals at episode end - plus roster verdicts;
    ``terminated`` the agent computes itself from the public goal
    atoms, and env-side code (BaseEnv, logging) is the sanctioned
    reader of the certificate's bool/reason.

    The evaluator rides on the agent-facing ``Task``, so instances must
    be leak-free by construction: no live env handle stored on the
    object, and NO oracle quantity anywhere on it (not even in
    ``offline_metrics`` - per-task oracle numbers like the domino K*
    belong in ``EnvironmentTask.offline_task_metrics``, which never
    reaches a ``Task``). Certificates that need a physics rollout (the
    domino counterfactual push probe) receive the caller's env as a
    transient ``sim_env`` argument per call instead - see ``_certify``.

    Subclasses override ``_certify`` for trajectory-level legitimacy
    rules, ``reward`` for cost terms, ``offline_metrics`` for
    experimenter-only episode statistics, and ``objective_description``
    for an agent-showable NL statement of the reward. Defaults
    reproduce the plain atom-set-goal semantics. ``solved`` is the
    public episode-success bit (the standard RL end-of-episode success
    flag; the roster's ``success=`` field): it never depends on a
    reward sign convention, so consumers that need "did this episode
    earn the success credit" (e.g. refinement's accept test) must read
    it rather than compare ``reward`` against zero. Reward contract: a
    certified success must yield strictly positive reward and anything
    else at most zero (the domino evaluator asserts this), so success
    and rejection are decodable from the (reward, terminated) pair
    alone - see ``EpisodeEvaluation.rejected``.
    """

    def __init__(self, goal: Set[GroundAtom]) -> None:
        self.goal = goal

    def terminated(self, state: State) -> bool:
        """Absorbing-state check: do the goal atoms hold?

        The evaluator-side view of ``Task.goal_holds`` (which is the
        public, per-state check and additionally handles VLM
        predicates); ``Task.__post_init__`` asserts the two judge one
        and the same goal-atom set. Subclasses may override to terminate
        on other absorbing states.
        """
        return all(atom.holds(state) for atom in self.goal)

    def reward(self,
               states: Sequence[State],
               step_options: Optional[Sequence[StepOption]],
               sim_env: Optional[Any] = None) -> float:
        """Episode reward: certified-success bonus (no cost by default)."""
        ok, _ = self._certify(states, step_options, sim_env=sim_env)
        return float(self.terminated(states[-1]) and ok)

    def solved(self,
               states: Sequence[State],
               step_options: Optional[Sequence[StepOption]],
               sim_env: Optional[Any] = None) -> bool:
        """Public episode-success bit: goal atoms hold at the end AND the
        success credit was awarded (the episode certifies)."""
        ok, _ = self._certify(states, step_options, sim_env=sim_env)
        return self.terminated(states[-1]) and ok

    def _certify(self,
                 states: Sequence[State],
                 step_options: Optional[Sequence[StepOption]],
                 sim_env: Optional[Any] = None) -> Tuple[bool, str]:
        """Trajectory-level legitimacy: (ok, human-readable reason).

        ``sim_env`` is the certifying caller's live environment (the
        true env in ``BaseEnv``, an agent's belief env in sandbox
        verdicts) for certificates that need a physics rollout - e.g.
        the domino counterfactual push probe. It is passed per call and
        MUST NOT be stored on the evaluator: the evaluator rides on the
        agent-facing ``Task`` and stays leak-free precisely because it
        holds no env handle. ``None`` (the default) runs whatever pure
        rules the certificate has.
        """
        del states, step_options, sim_env  # unused in the default
        return True, ""

    def offline_metrics(
            self, states: Sequence[State],
            step_options: Optional[Sequence[StepOption]]) -> Dict[str, float]:
        """Experimenter-only episode metrics (never shown to agents)."""
        del states, step_options  # unused in the default
        return {}

    def objective_description(self) -> str:
        """Agent-showable NL statement of the reward; '' = nothing to
        show."""
        return ""


@dataclass(frozen=True)
class EpisodeEvaluation:
    """A ``TaskEvaluator``'s verdict on one executed episode, as computed by
    ``BaseEnv.evaluate_episode``.

    ``reward``/``terminated`` are agent-visible by design; ``reason``
    and ``offline_metrics`` are env-side only (the agent gets at most
    the boolean rejection flag). There is deliberately no separate
    ``certified`` field: certification only gates the success bonus, so
    it carries information only when the episode terminated - and there
    the evaluator contract (a certified success strictly outscores any
    failure) makes it decodable as ``reward > 0``.
    """
    reward: float
    terminated: bool
    reason: str
    offline_metrics: Dict[str, float]

    @property
    def rejected(self) -> bool:
        """Terminated without the certified-success bonus: the goal atoms hold,
        but the episode broke the task rules (e.g. a reward-hacked topple).

        A non-terminated episode is never "rejected" - it is just a
        failure; whether its trajectory also broke rules is irrelevant
        because certification only gates the bonus.
        """
        return self.terminated and self.reward <= 0.0


@dataclass(frozen=True, eq=False)
class EnvironmentTask:
    """An initial observation and goal description.

    Environments produce environment tasks and agents produce and solve
    tasks.

    In fully observed settings, the init_obs will be a State and the
    goal_description will be a Set[GroundAtom]. For convenience, we can
    convert an EnvironmentTask into a Task in those cases.
    """
    init_obs: Observation
    goal_description: GoalDescription
    # See Task._alt_goal for the reason for this field.
    alt_goal_desc: Optional[GoalDescription] = field(default=None)
    # Optional natural language goal description (passed through to Task).
    goal_nl: Optional[str] = None
    # Optional per-task ground truth in the standard RL (reward, terminated)
    # shape. When None (every ordinary task), success = all goal_description
    # atoms hold and every trajectory is accepted. When set,
    # ``evaluator.terminated`` IS the success criterion consumed by
    # ``BaseEnv.goal_reached`` (purely physical: goal atoms, however
    # reached), while ``evaluator._certify`` constrains HOW the goal may be
    # reached and gates the success bonus inside ``evaluator.reward``
    # (consumed by ``BaseEnv.check_episode_trajectory`` /
    # ``BaseEnv.evaluate_episode``). Propagated into the agent-facing
    # ``Task`` (see Task.evaluator for why that is leak-free) and dropped
    # by replace_goal_with_alt_goal.
    evaluator: Optional[TaskEvaluator] = None
    # Experimenter-only per-task oracle quantities (e.g. the domino K*, the
    # searched minimum block count at the true physics), merged into the
    # PER_TASK results by main.py. Kept OFF the evaluator and never
    # propagated into ``Task``, so nothing agent-reachable encodes them.
    offline_task_metrics: Dict[str, float] = field(default_factory=dict)
    # Env-side early-stopping bar: when set, a solved training episode
    # counts toward online-learning early stopping only if its episode
    # reward reaches this value (minus
    # CFG.online_learning_early_stopping_reward_slack). Domains express
    # "solved well enough to stop training" in the one domain-general
    # currency, reward - e.g. domino min-block tasks set the optimal
    # reward 1 - block_cost * K*, so an over-built solve keeps training
    # going. May encode oracle quantities, so like offline_task_metrics
    # it never reaches the agent-facing ``Task``. None (the default)
    # keeps the plain solved criterion.
    early_stop_min_reward: Optional[float] = None

    @cached_property
    def task(self) -> Task:
        """Convenience method for environment tasks that are fully observed."""
        # If the environment task's goal is replaced with the alternative goal
        # before turning the environment task into a task, or no alternative
        # goal exists, then there's nothing particular to set the task's
        # alt_goal field to.
        if self.alt_goal_desc is None:
            return Task(self.init,
                        self.goal,
                        goal_nl=self.goal_nl,
                        evaluator=self.evaluator)
        # If we turn the environment task into a task before replacing the goal
        # with the alternative goal, we have to set the task's alt_goal field
        # accordingly to leave open the possibility of doing that replacement
        # later.
        # Assumption: we currently assume the alternative goal description is
        # always a set of ground atoms.
        assert isinstance(self.alt_goal_desc, set)
        for atom in self.alt_goal_desc:
            assert isinstance(atom, GroundAtom)
        return Task(self.init,
                    self.goal,
                    alt_goal=self.alt_goal_desc,
                    goal_nl=self.goal_nl,
                    evaluator=self.evaluator)

    @cached_property
    def init(self) -> State:
        """Convenience method for environment tasks that are fully observed."""
        assert isinstance(self.init_obs, State)
        return self.init_obs

    @cached_property
    def goal(self) -> Set[GroundAtom]:
        """Convenience method for environment tasks that are fully observed."""
        assert isinstance(self.goal_description, set)
        assert not self.goal_description or isinstance(
            next(iter(self.goal_description)), GroundAtom)
        return self.goal_description

    def replace_goal_with_alt_goal(self) -> EnvironmentTask:
        """Return an EnvironmentTask with the goal description replaced with
        the alternative goal description if it exists.

        See Task.replace_goal_with_alt_goal for the reason for this
        function.
        """
        # The evaluator is dropped along with the original goal: its
        # ``goal`` field holds exactly the atoms this replacement hides.
        # The early-stop reward bar goes with it (its value is only
        # meaningful under the dropped evaluator's reward). The env-side
        # offline metrics stay (they never reach a Task).
        if self.alt_goal_desc is not None:
            return EnvironmentTask(
                self.init_obs,
                goal_description=self.alt_goal_desc,
                offline_task_metrics=self.offline_task_metrics)
        return self


DefaultEnvironmentTask = EnvironmentTask(DefaultState, set())


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
    policy: ParameterizedPolicy = field(repr=False)
    # An initiation classifier maps a state, memory dict, objects, and
    # parameters to a bool, which is True iff the option can start
    # now. The objects' types will match those in self.types. The
    # parameters will be contained in params_space.
    initiable: ParameterizedInitiable = field(repr=False)
    # A termination condition maps a state, memory dict, objects, and
    # parameters to a bool, which is True iff the option should
    # terminate now. The objects' types will match those in
    # self.types. The parameters will be contained in params_space.
    terminal: ParameterizedTerminal = field(repr=False)
    params_description: Optional[Tuple[str, ...]] = field(default=None,
                                                          repr=False)

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
        if len(objects) != len(self.types):
            expected = [t.name for t in self.types]
            got = [f"{o.name}:{o.type.name}" for o in objects]
            raise ValueError(
                f"Cannot ground '{self.name}': expected {len(self.types)} "
                f"objects {expected}, got {len(objects)} {got}")
        for i, (obj, t) in enumerate(zip(objects, self.types)):
            if not obj.is_instance(t):
                raise TypeError(
                    f"Cannot ground '{self.name}': object '{obj.name}' at "
                    f"position {i} has type '{obj.type.name}', "
                    f"expected '{t.name}'")
        params = np.array(params, dtype=self.params_space.dtype)
        if not self.params_space.contains(params):
            # Values that passed through float32 (e.g. parsed agent plans)
            # can round a boundary value just past a float64 bound, since
            # float32(pi) > pi; treat within-precision violations as the
            # boundary itself and only reject genuine overshoots.
            low = self.params_space.low
            high = self.params_space.high
            tol = 1e-6 * np.maximum(1.0, np.maximum(np.abs(low), np.abs(high)))
            if np.all(params >= low - tol) and np.all(params <= high + tol):
                params = np.clip(params, low,
                                 high).astype(self.params_space.dtype)
            else:
                raise ValueError(
                    f"Cannot ground '{self.name}': params {params.tolist()} "
                    f"outside bounds low={self.params_space.low.tolist()}, "
                    f"high={self.params_space.high.tolist()}")
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

    def pddl_str(self) -> str:
        """Turn this option into a string that is PDDL-like."""
        params_str = " ".join(f"?x{i} - {t.name}"
                              for i, t in enumerate(self.types))
        return f"{self.name}({params_str})"


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

    def __str__(self) -> str:
        """Full spec including objects and parameters."""
        objects = ", ".join(o.name for o in self.objects)
        params = ", ".join(str(round(p, 2)) for p in self.params)
        return f"{self.name}({objects}, {params})"

    def simple_str(self, use_object_id: bool = False) -> str:
        """Simple spec without parameters."""
        if use_object_id:
            objects = ", ".join(
                [o.id_name + ":" + o.type.name for o in self.objects])
        else:
            objects = ", ".join(o.name for o in self.objects)
        return f"{self.name}({objects})"


DummyParameterizedOption: ParameterizedOption = ParameterizedOption(
    "DummyParameterizedOption", [], Box(0, 1, (0, )),
    lambda s, m, o, p: Action(np.array([0.0])), lambda s, m, o, p: False,
    lambda s, m, o, p: True)

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

    def make_endogenous_process(
        self,
        option: Optional[ParameterizedOption],
        option_vars: Optional[Sequence[Variable]],
        sampler: Optional[NSRTSampler],
        process_strength: Optional[float] = None,
        process_delay_params: Optional[Sequence[float]] = None,
        process_rng: Optional[np.random.Generator] = None,
    ) -> EndogenousProcess:
        """Make a CausalProcess out of this STRIPSOperator object."""
        assert option is not None and option_vars is not None and \
            sampler is not None
        if process_delay_params is None:
            process_delay_params = [5, 1]
        if process_strength is None:
            process_strength = 1.0
        if process_rng is None:
            process_rng = np.random.default_rng(CFG.seed)

        proc = EndogenousProcess(
            self.name,
            self.parameters,
            condition_at_start=self.preconditions
            if option.name != "Wait" else set(),
            condition_overall=set(),
            condition_at_end=set(),
            add_effects=self.add_effects if option.name != "Wait" else set(),
            delete_effects=self.delete_effects
            if option.name != "Wait" else set(),
            delay_distribution=utils.DiscreteGaussianDelay(
                torch.tensor(process_delay_params[0]),
                torch.tensor(process_delay_params[1])),
            strength=process_strength,  # type: ignore[arg-type]
            option=option,
            option_vars=option_vars,
            _sampler=sampler)
        return proc

    def make_exogenous_process(
        self,
        process_strength: Optional[float] = None,
        process_delay_params: Optional[Sequence[float]] = None,
        _process_rng: Optional[np.random.Generator] = None
    ) -> ExogenousProcess:
        """Make an ExogenousProcess out of this STRIPSOperator object."""
        if process_delay_params is None:
            process_delay_params = torch.tensor([1, 1
                                                 ])  # type: ignore[assignment]
        if process_strength is None:
            process_strength = torch.tensor(1.0)  # type: ignore[assignment]
        dist = utils.DiscreteGaussianDelay(torch.tensor(1), torch.tensor(1))

        proc = ExogenousProcess(
            self.name,
            self.parameters,
            condition_at_start=self.preconditions,
            condition_overall=self.preconditions,
            condition_at_end=set(),
            add_effects=self.add_effects,
            delete_effects=self.delete_effects,
            delay_distribution=dist,
            strength=process_strength)  # type: ignore[arg-type]
        return proc

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
    def short_str(self) -> str:
        """Abbreviated name, not necessarily unique."""
        obj_str = ", ".join([o.name for o in self.objects])
        return f"{self.name}({obj_str})"

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
    def short_str(self) -> str:
        """Abbreviated name, not necessarily unique."""
        obj_str = ", ".join([o.name for o in self.objects])
        return f"{self.name}({obj_str})"

    @property
    def ignore_effects(self) -> Set[Predicate]:
        """Ignore effects from the parent."""
        return self.parent.ignore_effects

    @property
    def op(self) -> _GroundSTRIPSOperator:
        """The corresponding ground operator."""
        return self.parent.op.ground(tuple(self.objects))

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
    # In rare cases, we want to associate additional information with an action
    # to control how it is executed in the environment. This is helpful if
    # actions are awkward to represent with continuous vectors, and if we have
    # no ambition to learn models over the actions directly.
    extra_info: Optional[Any] = None

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
    _source_simulator_version: Optional[str] = field(default=None)
    _source_predicates_version: Optional[str] = field(default=None)
    _source_samplers_version: Optional[str] = field(default=None)
    _env_reward: Optional[float] = field(default=None)
    _env_terminated: Optional[bool] = field(default=None)

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

    @property
    def source_simulator_version(self) -> Optional[str]:
        """Snapshot tag of the simulator that generated the plan that collected
        this trajectory (e.g. ``cycle_002_vers_005``), or ``None`` for offline
        demos / trajectories collected before the provenance tracking
        existed."""
        return self._source_simulator_version

    @property
    def source_predicates_version(self) -> Optional[str]:
        """Snapshot tag of the predicates set used to generate the plan that
        collected this trajectory, or ``None`` if not tracked."""
        return self._source_predicates_version

    @property
    def source_samplers_version(self) -> Optional[str]:
        """Snapshot tag of the per-skill samplers used to generate the plan
        that collected this trajectory, or ``None`` if not tracked."""
        return self._source_samplers_version

    @property
    def env_rejected(self) -> bool:
        """Whether the supervisor (the environment's evaluator) rejected the
        episode that produced this trajectory: the goal atoms held but the
        episode broke the task rules, so the success bonus was withheld.

        Derived from the stored (reward, terminated) pair - see
        ``EpisodeEvaluation.rejected`` for the decode contract; no
        separate flag is stored. Deliberately a bare boolean - this
        object is exposed to the agent's sandbox, and the agent must
        infer the violated rule from the task's NL goal description and
        the observed trajectory, not be told it.
        """
        return bool(self.env_terminated) and self.env_reward is not None \
            and self.env_reward <= 0.0

    @property
    def env_reward(self) -> Optional[float]:
        """The env evaluator's episode reward, or ``None`` if not evaluated.

        Computed by ``BaseEnv.evaluate_episode`` and passed through
        ``InteractionResult``. Agent-visible by design: the reward form
        is public and physics-independent, so the value leaks nothing
        about true dynamics. ``getattr`` guards keep pre-field pickles
        loadable.
        """
        return getattr(self, "_env_reward", None)

    @property
    def env_terminated(self) -> Optional[bool]:
        """The env evaluator's terminated verdict (goal atoms held in the final
        state, however reached), or ``None`` if not evaluated."""
        return getattr(self, "_env_terminated", None)


@dataclass(frozen=True, repr=False, eq=False)
class AtomOptionTrajectory:
    """A structure similar to a LowLevelTrajectory but save atoms at every
    state, as well as the option that was executed."""
    _low_level_states: List[State]
    _states: List[Set[GroundAtom]]
    _actions: List[_Option]
    _is_demo: bool = field(default=False)
    _train_task_idx: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        assert len(self._states) == len(self._actions) + 1
        if self._is_demo:
            assert self._train_task_idx is not None

    @property
    def states(self) -> List[Set[GroundAtom]]:
        """States in the trajectory."""
        return self._states

    @property
    def actions(self) -> List[_Option]:
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


@dataclass(frozen=True, repr=False, eq=False)
class ImageOptionTrajectory:
    """A structure similar to a LowLevelTrajectory where we record images at
    every state (i.e., observations), as well as the option that was executed
    to get between observation images. States are optionally included too.

    Invariant 1: If this trajectory is a demonstration, it must contain
    a train task idx and achieve the goal in the respective train task.
    This invariant is checked upon creation of the trajectory (in
    datasets) because the trajectory does not have a goal, it only has a
    train task idx. Invariant 2: The length of the state images sequence
    is always one greater than the length of the action sequence.
    """
    _objects: Collection[Object]
    _state_imgs: List[List[PIL.Image.Image]]
    _cropped_state_imgs: List[List[PIL.Image.Image]]
    _actions: List[_Option]
    _states: Optional[List[State]] = field(default=None)
    _is_demo: bool = field(default=False)
    _train_task_idx: Optional[int] = field(default=None)

    def __post_init__(self) -> None:
        assert len(self._state_imgs) == len(self._actions) + 1
        if self._is_demo:
            assert self._train_task_idx is not None
        if self._states is not None:
            assert len(self._states) == len(self._state_imgs)

    @property
    def imgs(self) -> List[List[PIL.Image.Image]]:
        """State images in the trajectory."""
        return self._state_imgs

    @property
    def cropped_imgs(self) -> List[List[PIL.Image.Image]]:
        """Cropped versions of state images in the trajectory."""
        return self._cropped_state_imgs

    @property
    def objects(self) -> Collection[Object]:
        """Objects important to the trajectory."""
        return self._objects

    @property
    def actions(self) -> List[_Option]:
        """Actions in the trajectory."""
        return self._actions

    @property
    def states(self) -> Optional[List[State]]:
        """States in the trajectory, if they exist."""
        return self._states

    @property
    def train_task_idx(self) -> Optional[int]:
        """Returns the idx of the train task."""
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


@dataclass(repr=False, eq=False)
class ClassificationDataset:
    """Maybe ultimately a collection of LowLevelTrajectory objects, and a list
    of labels, one per trajectory.

    There is List[Video] for each episode
    """
    task_names: List[str]
    support_videos: List[List[Video]]
    support_labels: List[List[int]]
    query_videos: List[List[Video]]
    query_labels: List[List[int]]
    seed: int
    _current_idx: int = field(default=0, init=False, repr=False)
    _rng: random.Random = field(
        default=None,  # type: ignore[assignment]
        init=False,
        repr=False)

    def __post_init__(self) -> None:
        assert len(self.support_videos) == len(self.support_labels) == \
                len(self.query_videos) == len(self.query_labels) == \
                len(self.task_names)
        self._current_idx = 0
        self._rng = random.Random(self.seed)

    def __iter__(self) -> "Iterator[ClassificationEpisode]":
        self._current_idx = 0
        return self

    def __next__(self) -> ClassificationEpisode:
        if self._current_idx >= len(self.support_videos):
            raise StopIteration

        episode_name = self.task_names[self._current_idx]
        episode_support_videos = self.support_videos[self._current_idx]
        episode_support_labels = self.support_labels[self._current_idx]
        episode_query_videos = self.query_videos[self._current_idx]
        episode_query_labels = self.query_labels[self._current_idx]

        assert len(episode_support_videos) == len(episode_support_labels)
        assert len(episode_query_videos) == len(episode_query_labels)

        # Generate a permutation index for shuffling
        perm = list(range(len(episode_query_videos)))
        perm.reverse()
        # self._rng.shuffle(perm)

        # Apply shuffle to query videos and labels
        episode_query_videos = [episode_query_videos[i] for i in perm]
        episode_query_labels = [episode_query_labels[i] for i in perm]

        episode: ClassificationEpisode = (episode_name, episode_support_videos,
                                          episode_support_labels,
                                          episode_query_videos,
                                          episode_query_labels)

        self._current_idx += 1
        return episode

    def __len__(self) -> int:
        """The number of episodes in the dataset."""
        return len(self.support_labels)


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
class PNAD:
    """PNAD: PartialNSRTAndDatastore.

    A helper class for NSRT learning that contains information
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
                         check_effect_equality: bool = True,
                         check_option_equality: bool = True) -> None:
        """Add a new member to self.datastore."""
        seg, var_obj_sub = member
        if len(self.datastore) > 0:
            # All variables should have a corresponding object.
            if CFG.exogenous_process_learner_do_intersect:
                # When we don't assume preconditions contain only atoms with
                # variables present in the effect, we would first include
                # all the variables in the op.parameters, and the var_obj_sub
                # only contain parameters that can be unified with the last
                # segment. So it can be a subset of the op.parameters.
                assert set(var_obj_sub).issubset(set(self.op.parameters))
            else:
                assert set(var_obj_sub) == set(self.op.parameters)
            # The effects should match.
            if check_effect_equality:
                obj_var_sub = {o: v for (v, o) in var_obj_sub.items()}
                lifted_add_effects = {
                    a.lift(obj_var_sub)
                    for a in seg.add_effects
                    if not isinstance(a.predicate, DerivedPredicate)
                }
                lifted_del_effects = {
                    a.lift(obj_var_sub)
                    for a in seg.delete_effects
                    if not isinstance(a.predicate, DerivedPredicate)
                }
                assert lifted_add_effects == self.op.add_effects
                assert lifted_del_effects == self.op.delete_effects
            if seg.has_option() and check_option_equality:
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

    def make_endogenous_process(self) -> EndogenousProcess:
        """Make an EndogenousProcess from this PNAD."""
        assert self.sampler is not None
        param_option, option_vars = self.option_spec
        return self.op.make_endogenous_process(param_option, option_vars,
                                               self.sampler)

    def make_exogenous_process(
        self,
        process_strength: Optional[float] = None,
        process_delay_params: Optional[Sequence[float]] = None,
        _process_rng: Optional[np.random.Generator] = None
    ) -> ExogenousProcess:
        """Make an ExogenousProcess from this PNAD."""
        return self.op.make_exogenous_process(
            process_strength=process_strength,
            process_delay_params=process_delay_params,
        )

    def copy(self) -> PNAD:
        """Make a copy of this PNAD object, taking care to ensure that
        modifying the original will not affect the copy."""
        new_op = self.op.copy_with()
        new_poss_keep_effects = set(self.poss_keep_effects)
        new_seg_to_keep_effects_sub = {}
        # NOTE: Below line effectively does a deep-copy of the nested dicts
        # here. This is crucial for the PNAD search learner (since otherwise,
        # updating a PNAD in a different set may change this dict for a PNAD
        # in the current set).
        for k, v in self.seg_to_keep_effects_sub.items():
            new_seg_to_keep_effects_sub[k] = dict(v)
        new_pnad = PNAD(new_op, self.datastore, self.option_spec)
        new_pnad.poss_keep_effects = new_poss_keep_effects
        new_pnad.seg_to_keep_effects_sub = new_seg_to_keep_effects_sub
        return new_pnad

    def __repr__(self) -> str:
        param_option, option_vars = self.option_spec
        vars_str = ", ".join(str(v) for v in option_vars)
        return f"{self.op}\n    Option Spec: {param_option.name}({vars_str})"

    def __str__(self) -> str:
        return repr(self)

    def __lt__(self, other: PNAD) -> bool:
        return repr(self) < repr(other)


@dataclass(eq=False, repr=False)
class PAPAD:
    """Partial Process and Datastore."""
    # The non option and sampler part of the CausalProcess
    pprocess: PartialProcess


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
    # Optional verdict from a planning explorer: did the *mental model*
    # (the learned simulator) reach the task goal when refining this
    # request's plan? ``None`` means "no verdict" (e.g. non-planning
    # explorers); online learning treats ``False`` as not-solved for
    # early stopping even if real-env execution happens to reach the
    # goal, so a model that executes-but-mispredicts isn't certified as
    # trained. See AgentBilevelExplorer / main._generate_interaction_results.
    mental_model_solved: Optional[bool] = None


@dataclass(frozen=True, eq=False, repr=False)
class InteractionResult:
    """The result of an InteractionRequest. Contains a list of states, a list
    of actions, and a list of responses to queries if provded.

    Invariant: len(states) == len(responses) == len(actions) + 1
    """
    states: List[State]
    actions: List[Action]
    responses: List[Optional[Response]]
    # The env evaluator's (reward, terminated) verdict on the executed
    # episode (see ``BaseEnv.evaluate_episode``); ``None`` when the env
    # defines no evaluator. Agent-visible by design: both are
    # physics-independent functions of the observed trajectory, and the
    # supervisor-rejection boolean is decodable from the pair (see
    # ``EpisodeEvaluation.rejected``) - the specific violation stays in
    # the env-side logs, since the rules themselves are stated in the
    # task's NL goal description.
    episode_reward: Optional[float] = None
    episode_terminated: Optional[bool] = None

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
    info: Optional[Dict] = field(default=None)

    @property
    def cost(self) -> float:
        return 1

    def get_info(self, key: Any) -> Any:
        """Access key from query info."""
        assert self.info is not None
        return self.info[key]


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
        parameter_str = "(" + " ".join(
            [f"{p.name} - {p.type.name}" for p in self.parameters]) + ")"

        def _atom_to_str(atom: LiftedAtom) -> str:
            args_str = " ".join([v.name for v in atom.variables])
            return f"({atom.predicate.name} {args_str})"

        inner_preconditions_strs = [
            _atom_to_str(a) for a in sorted(self.pos_state_preconditions)
        ]
        inner_preconditions_strs += [
            "(not " + _atom_to_str(a) + ")"
            for a in sorted(self.neg_state_preconditions)
        ]
        preconditions_str = " ".join(inner_preconditions_strs)
        if len(inner_preconditions_strs) > 1:
            preconditions_str = "(and " + preconditions_str + ")"
        elif not inner_preconditions_strs:
            preconditions_str = "()"
        goals_strs = [_atom_to_str(a) for a in sorted(self.goal_preconditions)]
        goals_str = " ".join(goals_strs)
        if len(goals_strs) > 1:
            goals_str = "(and " + goals_str + ")"
        elif not goals_strs:
            goals_str = "()"
        action_param_str = " ".join([v.name for v in self.nsrt.parameters])
        action_str = f"({self.nsrt.name} {action_param_str})"
        return f"""(:rule {self.name}
    :parameters {parameter_str}
    :preconditions {preconditions_str}
    :goals {goals_str}
    :action {action_str}
  )"""

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
        """Name of this ground LDL rule."""
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
        rule_str = "\n  ".join(str(r) for r in self.rules)
        return f"(define (policy)\n  {rule_str}\n)"


@dataclass(frozen=True, repr=False, eq=False)
class Macro:
    """A macro is a sequence of NSRTs with shared parameters."""
    parameters: Sequence[Variable]
    nsrts: Sequence[NSRT]
    nsrt_to_macro_params: Sequence[VarToVarSub]

    def __post_init__(self) -> None:
        assert len(self.nsrts) == len(self.nsrt_to_macro_params)
        for nsrt, subs in zip(self.nsrts, self.nsrt_to_macro_params):
            assert set(nsrt.parameters) == set(subs)
            assert set(subs.values()).issubset(self.parameters)
            assert all(p1.type == p2.type for p1, p2 in subs.items())

    @cached_property
    def preconditions(self) -> Set[LiftedAtom]:
        """The preconditions of this Macro."""
        # Map all NSRT preconditions and effects to the macro parameter space.
        macro_param_preconds: List[Set[LiftedAtom]] = []
        macro_param_add_effects: List[Set[LiftedAtom]] = []
        for nsrt, sub in zip(self.nsrts, self.nsrt_to_macro_params):
            preconds = {a.substitute(sub) for a in nsrt.preconditions}
            macro_param_preconds.append(preconds)
            add_effects = {a.substitute(sub) for a in nsrt.add_effects}
            macro_param_add_effects.append(add_effects)
        # Chain together the preconditions and add effects backwards.
        # To chain, shift the add effects back by one.
        empty_adds: Set[LiftedAtom] = set()
        macro_param_add_effects = [empty_adds] + macro_param_add_effects[:-1]
        final_macro_preconditions: Set[LiftedAtom] = set()
        while macro_param_preconds:
            final_macro_preconditions |= macro_param_preconds.pop()
            final_macro_preconditions -= macro_param_add_effects.pop()
        return final_macro_preconditions

    def ground(self, objects: Sequence[Object]) -> GroundMacro:
        """Ground into a GroundMacro, given objects."""
        return GroundMacro(self, objects)

    @cached_property
    def _str(self) -> str:
        member_strs = []
        for nsrt, sub in zip(self.nsrts, self.nsrt_to_macro_params):
            arg_str = ", ".join([sub[o].name for o in nsrt.parameters])
            nsrt_str = f"{nsrt.name}({arg_str})"
            member_strs.append(nsrt_str)
        members_str = ", ".join(member_strs)
        return f"Macro[{members_str}]"

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
        assert isinstance(other, Macro)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, Macro)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, Macro)
        return str(self) > str(other)


@dataclass(frozen=True, repr=False, eq=False)
class GroundMacro:
    """A sequence of ground NSRTs with shared objects."""
    parent: Macro
    objects: Sequence[Object]

    def __post_init__(self) -> None:
        assert len(self.objects) == len(self.parent.parameters)
        for o, p in zip(self.objects, self.parent.parameters):
            assert o.type == p.type

    @classmethod
    def from_ground_nsrts(cls,
                          ground_nsrts: Sequence[_GroundNSRT]) -> GroundMacro:
        """Create a GroundMacro from a sequence of _GroundNSRTs."""
        obj_to_macro_param: ObjToVarSub = {}
        nsrts: List[NSRT] = []
        nsrt_to_macro_params: List[VarToVarSub] = []
        var_count = itertools.count()
        for ground_nsrt in ground_nsrts:
            nsrt = ground_nsrt.parent
            sub: VarToVarSub = {}
            for nsrt_var, obj in zip(nsrt.parameters, ground_nsrt.objects):
                if obj not in obj_to_macro_param:
                    new_var = Variable(f"?x{next(var_count)}", obj.type)
                    obj_to_macro_param[obj] = new_var
                sub[nsrt_var] = obj_to_macro_param[obj]
            nsrts.append(nsrt)
            nsrt_to_macro_params.append(sub)
        parameters = sorted(obj_to_macro_param.values())
        macro = Macro(parameters, nsrts, nsrt_to_macro_params)
        macro_param_to_obj = {v: k for k, v in obj_to_macro_param.items()}
        objects = [macro_param_to_obj[p] for p in macro.parameters]
        return macro.ground(objects)

    @cached_property
    def preconditions(self) -> Set[GroundAtom]:
        """The preconditions of the ground macro."""
        lifted_preconds = self.parent.preconditions
        sub = dict(zip(self.parent.parameters, self.objects))
        ground_preconds = {a.ground(sub) for a in lifted_preconds}
        return ground_preconds

    @cached_property
    def ground_nsrts(self) -> List[_GroundNSRT]:
        """The _GroundNSRTs for this GroundMacro."""
        ground_nsrts: List[_GroundNSRT] = []
        parent = self.parent
        macro_param_to_obj = dict(zip(parent.parameters, self.objects))
        for nsrt, sub in zip(parent.nsrts, parent.nsrt_to_macro_params):
            objs = tuple(macro_param_to_obj[sub[p]] for p in nsrt.parameters)
            ground_nsrt = nsrt.ground(objs)
            ground_nsrts.append(ground_nsrt)
        return ground_nsrts

    def pop(self) -> Tuple[_GroundNSRT, GroundMacro]:
        """Get the next ground NSRT and the remaining ground macro."""
        ground_nsrt_queue = list(self.ground_nsrts)
        next_ground_nsrt = ground_nsrt_queue.pop(0)
        remaining_ground_macro = GroundMacro.from_ground_nsrts(
            ground_nsrt_queue)
        return next_ground_nsrt, remaining_ground_macro

    @cached_property
    def _str(self) -> str:
        member_strs = []
        for nsrt in self.ground_nsrts:
            arg_str = ", ".join([o.name for o in nsrt.objects])
            nsrt_str = f"{nsrt.name}({arg_str})"
            member_strs.append(nsrt_str)
        members_str = ", ".join(member_strs)
        return f"GroundMacro[{members_str}]"

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
        assert isinstance(other, GroundMacro)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, GroundMacro)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, GroundMacro)
        return str(self) > str(other)

    def __len__(self) -> int:
        return len(self.ground_nsrts)


@dataclass(frozen=False, repr=False, eq=False)
class DelayDistribution:
    """Base class for delay distributions."""

    def set_parameters(self, parameters: Sequence[torch.Tensor],
                       **kwargs: Any) -> None:
        """Set the parameters of this distribution."""
        raise NotImplementedError

    def get_parameters(self) -> Sequence[float]:
        """Get the parameters of this distribution."""
        raise NotImplementedError

    def sample(self) -> int:
        """Sample a delay from this distribution."""
        raise NotImplementedError

    def log_prob(self, k: Union[int, torch.Tensor]) -> torch.Tensor:
        """Compute the log probability of a delay value."""
        raise NotImplementedError

    def probability(self, k: int) -> float:
        """Compute the probability of a delay value."""
        raise NotImplementedError

    def copy(self) -> DelayDistribution:
        """Create a copy of this distribution."""
        raise NotImplementedError

    def __str__(self) -> str:
        return self._str

    @cached_property
    def _str(self) -> str:
        raise NotImplementedError


@dataclass(frozen=False, repr=False, eq=False)
class PartialProcess:
    """A partial process placeholder."""


@dataclass(frozen=False, repr=False, eq=False)
class CausalProcess(abc.ABC):
    """Abstract base class for causal processes."""
    name: str
    parameters: Sequence[Variable]
    condition_at_start: Set[LiftedAtom]
    condition_overall: Set[LiftedAtom]
    condition_at_end: Set[LiftedAtom]
    add_effects: Set[LiftedAtom]
    delete_effects: Set[LiftedAtom]
    delay_distribution: DelayDistribution
    strength: torch.Tensor

    @abc.abstractmethod
    def ground(self, objects: Sequence[Object]) -> _GroundCausalProcess:
        """Ground this process with the given objects."""

    @abc.abstractmethod
    def copy(self) -> CausalProcess:
        """Create a deep copy of this causal process."""

    @abc.abstractmethod
    def filter_predicates(self, kept: Collection[Predicate]) -> CausalProcess:
        """Keep only the given predicates in the preconditions, add effects,
        delete effects, and ignore effects.

        Note that the parameters must stay the same for the sake of the
        sampler inputs.
        """

    def _set_parameters(self, parameters: Sequence[float],
                        **kwargs: Any) -> None:
        self.strength = parameters[0]  # type: ignore[assignment]
        self.delay_distribution.set_parameters(
            parameters[1:], **kwargs)  # type: ignore[arg-type]
        # Invalidate cached properties
        if '_str' in self.__dict__:
            del self.__dict__['_str']
        if '_hash' in self.__dict__:
            del self.__dict__['_hash']

    def _get_parameters(self) -> Sequence[float]:
        """Get the parameters of this CausalProcess.

        The first parameter is the strength, and the rest are the delay
        distribution parameters.
        """
        return [
            self.strength
        ] + self.delay_distribution.get_parameters()  # type: ignore[operator]

    def delay_probability(self, delay: int) -> float:
        """Compute the probability of a given delay."""
        return self.delay_distribution.probability(delay)

    @cached_property
    def _hash(self) -> int:
        return hash(str(self))

    def __hash__(self) -> int:
        return self._hash

    @cached_property
    def _str(self) -> str:
        ignore_effects_str = ""
        ign = getattr(self, 'ignore_effects', None)
        if isinstance(ign, set):
            ign_sorted = sorted(ign, key=str)
            ignore_effects_str = (f"\n    Ignore Effects: {ign_sorted}")
        return f"""    Parameters: {self.parameters}
    Conditions at start: {sorted(self.condition_at_start, key=str)}
    Conditions overall: {sorted(self.condition_overall, key=str)}
    Conditions at end: {sorted(self.condition_at_end, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}{ignore_effects_str}
    Log Strength: {self.strength:.4f}
    Delay Distribution: {self.delay_distribution}"""

    @cached_property
    def _str_wo_params(self) -> str:
        return f"""    Parameters: {self.parameters}
    Conditions at start: {sorted(self.condition_at_start, key=str)}
    Conditions overall: {sorted(self.condition_overall, key=str)}
    Conditions at end: {sorted(self.condition_at_end, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}"""

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return str(self)

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, CausalProcess)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, CausalProcess)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, CausalProcess)
        return str(self) > str(other)

    def get_complexity(self) -> float:
        """Get the complexity of this operator.

        We only care about the arity of the operator, since that is what
        affects grounding. We'll use 2^arity as a measure of grounding
        effort.
        """
        return float(2**len(self.parameters))


@dataclass(frozen=False, repr=False, eq=False)
class ExogenousProcess(CausalProcess):
    """An exogenous causal process."""

    def copy(self) -> ExogenousProcess:
        """Create a deep copy of this exogenous process."""
        return ExogenousProcess(
            name=self.name,
            parameters=list(self.parameters),
            condition_at_start=self.condition_at_start.copy(),
            condition_overall=self.condition_overall.copy(),
            condition_at_end=self.condition_at_end.copy(),
            add_effects=self.add_effects.copy(),
            delete_effects=self.delete_effects.copy(),
            delay_distribution=self.delay_distribution.copy(),
            strength=self.strength.clone())

    def filter_predicates(self,
                          kept: Collection[Predicate]) -> ExogenousProcess:
        condition_at_start = {a for a in self.condition_at_start if a.predicate\
                                in kept}
        condition_overall = {a for a in self.condition_overall if a.predicate\
                                in kept}
        condition_at_end = {a for a in self.condition_at_end if a.predicate\
                                in kept}
        add_effects = {a for a in self.add_effects if a.predicate in kept}
        delete_effects = {
            a
            for a in self.delete_effects if a.predicate in kept
        }

        return ExogenousProcess(self.name, self.parameters, condition_at_start,
                                condition_overall, condition_at_end,
                                add_effects, delete_effects,
                                self.delay_distribution, self.strength)

    @cached_property
    def _str(self) -> str:
        process_str = super()._str
        return f"""ExogenousProcess-{self.name}:
{process_str}"""

    @cached_property
    def _str_wo_params(self) -> str:
        process_str = super()._str_wo_params
        return f"""ExogenousProcess-{self.name}:
{process_str}"""

    def ground(self, objects: Sequence[Object]) -> _GroundExogenousProcess:
        assert len(objects) == len(self.parameters)
        assert all(
            o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        condition_at_start = {a.ground(sub) for a in self.condition_at_start}
        condition_overall = {a.ground(sub) for a in self.condition_overall}
        condition_at_end = {a.ground(sub) for a in self.condition_at_end}
        add_effects = {a.ground(sub) for a in self.add_effects}
        delete_effects = {a.ground(sub) for a in self.delete_effects}
        return _GroundExogenousProcess(self, objects, condition_at_start,
                                       condition_overall, condition_at_end,
                                       add_effects, delete_effects)


@dataclass(frozen=False, repr=False, eq=False)
class EndogenousProcess(CausalProcess):
    """An endogenous causal process tied to an option."""
    option: ParameterizedOption
    option_vars: Sequence[Variable]
    _sampler: NSRTSampler = field(repr=False)
    ignore_effects: Set[Predicate] = field(default_factory=set)

    def copy(self) -> EndogenousProcess:
        """Create a deep copy of this endogenous process."""
        return EndogenousProcess(
            name=self.name,
            parameters=list(self.parameters),
            condition_at_start=self.condition_at_start.copy(),
            condition_overall=self.condition_overall.copy(),
            condition_at_end=self.condition_at_end.copy(),
            add_effects=self.add_effects.copy(),
            delete_effects=self.delete_effects.copy(),
            delay_distribution=self.delay_distribution.copy(),
            strength=self.strength.clone(),
            option=copy.copy(self.option),
            option_vars=self.option_vars.copy(),  # type: ignore[attr-defined]
            _sampler=self._sampler.copy(),  # type: ignore[attr-defined]
            ignore_effects=self.ignore_effects.copy(),
        )

    def filter_predicates(self,
                          kept: Collection[Predicate]) -> EndogenousProcess:
        """Keep only the given predicates in the preconditions, add effects,
        delete effects, and ignore effects.

        Note that the parameters must stay the same for the sake of the
        sampler inputs.
        """
        condition_at_start = {a for a in self.condition_at_start if a.predicate\
                                in kept}
        condition_overall = {a for a in self.condition_overall if a.predicate\
                                in kept}
        condition_at_env = {a for a in self.condition_at_end if a.predicate\
                                in kept}
        add_effects = {a for a in self.add_effects if a.predicate in kept}
        delete_effects = {
            a
            for a in self.delete_effects if a.predicate in kept
        }
        ignore_effects = {a for a in self.ignore_effects if a in kept}

        return EndogenousProcess(self.name, self.parameters,
                                 condition_at_start, condition_overall,
                                 condition_at_env, add_effects, delete_effects,
                                 self.delay_distribution, self.strength,
                                 self.option, self.option_vars, self._sampler,
                                 ignore_effects)

    def ground(self, objects: Sequence[Object]) -> _GroundEndogenousProcess:
        assert len(objects) == len(self.parameters)
        assert all(
            o.is_instance(p.type) for o, p in zip(objects, self.parameters))
        sub = dict(zip(self.parameters, objects))
        condition_at_start = {a.ground(sub) for a in self.condition_at_start}
        condition_overall = {a.ground(sub) for a in self.condition_overall}
        condition_at_end = {a.ground(sub) for a in self.condition_at_end}
        add_effects = {a.ground(sub) for a in self.add_effects}
        delete_effects = {a.ground(sub) for a in self.delete_effects}
        option_objs = [sub[v] for v in self.option_vars]
        return _GroundEndogenousProcess(self, objects, condition_at_start,
                                        condition_overall, condition_at_end,
                                        add_effects, delete_effects,
                                        self.option, option_objs,
                                        self._sampler)

    @cached_property
    def _str(self) -> str:
        option_var_str = ", ".join([str(v) for v in self.option_vars])
        process_str = super()._str
        return f"""EndogenousProcess-{self.name}:
{process_str}
    Option Spec: {self.option.name}({option_var_str})"""


@dataclass(frozen=False, repr=False, eq=False)
class _GroundCausalProcess:
    parent: CausalProcess
    objects: Sequence[Object]
    condition_at_start: Set[GroundAtom]
    condition_overall: Set[GroundAtom]
    condition_at_end: Set[GroundAtom]
    add_effects: Set[GroundAtom]
    delete_effects: Set[GroundAtom]

    @property
    def delay_distribution(self) -> DelayDistribution:
        """The delay distribution of the parent CausalProcess."""
        return self.parent.delay_distribution

    @property
    def strength(self) -> float:
        """The strength of the parent CausalProcess."""
        return self.parent.strength  # type: ignore[return-value]

    @abc.abstractmethod
    def cause_triggered(self, state_history: List[Set[GroundAtom]],
                        action_history: List[_Option]) -> bool:
        """Check if this process's cause was triggered."""
        raise NotImplementedError

    def effect_factor(self, state: Set[GroundAtom]) -> float:
        """Compute the effect factor of this ground causal process on the
        state."""
        return int(
            self.add_effects.issubset(state)
            and not self.delete_effects.issubset(state)) * self.strength

    def factored_effect_factor(self, y_tj: bool, factor_atom: GroundAtom,
                               prev_val: bool) -> Tensor:
        """If x_tj is True, we say that x_tj would get the effect factor of a
        process if at this time step, factor_atom is in the add effects and not
        in the delete effects of the process.

        If x_tj is False in the current step t, then we say that x_tj
        would get effect from the effect factor of a process if at this
        time step, x_tj is in the delete effects and not in the add
        effects of the process.
        """
        # match1 requires in the x_tj = False case because match1 requires that
        # (atom in not add_effects or in delete_effects) simply be true,
        # whereas match2 requires specifically that
        # (atom in delete_effects and not in add_effects) be true.
        # match1 = (factor_atom in self.add_effects and
        #          factor_atom not in self.delete_effects) == x_tj
        if y_tj:
            match = int(y_tj != prev_val and factor_atom in self.add_effects
                        and factor_atom not in self.delete_effects)
        else:
            match = int(y_tj != prev_val and factor_atom in self.delete_effects
                        and factor_atom not in self.add_effects)
        return match * self.strength  # type: ignore[return-value]

    @property
    def name(self) -> str:
        """Name of this ground causal process."""
        return self.parent.name

    @cached_property
    def _str(self) -> str:
        return f"""GroundProcess-{self.name}:
    Parameters: {self.objects}
    Conditions at start: {sorted(self.condition_at_start, key=str)}
    Conditions overall: {sorted(self.condition_overall, key=str)}
    Conditions at end: {sorted(self.condition_at_end, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}"""

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
        assert isinstance(other, _GroundCausalProcess)
        return str(self) == str(other)

    def __lt__(self, other: object) -> bool:
        assert isinstance(other, _GroundCausalProcess)
        return str(self) < str(other)

    def __gt__(self, other: object) -> bool:
        assert isinstance(other, _GroundCausalProcess)
        return str(self) > str(other)

    def name_and_objects_str(self) -> str:
        """Return a string with the process name and objects."""
        return f"{self.name}({', '.join([str(o) for o in self.objects])})"


@dataclass(frozen=False, repr=False, eq=False)
class _GroundEndogenousProcess(_GroundCausalProcess):
    option: ParameterizedOption
    option_objs: Sequence[Object]
    _sampler: NSRTSampler = field(repr=False)

    @property
    def ignore_effects(self) -> Set[Predicate]:
        """Ignore effects from the parent."""
        # pylint: disable-next=no-member
        return self.parent.ignore_effects  # type: ignore

    @cached_property
    def _str(self) -> str:
        return f"""Process-{self.name}:
    Parameters: {self.objects}
    Conditions at start: {sorted(self.condition_at_start, key=str)}
    Conditions overall: {sorted(self.condition_overall, key=str)}
    Conditions at end: {sorted(self.condition_at_end, key=str)}
    Add Effects: {sorted(self.add_effects, key=str)}
    Delete Effects: {sorted(self.delete_effects, key=str)}
    Ignore Effects: {sorted(self.ignore_effects, key=str)}
    Option: {self.option}
    Option Objects: {self.option_objs}"""

    def cause_triggered(self, state_history: List[Set[GroundAtom]],
                        action_history: List[_Option]) -> bool:
        """Check if this endogenous process was triggered by the last
        action."""

        def check_wo_s(_state: Set[GroundAtom], action: _Option) -> bool:
            return (action.parent == self.option
                    and action.objects == self.option_objs)

        def check_w_s(state: Set[GroundAtom], action: _Option) -> bool:
            return (action.parent == self.option
                    and action.objects == self.option_objs
                    and self.condition_at_start.issubset(state))

        # if self.name == "SwitchFaucetOff" and check_wo_s(state_history[-1],
        #                                                  action_history[-1]):
        #     breakpoint()
        return check_w_s(state_history[-1], action_history[-1]) and (
            len(state_history) == 1
            or not check_wo_s(state_history[-2], action_history[-2]))

    def copy(self) -> _GroundEndogenousProcess:
        """Make a copy of this _GroundEndogenousProcess object."""
        new_condition_at_start = set(self.condition_at_start)
        new_condition_overall = set(self.condition_overall)
        new_condition_at_end = set(self.condition_at_end)
        new_add_effects = set(self.add_effects)
        new_delete_effects = set(self.delete_effects)
        return _GroundEndogenousProcess(self.parent, self.objects,
                                        new_condition_at_start,
                                        new_condition_overall,
                                        new_condition_at_end, new_add_effects,
                                        new_delete_effects, self.option,
                                        self.option_objs, self._sampler)

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


@dataclass(frozen=False, repr=False, eq=False)
class _GroundExogenousProcess(_GroundCausalProcess):

    def cause_triggered(self, state_history: List[Set[GroundAtom]],
                        action_history: List[_Option]) -> bool:
        """Check if this exogenous process was triggered by the last action."""

        def check(state: Set[GroundAtom]) -> bool:
            return self.condition_at_start.issubset(state)

        return check(state_history[-1]) and (len(state_history) == 1
                                             or not check(state_history[-2]))

    def copy(self) -> _GroundExogenousProcess:
        """Make a copy of this _GroundExogenousProcess object."""
        new_condition_at_start = set(self.condition_at_start)
        new_condition_overall = set(self.condition_overall)
        new_condition_at_end = set(self.condition_at_end)
        new_add_effects = set(self.add_effects)
        new_delete_effects = set(self.delete_effects)
        return _GroundExogenousProcess(self.parent, self.objects,
                                       new_condition_at_start,
                                       new_condition_overall,
                                       new_condition_at_end, new_add_effects,
                                       new_delete_effects)


# Convenience higher-order types useful throughout the code
Observation = Any
GoalDescription = Any
OptionSpec = Tuple[ParameterizedOption, List[Variable]]
GroundAtomTrajectory = Tuple[LowLevelTrajectory, List[Set[GroundAtom]]]
Image = NDArray[np.uint8]
ImageInput = NDArray[np.float32]
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
# NSRT Sampler that also returns a boolean indicating whether the sample was
# generated randomly (for exploration) or from the current learned
# distribution.
NSRTSamplerWithEpsilonIndicator = Callable[
    [State, Set[GroundAtom], np.random.Generator, Sequence[Object]],
    Tuple[Array, bool]]
# Parameterized (per-skill) sampler consulted during bilevel-sketch
# refinement: keyed by ParameterizedOption name, authored/learned once, and
# consulted for every ground call of that option - though each call passes
# the ground binding, so the function can (and should) specialize per
# grounding. Shares NSRTSampler's call signature (state, atoms, rng,
# objects) so the two are interchangeable, but the GroundAtom set it
# receives is the step's *subgoal* (not the task goal), letting it aim
# continuous params at the subgoal instead of drawing uniformly; at steps
# with no subgoal annotation the set is empty, which the sampler must
# tolerate. Returns a params array matching the option's params_space;
# refinement clips it to that box and falls back to uniform on a
# wrong-shaped return. The ground level of the hierarchy is
# bilevel_sketch.GroundSampler, a per-step distribution compiled from a
# `~ [widths]` region annotation (precedence: ground sampler >
# parameterized sampler > uniform).
ParameterizedSampler = Callable[
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
RefinementDatapoint = Tuple[Task, List[_GroundNSRT], List[Set[GroundAtom]],
                            bool, List[float], List[int]]
# For PDDLEnv environments, given a desired number of problems and an rng,
# returns a list of that many PDDL problem strings.
PDDLProblemGenerator = Callable[[int, np.random.Generator], List[str]]
# Used in ml_models.py. Either the maximum number of training iterations for
# a model, or a function that produces this number given the amount of data.
MaxTrainIters = Union[int, Callable[[int], int]]
ExplorationStrategy = Tuple[Callable[[State], Action], Callable[[State], bool]]
ParameterizedPolicy = Callable[[State, Dict, Sequence[Object], Array], Action]
ParameterizedInitiable = Callable[[State, Dict, Sequence[Object], Array], bool]
ParameterizedTerminal = Callable[[State, Dict, Sequence[Object], Array], bool]
AbstractPolicy = Callable[[Set[GroundAtom], Set[Object], Set[GroundAtom]],
                          Optional[_GroundNSRT]]
AbstractProcessPolicy = Callable[
    [Set[GroundAtom], Set[Object], Set[GroundAtom]],
    Optional[_GroundEndogenousProcess]]
RGBA = Tuple[float, float, float, float]
BridgePolicy = Callable[[State, Set[GroundAtom], List[_Option]], _Option]
BridgeDataset = List[Tuple[Set[_Option], _GroundNSRT, Set[GroundAtom], State]]
Mask = NDArray[np.bool_]
ClassificationEpisode = Tuple[str, List[Video], List[int], List[Video],
                              List[int]]
