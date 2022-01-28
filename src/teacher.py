"""Contains classes defining a teacher that can supply additional, oracular
information to assist an agent during online learning."""

from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import List
from predicators.src.structs import State, Object
from predicators.src.settings import CFG, get_allowed_query_type_names
from predicators.src.envs import create_env

################################################################################
#                                Core classes                                  #
################################################################################


class Teacher:
    """The teacher can respond to queries of various types."""

    def __init__(self) -> None:
        env = create_env(CFG.env)
        self._pred_name_to_pred = {pred.name: pred for pred in env.predicates}
        self._allowed_query_type_names = get_allowed_query_type_names()

    def answer_query(self, state: State, query: Query) -> Response:
        """The key method that the teacher defines."""
        assert query.__class__.__name__ in self._allowed_query_type_names, \
            f"Disallowed query: {query}"
        if isinstance(query, GroundAtomHoldsQuery):
            return self._answer_GroundAtomHolds_query(state, query)
        raise ValueError(f"Unrecognized query: {query}")

    def _answer_GroundAtomHolds_query(
            self, state: State,
            query: GroundAtomHoldsQuery) -> GroundAtomHoldsResponse:
        pred = self._pred_name_to_pred[query.predicate_name]
        holds = pred.holds(state, query.objects)
        return GroundAtomHoldsResponse(query, holds)


@dataclass(frozen=True, eq=False, repr=False)
class Query(abc.ABC):
    """Base class for a Query.

    Has no API.
    """


@dataclass(frozen=True, eq=False, repr=False)
class Response(abc.ABC):
    """Base class for a Response to a query.

    All responses contain the Query object itself, for convenience.
    """
    query: Query


################################################################################
#                        Query and Response subclasses                         #
################################################################################


@dataclass(frozen=True, eq=False, repr=False)
class GroundAtomHoldsQuery(Query):
    """A query for whether a grounding of a predicate holds in the state."""
    predicate_name: str
    objects: List[Object]


@dataclass(frozen=True, eq=False, repr=False)
class GroundAtomHoldsResponse(Response):
    """A response to a GroundAtomHoldsQuery, providing a boolean answer."""
    holds: bool
