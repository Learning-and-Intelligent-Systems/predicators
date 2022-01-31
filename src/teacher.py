"""Contains classes defining a teacher that can supply additional, oracular
information to assist an agent during online learning."""

from __future__ import annotations
import abc
from dataclasses import dataclass
from typing import Collection, Dict
from predicators.src.structs import State, GroundAtom
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
        assert isinstance(query, GroundAtomsHoldQuery)
        return self._answer_GroundAtomsHold_query(state, query)

    def _answer_GroundAtomsHold_query(
            self, state: State,
            query: GroundAtomsHoldQuery) -> GroundAtomsHoldResponse:
        holds = {}
        for ground_atom in query.ground_atoms:
            pred = self._pred_name_to_pred[ground_atom.predicate.name]
            holds[ground_atom] = pred.holds(state, ground_atom.objects)
        return GroundAtomsHoldResponse(query, holds)


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
class GroundAtomsHoldQuery(Query):
    """A query for whether ground atoms hold in the state."""
    ground_atoms: Collection[GroundAtom]


@dataclass(frozen=True, eq=False, repr=False)
class GroundAtomsHoldResponse(Response):
    """A response to a GroundAtomsHoldQuery, providing boolean answers."""
    holds: Dict[GroundAtom, bool]
