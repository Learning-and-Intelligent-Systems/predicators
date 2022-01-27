"""Contains classes defining a teacher that can supply additional, oracular
information to assist an agent during online learning.
"""

from __future__ import annotations
import abc
from dataclasses import dataclass
from predicators.src.structs import State


class Teacher:
    """The teacher can respond to queries of various types.
    """
    def __init__(self) -> None:
        pass  # TODO: instantiate an oracle approach

    def ask(self, state: State, query: Query) -> QueryResponse:
        """The key method that a teacher defines.
        """
        raise NotImplementedError


@dataclass(frozen=True, eq=False, repr=False)
class Query(abc.ABC):
    """Base class for a Query. Has no API.
    """


@dataclass(frozen=True, eq=False, repr=False)
class QueryResponse(abc.ABC):
    """Base class for a Response to a query. All responses contain the
    Query object itself, for convenience.
    """
    query: Query
