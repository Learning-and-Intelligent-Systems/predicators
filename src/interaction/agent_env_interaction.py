"""Contains classes for handling the interaction between an agent and
the environment, during the online learning loop in main.py that occurs
after learning from the offline dataset.
"""

from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional
from predicators.src.structs import State, Action
from predicators.src.interaction.teacher import Query, QueryResponse


@dataclass(frozen=True, eq=False, repr=False)
class InteractionRequest:
    """A request for interacting with a training task. Contains an index
    for that training task, an acting policy, a query policy, and a
    termination function.
    """
    train_task_idx: int
    act_policy: Callable[[State], Action]
    query_policy: Callable[[State], Optional[Query]]  # query can be None
    termination_function: Callable[[State], bool]


# A response to an InteractionRequest. This response is a sequence
# of (state, action, response to query if provided, next_state) tuples.
InteractionResponse = List[Tuple[State, Action, Optional[QueryResponse], State]]
