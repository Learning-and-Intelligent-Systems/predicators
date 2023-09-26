"""Base class for execution monitors."""

import abc
from typing import Any, List, Optional

from predicators.structs import Action, State, Task


class BaseExecutionMonitor(abc.ABC):
    """An execution monitor consumes states and decides whether to replan."""

    def __init__(self) -> None:
        self._approach_info: List[Any] = []
        self._curr_plan_timestep = 0
        self._prev_action: Optional[Action] = None

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this execution monitor."""

    def reset(self, task: Task) -> None:
        """Reset after replanning."""
        del task  # unused
        self._curr_plan_timestep = 0
        self._prev_action = None

    @abc.abstractmethod
    def step(self, state: State) -> bool:
        """Return true if the agent should replan."""

    def update_with_action(self, action: Action) -> None:
        """Called after each action is executed."""
        self._prev_action = action

    def update_approach_info(self, info: List[Any]) -> None:
        """Update internal info received from approach."""
        self._approach_info = info
