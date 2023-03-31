"""Base class for execution monitors."""

import abc

from predicators.structs import State, Task


class BaseExecutionMonitor(abc.ABC):
    """An execution monitor consumes states and decides whether to replan."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this execution monitor."""

    @abc.abstractmethod
    def reset(self, task: Task) -> None:
        """Reset after replanning."""

    @abc.abstractmethod
    def step(self, state: State) -> bool:
        """Return true if the agent should replan."""
