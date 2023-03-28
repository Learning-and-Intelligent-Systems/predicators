"""Base class for perceivers."""

import abc

from predicators.structs import EnvironmentTask, Observation, State, Task


class BasePerceiver(abc.ABC):
    """A perceiver consumes observations and produces states."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this perceiver."""

    @abc.abstractmethod
    def reset(self, env_task: EnvironmentTask) -> Task:
        """Start a new episode of environment interaction."""

    @abc.abstractmethod
    def step(self, observation: Observation) -> State:
        """Produce a State given the current and past observations."""
