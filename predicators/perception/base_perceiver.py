"""Base class for perceivers."""

import abc

from predicators.structs import Observation, State


class BasePerceiver(abc.ABC):
    """A perceiver consumes observations and produces states."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this perceiver."""

    @abc.abstractmethod
    def reset(self, observation: Observation) -> State:
        """Start a new episode of environment interaction."""

    @abc.abstractmethod
    def step(self, observation: Observation) -> State:
        """Produce a State given the current and past observations."""
