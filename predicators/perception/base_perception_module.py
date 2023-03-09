"""Base class for a perception module."""

import abc

class BasePerceptionModule(abc.ABC):
    """Base class for a perception module."""

    @abc.abstractmethod
    def observe(self, observation: Observation) -> State:
        """Incorporate the given state and return the state."""
        
