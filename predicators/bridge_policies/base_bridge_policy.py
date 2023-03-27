"""Base class for a bridge policy."""

import abc
from typing import Callable, Set

from predicators.structs import GroundAtom, State, _GroundNSRT, _Option, Action


class BaseBridgePolicy(abc.ABC):
    """Base bridge policy."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this bridge policy, for future use as the
        argument to `--bridge_policy`."""
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def get_policy(self, state: State, atoms: Set[GroundAtom],
                 failed_nsrt: _GroundNSRT) -> Callable[[State], Action]:
        """The main method creating the bridge policy."""
        raise NotImplementedError("Override me!")
