"""Handle creation of perceivers."""

from predicators import utils
from predicators.perception.base_perceiver import BasePerceiver

__all__ = ["BasePerceiver"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_perceiver(name: str, ) -> BasePerceiver:
    """Create a perceiver given its name."""
    for cls in utils.get_all_subclasses(BasePerceiver):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            perceiver = cls()
            break
    else:
        raise NotImplementedError(f"Unrecognized perceiver: {name}")
    return perceiver
