"""Handle creation of bridge policies."""

from predicators import utils
from predicators.bridge_policies.base_bridge_policy import BaseBridgePolicy

__all__ = ["BaseBridgePolicy", "create_bridge_policy"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_bridge_policy(name: str) -> BaseBridgePolicy:
    """Create a bridge policy given its name."""
    for cls in utils.get_all_subclasses(BaseBridgePolicy):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            bridge_policy = cls()
            break
    else:
        raise NotImplementedError(f"Unknown bridge policy: {name}")
    return bridge_policy
