"""Default imports for envs folder.
"""

from predicators.src.envs.base_env import BaseEnv
from predicators.src.envs.cover import CoverEnv, CoverEnvTypedOptions

__all__ = [
    "BaseEnv",
    "CoverEnv",
    "CoverEnvTypedOptions",
]


def create_env(name: str) -> BaseEnv:
    """Create an environment given its name.
    """
    if name == "cover":
        return CoverEnv()
    if name == "cover_typed":
        return CoverEnvTypedOptions()
    raise NotImplementedError(f"Unknown env: {name}")
