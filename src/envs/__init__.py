"""Default imports for envs folder.
"""

from predicators.src.envs.base_env import BaseEnv, EnvironmentFailure
from predicators.src.envs.cover import CoverEnv, CoverEnvTypedOptions
from predicators.src.envs.cluttered_table import ClutteredTableEnv

__all__ = [
    "BaseEnv",
    "EnvironmentFailure",
    "CoverEnv",
    "CoverEnvTypedOptions",
    "ClutteredTableEnv",
]


def create_env(name: str) -> BaseEnv:
    """Create an environment given its name.
    """
    if name == "cover":
        return CoverEnv()
    if name == "cover_typed":
        return CoverEnvTypedOptions()
    if name == "cluttered_table":
        return ClutteredTableEnv()
    raise NotImplementedError(f"Unknown env: {name}")
