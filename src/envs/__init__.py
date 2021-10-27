"""Default imports for envs folder.
"""

from predicators.src.envs.base_env import BaseEnv, EnvironmentFailure
from predicators.src.envs.cover import CoverEnv, CoverEnvTypedOptions
from predicators.src.envs.cluttered_table import ClutteredTableEnv
from predicators.src.envs.blocks import BlocksEnv

__all__ = [
    "BaseEnv",
    "EnvironmentFailure",
    "CoverEnv",
    "CoverEnvTypedOptions",
    "ClutteredTableEnv",
    "BlocksEnv",
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
    if name == "blocks":
        return BlocksEnv()
    raise NotImplementedError(f"Unknown env: {name}")
