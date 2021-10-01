"""Default imports for envs folder.
"""

from predicators.src.envs.base_env import BaseEnv
from predicators.src.envs.cover import CoverEnv

__all__ = [
    "BaseEnv",
    "CoverEnv",
]


def create_env(name: str) -> BaseEnv:
    """Create an environment given its name.
    """
    if name == "cover":
        return CoverEnv()
    raise NotImplementedError(f"Unknown env: {name}")
