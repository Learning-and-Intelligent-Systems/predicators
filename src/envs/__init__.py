"""Default imports for envs folder.
"""

from predicators.src.envs.base_env import BaseEnv
from predicators.src.envs.cover import CoverEnv, CoverEnvTypedOptions
from predicators.src.envs.behavior import BehaviorEnv

__all__ = [
    "BaseEnv",
    "CoverEnv",
    "CoverEnvTypedOptions",
    "BehaviorEnv",
]


def create_env(name: str) -> BaseEnv:
    """Create an environment given its name.
    """
    if name == "cover":
        return CoverEnv()
    if name == "cover_typed":
        return CoverEnvTypedOptions()
    if name == "behavior":
        return BehaviorEnv()
    raise NotImplementedError(f"Unknown env: {name}")
