"""Default imports for envs folder.
"""

from predicators.src.envs.base_env import BaseEnv, EnvironmentFailure
from predicators.src.envs.cover import (
    CoverEnv,
    CoverEnvTypedOptions,
    CoverEnvHierarchicalTypes,
    CoverMultistepOptions,
)
from predicators.src.envs.behavior import BehaviorEnv
from predicators.src.envs.cluttered_table import ClutteredTableEnv
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.envs.playroom import PlayroomEnv

__all__ = [
    "BaseEnv",
    "EnvironmentFailure",
    "CoverEnv",
    "CoverEnvTypedOptions",
    "CoverEnvHierarchicalTypes",
    "CoverMultistepOptions",
    "ClutteredTableEnv",
    "BlocksEnv",
    "PlayroomEnv",
    "BehaviorEnv",
]


_MOST_RECENT_ENV_INSTANCE = {}

def create_env(name: str) -> BaseEnv:
    """Create an environment given its name."""
    if name == "cover_typed_options":
        env = CoverEnvTypedOptions() # type: ignore
    elif name == "cover_hierarchical_types":
        env = CoverEnvHierarchicalTypes() # type: ignore
    elif name == "cluttered_table":
        env = ClutteredTableEnv() # type: ignore
    elif name == "blocks":
        env = BlocksEnv() # type: ignore
    elif name == "behavior":
        env = BehaviorEnv() # type: ignore # pragma: no cover
    else:
        raise NotImplementedError(f"Unknown env: {name}")

    _MOST_RECENT_ENV_INSTANCE[name] = env
    return env


def get_env_instance(name: str) -> BaseEnv:
    """Get the most recent env instance, or make a new one."""
    if name in _MOST_RECENT_ENV_INSTANCE:
        return _MOST_RECENT_ENV_INSTANCE[name]
    return create_env(name)
