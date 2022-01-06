"""Default imports for envs folder."""

from predicators.src.envs.base_env import BaseEnv, EnvironmentFailure
from predicators.src.envs.cover import CoverEnv, CoverEnvTypedOptions, \
    CoverEnvHierarchicalTypes, CoverMultistepOptions
from predicators.src.envs.behavior import BehaviorEnv
from predicators.src.envs.cluttered_table import ClutteredTableEnv
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.envs.painting import PaintingEnv
from predicators.src.envs.playroom import PlayroomEnv
from predicators.src.envs.repeated_nextto import RepeatedNextToEnv

__all__ = [
    "BaseEnv",
    "EnvironmentFailure",
    "CoverEnv",
    "CoverEnvTypedOptions",
    "CoverEnvHierarchicalTypes",
    "CoverMultistepOptions",
    "ClutteredTableEnv",
    "BlocksEnv",
    "PaintingEnv",
    "PlayroomEnv",
    "BehaviorEnv",
    "RepeatedNextToEnv",
]

_MOST_RECENT_ENV_INSTANCE = {}


def _create_new_env_instance(name: str) -> BaseEnv:
    """Create a new instance of an environment from its name.

    Note that this env instance will not be cached.
    """
    if name == "cover":
        return CoverEnv()
    if name == "cover_typed_options":
        return CoverEnvTypedOptions()
    if name == "cover_hierarchical_types":
        return CoverEnvHierarchicalTypes()
    if name == "cover_multistep_options":
        return CoverMultistepOptions()
    if name == "cluttered_table":
        return ClutteredTableEnv()
    if name == "blocks":
        return BlocksEnv()
    if name == "painting":
        return PaintingEnv()
    if name == "playroom":
        return PlayroomEnv()
    if name == "behavior":
        return BehaviorEnv() # pragma: no cover
    if name == "repeated_nextto":
        return RepeatedNextToEnv()
    raise NotImplementedError(f"Unknown env: {name}")


def create_env(name: str) -> BaseEnv:
    """Create an environment instance given its name and cache it."""
    env = _create_new_env_instance(name)
    _MOST_RECENT_ENV_INSTANCE[name] = env
    return env


def get_cached_env_instance(name: str) -> BaseEnv:
    """Get the most recent cached env instance (env must have been previously
    created with create_env() to exist in the cache)."""
    assert name in _MOST_RECENT_ENV_INSTANCE
    return _MOST_RECENT_ENV_INSTANCE[name]
