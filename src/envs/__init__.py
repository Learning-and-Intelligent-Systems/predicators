"""Default imports for envs folder."""

from predicators.src.envs.base_env import BaseEnv
from predicators.src.envs.cover import CoverEnv, CoverEnvTypedOptions, \
    CoverEnvHierarchicalTypes, CoverMultistepOptions, \
    CoverMultistepOptionsFixedTasks, CoverEnvRegrasp
from predicators.src.envs.behavior import BehaviorEnv
from predicators.src.envs.cluttered_table import ClutteredTableEnv, \
    ClutteredTablePlaceEnv
from predicators.src.envs.blocks import BlocksEnv
from predicators.src.envs.painting import PaintingEnv
from predicators.src.envs.tools import ToolsEnv
from predicators.src.envs.playroom import PlayroomEnv
from predicators.src.envs.repeated_nextto import RepeatedNextToEnv
from predicators.src.envs.repeated_nextto_painting import \
    RepeatedNextToPaintingEnv

__all__ = [
    "BaseEnv",
    "CoverEnv",
    "CoverEnvTypedOptions",
    "CoverEnvHierarchicalTypes",
    "CoverEnvRegrasp",
    "CoverMultistepOptions",
    "CoverMultistepOptionsFixedTasks",
    "ClutteredTableEnv",
    "BlocksEnv",
    "PaintingEnv",
    "ToolsEnv",
    "PlayroomEnv",
    "BehaviorEnv",
    "RepeatedNextToEnv",
    "RepeatedNextToPaintingEnv",
]

_MOST_RECENT_ENV_INSTANCE = {}


def create_new_env(name: str, do_cache: bool = False) -> BaseEnv:
    """Create a new instance of an environment from its name.

    If do_cache is True, then cache this env instance so that it can
    later be loaded using get_or_create_env().
    """
    if name == "cover":
        return CoverEnv()
    if name == "cover_typed_options":
        return CoverEnvTypedOptions()
    if name == "cover_hierarchical_types":
        return CoverEnvHierarchicalTypes()
    if name == "cover_regrasp":
        return CoverEnvRegrasp()
    if name == "cover_multistep_options":
        return CoverMultistepOptions()
    if name == "cover_multistep_options_fixed_tasks":
        return CoverMultistepOptionsFixedTasks()
    if name == "cluttered_table":
        return ClutteredTableEnv()
    if name == "cluttered_table_place":
        return ClutteredTablePlaceEnv()
    if name == "blocks":
        return BlocksEnv()
    if name == "painting":
        return PaintingEnv()
    if name == "tools":
        return ToolsEnv()
    if name == "playroom":
        return PlayroomEnv()
    if name == "behavior":
        return BehaviorEnv()  # pragma: no cover
    if name == "repeated_nextto":
        return RepeatedNextToEnv()
    if name == "repeated_nextto_painting":
        return RepeatedNextToPaintingEnv()
    raise NotImplementedError(f"Unknown env: {name}")


def create_env(name: str) -> BaseEnv:
    """Create an environment instance given its name and cache it."""
    env = _create_new_env_instance(name)
    _MOST_RECENT_ENV_INSTANCE[name] = env
    return env


def get_or_create_env(name: str) -> BaseEnv:
    """Get the most recent cached env instance. If one does not exist in the
    cache, create it using create_new_env().

    If you use this function, you should NOT be doing anything that
    relies on the environment's internal state (i.e., you should not
    call reset() or step()).
    """
    if name not in _MOST_RECENT_ENV_INSTANCE:
        print("WARNING: you called get_or_create_env, but I couldn't find "
              f"{name} in the cache. Making a new environment instance.")
        create_new_env(name, do_cache=True)
    return _MOST_RECENT_ENV_INSTANCE[name]
