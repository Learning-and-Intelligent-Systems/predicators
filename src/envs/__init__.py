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
]

_MOST_RECENT_ENV_INSTANCE = {}


def create_new_env(name: str, do_cache: bool = False) -> BaseEnv:
    """Create a new instance of an environment from its name.

    If do_cache is True, then cache this env instance so that it can
    later be loaded using get_or_create_env().
    """
    if name == "cover":
        env: BaseEnv = CoverEnv()
    elif name == "cover_typed_options":
        env = CoverEnvTypedOptions()
    elif name == "cover_hierarchical_types":
        env = CoverEnvHierarchicalTypes()
    elif name == "cover_regrasp":
        env = CoverEnvRegrasp()
    elif name == "cover_multistep_options":
        env = CoverMultistepOptions()
    elif name == "cover_multistep_options_fixed_tasks":
        env = CoverMultistepOptionsFixedTasks()
    elif name == "cluttered_table":
        env = ClutteredTableEnv()
    elif name == "cluttered_table_place":
        env = ClutteredTablePlaceEnv()
    elif name == "blocks":
        env = BlocksEnv()
    elif name == "painting":
        env = PaintingEnv()
    elif name == "tools":
        env = ToolsEnv()
    elif name == "playroom":
        env = PlayroomEnv()
    elif name == "behavior":
        env = BehaviorEnv()  # pragma: no cover
    elif name == "repeated_nextto":
        env = RepeatedNextToEnv()
    else:
        raise NotImplementedError(f"Unknown env: {name}")
    if do_cache:
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
