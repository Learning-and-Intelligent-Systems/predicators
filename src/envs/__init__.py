"""Default imports for envs folder.
"""

from predicators.src.envs.base_env import BaseEnv, EnvironmentFailure
from predicators.src.envs.cover import CoverEnv, CoverEnvTypedOptions, \
    CoverEnvHierarchicalTypes, CoverMultistepOptions
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
    "CoverMultistepOptions",
    "CoverEnvHierarchicalTypes",
    "ClutteredTableEnv",
    "BlocksEnv",
    "PaintingEnv",
    "PlayroomEnv",
    "RepeatedNextToEnv",
]


_MOST_RECENT_ENV_INSTANCE = {}


def create_env(name: str) -> BaseEnv:
    """Create an environment given its name."""
    # NOTE: All the type ignore comments below
    # are necessary because mypy gets confused
    # by the type of the 'env' variable and
    # complains otherwise.
    if name == "cover":
        env = CoverEnv() # type: ignore
    elif name == "cover_typed_options":
        env = CoverEnvTypedOptions() # type: ignore
    elif name == "cover_multistep_options":
        env = CoverMultistepOptions() # type: ignore
    elif name == "cover_hierarchical_types":
        env = CoverEnvHierarchicalTypes() # type: ignore
    elif name == "cluttered_table":
        env = ClutteredTableEnv() # type: ignore
    elif name == "blocks":
        env = BlocksEnv() # type: ignore
    elif name == "painting":
        env = PaintingEnv() # type: ignore
    elif name == "playroom":
        env = PlayroomEnv() # type: ignore
    elif name == "repeated_nextto":
        return RepeatedNextToEnv() # type: ignore
    else:
        raise NotImplementedError(f"Unknown env: {name}")

    _MOST_RECENT_ENV_INSTANCE[name] = env
    return env


# NOTE: we don't want to cover this because it's only useful
# for the BEHAVIOR environment
def get_env_instance(name: str) -> BaseEnv: # pragma: no cover
    """Get the most recent env instance, or make a new one."""
    if name in _MOST_RECENT_ENV_INSTANCE:
        return _MOST_RECENT_ENV_INSTANCE[name]
    return create_env(name)
