"""Default imports for envs folder.
"""

from predicators.src.envs.base_env import BaseEnv, EnvironmentFailure
from predicators.src.envs.cover import CoverEnv, CoverEnvTypedOptions
from predicators.src.envs.behavior import BehaviorEnv
from predicators.src.envs.cluttered_table import ClutteredTableEnv

__all__ = [
    "BaseEnv",
    "EnvironmentFailure",
    "CoverEnv",
    "CoverEnvTypedOptions",
    "BehaviorEnv",
    "ClutteredTableEnv",
]


_MOST_RECENT_ENV_INSTANCE = {}


def create_env(name: str) -> BaseEnv:
    """Create an environment given its name.
    """
    if name == "cover":
        env = CoverEnv()
    elif name == "cover_typed":
        env = CoverEnvTypedOptions()
    elif name == "behavior":
        env = BehaviorEnv()
    elif name == "cluttered_table":
        env = ClutteredTableEnv()
    else:
        raise NotImplementedError(f"Unknown env: {name}")

    _MOST_RECENT_ENV_INSTANCE[name] = env

    return env


def get_env_instance(name: str) -> BaseEnv:
    """Get the most recent env instance, or make a new one.
    """
    if name in _MOST_RECENT_ENV_INSTANCE:
        return _MOST_RECENT_ENV_INSTANCE[name]
    return create_env(name)
