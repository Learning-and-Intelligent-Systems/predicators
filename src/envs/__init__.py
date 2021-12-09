"""Default imports for envs folder.
"""

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


def create_env(name: str) -> BaseEnv:
    """Create an environment given its name.
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
        return BehaviorEnv()  # pragma: no cover
    if name == "repeated_nextto":
        return RepeatedNextToEnv()
    raise NotImplementedError(f"Unknown env: {name}")
