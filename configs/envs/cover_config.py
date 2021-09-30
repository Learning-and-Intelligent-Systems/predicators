"""Config for CoverEnv.
"""
import ml_collections
from predicators.configs.envs import default_env_config


def get_config() -> ml_collections.ConfigDict:
    """Create config dict.
    """
    config = default_env_config.get_config()
    config.name = "Cover"
    return config
