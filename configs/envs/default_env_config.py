"""Default config for envs.
"""
import ml_collections


def get_config() -> ml_collections.ConfigDict:
    """Create config dict.
    """
    config = ml_collections.ConfigDict()
    return config
