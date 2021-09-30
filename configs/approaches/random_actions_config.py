"""Config for RandomActionsApproach.
"""
import ml_collections
from predicators.configs.approaches import default_approach_config


def get_config() -> ml_collections.ConfigDict:
    """Create config dict.
    """
    config = default_approach_config.get_config()
    config.name = "Random Actions"
    return config
