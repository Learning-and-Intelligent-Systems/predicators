"""Config for TAMPApproach.
"""
import ml_collections
from predicators.configs.approaches import default_approach_config


def get_config() -> ml_collections.ConfigDict:
    """Create config dict.
    """
    config = default_approach_config.get_config()
    config.name = "TAMP Approach"
    config.max_samples_per_step = 10
    config.max_num_steps_option_rollout = 100
    return config
