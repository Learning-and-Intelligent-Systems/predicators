"""Config for CoverEnv.
"""
import ml_collections
from predicators.configs.envs import default_env_config


def get_config() -> ml_collections.ConfigDict:
    """Create config dict.
    """
    config = default_env_config.get_config()
    config.name = "Cover"
    config.num_blocks = 2
    config.num_targets = 2
    config.num_train_tasks = 5
    config.num_test_tasks = 10
    config.block_widths = [0.1, 0.07]
    config.target_widths = [0.05, 0.03]
    return config
