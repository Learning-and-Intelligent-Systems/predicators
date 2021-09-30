"""Test cases for the oracle approach class.
"""
from absl import flags
from predicators.configs.envs import cover_config
from predicators.src.approaches import OracleApproach
from predicators.src.envs import CoverEnv


def test_oracle_approach():
    """Tests for OracleApproach class.
    """
    flags.env = cover_config.get_config()
    env = CoverEnv()
    approach = OracleApproach(env.simulate, env.predicates,
                              env.options, env.action_space)
    # Test get gt operators
    operators = approach._get_current_operators()
    
