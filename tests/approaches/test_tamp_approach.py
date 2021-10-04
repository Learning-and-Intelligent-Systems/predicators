"""Test cases for the TAMP approach.
"""

import pytest
from predicators.src.envs import CoverEnv
from predicators.src.approaches import TAMPApproach


def test_tamp_approach():
    """Tests for TAMPApproach class.
    """
    env = CoverEnv()
    approach = TAMPApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    for task in env.get_train_tasks():
        with pytest.raises(NotImplementedError):
            approach.solve(task, timeout=500)
