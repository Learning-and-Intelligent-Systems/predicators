"""Test cases for the TAMP approach."""

import pytest
from predicators.src.envs import CoverEnv
from predicators.src.approaches import TAMPApproach


def test_tamp_approach():
    """Tests for TAMPApproach class."""
    env = CoverEnv()
    approach = TAMPApproach(env.simulate, env.predicates, env.options,
                            env.types, env.action_space)
    for task in next(env.train_tasks_generator()):
        with pytest.raises(NotImplementedError):
            approach.solve(task, timeout=500)
