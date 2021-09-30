"""Test cases for the TAMP approach.
"""

from predicators.src.envs import CoverEnv
from predicators.src.approaches import TAMPApproach
from predicators.src import utils


def test_tamp_approach():
    """Tests for TAMPApproach class.
    """
    env = CoverEnv()
    env.seed(123)
    approach = TAMPApproach(env.simulate, env.predicates, env.options,
                            env.action_space)
    for task in env.get_train_tasks():
        policy = approach.solve(task, timeout=500)
        _check_policy(task, env.simulate, env.predicates, policy)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        _check_policy(task, env.simulate, env.predicates, policy)


def _check_policy(task, simulator, predicates, policy):
    state = task.init
    atoms = utils.abstract(state, predicates)
    assert not task.goal.issubset(atoms)  # goal shouldn't be already satisfied
    for _ in range(100):
        act = policy(state)
        state = simulator(state, act)
        atoms = utils.abstract(state, predicates)
        if task.goal.issubset(atoms):
            break
    assert task.goal.issubset(atoms)
