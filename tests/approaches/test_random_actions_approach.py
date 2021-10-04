"""Test cases for the random actions approach class.
"""
from predicators.src.approaches import RandomActionsApproach
from predicators.src.envs import CoverEnv


def test_random_actions_approach():
    """Tests for RandomActionsApproach class.
    """
    env = CoverEnv()
    tasks = env.get_train_tasks()
    task = tasks[0]
    approach = RandomActionsApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    approach.seed(123)
    policy = approach.solve(task, 500)
    for _ in range(10):
        act = policy(task.init)
        assert env.action_space.contains(act.arr)
