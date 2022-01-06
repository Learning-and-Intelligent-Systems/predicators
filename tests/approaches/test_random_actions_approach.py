"""Test cases for the random actions approach class."""

from predicators.src.approaches import RandomActionsApproach
from predicators.src.envs import CoverEnv
from predicators.src import utils


def test_random_actions_approach():
    """Tests for RandomActionsApproach class."""
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "seed": 123
    })
    env = CoverEnv()
    task = next(env.train_tasks_generator())[0]
    approach = RandomActionsApproach(env.simulate, env.predicates, env.options,
                                     env.types, env.action_space)
    approach.seed(123)
    policy = approach.solve(task, 500)
    actions = []
    for _ in range(10):
        act = policy(task.init)
        actions.append(act)
        assert env.action_space.contains(act.arr)
    # Test reproducibility
    assert str(actions) == "[Action(_arr=array([0.70787615], dtype=float32)), Action(_arr=array([0.3698764], dtype=float32)), Action(_arr=array([0.29010695], dtype=float32)), Action(_arr=array([0.10647454], dtype=float32)), Action(_arr=array([0.9975787], dtype=float32)), Action(_arr=array([0.9942262], dtype=float32)), Action(_arr=array([0.98252517], dtype=float32)), Action(_arr=array([0.55868745], dtype=float32)), Action(_arr=array([0.68523175], dtype=float32)), Action(_arr=array([0.99104315], dtype=float32))]" # pylint: disable=line-too-long
