"""Test cases for the random actions approach class."""
from predicators import utils
from predicators.approaches.random_actions_approach import \
    RandomActionsApproach
from predicators.envs.cover import CoverEnv
from predicators.ground_truth_models import get_gt_options


def test_random_actions_approach():
    """Tests for RandomActionsApproach class."""
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    task = train_tasks[0]
    approach = RandomActionsApproach(env.predicates,
                                     get_gt_options(env.get_name()), env.types,
                                     env.action_space, train_tasks)
    assert not approach.is_learning_based
    policy = approach.solve(task, 500)
    actions = []
    for _ in range(10):
        act = policy(task.init)
        actions.append(act)
        assert env.action_space.contains(act.arr)
    # Test reproducibility
    assert str(actions) == "[Action(_arr=array([0.6823519], dtype=float32), extra_info=None), Action(_arr=array([0.05382102], dtype=float32), extra_info=None), Action(_arr=array([0.22035988], dtype=float32), extra_info=None), Action(_arr=array([0.18437181], dtype=float32), extra_info=None), Action(_arr=array([0.1759059], dtype=float32), extra_info=None), Action(_arr=array([0.8120945], dtype=float32), extra_info=None), Action(_arr=array([0.92334497], dtype=float32), extra_info=None), Action(_arr=array([0.2765744], dtype=float32), extra_info=None), Action(_arr=array([0.81975454], dtype=float32), extra_info=None), Action(_arr=array([0.8898927], dtype=float32), extra_info=None)]"  # pylint: disable=line-too-long
