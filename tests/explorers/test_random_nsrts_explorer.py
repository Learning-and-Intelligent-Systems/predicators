"""Test cases for the random nsrts explorer class."""
import pytest

from predicators import utils
from predicators.envs.touch_point import TouchOpenEnv
from predicators.explorers import create_explorer
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options


def test_random_nsrts_explorer():
    """Tests for RandomNSRTsExplorer class."""
    utils.reset_config({
        "env": "touch_open",
        "explorer": "random_nsrts",
    })
    env = TouchOpenEnv()
    train_tasks = env.get_train_tasks()
    task_idx = 0
    task = train_tasks[task_idx]
    options = get_gt_options(env.get_name())
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    explorer = create_explorer("random_nsrts", env.predicates, options,
                               env.types, env.action_space, train_tasks, nsrts)
    policy, termination_function = explorer.get_exploration_strategy(
        task_idx, 500)

    # Test the general use of the policy and termination function.
    assert not termination_function(task.init)
    state = task.init
    for _ in range(5):
        act = policy(state)
        state = env.simulate(state, act)
        assert env.action_space.contains(act.arr)

    # Test case where no applicable nsrt can be found.
    insufficient_nsrts = {sorted(nsrts)[1]}  #  OpenDoor
    explorer = create_explorer("random_nsrts", env.predicates, options,
                               env.types, env.action_space, train_tasks,
                               insufficient_nsrts)
    policy, _ = explorer.get_exploration_strategy(task_idx, 500)
    with pytest.raises(utils.RequestActPolicyFailure) as e:
        policy(task.init)
    assert "No applicable NSRT in this state!" in str(e)
