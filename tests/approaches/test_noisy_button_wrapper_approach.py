"""Test cases for the noisy button wrapper approach."""

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs.noisy_button import NoisyButtonEnv
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG


def test_noisy_button_wrapper_approach():
    """Test for NoisyButtonWrapperApproach class."""
    utils.reset_config({
        "env": "noisy_button",
        "approach": "noisy_button_wrapper[oracle]",
        "timeout": 10,
        "num_train_tasks": 1,
        "num_test_tasks": 10,
    })
    env = NoisyButtonEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = create_approach(CFG.approach, env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space, train_tasks)
    assert not approach.is_learning_based
    for task in [t.task for t in env.get_test_tasks()]:
        policy = approach.solve(task, timeout=CFG.timeout)
        traj = utils.run_policy_with_simulator(policy,
                                               env.simulate,
                                               task.init,
                                               task.goal_holds,
                                               max_num_steps=CFG.horizon)
        assert task.goal_holds(traj.states[-1])
