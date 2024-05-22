"""Test cases for the noisy button wrapper approach."""

import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs.noisy_button import NoisyButtonEnv
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import Action


@pytest.mark.parametrize("base_name,check_solved", [("oracle", True),
                                                    ("random_options", False)])
def test_noisy_button_wrapper_approach(base_name, check_solved):
    """Test for NoisyButtonWrapperApproach class."""
    utils.reset_config({
        "env": "noisy_button",
        "approach": f"noisy_button_wrapper[{base_name}]",
        "timeout": 10,
        "num_train_tasks": 1,
        "num_test_tasks": 10,
    })
    env = NoisyButtonEnv()
    assert env.goal_predicates == env.predicates
    assert len(env.predicates) == 1
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
        if check_solved:
            assert task.goal_holds(traj.states[-1])
    task = train_tasks[0]
    assert env.reset("train", 0).allclose(task.init)
    env.render_state_plt(task.init, task, caption="caption")
    env.render_state_plt(task.init, task, action=Action([0.5, 0.0]))
    # Cover the inherited methods.
    approach.learn_from_offline_dataset([])
    approach.load(None)
    approach.get_interaction_requests()
    approach.learn_from_interaction_results([])
