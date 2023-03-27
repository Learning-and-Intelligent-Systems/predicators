"""Test cases for the BridgePolicyApproach class."""
from predicators import utils
from predicators.approaches.bridge_policy_approach import BridgePolicyApproach
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG


def _policy_solves_task(policy, task, simulator):
    """Helper method used in this file, copied from test_oracle_approach.py."""
    traj = utils.run_policy_with_simulator(policy,
                                           simulator,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=CFG.horizon)
    return task.goal_holds(traj.states[-1])


def test_bridge_policy_approach():
    """Tests for BridgePolicyApproach class."""
    args = {
        "refinement_estimator": "bridge_policy",
        "env": "painting",
        "painting_lid_open_prob": 0.0,
        "painting_raise_environment_failure": False,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
    }
    utils.reset_config(args)
    env = get_or_create_env(CFG.env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    test_tasks = [t.task for t in env.get_test_tasks()]
    approach = BridgePolicyApproach(env.predicates,
                                    get_gt_options(env.get_name()), env.types,
                                    env.action_space, train_tasks)
    assert approach.get_name() == "bridge_policy"
    for task in test_tasks:
        policy = approach.solve(task, timeout=500)
        assert _policy_solves_task(policy, task, env.simulate)
