"""Test cases for the BridgePolicyApproach class."""

from unittest.mock import patch

import predicators.bridge_policies.oracle_bridge_policy
from predicators import utils
from predicators.approaches.bridge_policy_approach import BridgePolicyApproach
from predicators.bridge_policies import BridgePolicyDone
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG

_MODULE_PATH = predicators.bridge_policies.oracle_bridge_policy.__name__


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
    task = test_tasks[0]
    policy = approach.solve(task, timeout=500)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=CFG.horizon)
    assert task.goal_holds(traj.states[-1])

    # Test case where bridge policy hands back control to planner immediately.
    # The policy should get stuck and not achieve the goal, but not crash.
    def done_option_policy(s):
        del s  # ununsed
        raise BridgePolicyDone()

    with patch(f"{_MODULE_PATH}.OracleBridgePolicy.get_policy") as m:
        m.return_value = done_option_policy
        policy = approach.solve(task, timeout=500)
        traj = utils.run_policy_with_simulator(policy,
                                               env.simulate,
                                               task.init,
                                               task.goal_holds,
                                               max_num_steps=25)
        assert not task.goal_holds(traj.states[-1])
        for t in range(-1, -5, 1):
            assert traj.actions[t].get_option().name == "Place"
