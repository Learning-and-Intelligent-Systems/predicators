"""Test cases for the oracle bridge policy class."""

import numpy as np
import pytest

from predicators import utils
from predicators.bridge_policies.oracle_bridge_policy import \
    OracleBridgePolicy, _create_oracle_bridge_policy
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.settings import CFG
from predicators.structs import DummyOption


def test_oracle_bridge_policy():
    """Tests for OracleBridgePolicy."""
    utils.reset_config({"env": "painting"})
    env = get_or_create_env("painting")
    options = get_gt_options("painting")
    nsrts = get_gt_nsrts("painting", env.predicates, options)
    bridge_policy = OracleBridgePolicy(env.predicates, nsrts)
    rng = np.random.default_rng(123)

    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    test_tasks = [t.task for t in env.get_test_tasks()]
    task = test_tasks[0]
    state = task.init
    held_obj = next(o for o in state if o.type.name == "obj")
    box = next(o for o in state if o.type.name == "box")
    robot = next(o for o in state if o.type.name == "robot")
    failed_nsrt = nsrt_name_to_nsrt["PlaceInBox"].ground(
        [held_obj, box, robot])
    failed_option = failed_nsrt.sample_option(state, task.goal, rng)

    # Test case where bridge policy returns an invalid option.
    def invalid_option_policy(s, a, fo):
        del s, a, fo  # ununsed
        return DummyOption

    bridge_policy._oracle_bridge_policy = invalid_option_policy  # pylint: disable=protected-access

    bridge_policy.reset()
    bridge_policy.record_failed_option(failed_option)
    option_policy = bridge_policy.get_option_policy()
    policy = utils.option_policy_to_policy(option_policy)
    with pytest.raises(utils.OptionExecutionFailure) as e:
        utils.run_policy_with_simulator(policy,
                                        env.simulate,
                                        task.init,
                                        task.goal_holds,
                                        max_num_steps=CFG.horizon)
    assert "Unsound option policy" in str(e)

    with pytest.raises(NotImplementedError):
        _create_oracle_bridge_policy("not a real env", nsrts, env.predicates,
                                     rng)
