"""Test cases for the base bridge policy class."""

import pytest

from predicators import utils
from predicators.bridge_policies import BaseBridgePolicy, create_bridge_policy
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options


def test_bridge_policy_creation():
    """Tests for create_bridge_policy()."""
    utils.reset_config({"env": "painting"})
    env = get_or_create_env("painting")
    options = get_gt_options("painting")
    nsrts = get_gt_nsrts("painting", env.predicates, options)
    bridge_policy = create_bridge_policy("oracle", env.types, env.predicates,
                                         options, nsrts)
    assert isinstance(bridge_policy, BaseBridgePolicy)
    assert bridge_policy.get_name() == "oracle"
    with pytest.raises(NotImplementedError):
        create_bridge_policy("not a real bridge policy", env.types,
                             env.predicates, options, nsrts)
