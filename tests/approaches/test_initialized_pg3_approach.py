"""Test cases for the PG3 approach."""

import tempfile

import dill as pkl
import pytest

from predicators import utils
from predicators.approaches.initialized_pg3_approach import \
    InitializedPG3Approach
from predicators.envs import create_new_env
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.structs import LDLRule, LiftedDecisionList


@pytest.mark.parametrize("approach_name,approach_cls",
                         [("initialized_pg3", InitializedPG3Approach)])
def test_initialized_pg3_approach(approach_name, approach_cls):
    """Tests for InitializedPG3Approach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": approach_name,
    })

    env = create_new_env(env_name)
    nsrts = get_gt_nsrts(env.predicates, env.options)
    train_tasks = env.get_train_tasks()

    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
    approach = approach_cls(env.predicates, env.options, env.types,
                            env.action_space, train_tasks)
    approach._nsrts = nsrts  # pylint: disable=protected-access

    # Create dummy policy
    pick_up_nsrt = name_to_nsrt["pick-up"]

    pick_up_rule = LDLRule(name="PickUp",
                           parameters=pick_up_nsrt.parameters,
                           pos_state_preconditions=set(
                               pick_up_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           nsrt=pick_up_nsrt)

    ldl = LiftedDecisionList([
        pick_up_rule,
    ])

    ldl_policy_file = tempfile.NamedTemporaryFile()
    pkl.dump(ldl, ldl_policy_file)

    utils.reset_config({
        "env": "blocks",
        "approach": approach_name,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "pg3_heuristic": "demo_plan_comparison",  # faster for tests
        "pg3_search_method": "hill_climbing",
        "pg3_hc_enforced_depth": 0,
        "pg3_init_policy": ldl_policy_file.name
    })
    approach = approach_cls(env.predicates, env.options, env.types,
                            env.action_space, train_tasks)
    assert approach.get_name() == approach_name

    assert approach._get_policy_search_initial_ldl() == ldl  # pylint: disable=protected-access
