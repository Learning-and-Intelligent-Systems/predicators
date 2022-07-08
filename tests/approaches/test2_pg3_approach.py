"""Test cases for the PG3 approach."""

import os

import pytest

from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.pg3_approach import PG3Approach, \
    _AddConditionPG3SearchOperator, _AddRulePG3SearchOperator, \
    _DemoPlanComparisonPG3Heuristic, _PolicyEvaluationPG3Heuristic, \
    _PolicyGuidedPG3Heuristic
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.structs import LDLRule, LiftedDecisionList


def test_pg3_approach():
    """Tests for PG3Approach()."""
    env_name = "painting"
    utils.reset_config({
        "env": env_name,
        "approach": "pg3",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "pg3_heuristic": "demo_plan_comparison",  # faster for tests
        "pg3_search_method": "hill_climbing",
        "pg3_hc_enforced_depth": 0,
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    approach = PG3Approach(env.predicates, env.options, env.types,
                           env.action_space, train_tasks)
    assert approach.get_name() == "pg3"

    # Test prediction with a good policy.
    nsrts = get_gt_nsrts(env.predicates, env.options)
    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
    paint_to_box_nsrt = name_to_nsrt["PaintToBox"]
    paint_to_shelf_nsrt = name_to_nsrt["PaintToShelf"]
    place_in_box_nsrt = name_to_nsrt["PlaceInBox"]
    place_in_shelf_nsrt = name_to_nsrt["PlaceInShelf"]
    place_on_table_nsrt = name_to_nsrt["PlaceOnTable"]
    pick_from_top_nsrt = name_to_nsrt["PickFromTop"]
    pick_from_side_nsrt = name_to_nsrt["PickFromSide"]
    wash_nsrt = name_to_nsrt["Wash"]
    dry_nsrt = name_to_nsrt["Dry"]
    name_to_pred = {pred.name: pred for pred in env.predicates}
    is_wet = name_to_pred["IsWet"]
    is_dirty = name_to_pred["IsDirty"]
    is_clean = name_to_pred["IsClean"]
    is_dry = name_to_pred["IsDry"]
    is_box_color = name_to_pred["IsBoxColor"]
    is_shelf_color = name_to_pred["IsShelfColor"]
    in_shelf = name_to_pred["InShelf"]
    in_box = name_to_pred["InBox"]
    on_table = name_to_pred["OnTable"]
    not_on_table = name_to_pred["NotOnTable"]
    holding = name_to_pred["Holding"]
    holding_top = name_to_pred["HoldingTop"]
    holding_side = name_to_pred["HoldingSide"]
    gripper_open = name_to_pred["GripperOpen"]

    obj, box, robot = paint_to_box_nsrt.parameters
    assert "obj" in str(object)
    assert "box" in str(box)
    assert "robot" in str(robot)

    _, shelf,_ = paint_to_shelf_nsrt.parameters
    assert "shelf" in str(shelf)

    paint_to_box_rule = LDLRule(name="PaintToBox",
                           parameters=[obj,box,robot],
                           pos_state_preconditions=set(
                               paint_to_box_nsrt.preconditions),
                           neg_state_preconditions={is_box_color([obj,box])},
                           goal_preconditions={is_box_color([obj,box])},
                           nsrt=paint_to_box_nsrt)

    paint_to_shelf_rule = LDLRule(name="PaintToShelf",
                           parameters=[obj,shelf,robot],
                           pos_state_preconditions=set(
                               paint_to_shelf_nsrt.preconditions),
                           neg_state_preconditions={is_shelf_color([obj,shelf])},
                           goal_preconditions={is_shelf_color([obj,shelf])},
                           nsrt=paint_to_shelf_nsrt)

    place_in_box_rule = LDLRule(name="PlaceInBox",
                           parameters=[obj,box,robot],
                           pos_state_preconditions=set(
                               place_in_box_nsrt.preconditions) | \
                               {is_box_color([obj,box])},
                           neg_state_preconditions=set(),
                           goal_preconditions={in_box([obj,box])},
                           nsrt=place_in_box_nsrt)

    place_in_shelf_rule = LDLRule(name="PlaceInShelf",
                           parameters=[obj,shelf,robot],
                           pos_state_preconditions=set(
                               place_in_shelf_nsrt.preconditions) | \
                               {is_shelf_color([obj,shelf])},
                           neg_state_preconditions=set(),
                           goal_preconditions={in_shelf([obj,shelf])},
                           nsrt=place_in_shelf_nsrt)

    # place_on_table_rule = LDLRule(name="PlaceOnTable",
    #                        parameters=[obj,robot],
    #                        pos_state_preconditions=set(
    #                            place_on_table_nsrt.preconditions),
    #                        neg_state_preconditions=set(),
    #                        goal_preconditions=set(),
    #                        nsrt=place_on_table_nsrt)

    pick_from_top_rule = LDLRule(name="PickFromTop",
                           parameters=[obj,box,robot],
                           pos_state_preconditions=set(
                               pick_from_top_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions={in_box([obj,box])},
                           nsrt=pick_from_top_nsrt)

    pick_from_side_rule = LDLRule(name="PickFromSide",
                           parameters=[obj,shelf,robot],
                           pos_state_preconditions=set(
                               pick_from_side_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions={in_shelf([obj,shelf])},
                           nsrt=pick_from_side_nsrt)

    wash_from_top_rule = LDLRule(name="WashFromTop",
                           parameters=[obj,box,robot],
                           pos_state_preconditions=set(
                               wash_nsrt.preconditions) | \
                               {holding_top([obj])},
                           neg_state_preconditions=set(),
                           goal_preconditions={is_box_color([obj,box])},
                           nsrt=wash_nsrt)

    wash_from_side_rule = LDLRule(name="WashFromSide",
                           parameters=[obj,shelf,robot],
                           pos_state_preconditions=set(
                               wash_nsrt.preconditions) | \
                               {holding_side([obj])},
                           neg_state_preconditions=set(),
                           goal_preconditions={is_shelf_color([obj,shelf])},
                           nsrt=wash_nsrt)

    dry_from_top_rule = LDLRule(name="DryFromTop",
                           parameters=[obj,box,robot],
                           pos_state_preconditions=set(
                               dry_nsrt.preconditions) | \
                               {holding_top([obj])},
                           neg_state_preconditions=set(),
                           goal_preconditions={is_box_color([obj,box])},
                           nsrt=dry_nsrt)
    dry_from_side_rule = LDLRule(name="DryFromSide",
                           parameters=[obj,shelf,robot],
                           pos_state_preconditions=set(
                               dry_nsrt.preconditions) | \
                               {holding_side([obj])},
                           neg_state_preconditions=set(),
                           goal_preconditions={is_shelf_color([obj,shelf])},
                           nsrt=dry_nsrt)

    ldl = LiftedDecisionList([paint_to_box_rule, paint_to_shelf_rule,
                            place_in_box_rule, place_in_shelf_rule,
                            pick_from_top_rule,
                            pick_from_side_rule, wash_from_top_rule,
                            wash_from_side_rule, dry_from_top_rule,
                            dry_from_side_rule])
    approach._current_ldl = ldl  # pylint: disable=protected-access
    task = train_tasks[0]
    policy = approach.solve(task, timeout=500)
    act = policy(task.init)
    option = act.get_option()
    # assert option.name == "Pick"
    # assert str(option.objects) == "[paper-0:paper, loc-0:loc]"
    ldl = LiftedDecisionList([])
    approach._current_ldl = ldl  # pylint: disable=protected-access
    with pytest.raises(ApproachFailure) as e:
        approach.solve(task, timeout=500)
    assert "PG3 policy was not applicable!" in str(e)

    # Test learning with a fast heuristic.
    dataset = create_dataset(env, train_tasks, env.options)
    approach.learn_from_offline_dataset(dataset)
    load_path = utils.get_approach_load_path_str()
    expected_policy_file = f"{load_path}_None.ldl"
    assert os.path.exists(expected_policy_file)
    approach.load(None)
    # Test with GBFS instead of hill climbing.
    utils.update_config({
        "pg3_search_method": "gbfs",
        "pg3_gbfs_max_expansions": 1
    })
    approach.learn_from_offline_dataset(dataset)
    # Test invalid search method.
    utils.update_config({
        "pg3_search_method": "not a real search method",
    })
    with pytest.raises(NotImplementedError):
        approach.learn_from_offline_dataset(dataset)
