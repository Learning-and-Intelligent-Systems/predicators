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
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "pg3",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "pg3_heuristic": "plan_comparison",  # faster for tests
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    approach = PG3Approach(env.predicates, env.options, env.types,
                           env.action_space, train_tasks)
    assert approach.get_name() == "pg3"

    # Test meta-controller prediction.
    nsrts = get_gt_nsrts(env.predicates, env.options)
    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
    pick_up_nsrt = name_to_nsrt["pick-up"]
    pick_up_rule = LDLRule(name="MyPickUp",
                           parameters=pick_up_nsrt.parameters,
                           pos_state_preconditions=set(
                               pick_up_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           nsrt=pick_up_nsrt)
    ldl = LiftedDecisionList([pick_up_rule])
    approach._current_ldl = ldl  # pylint: disable=protected-access
    task = train_tasks[0]
    policy = approach.solve(task, timeout=500)
    act = policy(task.init)
    option = act.get_option()
    assert option.name == "pick-up"
    assert str(option.objects) == "[paper-0:paper, loc-0:loc]"
    ldl = LiftedDecisionList([])
    approach._current_ldl = ldl  # pylint: disable=protected-access
    policy = approach.solve(task, timeout=500)
    with pytest.raises(ApproachFailure) as e:
        _ = policy(task.init)
    assert "PG3 policy was not applicable!" in str(e)

    # Test learning with a fast heuristic.
    dataset = create_dataset(env, train_tasks, env.options)
    approach.learn_from_offline_dataset(dataset)
    load_path = utils.get_approach_load_path_str()
    expected_policy_file = f"{load_path}_None.ldl"
    assert os.path.exists(expected_policy_file)
    approach.load(None)


def test_pg3_search_operators():
    """Tests for PG3 search operator classes."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({"env": env_name})
    env = create_new_env(env_name)
    nsrts = get_gt_nsrts(env.predicates, env.options)
    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
    pick_up_nsrt = name_to_nsrt["pick-up"]
    preds = env.predicates

    pick_up_rule = LDLRule(name="MyPickUp",
                           parameters=pick_up_nsrt.parameters,
                           pos_state_preconditions=set(
                               pick_up_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           nsrt=pick_up_nsrt)

    ldl1 = LiftedDecisionList([])
    ldl2 = LiftedDecisionList([pick_up_rule])

    # _AddRulePG3SearchOperator
    op = _AddRulePG3SearchOperator(preds, nsrts)

    succ1 = list(op.get_successors(ldl1))
    assert len(succ1) == 3
    ldl1_1, ldl1_2, ldl1_3 = sorted(succ1, key=lambda l: l.rules[0].name)
    assert str(ldl1_1) == """LiftedDecisionList[
LDLRule-deliver:
    Parameters: [?paper:paper, ?loc:loc]
    Pos State Pre: [at(?loc:loc), carrying(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: deliver(?paper:paper, ?loc:loc)
]"""

    assert str(ldl1_2) == """LiftedDecisionList[
LDLRule-move:
    Parameters: [?from:loc, ?to:loc]
    Pos State Pre: [at(?from:loc), safe(?from:loc)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: move(?from:loc, ?to:loc)
]"""

    assert str(ldl1_3) == """LiftedDecisionList[
LDLRule-pick-up:
    Parameters: [?paper:paper, ?loc:loc]
    Pos State Pre: [at(?loc:loc), ishomebase(?loc:loc), unpacked(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: pick-up(?paper:paper, ?loc:loc)
]"""

    succ2 = list(op.get_successors(ldl2))
    assert len(succ2) == 6
    ldl2_1 = min(succ2, key=lambda l: l.rules[0].name)
    assert str(ldl2_1) == """LiftedDecisionList[
LDLRule-MyPickUp:
    Parameters: [?paper:paper, ?loc:loc]
    Pos State Pre: [at(?loc:loc), ishomebase(?loc:loc), unpacked(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: pick-up(?paper:paper, ?loc:loc)
LDLRule-deliver:
    Parameters: [?paper:paper, ?loc:loc]
    Pos State Pre: [at(?loc:loc), carrying(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: deliver(?paper:paper, ?loc:loc)
]"""

    # _AddConditionPG3SearchOperator
    op = _AddConditionPG3SearchOperator(preds, nsrts)

    succ1 = list(op.get_successors(ldl1))
    assert len(succ1) == 0

    succ2 = list(op.get_successors(ldl2))
    assert len(succ2) == 36
    ldl2_1 = min(succ2, key=lambda l: l.rules[0].name)
    assert str(ldl2_1) == """LiftedDecisionList[
LDLRule-MyPickUp:
    Parameters: [?loc:loc, ?paper:paper, ?x0:loc]
    Pos State Pre: [at(?loc:loc), at(?x0:loc), ishomebase(?loc:loc), unpacked(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: pick-up(?paper:paper, ?loc:loc)
]"""


def test_pg3_heuristics():
    """Tests for PG3 heuristic classes."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "pg3",
        "num_train_tasks": 10,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "pg3_heuristic": "policy_guided",
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    ground_atom_trajectories = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    nsrts = get_gt_nsrts(env.predicates, env.options)
    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
    deliver_nsrt = name_to_nsrt["deliver"]
    pick_up_nsrt = name_to_nsrt["pick-up"]
    move_nsrt = name_to_nsrt["move"]
    name_to_pred = {pred.name: pred for pred in env.predicates}
    satisfied = name_to_pred["satisfied"]
    wantspaper = name_to_pred["wantspaper"]

    pick_up_rule = LDLRule(name="PickUp",
                           parameters=pick_up_nsrt.parameters,
                           pos_state_preconditions=set(
                               pick_up_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           nsrt=pick_up_nsrt)

    paper, loc = deliver_nsrt.parameters
    assert "paper" in str(paper)
    assert "loc" in str(loc)

    deliver_rule1 = LDLRule(name="Deliver",
                            parameters=[loc, paper],
                            pos_state_preconditions=set(
                                deliver_nsrt.preconditions),
                            neg_state_preconditions=set(),
                            goal_preconditions=set(),
                            nsrt=deliver_nsrt)

    deliver_rule2 = LDLRule(
        name="Deliver",
        parameters=[loc, paper],
        pos_state_preconditions=set(deliver_nsrt.preconditions),
        neg_state_preconditions={satisfied([loc])},  # different
        goal_preconditions=set(),
        nsrt=deliver_nsrt)

    from_loc, to_loc = move_nsrt.parameters
    assert "from" in str(from_loc)
    assert "to" in str(to_loc)

    move_rule1 = LDLRule(name="Move",
                         parameters=[from_loc, to_loc],
                         pos_state_preconditions=set(move_nsrt.preconditions),
                         neg_state_preconditions=set(),
                         goal_preconditions=set(),
                         nsrt=move_nsrt)

    move_rule2 = LDLRule(
        name="Move",
        parameters=[from_loc, to_loc],
        pos_state_preconditions=set(move_nsrt.preconditions) | \
                                {wantspaper([to_loc])},  # different
        neg_state_preconditions=set(),
        goal_preconditions=set(),
        nsrt=move_nsrt
    )

    # Scores should monotonically decrease.
    policy_sequence = [
        LiftedDecisionList([]),
        LiftedDecisionList([pick_up_rule]),
        LiftedDecisionList([pick_up_rule, deliver_rule1]),
        LiftedDecisionList([pick_up_rule, deliver_rule2]),
        LiftedDecisionList([pick_up_rule, deliver_rule2, move_rule1]),
        LiftedDecisionList([pick_up_rule, deliver_rule2, move_rule2]),
    ]

    # The policy-guided heuristic should strictly decrease.
    heuristic = _PolicyGuidedPG3Heuristic(env.predicates, nsrts,
                                          ground_atom_trajectories,
                                          train_tasks)
    score_sequence = [heuristic(ldl) for ldl in policy_sequence]
    for i in range(len(score_sequence) - 1):
        assert score_sequence[i] > score_sequence[i + 1]

    # The baseline score functions should decrease (not strictly).
    for heuristic_cls in [
            _PolicyEvaluationPG3Heuristic, _DemoPlanComparisonPG3Heuristic
    ]:
        heuristic = heuristic_cls(env.predicates, nsrts,
                                  ground_atom_trajectories, train_tasks)
        score_sequence = [heuristic(ldl) for ldl in policy_sequence]
        for i in range(len(score_sequence) - 1):
            assert score_sequence[i] >= score_sequence[i + 1]
