"""Test cases for the PG3 approach."""

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, create_approach
from predicators.src.approaches.pg3_approach import _PolicyGuidedPG3Heuristic, _PolicyEvaluationPG3Heuristic, _DemoPlanComparisonPG3Heuristic, _AddConditionPG3SearchOperator, _AddRulePG3SearchOperator
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_new_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.settings import CFG
from predicators.src.structs import LDLRule, LiftedDecisionList


def test_pg3_approach():
    """Tests for PG3Approach()."""
    import ipdb; ipdb.set_trace()


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
    for i in range(len(score_sequence)-1):
        assert score_sequence[i] > score_sequence[i+1]

    # The baseline score functions should decrease (not strictly).
    for heuristic_cls in [_PolicyEvaluationPG3Heuristic,
                          _DemoPlanComparisonPG3Heuristic]:
        heuristic = heuristic_cls(env.predicates, nsrts,
                                              ground_atom_trajectories,
                                              train_tasks)    
        score_sequence = [heuristic(ldl) for ldl in policy_sequence]
        for i in range(len(score_sequence)-1):
            assert score_sequence[i] >= score_sequence[i+1]
