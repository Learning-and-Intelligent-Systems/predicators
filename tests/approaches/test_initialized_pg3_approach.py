"""Test cases for the Initialized PG3 approach."""

import tempfile
from unittest.mock import patch

import dill as pkl

import predicators.approaches.initialized_pg3_approach
from predicators import utils
from predicators.approaches.initialized_pg3_approach import \
    InitializedPG3Approach, _Analogy, _apply_analogy_to_ldl
from predicators.envs import create_new_env
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.structs import LDLRule, LiftedDecisionList

_MODULE_PATH = predicators.approaches.initialized_pg3_approach.__name__


def test_initialized_pg3_approach():
    """Tests for InitializedPG3Approach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "initialized_pg3",
    })

    env = create_new_env(env_name)
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, env.options)
    train_tasks = env.get_train_tasks()

    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}

    # Create dummy policy
    pick_up_nsrt = name_to_nsrt["pick-up"]

    pick_up_rule = LDLRule(name="PickUp",
                           parameters=sorted(pick_up_nsrt.parameters),
                           pos_state_preconditions=set(
                               pick_up_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           nsrt=pick_up_nsrt)

    ldl = LiftedDecisionList([pick_up_rule])

    ldl_policy_file = tempfile.NamedTemporaryFile(suffix=".ldl")
    pkl.dump(ldl, ldl_policy_file)

    utils.reset_config({
        "env": env_name,
        "approach": "initialized_pg3",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "pg3_heuristic": "demo_plan_comparison",  # faster for tests
        "pg3_search_method": "hill_climbing",
        "pg3_hc_enforced_depth": 0,
        "pg3_init_policy": ldl_policy_file.name,
        "pg3_init_base_env": env_name,
    })

    approach = InitializedPG3Approach(env.predicates, env.options, env.types,
                                      env.action_space, train_tasks)
    assert approach.get_name() == "initialized_pg3"

    predicate_map = {p: p for p in env.predicates}
    nsrt_map = {n: n for n in nsrts}
    variable_map = {p: p for n in nsrts for p in n.parameters}
    identity_analogy = _Analogy(predicate_map, nsrt_map, variable_map)

    with patch(f"{_MODULE_PATH}._find_env_analogies") as mocker:
        mocker.return_value = [identity_analogy]
        init_ldls = approach._get_policy_search_initial_ldls()  # pylint: disable=protected-access
        for init_ldl in init_ldls:
            print(init_ldl)
        assert len(init_ldls) == 1
        assert init_ldls[0] == ldl

    # Test loading from file.
    ldl_str = """(define (policy delivery-individual-policy)
	(:rule rule1 
		:parameters (?loc - loc ?paper - paper)
        :preconditions (and (at ?loc) (ishomebase ?loc) (unpacked ?paper))
        :goals ()
		:action (pick-up ?paper ?loc)
	)
	(:rule rule2 
		:parameters (?loc - loc ?paper - paper)
        :preconditions (and (at ?loc) (carrying ?paper) (not (satisfied ?loc)))
        :goals ()
		:action (deliver ?paper ?loc)
	)
	(:rule rule3
		:parameters (?from - loc ?to - loc)
        :preconditions (and (at ?from) (safe ?from) (wantspaper ?to))
        :goals ()
		:action (move ?from ?to)
	)
)
"""
    ldl_policy_txt_file = tempfile.NamedTemporaryFile(suffix=".txt").name
    with open(ldl_policy_txt_file, "w", encoding="utf-8") as f:
        f.write(ldl_str)
        utils.reset_config({
            "env": env_name,
            "approach": "initialized_pg3",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "strips_learner": "oracle",
            "pg3_heuristic": "demo_plan_comparison",  # faster for tests
            "pg3_search_method": "hill_climbing",
            "pg3_hc_enforced_depth": 0,
            "pg3_init_policy": ldl_policy_txt_file,
            "pg3_init_base_env": env_name,
        })
    approach = InitializedPG3Approach(env.predicates, env.options, env.types,
                                      env.action_space, train_tasks)
    with patch(f"{_MODULE_PATH}._find_env_analogies") as mocker:
        mocker.return_value = [identity_analogy]
        init_ldls = approach._get_policy_search_initial_ldls()  # pylint: disable=protected-access
    assert len(init_ldls) == 1
    assert str(init_ldls[0]) == """LiftedDecisionList[
LDLRule-rule1:
    Parameters: [?loc:loc, ?paper:paper]
    Pos State Pre: [at(?loc:loc), ishomebase(?loc:loc), unpacked(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: pick-up(?paper:paper, ?loc:loc)
LDLRule-rule2:
    Parameters: [?loc:loc, ?paper:paper]
    Pos State Pre: [at(?loc:loc), carrying(?paper:paper)]
    Neg State Pre: [satisfied(?loc:loc)]
    Goal Pre: []
    NSRT: deliver(?paper:paper, ?loc:loc)
LDLRule-rule3:
    Parameters: [?from:loc, ?to:loc]
    Pos State Pre: [at(?from:loc), safe(?from:loc), wantspaper(?to:loc)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: move(?from:loc, ?to:loc)
]"""


def test_apply_analogy_to_ldl():
    """Tests for _apply_analogy_to_ldl()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "initialized_pg3",
    })
    env = create_new_env(env_name)
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, env.options)
    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
    pick_up_nsrt = name_to_nsrt["pick-up"]
    pick_up_rule = LDLRule(name="PickUp",
                           parameters=sorted(pick_up_nsrt.parameters),
                           pos_state_preconditions=set(
                               pick_up_nsrt.preconditions),
                           neg_state_preconditions=set(),
                           goal_preconditions=set(),
                           nsrt=pick_up_nsrt)
    ldl = LiftedDecisionList([pick_up_rule])
    nsrt_map = {n: n for n in nsrts}
    variable_map = {p: p for n in nsrts for p in n.parameters}

    # Test that an empty analogy results in an empty LDL.
    analogy = _Analogy(predicates={}, nsrts={}, variables={})
    new_ldl = _apply_analogy_to_ldl(analogy, ldl)
    assert len(new_ldl.rules) == 0

    # Test that an analogy with no predicates results in an LDL with just the
    # NSRT preconditions as positive preconditions.
    analogy = _Analogy(predicates={}, nsrts=nsrt_map, variables=variable_map)
    assert analogy.types == {t: t for t in env.types if t.name != "object"}
    new_ldl = _apply_analogy_to_ldl(analogy, ldl)
    assert str(new_ldl) == """LiftedDecisionList[
LDLRule-PickUp:
    Parameters: [?loc:loc, ?paper:paper]
    Pos State Pre: [at(?loc:loc), ishomebase(?loc:loc), unpacked(?paper:paper)]
    Neg State Pre: []
    Goal Pre: []
    NSRT: pick-up(?paper:paper, ?loc:loc)
]"""
