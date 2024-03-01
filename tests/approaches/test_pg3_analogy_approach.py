"""Test cases for the Initialized PG3 approach."""
import tempfile
from unittest.mock import patch

import dill as pkl
import smepy
from smepy.struct_case import Entity as SmepyEntity

import predicators.approaches.sme_pg3_analogy_approach
from predicators import utils
from predicators.approaches.sme_pg3_analogy_approach import \
    SMEPG3AnalogyApproach, _Analogy, _apply_analogy_to_ldl, \
    _find_env_analogies
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import LDLRule, LiftedAtom, LiftedDecisionList, \
    Variable

_MODULE_PATH = predicators.approaches.sme_pg3_analogy_approach.__name__


def test_pg3_analogy_approach():
    """Tests for SMEPG3AnalogyApproach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "sme_pg3",
    })

    env = create_new_env(env_name)
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    train_tasks = [t.task for t in env.get_train_tasks()]

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
        "approach": "sme_pg3",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "pg3_heuristic": "demo_plan_comparison",  # faster for tests
        "pg3_search_method": "hill_climbing",
        "pg3_hc_enforced_depth": 0,
        "pg3_init_policy": ldl_policy_file.name,
        "pg3_init_base_env": env_name,
    })

    approach = SMEPG3AnalogyApproach(env.predicates,
                                     get_gt_options(env.get_name()), env.types,
                                     env.action_space, train_tasks)
    assert approach.get_name() == "sme_pg3"

    predicate_map = {p: p for p in env.predicates}
    nsrt_map = {n: n for n in nsrts}
    nsrt_var_map = {(n, p): (n, p) for n in nsrts for p in n.parameters}
    identity_analogy = _Analogy(predicate_map, nsrt_map, nsrt_var_map)

    with patch(f"{_MODULE_PATH}._find_env_analogies") as mocker:
        mocker.return_value = [identity_analogy]
        init_ldls = approach._get_policy_search_initial_ldls()  # pylint: disable=protected-access
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
            "approach": "sme_pg3",
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "strips_learner": "oracle",
            "pg3_heuristic": "demo_plan_comparison",  # faster for tests
            "pg3_search_method": "hill_climbing",
            "pg3_hc_enforced_depth": 0,
            "pg3_init_policy": ldl_policy_txt_file,
            "pg3_init_base_env": env_name,
        })
    approach = SMEPG3AnalogyApproach(env.predicates,
                                     get_gt_options(env.get_name()), env.types,
                                     env.action_space, train_tasks)
    with patch(f"{_MODULE_PATH}._find_env_analogies") as mocker:
        mocker.return_value = [identity_analogy]
        init_ldls = approach._get_policy_search_initial_ldls()  # pylint: disable=protected-access
    assert len(init_ldls) == 1
    assert str(init_ldls[0]) == """(define (policy)
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
)"""


def _disabled_test_find_env_analogies():  # pragma: no cover
    """Tests for _find_env_analogies().

    NOTE: this test is currently disabled because of sudden flakiness in the
    SME depedency, despite no changes for months. Since we're not actively
    using this code, we're just disabling it, but leaving it here in case we
    do want to resurrect the code in the future.
    """
    # Test for gripper -> ferry.
    base_env = create_new_env("pddl_gripper_procedural_tasks")
    base_nsrts = get_gt_nsrts(base_env.get_name(), base_env.predicates,
                              get_gt_options(base_env.get_name()))
    target_env = create_new_env("pddl_ferry_procedural_tasks")
    target_nsrts = get_gt_nsrts(target_env.get_name(), target_env.predicates,
                                get_gt_options(target_env.get_name()))

    # Mock SME because it's potentially slow.
    mock_match_strs = [
        # Operators
        ("move", "sail"),
        ("pick", "board"),
        ("drop", "debark"),
        # Variables
        ("move-to", "sail-to"),
        ("move-from", "sail-from"),
        ("pick-obj", "board-car"),
        ("pick-room", "board-loc"),
        ("drop-obj", "debark-car"),
        ("drop-room", "debark-loc"),
        # Include a bogus variable mapping that should get ignored because the
        # NSRTS are different.
        ("move-to", "board-car"),
        # Predicates
        ("ball", "car"),
        ("room", "location"),
        ("at", "at"),
        ("at-robby", "at-ferry"),
    ]
    mock_sme_output = smepy.Mapping(matches=[
        smepy.Match(SmepyEntity(b), SmepyEntity(t)) for b, t in mock_match_strs
    ])
    with patch(f"{_MODULE_PATH}._query_sme") as mocker:
        mocker.return_value = [mock_sme_output]
        analogies = _find_env_analogies(base_env, target_env, base_nsrts,
                                        target_nsrts)

    assert len(analogies) == 1
    analogy = analogies[0]

    # Verify NSRT matches.
    base_nsrt_name_to_nsrt = {n.name: n for n in base_nsrts}
    target_nsrt_name_to_nsrt = {n.name: n for n in target_nsrts}
    move = base_nsrt_name_to_nsrt["move"]
    pick = base_nsrt_name_to_nsrt["pick"]
    drop = base_nsrt_name_to_nsrt["drop"]
    sail = target_nsrt_name_to_nsrt["sail"]
    board = target_nsrt_name_to_nsrt["board"]
    debark = target_nsrt_name_to_nsrt["debark"]
    assert analogy.nsrts == {move: sail, pick: board, drop: debark}

    # Verify predicate matches.
    base_pred_name_to_pred = {n.name: n for n in base_env.predicates}
    target_pred_name_to_pred = {n.name: n for n in target_env.predicates}
    ball = base_pred_name_to_pred["ball"]
    room = base_pred_name_to_pred["room"]
    at_base = base_pred_name_to_pred["at"]
    at_robby = base_pred_name_to_pred["at-robby"]
    car = target_pred_name_to_pred["car"]
    location = target_pred_name_to_pred["location"]
    at_target = target_pred_name_to_pred["at"]
    at_ferry = target_pred_name_to_pred["at-ferry"]
    assert analogy.predicates == {
        ball: car,
        room: location,
        at_base: at_target,
        at_robby: at_ferry
    }

    # Test base_nsrt_to_variable_analogy.
    old_var_to_new_var = analogy.base_nsrt_to_variable_analogy(move)
    assert len(old_var_to_new_var) == 1  # exclude the bogus entry


def test_apply_analogy_to_ldl():
    """Tests for _apply_analogy_to_ldl()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "sme_pg3",
    })
    env = create_new_env(env_name)
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
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
    nsrt_var_map = {(n, p): (n, p) for n in nsrts for p in n.parameters}

    # Test that an empty analogy results in an empty LDL.
    analogy = _Analogy(predicates={}, nsrts={}, nsrt_variables={})
    new_ldl = _apply_analogy_to_ldl(analogy, ldl)
    assert len(new_ldl.rules) == 0

    # Test that an analogy with no predicates results in an LDL with just the
    # NSRT preconditions as positive preconditions.
    analogy = _Analogy(predicates={},
                       nsrts=nsrt_map,
                       nsrt_variables=nsrt_var_map)
    assert analogy.types == {t: t for t in env.types if t.name != "object"}
    new_ldl = _apply_analogy_to_ldl(analogy, ldl)
    assert str(new_ldl) == """(define (policy)
  (:rule PickUp
    :parameters (?loc - loc ?paper - paper)
    :preconditions (and (at ?loc) (ishomebase ?loc) (unpacked ?paper))
    :goals ()
    :action (pick-up ?paper ?loc)
  )
)"""

    # Test case where there is a variable in the LDL rule that doesn't appear
    # in the NSRT.
    predicate_map = {p: p for p in env.predicates}
    nsrt_map = {n: n for n in nsrts}
    nsrt_var_map = {(n, p): (n, p) for n in nsrts for p in n.parameters}
    identity_analogy = _Analogy(predicate_map, nsrt_map, nsrt_var_map)

    type_name_to_type = {t.name: t for t in env.types}
    pred_name_to_pred = {p.name: p for p in env.predicates}
    new_var = Variable("?extra", type_name_to_type["paper"])
    unpacked = pred_name_to_pred["unpacked"]
    params = [new_var] + sorted(pick_up_nsrt.parameters)
    pick_up_extra_param_rule = LDLRule(
        name="PickUp",
        parameters=params,
        pos_state_preconditions=set(pick_up_nsrt.preconditions),
        neg_state_preconditions={LiftedAtom(unpacked, [new_var])},
        goal_preconditions=set(),
        nsrt=pick_up_nsrt)
    ldl_extra_param = LiftedDecisionList([pick_up_extra_param_rule])
    new_ldl = _apply_analogy_to_ldl(identity_analogy, ldl_extra_param)
    assert str(new_ldl) == """(define (policy)
  (:rule PickUp
    :parameters (?extra - paper ?loc - loc ?paper - paper)
    :preconditions (and (at ?loc) (ishomebase ?loc) (unpacked ?paper) (not (unpacked ?extra)))
    :goals ()
    :action (pick-up ?paper ?loc)
  )
)"""
