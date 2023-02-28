"""Test cases for the PG4 approach."""
import pytest

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.pg4_approach import PG4Approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import LDLRule, LiftedAtom, LiftedDecisionList, Task


def test_pg4_approach():
    """Tests for PG4Approach().

    Additional tests are in test_pg3_approach().
    """
    env_name = "cover"
    utils.reset_config({
        "env": env_name,
        "approach": "pg4",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "cover_initial_holding_prob": 1.0,
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    approach = PG4Approach(env.predicates, get_gt_options(env.get_name()),
                           env.types, env.action_space, train_tasks)
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    approach._nsrts = nsrts  # pylint: disable=protected-access
    task = train_tasks[0]

    nsrt_name_to_nsrt = {n.name: n for n in nsrts}
    pick_nsrt = nsrt_name_to_nsrt["Pick"]
    place_nsrt = nsrt_name_to_nsrt["Place"]
    block, target = place_nsrt.parameters
    pred_name_to_pred = {p.name: p for p in env.predicates}
    Covers = pred_name_to_pred["Covers"]
    HandEmpty = pred_name_to_pred["HandEmpty"]
    Holding = pred_name_to_pred["Holding"]
    IsBlock = pred_name_to_pred["IsBlock"]
    IsTarget = pred_name_to_pred["IsTarget"]

    # Test successful planning.
    pick_rule = LDLRule("Pick", [block, target],
                        {LiftedAtom(HandEmpty, []),
                         IsBlock([block])}, {Covers([block, target])},
                        {Covers([block, target])}, pick_nsrt)
    place_rule = LDLRule(
        "Place", [block, target],
        {Holding([block]),
         IsBlock([block]),
         IsTarget([target])}, set(), {Covers([block, target])}, place_nsrt)

    ldl = LiftedDecisionList([pick_rule, place_rule])
    approach._current_ldl = ldl  # pylint: disable=protected-access
    approach.solve(task, 500)

    # Test option execution failure.
    trivial_task = Task(task.init, set())
    policy = approach.solve(trivial_task, 500)
    with pytest.raises(ApproachFailure) as e:
        policy(task.init)
    assert "Option plan exhausted!" in str(e)

    # Test planning timeout.
    with pytest.raises(ApproachTimeout):
        approach.solve(task, -1)

    # Test planning failure.
    approach._nsrts = set()  # pylint: disable=protected-access
    with pytest.raises(ApproachFailure):
        approach.solve(task, 1)
