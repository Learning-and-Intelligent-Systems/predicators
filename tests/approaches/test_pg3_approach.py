"""Test cases for the PG3 approach."""
import os
import tempfile

import pytest

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.pg3_approach import PG3Approach
from predicators.approaches.pg4_approach import PG4Approach
from predicators.datasets import create_dataset
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.option_model import _OptionModelBase
from predicators.structs import LDLRule, LiftedDecisionList


class _MockOptionModel(_OptionModelBase):

    def __init__(self, simulator):
        self._simulator = simulator

    def get_next_state_and_num_actions(self, state, option):
        return state.copy(), 0


@pytest.mark.parametrize("approach_name,approach_cls", [("pg3", PG3Approach),
                                                        ("pg4", PG4Approach)])
def test_pg3_approach(approach_name, approach_cls):
    """Tests for PG3Approach() and PG4Approach()."""
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": approach_name,
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
        "pg3_heuristic": "demo_plan_comparison",  # faster for tests
        "pg3_search_method": "hill_climbing",
        "pg3_hc_enforced_depth": 0,
    })
    env = create_new_env(env_name)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = approach_cls(env.predicates, get_gt_options(env.get_name()),
                            env.types, env.action_space, train_tasks)
    assert approach.get_name() == approach_name
    nsrts = get_gt_nsrts(env.get_name(), env.predicates,
                         get_gt_options(env.get_name()))
    name_to_nsrt = {nsrt.name: nsrt for nsrt in nsrts}
    approach._nsrts = nsrts  # pylint: disable=protected-access

    # Test prediction with a good policy.
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

    deliver_rule = LDLRule(name="Deliver",
                           parameters=[loc, paper],
                           pos_state_preconditions=set(
                               deliver_nsrt.preconditions),
                           neg_state_preconditions={satisfied([loc])},
                           goal_preconditions=set(),
                           nsrt=deliver_nsrt)

    from_loc, to_loc = move_nsrt.parameters
    assert "from" in str(from_loc)
    assert "to" in str(to_loc)

    move_rule = LDLRule(
        name="Move",
        parameters=[from_loc, to_loc],
        pos_state_preconditions=set(move_nsrt.preconditions) | \
                                {wantspaper([to_loc])},
        neg_state_preconditions=set(),
        goal_preconditions=set(),
        nsrt=move_nsrt
    )

    ldl = LiftedDecisionList([pick_up_rule, deliver_rule, move_rule])
    approach._current_ldl = ldl  # pylint: disable=protected-access
    task = train_tasks[0]
    policy = approach.solve(task, timeout=500)
    act = policy(task.init)
    option = act.get_option()
    assert option.name == "pick-up"

    # Test loading from a saved init policy file.
    if approach_name == "pg3":
        pg3_init_file = tempfile.NamedTemporaryFile().name
        with open(pg3_init_file, "w", encoding="utf-8") as f:
            # Write the good policy to a file.
            f.write(str(ldl))
        # Create a new approach.
        utils.reset_config({
            "env": env_name,
            "approach": approach_name,
            "num_train_tasks": 1,
            "num_test_tasks": 1,
            "strips_learner": "oracle",
            "pg3_heuristic": "demo_plan_comparison",  # faster for tests
            "pg3_search_method": "hill_climbing",
            "pg3_hc_enforced_depth": 0,
            "pg3_init_policy": pg3_init_file,
        })
        new_approach = approach_cls(env.predicates,
                                    get_gt_options(env.get_name()), env.types,
                                    env.action_space, train_tasks)
        new_approach._nsrts = nsrts  # pylint: disable=protected-access
        recovered_ldls = new_approach._get_policy_search_initial_ldls()  # pylint: disable=protected-access
        assert len(recovered_ldls) == 1
        recovered_ldl = recovered_ldls[0]
        assert len(recovered_ldl.rules) == len(ldl.rules)

    # Test case where low-level search fails in PG3.
    if approach_name == "pg3":
        approach._option_model = _MockOptionModel(env.simulate)  # pylint: disable=protected-access
        with pytest.raises(ApproachFailure) as e:
            approach.solve(task, timeout=500)
        assert "Low-level search failed" in str(e)
    ldl = LiftedDecisionList([])
    approach._current_ldl = ldl  # pylint: disable=protected-access
    # PG3 alone fails.
    if approach_name == "pg3":
        with pytest.raises(ApproachFailure) as e:
            approach.solve(task, timeout=500)
        assert "PG3 policy was not applicable!" in str(e)
        # Test case where PG3 times out during planning.
        with pytest.raises(ApproachFailure) as e:
            approach.solve(task, timeout=-1)
        assert "Timeout exceeded" in str(e)
    # PG4 falls back to sesame, so succeeds.
    else:
        assert approach_name == "pg4"
        policy = approach.solve(task, timeout=500)
        act = policy(task.init)
        option = act.get_option()
        assert option.name == "pick-up"
    # Test learning with a fast heuristic.
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
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


def test_cluttered_table_pg3_approach():
    """Tests for PG3Approach() in cluttered_table."""
    # Learning is very fast, so we can run the full pipeline. This also covers
    # an important case where a discovered failure is raised during low-level
    # search.
    env_name = "cluttered_table"
    utils.reset_config({
        "env": env_name,
        "approach": "pg3",
        # Needs to be large to see example of dumping.
        "num_train_tasks": 20,
        "num_test_tasks": 10,
        "strips_learner": "oracle",
        "sampler_learner": "oracle",
    })
    env = create_new_env(env_name)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = PG3Approach(env.predicates, get_gt_options(env.get_name()),
                           env.types, env.action_space, train_tasks)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             env.predicates)
    approach.learn_from_offline_dataset(dataset)
    # Test several tasks to make sure we encounter at least one discovered
    # failure.
    for task in env.get_test_tasks():
        # Should not crash, but may not solve the task.
        try:
            policy = approach.solve(task, timeout=500)
            act = policy(task.init)
            assert act is not None
        except ApproachFailure as e:
            assert "Discovered a failure" in str(e)
