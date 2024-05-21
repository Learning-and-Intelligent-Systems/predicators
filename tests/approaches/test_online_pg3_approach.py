"""Test cases for the online PG3 approach."""
import pytest

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.online_pg3_approach import OnlinePG3Approach
from predicators.cogman import CogMan
from predicators.datasets import create_dataset
from predicators.envs.cover import CoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.main import _generate_interaction_results
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Dataset
from predicators.teacher import Teacher


def test_online_pg3_approach():
    """Test for OnlinePG3Approach class, entire pipeline."""
    utils.reset_config({
        "env": "cover",
        "approach": "online_pg3",
        "timeout": 10,
        "sampler_mlp_classifier_max_itr": 10,
        "neural_gaus_regressor_max_itr": 10,
        "num_online_learning_cycles": 1,
        "online_nsrt_learning_requests_per_cycle": 1,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "explorer": "random_options",
        "pg3_heuristic": "demo_plan_comparison",  # faster for tests
        "pg3_search_method": "gbfs",
        "pg3_gbfs_max_expansions": 1
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = OnlinePG3Approach(env.predicates,
                                 get_gt_options(env.get_name()), env.types,
                                 env.action_space, train_tasks)
    predicates, _ = utils.parse_config_excluded_predicates(env)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()),
                             predicates)
    assert approach.is_learning_based
    # Learning with an empty dataset should not crash.
    approach.learn_from_offline_dataset(Dataset([]))
    # Learning with the actual dataset.
    approach.learn_from_offline_dataset(dataset)
    approach.load(online_learning_cycle=None)
    interaction_requests = approach.get_interaction_requests()
    teacher = Teacher(train_tasks)
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("trivial")
    cogman = CogMan(approach, perceiver, exec_monitor)
    interaction_results, _ = _generate_interaction_results(
        cogman, env, teacher, interaction_requests)
    approach.learn_from_interaction_results(interaction_results)
    approach.load(online_learning_cycle=0)
    with pytest.raises(FileNotFoundError):
        approach.load(online_learning_cycle=1)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            pass
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.
