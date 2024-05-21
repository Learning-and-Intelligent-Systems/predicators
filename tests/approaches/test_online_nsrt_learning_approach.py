"""Test cases for the online NSRT learning approach."""
import pytest

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.cogman import CogMan
from predicators.datasets import create_dataset
from predicators.envs.cover import CoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_options
from predicators.main import _generate_interaction_results
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Dataset


def test_online_nsrt_learning_approach():
    """Test for OnlineNSRTLearningApproach class, entire pipeline."""
    utils.reset_config({
        "env": "cover",
        "approach": "online_nsrt_learning",
        "timeout": 10,
        "sampler_mlp_classifier_max_itr": 10,
        "predicate_mlp_classifier_max_itr": 10,
        "neural_gaus_regressor_max_itr": 10,
        "num_online_learning_cycles": 1,
        "online_nsrt_learning_requests_per_cycle": 1,
        "num_train_tasks": 3,
        "num_test_tasks": 3,
        "explorer": "random_options",
        "online_learning_max_novelty_count": float("inf"),
        "make_interaction_videos": True,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = OnlineNSRTLearningApproach(env.predicates,
                                          get_gt_options(env.get_name()),
                                          env.types, env.action_space,
                                          train_tasks)
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
    teacher = None
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
    # Tests for novelty scoring.
    pred_name_to_pred = {p.name: p for p in env.predicates}
    type_name_to_type = {t.name: t for t in env.types}
    Covers = pred_name_to_pred["Covers"]
    IsBlock = pred_name_to_pred["IsBlock"]
    IsTarget = pred_name_to_pred["IsTarget"]
    block_type = type_name_to_type["block"]
    target_type = type_name_to_type["target"]
    # Covers should appear more interesting than the IsBlock and IsTarget.
    task = train_tasks[0]
    block = task.init.get_objects(block_type)[0]
    target = task.init.get_objects(target_type)[0]
    covers = {Covers([block, target])}
    is_block = {IsBlock([block])}
    is_target = {IsTarget([target])}
    covers_score = approach._score_atoms_novelty(covers)  # pylint: disable=protected-access
    is_block_score = approach._score_atoms_novelty(is_block)  # pylint: disable=protected-access
    is_target_score = approach._score_atoms_novelty(is_target)  # pylint: disable=protected-access
    assert covers_score > is_block_score
    assert covers_score > is_target_score
    # Scores should now be -inf.
    utils.update_config({
        "online_learning_max_novelty_count": 0,
    })
    covers_score = approach._score_atoms_novelty(covers)  # pylint: disable=protected-access
    is_block_score = approach._score_atoms_novelty(is_block)  # pylint: disable=protected-access
    is_target_score = approach._score_atoms_novelty(is_target)  # pylint: disable=protected-access
    assert covers_score == is_block_score == is_target_score == -float("inf")
