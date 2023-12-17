"""Test cases for the active sampler learning approach."""
import pytest

from predicators import utils
from predicators.approaches.active_sampler_learning_approach import \
    ActiveSamplerLearningApproach
from predicators.cogman import CogMan
from predicators.datasets import create_dataset
from predicators.envs.cover import BumpyCoverEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.main import _generate_interaction_results
from predicators.perception import create_perceiver
from predicators.settings import CFG
from predicators.structs import Dataset
from predicators.teacher import Teacher


@pytest.mark.parametrize(
    "model_name,right_targets,num_demo,feat_type,object_specific",
    [("myopic_classifier_mlp", False, 0, "all", False),
     ("myopic_classifier_mlp", True, 1, "all", False),
     ("myopic_classifier_ensemble", False, 0, "all", False),
     ("myopic_classifier_ensemble", False, 1, "all", False),
     ("fitted_q", False, 0, "all", False), ("fitted_q", True, 0, "all", False),
     ("myopic_classifier_knn", False, 0, "oracle", False),
     ("myopic_classifier_mlp", False, 0, "all", True)])
def test_active_sampler_learning_approach(model_name, right_targets, num_demo,
                                          feat_type, object_specific):
    """Test for ActiveSamplerLearningApproach class, entire pipeline."""
    utils.reset_config({
        "env":
        "bumpy_cover",
        "approach":
        "active_sampler_learning",
        "active_sampler_learning_model":
        model_name,
        "active_sampler_learning_feature_selection":
        feat_type,
        "timeout":
        10,
        "strips_learner":
        "oracle",
        "sampler_learner":
        "oracle",
        "sampler_disable_classifier":
        True,
        "num_online_learning_cycles":
        1,
        "active_sampler_learning_explore_length_base":
        4,
        "max_num_steps_interaction_request":
        5,
        "online_nsrt_learning_requests_per_cycle":
        1,
        "sampler_mlp_classifier_max_itr":
        10,
        "mlp_regressor_max_itr":
        10,
        "num_train_tasks":
        3,
        "max_initial_demos":
        num_demo,
        "num_test_tasks":
        1,
        "explorer":
        "active_sampler",
        "active_sampler_learning_num_samples":
        5,
        "active_sampler_learning_fitted_q_iters":
        2,
        "active_sampler_learning_num_lookahead_samples":
        2,
        "bumpy_cover_right_targets":
        right_targets,
        "active_sampler_learning_num_ensemble_members":
        2,
        "bilevel_plan_without_sim":
        True,
        "active_sampler_learning_object_specific_samplers":
        object_specific,
    })
    env = BumpyCoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = get_gt_options(env.get_name())
    approach = ActiveSamplerLearningApproach(env.predicates, options,
                                             env.types, env.action_space,
                                             train_tasks)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
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
    # Add some nontrivial ground operator history.
    nsrts = sorted(get_gt_nsrts(env.get_name(), env.predicates, options))
    objs = set(env.get_test_tasks()[0].task.init)
    ground_nsrts = utils.all_ground_nsrts(nsrts[0], objs)
    ground_op_hist = {n.op: [True, False, False] for n in ground_nsrts}
    approach._ground_op_hist = ground_op_hist  # pylint: disable=protected-access
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=CFG.timeout)
        # We won't fully check the policy here because we don't want
        # tests to have to train very good models, since that would
        # be slow. But we will test that the policy at least produces
        # an action.
        action = policy(task.init)
        assert env.action_space.contains(action.arr)
