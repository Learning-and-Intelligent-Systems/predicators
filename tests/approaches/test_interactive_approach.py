"""Test cases for the interactive learning approach."""

from predicators.src.approaches import InteractiveLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv
from predicators.src.settings import CFG
from predicators.src.main import _generate_interaction_results
from predicators.src.teacher import Teacher
from predicators.src import utils


def test_interactive_learning_approach():
    """Test for InteractiveLearningApproach class, entire pipeline."""
    utils.reset_config({
        "env": "cover",
        "approach": "interactive_learning",
        "offline_data_method": "demo+ground_atoms",
        "excluded_predicates": "IsBlock,Covers",
        "timeout": 10,
        "sampler_mlp_classifier_max_itr": 200,
        "predicate_mlp_classifier_max_itr": 200,
        "neural_gaus_regressor_max_itr": 200,
        "num_online_learning_cycles": 1,
        "num_train_tasks": 5,
        "num_test_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    initial_predicates = {
        p
        for p in env.predicates if p.name not in ["IsBlock", "Covers"]
    }
    approach = InteractiveLearningApproach(initial_predicates, env.options,
                                           env.types, env.action_space,
                                           train_tasks)
    teacher = Teacher()
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    interaction_requests = approach.get_interaction_requests()
    interaction_results = _generate_interaction_results(
        env.simulate, teacher, train_tasks, interaction_requests)
    approach.learn_from_interaction_results(interaction_results)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            pass
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.
