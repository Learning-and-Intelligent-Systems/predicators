"""Test cases for the interactive learning approach."""

from predicators.src.approaches import InteractiveLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv
from predicators.src.settings import CFG
from predicators.src import utils


def test_interactive_learning_approach():
    """Test for InteractiveLearningApproach class, entire pipeline."""
    utils.update_config({"env": "cover"})
    utils.update_config({
        "approach": "interactive_learning",
        "offline_data_method": "demo+ground_atoms",
        "excluded_predicates": "IsBlock,IsTarget",
        "timeout": 10,
        "max_samples_per_step": 10,
        "seed": 123,
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
        for p in env.predicates if p.name not in ["IsBlock", "IsTarget"]
    }
    approach = InteractiveLearningApproach(initial_predicates, env.options,
                                           env.types, env.action_space,
                                           train_tasks)
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            pass
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.
