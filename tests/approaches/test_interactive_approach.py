"""Test cases for the interactive learning approach.
"""

import pytest
from predicators.src.approaches import InteractiveLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.approaches.interactive_learning_approach import \
    create_teacher_dataset
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv
from predicators.src.settings import CFG
from predicators.src import utils


def test_teacher_dataset():
    """Test teacher dataset creation with Covers env.
    """
    # Test that data does not contain options since approach is random
    utils.update_config({
        "env": "cover",
        "approach": "random",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "seed": 123,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    dataset = create_dataset(env)
    teacher_dataset = create_teacher_dataset(env.predicates, dataset)
    assert len(teacher_dataset) == 7
    for _, actions in dataset:
        for action in actions:
            assert not action.has_option()

    # Test the first trajectory for correct usage of ratio
    # Generate groundatoms
    (ss, _) = dataset[0]
    ground_atoms_traj = []
    for s in ss:
        ground_atoms = list(utils.abstract(s, env.predicates))
        ground_atoms_traj.append(ground_atoms)
    # Check that numbers of groundatoms are as expected
    lengths = [len(e) for e in ground_atoms_traj]
    teacher_lengths = [len(e) for e in teacher_dataset[0]]
    assert len(lengths) == len(teacher_lengths)
    ratio = CFG.teacher_dataset_label_ratio
    for i in range(len(lengths)):
        assert teacher_lengths[i] == int(ratio * lengths[i])


def test_interactive_learning_approach():
    """Tests for InteractiveLearningApproach class.
    """
    utils.update_config({"env": "cover", "approach": "interactive_learning",
                         "timeout": 10, "max_samples_per_step": 10,
                         "seed": 12345, "classifier_max_itr": 50,
                         "regressor_max_itr": 50})
    env = CoverEnv()
    approach = InteractiveLearningApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    dataset = create_dataset(env)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    for task in env.get_test_tasks():
        try:
            approach.solve(task, timeout=CFG.timeout)
        except (ApproachTimeout, ApproachFailure):  # pragma: no cover
            pass
        # We won't check the policy here because we don't want unit tests to
        # have to train very good models, since that would be slow.

    # Test teacher
    (ss, _) = dataset[0]
    for s in ss:
        ground_atoms = sorted(utils.abstract(s, env.predicates))
        for g in ground_atoms:
            assert approach.ask_teacher(s, g)


def test_interactive_learning_approach_exception():
    """Test for failure when teacher dataset contains set of 0 ground atoms.
    """
    utils.update_config({"env": "cover", "approach": "interactive_learning",
                         "timeout": 10, "max_samples_per_step": 10,
                         "seed": 12345, "classifier_max_itr": 50,
                         "regressor_max_itr": 50,
                         "teacher_dataset_label_ratio": 0})
    env = CoverEnv()
    approach = InteractiveLearningApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    dataset = create_dataset(env)
    assert approach.is_learning_based
    with pytest.raises(ApproachFailure):
        approach.learn_from_offline_dataset(dataset)
