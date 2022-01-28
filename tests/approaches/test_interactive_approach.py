"""Test cases for the interactive learning approach."""

import pytest
from predicators.src.approaches import InteractiveLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv
from predicators.src.settings import CFG
from predicators.src.structs import State, GroundAtom
from predicators.src import utils


class _DummyInteractiveLearningApproach(InteractiveLearningApproach):
    """An approach that learns predicates from a teacher."""

    def ask_teacher(self, state: State, ground_atom: GroundAtom) -> bool:
        """Returns whether the ground atom is true in the state."""
        return super()._ask_teacher(state, ground_atom)


def test_interactive_learning_approach():
    """Test for InteractiveLearningApproach class, entire pipeline."""
    utils.update_config({"env": "cover"})
    utils.update_config({
        "approach": "interactive_learning",
        "excluded_predicates": "",
        "timeout": 10,
        "max_samples_per_step": 10,
        "seed": 123,
        "sampler_mlp_classifier_max_itr": 500,
        "predicate_mlp_classifier_max_itr": 500,
        "neural_gaus_regressor_max_itr": 500,
        "interactive_num_episodes": 1,
        "interactive_relearn_every": 1,
        "interactive_known_predicates": "HandEmpty,Holding,Covers",
        "offline_data_method": "demo+ground_atoms",
        "num_train_tasks": 5,
        "num_test_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    approach = _DummyInteractiveLearningApproach(env.predicates, env.options,
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

    # Test teacher
    traj = dataset.trajectories[0]
    for s in traj.states:
        ground_atoms = sorted(utils.abstract(s, env.predicates))
        for g in ground_atoms:
            assert approach.ask_teacher(s, g)


def test_interactive_learning_approach_no_ground_atoms():
    """Test for InteractiveLearningApproach class where the dataset contains an
    empty ground atom set."""
    utils.update_config({"env": "cover"})
    utils.update_config({
        "env": "cover",
        "approach": "interactive_learning",
        "timeout": 10,
        "max_samples_per_step": 10,
        "seed": 123,
        "interactive_num_episodes": 0,
        "teacher_dataset_label_ratio": 0.0,
        "interactive_known_predicates": "HandEmpty,IsBlock,IsTarget,Holding",
        "num_train_tasks": 5,
        "num_test_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    approach = _DummyInteractiveLearningApproach(env.predicates, env.options,
                                                 env.types, env.action_space,
                                                 train_tasks)
    dataset = create_dataset(env, train_tasks)
    assert approach.is_learning_based
    # MLP training fails since there are 0 positive examples
    with pytest.raises(RuntimeError):
        approach.learn_from_offline_dataset(dataset)
