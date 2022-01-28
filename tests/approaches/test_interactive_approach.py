"""Test cases for the interactive learning approach."""

import pytest
import numpy as np
from predicators.src.approaches import InteractiveLearningApproach, \
    ApproachTimeout, ApproachFailure
from predicators.src.approaches.interactive_learning_approach import \
    create_teacher_dataset
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


def test_create_teacher_dataset():
    """Test teacher dataset creation with Covers env."""
    # Test that data does not contain options since approach is random
    utils.update_config({"env": "cover"})
    utils.update_config({
        "approach": "interactive_learning",
        "seed": 123,
        "num_train_tasks": 15,
        "num_test_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    teacher_dataset = create_teacher_dataset(env.predicates, dataset)
    assert len(teacher_dataset) == 15

    # Test the first trajectory for correct usage of ratio
    # Generate and count groundatoms
    traj = dataset.trajectories[0]
    ground_atoms_traj = []
    totals = {p: 0 for p in env.predicates}
    for s in traj.states:
        ground_atoms = list(utils.abstract(s, env.predicates))
        for ga in ground_atoms:
            totals[ga.predicate] += 1
        ground_atoms_traj.append(ground_atoms)
    lengths = [len(elt) for elt in ground_atoms_traj]
    # Check that numbers of groundatoms are as expected
    _, traj = teacher_dataset[0]
    labeleds = {p: 0 for p in env.predicates}
    for ground_atom_set in traj:
        for ga in ground_atom_set:
            labeleds[ga.predicate] += 1
    teacher_lengths = [len(elt) for elt in traj]
    assert len(lengths) == len(teacher_lengths)
    # Check overall ratio
    ratio = CFG.teacher_dataset_label_ratio
    for i in range(len(lengths)):
        assert np.allclose(ratio, teacher_lengths[i] / lengths[i])
    # Check ratios for each predicate
    for p in env.predicates:
        assert np.allclose(ratio, labeleds[p] / totals[p])


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
