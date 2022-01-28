"""Test cases for dataset generation."""

import pytest
from predicators.src.datasets import create_dataset
from predicators.src.envs import CoverEnv, ClutteredTableEnv
from predicators.src.structs import Dataset
from predicators.src import utils
from predicators.src.ground_truth_nsrts import _get_predicates_by_names
from predicators.src.settings import CFG


def test_demo_dataset():
    """Test demo-only dataset creation with Covers env."""
    # Test that data does not contain options since
    # option_learner is not "no_learning"
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
    })
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "option_learner": "arbitrary_dummy",
        "seed": 123,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 7
    assert len(dataset.trajectories[0].states) == 3
    assert len(dataset.trajectories[0].actions) == 2
    for traj in dataset.trajectories:
        assert traj.is_demo
        for action in traj.actions:
            assert not action.has_option()
    # Test that data contains options since option_learner is "no_learning"
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "option_learner": "no_learning",
        "seed": 123,
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 7
    assert len(dataset.trajectories[0].states) == 3
    assert len(dataset.trajectories[0].actions) == 2
    for traj in dataset.trajectories:
        assert traj.is_demo
        for action in traj.actions:
            assert action.has_option()
    with pytest.raises(AssertionError):
        _ = dataset.annotations
    utils.update_config({
        "offline_data_method": "not a real method",
    })
    with pytest.raises(NotImplementedError):
        create_dataset(env, train_tasks)


def test_demo_replay_dataset():
    """Test demo+replay dataset creation with Covers env."""
    # Test that data contains options since
    # option_learner is "no_learning"
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "no_learning",
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 5 + 3
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1
    num_demos = 0
    for traj in dataset.trajectories:
        num_demos += int(traj.is_demo)
        for action in traj.actions:
            assert action.has_option()
    assert num_demos == 5
    # Test that data does not contain options since
    # option_learner is not "no_learning"
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "arbitrary_dummy",
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 5 + 3
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1
    num_demos = 0
    for traj in dataset.trajectories:
        num_demos += int(traj.is_demo)
        for action in traj.actions:
            assert not action.has_option()
    assert num_demos == 5
    # Test cluttered table data collection
    utils.update_config({
        "env": "cluttered_table",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 10,
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = ClutteredTableEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1


def test_demo_nonoptimal_replay_dataset():
    """Test demo+nonoptimalreplay dataset creation with Covers env."""
    # Note that the planning timeout is intentionally set low enough that we
    # cover some failures to plan from the replays, but not so low that
    # planning always fails. Also the number of replays is set high enough
    # that we consistently cover the failure case.
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+nonoptimalreplay",
        "offline_data_planning_timeout": 1e-1,
        "offline_data_num_replays": 50,
        "option_learner": "no_learning",
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 5 + 50
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1


def test_dataset_with_annotations():
    """Test the creation of a Dataset with annotations."""
    utils.update_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "no_learning",
        "seed": 123,
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    trajectories = create_dataset(env, train_tasks).trajectories
    # The annotations and trajectories need to be the same length.
    with pytest.raises(AssertionError):
        dataset = Dataset(trajectories, [])
    annotations = ["label" for _ in trajectories]
    dataset = Dataset(trajectories, annotations)
    assert dataset.annotations == annotations
    # Can't add a data point without an annotation.
    with pytest.raises(AssertionError):
        dataset.append(trajectories)
    dataset.append(trajectories[0], annotations[0])
    assert len(dataset.trajectories) == len(dataset.annotations) == \
        len(trajectories) + 1


def test_ground_atom_dataset():
    """Test creation of demo+ground_atoms dataset."""
    utils.update_config({"env": "cover"})
    utils.update_config({
        "approach":
        "interactive_learning",
        "num_train_tasks":
        15,
        "offline_data_method":
        "demo+ground_atoms",
        "teacher_dataset_label_ratio":
        0.5,
        "interactive_known_predicates":
        "HandEmpty,IsBlock,IsTarget",
    })
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks)
    assert len(dataset.trajectories) == 15
    assert len(dataset.annotations) == 15
    Covers, HandEmpty, Holding = _get_predicates_by_names(
        "cover", ["Covers", "HandEmpty", "Holding"])
    all_predicates = {Covers, HandEmpty, Holding}
    # Test that the approximately correct ratio of atoms are annotated.
    pred_name_to_total = {p.name: 0 for p in all_predicates}
    pred_name_to_labels = {p.name: 0 for p in all_predicates}
    for traj, ground_atom_seq in zip(dataset.trajectories,
                                     dataset.annotations):
        assert len(traj.states) == len(ground_atom_seq)
        for ground_atoms, s in zip(ground_atom_seq, traj.states):
            all_ground_atoms = utils.abstract(s, all_predicates)
            all_ground_atom_names = set()
            for ground_truth_atom in all_ground_atoms:
                pred_name_to_total[ground_truth_atom.predicate.name] += 1
                all_ground_atom_names.add((ground_truth_atom.predicate.name,
                                           tuple(ground_truth_atom.objects)))
            for annotated_atom in ground_atoms:
                pred_name_to_labels[annotated_atom.predicate.name] += 1
                # Make sure the annotations are correct.
                annotated_atom_name = (annotated_atom.predicate.name,
                                       tuple(annotated_atom.objects))
                assert annotated_atom_name in all_ground_atom_names
                # Make sure we're not leaking information.
                assert not annotated_atom.holds(s)
    # HandEmpty was excluded.
    assert pred_name_to_labels["HandEmpty"] == 0
    assert pred_name_to_total["HandEmpty"] > 0
    # Holding and Covers were included.
    target_ratio = CFG.teacher_dataset_label_ratio
    for name in ["Holding", "Covers"]:
        ratio = pred_name_to_labels[name] / pred_name_to_total[name]
        assert abs(target_ratio - ratio) < 0.05
