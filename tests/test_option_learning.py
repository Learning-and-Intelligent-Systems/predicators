"""Tests for option learning."""

import pytest
import numpy as np
from predicators.src.envs import create_env
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.nsrt_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src.option_learning import create_option_learner
from predicators.src import utils


def test_known_options_option_learner():
    """Tests for _KnownOptionsOptionLearner."""
    utils.update_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "seed": 123
    })
    utils.update_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "seed": 123,
        "num_train_tasks": 3,
        "option_learner": "no_learning"
    })
    env = create_env("cover")
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert act.has_option()
    segments = [
        seg for traj in ground_atom_dataset for seg in segment_trajectory(traj)
    ]
    pnads = learn_strips_operators(segments)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 4
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
    assert len(option_specs) == len(strips_ops) == 4
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []) for _ in range(4)]
    for datastore, spec in zip(datastores, option_specs):
        for (segment, _) in datastore:
            assert segment.has_option()
            option = segment.get_option()
            # This call should be a no-op when options are known.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            assert segment.get_option() == option
    # Reset configuration.
    utils.update_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "seed": 123,
        "option_learner": "no_learning"
    })


def test_oracle_option_learner_cover():
    """Tests for _OracleOptionLearner for the cover environment."""
    utils.update_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "seed": 123
    })
    utils.update_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "seed": 123,
        "num_train_tasks": 3,
        "option_learner": "oracle"
    })
    env = create_env("cover")
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert not act.has_option()
    segments = [
        seg for traj in ground_atom_dataset for seg in segment_trajectory(traj)
    ]
    pnads = learn_strips_operators(segments)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 3
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
    assert len(option_specs) == len(strips_ops) == 3
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []), (PickPlace, []), (PickPlace, [])]
    for datastore, spec in zip(datastores, option_specs):
        for (segment, _) in datastore:
            assert not segment.has_option()
            # This call should add an option to the segment.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            option = segment.get_option()
            # In cover env, param == action array.
            assert option.parent == PickPlace
            assert np.allclose(option.params, segment.actions[0].arr)
    # Reset configuration.
    utils.update_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "seed": 123,
        "option_learner": "no_learning"
    })


def test_oracle_option_learner_blocks():
    """Tests for _OracleOptionLearner for the blocks environment."""
    utils.update_config({
        "env": "blocks",
        "approach": "nsrt_learning",
        "seed": 123
    })
    utils.update_config({
        "env": "blocks",
        "approach": "nsrt_learning",
        "seed": 123,
        "num_train_tasks": 3,
        "option_learner": "oracle"
    })
    env = create_env("blocks")
    train_tasks = next(env.train_tasks_generator())
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert not act.has_option()
    segments = [
        seg for traj in ground_atom_dataset for seg in segment_trajectory(traj)
    ]
    pnads = learn_strips_operators(segments)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 4
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
    assert len(option_specs) == len(strips_ops) == 4
    assert len(env.options) == 3
    Pick = [option for option in env.options if option.name == "Pick"][0]
    Stack = [option for option in env.options if option.name == "Stack"][0]
    PutOnTable = [
        option for option in env.options if option.name == "PutOnTable"
    ][0]
    param_opts = [spec[0] for spec in option_specs]
    assert param_opts.count(Pick) == 2
    assert param_opts.count(Stack) == 1
    assert param_opts.count(PutOnTable) == 1
    for datastore, spec in zip(datastores, option_specs):
        for (segment, _) in datastore:
            assert not segment.has_option()
            # This call should add an option to the segment.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            option = segment.get_option()
            assert option.parent in (Pick, Stack, PutOnTable)
            assert [obj.type for obj in option.objects] == option.parent.types
    # Reset configuration.
    utils.update_config({
        "env": "blocks",
        "approach": "nsrt_learning",
        "seed": 123,
        "option_learner": "no_learning"
    })


def test_create_option_learner():
    """Tests for create_option_learner()."""
    utils.update_config({
        "env": "not a real env",
        "approach": "nsrt_learning",
        "seed": 123
    })
    utils.update_config({
        "env": "blocks",
        "approach": "nsrt_learning",
        "seed": 123,
        "num_train_tasks": 3,
        "option_learner": "not a real option learner"
    })
    with pytest.raises(NotImplementedError):
        create_option_learner()
    # Reset configuration.
    utils.update_config({
        "env": "blocks",
        "approach": "nsrt_learning",
        "seed": 123,
        "option_learner": "no_learning"
    })
