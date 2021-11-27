"""Tests for option learning.
"""

import time
from gym.spaces import Box
import numpy as np
from predicators.src.envs import create_env
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.nsrt_learning import learn_nsrts_from_data, \
    unify_effects_and_options, segment_trajectory, learn_strips_operators
from predicators.src.option_learning import create_option_learner
from predicators.src.structs import Type, Predicate, State, Action, \
    ParameterizedOption
from predicators.src import utils


def test_known_options_option_learner():
    """Tests for _KnownOptionsOptionLearner.
    """
    env = create_env("cover")
    # We need to call update_config twice because the first call sets
    # some variables whose values we can then change in the second call.
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123})
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "num_train_tasks": 3,
                         "do_option_learning": False})
    dataset = create_demo_replay_data(env)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for _, actions, _ in ground_atom_dataset:
        for act in actions:
            assert act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    strips_ops, partitions = learn_strips_operators(segments)
    assert len(strips_ops) == len(partitions) == 2
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops) == 2
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []), (PickPlace, [])]
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            assert segment.has_option()
            option = segment.get_option()
            # This call should be a no-op when options are known.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            assert segment.get_option() == option


def test_oracle_option_learner_cover():
    """Tests for _OracleOptionLearner for the cover environment.
    """
    env = create_env("cover")
    # We need to call update_config twice because the first call sets
    # some variables whose values we can then change in the second call.
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123})
    utils.update_config({"env": "cover",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "num_train_tasks": 3,
                         "do_option_learning": True,
                         "option_learner": "oracle"})
    dataset = create_demo_replay_data(env)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for _, actions, _ in ground_atom_dataset:
        for act in actions:
            assert not act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    strips_ops, partitions = learn_strips_operators(segments)
    assert len(strips_ops) == len(partitions) == 2
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops) == 2
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []), (PickPlace, [])]
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            assert not segment.has_option()
            # This call should add an option to the segment.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            option = segment.get_option()
            # In cover env, param == action array.
            assert option.parent == PickPlace
            assert np.allclose(option.params, segment.actions[0].arr)


def test_oracle_option_learner_blocks():
    """Tests for _OracleOptionLearner for the blocks environment.
    """
    env = create_env("blocks")
    # We need to call update_config twice because the first call sets
    # some variables whose values we can then change in the second call.
    utils.update_config({"env": "blocks",
                         "approach": "nsrt_learning",
                         "seed": 123})
    utils.update_config({"env": "blocks",
                         "approach": "nsrt_learning",
                         "seed": 123,
                         "num_train_tasks": 3,
                         "do_option_learning": True,
                         "option_learner": "oracle"})
    dataset = create_demo_replay_data(env)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset, env.predicates)
    for _, actions, _ in ground_atom_dataset:
        for act in actions:
            assert not act.has_option()
    segments = [seg for traj in ground_atom_dataset
                for seg in segment_trajectory(traj)]
    strips_ops, partitions = learn_strips_operators(segments)
    assert len(strips_ops) == len(partitions) == 4
    option_learner = create_option_learner()
    option_specs = option_learner.learn_option_specs(strips_ops, partitions)
    assert len(option_specs) == len(strips_ops) == 4
    assert len(env.options) == 3
    Pick = [option for option in env.options
            if option.name == "Pick"][0]
    Stack = [option for option in env.options
             if option.name == "Stack"][0]
    PutOnTable = [option for option in env.options
                  if option.name == "PutOnTable"][0]
    param_opts = [spec[0] for spec in option_specs]
    assert param_opts.count(Pick) == 2
    assert param_opts.count(Stack) == 1
    assert param_opts.count(PutOnTable) == 1
    for partition, spec in zip(partitions, option_specs):
        for (segment, _) in partition:
            assert not segment.has_option()
            # This call should add an option to the segment.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            option = segment.get_option()
            assert option.parent in (Pick, Stack, PutOnTable)
            assert [obj.type for obj in option.objects] == option.parent.types
