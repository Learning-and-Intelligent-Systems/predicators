"""Tests for option learning."""

import pytest
import numpy as np
from predicators.src.approaches import ApproachFailure
from predicators.src.envs import create_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.datasets.demo_replay import create_demo_replay_data
from predicators.src.nsrt_learning.strips_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src.structs import STRIPSOperator
from predicators.src.torch_models import MLPRegressor
from predicators.src.nsrt_learning.option_learning import \
    create_option_learner, _LearnedNeuralParameterizedOption
from predicators.src import utils


def test_known_options_option_learner():
    """Tests for _KnownOptionsOptionLearner."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "num_train_tasks": 3,
        "option_learner": "no_learning"
    })
    env = create_env("cover")
    train_tasks = env.get_train_tasks()
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
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


def test_oracle_option_learner_cover():
    """Tests for _OracleOptionLearner for the cover environment."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "num_train_tasks": 3,
        "option_learner": "oracle"
    })
    env = create_env("cover")
    train_tasks = env.get_train_tasks()
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
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


def test_oracle_option_learner_blocks():
    """Tests for _OracleOptionLearner for the blocks environment."""
    utils.reset_config({
        "env": "blocks",
        "approach": "nsrt_learning",
        "seed": 123,
        "num_train_tasks": 3,
        "option_learner": "oracle"
    })
    env = create_env("blocks")
    train_tasks = env.get_train_tasks()
    dataset = create_demo_replay_data(env, train_tasks)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
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


def test_learned_neural_parameterized_option():
    """Tests for _LearnedNeuralParameterizedOption()."""
    # Create a _LearnedNeuralParameterizedOption() for the cover Pick operator.
    utils.reset_config({
        "env": "cover_multistep_options",
        "option_learner": "neural",
        "mlp_regressor_max_itr": 10,
    })
    env = create_env("cover_multistep_options")
    nsrts = get_gt_nsrts(env.predicates, env.options)
    assert len(nsrts) == 2
    pick_nsrt = min(nsrts, key=lambda o: o.name)
    pick_operator = STRIPSOperator(pick_nsrt.name, pick_nsrt.parameters,
                                   pick_nsrt.preconditions,
                                   pick_nsrt.add_effects,
                                   pick_nsrt.delete_effects,
                                   pick_nsrt.side_predicates)
    # In this example, both of the parameters (block and robot) are changing.
    changing_parameters = pick_operator.parameters
    # Create a dummy regressor but with the right shapes.
    regressor = MLPRegressor()
    param_dim = sum([p.type.dim for p in changing_parameters])
    input_dim = sum([p.type.dim for p in pick_operator.parameters]) + param_dim
    # The plus 1 is for the bias term.
    X_arr_regressor = np.zeros((1, 1 + input_dim), dtype=np.float32)
    Y_arr_regressor = np.zeros((1, ) + env.action_space.shape,
                               dtype=np.float32)
    regressor.fit(X_arr_regressor, Y_arr_regressor)
    param_option = _LearnedNeuralParameterizedOption("LearnedOption1",
                                                     pick_operator, regressor,
                                                     changing_parameters)
    assert param_option.name == "LearnedOption1"
    assert param_option.types == [p.type for p in pick_operator.parameters]
    assert param_option.params_space.shape == (param_dim, )
    # Get an initial state where picking should be possible.
    env.seed(123)
    task = env.get_test_tasks()[0]

    state = task.init.copy()
    block0, _, block1, _, robot, _, _, _, _ = sorted(state)
    assert block0.name == "block0"
    assert robot.name == "robby"
    option = param_option.ground([block0, robot],
                                 np.zeros(param_dim, dtype=np.float32))
    assert option.initiable(state)
    action = option.policy(state)
    assert env.action_space.contains(action.arr)
    assert not option.terminal(state)
    assert state.get(block0, "grasp") == -1.0
    state.set(block0, "grasp", 1.0)
    state.set(robot, "holding", 1.0)
    assert option.terminal(state)
    # Option should also terminate if it's outside of the precondition set.
    state.set(block0, "grasp", -1.0)
    state.set(block1, "grasp", 1.0)
    assert option.terminal(state)
    # Cover case where regressor returns nan.
    with pytest.raises(ApproachFailure):
        state.set(block0, "x", np.nan)
        action = option.policy(state)
    # Test that the option terminates early if it encounters the same state
    # two times in a row.
    state = task.init
    option = param_option.ground([block0, robot],
                                 np.zeros(param_dim, dtype=np.float32))
    assert option.initiable(state)
    assert not option.terminal(state)
    assert option.terminal(state)


def test_create_option_learner():
    """Tests for create_option_learner()."""
    utils.reset_config({
        "env": "blocks",
        "approach": "nsrt_learning",
        "num_train_tasks": 3,
        "option_learner": "not a real option learner"
    })
    with pytest.raises(NotImplementedError):
        create_option_learner()
