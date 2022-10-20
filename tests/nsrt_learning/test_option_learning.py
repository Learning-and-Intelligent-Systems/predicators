"""Tests for option learning."""

import tempfile
from unittest.mock import patch

import dill as pkl
import numpy as np
import pybullet as p
import pytest

import predicators.nsrt_learning.option_learning
from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout, \
    create_approach
from predicators.datasets import create_dataset
from predicators.datasets.demo_replay import create_demo_replay_data
from predicators.envs import create_new_env
from predicators.ground_truth_nsrts import get_gt_nsrts
from predicators.ml_models import MLPRegressor
from predicators.nsrt_learning.option_learning import _ActionConverter, \
    _LearnedNeuralParameterizedOption, create_action_converter, \
    create_option_learner, create_rl_option_learner
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.nsrt_learning.strips_learning import learn_strips_operators
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot
from predicators.settings import CFG
from predicators.structs import STRIPSOperator

_MODULE_PATH = predicators.nsrt_learning.option_learning.__name__
longrun = pytest.mark.skipif("not config.getoption('longrun')")


def test_known_options_option_learner():
    """Tests for _KnownOptionsOptionLearner."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "num_train_tasks": 3,
        "option_learner": "no_learning",
    })
    env = create_new_env("cover")
    train_tasks = env.get_train_tasks()
    dataset = create_demo_replay_data(env, train_tasks, env.options)
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert act.has_option()
    segmented_trajs = [
        segment_trajectory(traj) for traj in ground_atom_dataset
    ]
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   env.predicates,
                                   segmented_trajs,
                                   verify_harmlessness=True)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 5
    option_learner = create_option_learner(env.action_space)
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
    assert len(option_specs) == len(strips_ops) == 5
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []) for _ in range(5)]
    for datastore, spec in zip(datastores, option_specs):
        for (segment, _) in datastore:
            assert segment.has_option()
            option = segment.get_option()
            # This call should be a noop when options are known.
            option_learner.update_segment_from_option_spec(segment, spec)
            assert segment.has_option()
            assert segment.get_option() == option


def test_oracle_option_learner_cover():
    """Tests for _OracleOptionLearner for the cover environment."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "num_train_tasks": 3,
        "option_learner": "oracle",
        "segmenter": "atom_changes",
    })
    env = create_new_env("cover")
    train_tasks = env.get_train_tasks()
    dataset = create_demo_replay_data(env, train_tasks, known_options=set())
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert not act.has_option()
    segmented_trajs = [
        segment_trajectory(traj) for traj in ground_atom_dataset
    ]
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   env.predicates,
                                   segmented_trajs,
                                   verify_harmlessness=True)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 4
    option_learner = create_option_learner(env.action_space)
    option_specs = option_learner.learn_option_specs(strips_ops, datastores)
    assert len(option_specs) == len(strips_ops) == 4
    assert len(env.options) == 1
    PickPlace = next(iter(env.options))
    assert option_specs == [(PickPlace, []) for _ in range(4)]
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
        "option_learner": "oracle",
        "segmenter": "atom_changes",
        "blocks_num_blocks_train": [3],
        "blocks_num_blocks_test": [4],
    })
    env = create_new_env("blocks")
    train_tasks = env.get_train_tasks()
    dataset = create_demo_replay_data(env, train_tasks, known_options=set())
    ground_atom_dataset = utils.create_ground_atom_dataset(
        dataset.trajectories, env.predicates)
    for traj, _ in ground_atom_dataset:
        for act in traj.actions:
            assert not act.has_option()
    segmented_trajs = [
        segment_trajectory(traj) for traj in ground_atom_dataset
    ]
    pnads = learn_strips_operators(dataset.trajectories,
                                   train_tasks,
                                   env.predicates,
                                   segmented_trajs,
                                   verify_harmlessness=True)
    strips_ops = [pnad.op for pnad in pnads]
    datastores = [pnad.datastore for pnad in pnads]
    assert len(strips_ops) == len(datastores) == 4
    option_learner = create_option_learner(env.action_space)
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
        "option_learner": "direct_bc",
        "segmenter": "atom_changes",
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99,
    })
    action_converter = create_action_converter()
    env = create_new_env("cover_multistep_options")
    nsrts = get_gt_nsrts(env.predicates, env.options)
    assert len(nsrts) == 2
    pick_nsrt = min(nsrts, key=lambda o: o.name)
    pick_operator = STRIPSOperator(pick_nsrt.name, pick_nsrt.parameters,
                                   pick_nsrt.preconditions,
                                   pick_nsrt.add_effects,
                                   pick_nsrt.delete_effects,
                                   pick_nsrt.ignore_effects)
    # In this example, both of the parameters (block and robot) are changing.
    # For simplicity, assume that the first and third features for the
    # block and the robot are changing.
    changing_var_to_feat = {p: [0, 2] for p in pick_operator.parameters}
    changing_var_order = list(pick_operator.parameters)
    # Create a dummy regressor but with the right shapes.
    regressor = MLPRegressor(seed=123,
                             hid_sizes=[32, 32],
                             max_train_iters=10,
                             clip_gradients=False,
                             clip_value=5,
                             learning_rate=1e-3)
    param_dim = sum(len(v) for v in changing_var_to_feat.values())
    input_dim = sum(p.type.dim for p in pick_operator.parameters) + param_dim
    # The plus 1 is for the bias term.
    X_arr_regressor = np.zeros((1, 1 + input_dim), dtype=np.float32)
    Y_arr_regressor = np.zeros((1, ) + env.action_space.shape,
                               dtype=np.float32)
    regressor.fit(X_arr_regressor, Y_arr_regressor)
    param_option = _LearnedNeuralParameterizedOption(
        "LearnedOption1", pick_operator, regressor, changing_var_to_feat,
        changing_var_order, env.action_space, action_converter)
    assert param_option.name == "LearnedOption1"
    assert param_option.types == [p.type for p in pick_operator.parameters]
    assert param_option.params_space.shape == (param_dim, )
    # Get an initial state where picking should be possible.
    task = env.get_test_tasks()[0]

    state = task.init.copy()
    block0, _, block1, _, robot, _, _, _, _ = list(state)
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
    with pytest.raises(utils.OptionExecutionFailure) as e:
        state.set(block0, "x", np.nan)
        action = option.policy(state)
    assert "Option policy returned nan" in str(e)
    # Test that the option terminates early if it encounters the same state
    # two times in a row.
    state = task.init
    option = param_option.ground([block0, robot],
                                 np.zeros(param_dim, dtype=np.float32))
    assert option.initiable(state)
    assert not option.terminal(state)
    assert option.terminal(state)
    # Test the method to get the relative parameter from the current state.
    random_param = np.random.rand(param_dim)
    option = param_option.ground([block0, robot], random_param)
    assert option.initiable(state)
    assert np.allclose(random_param,
                       option.parent.get_rel_option_param_from_state(
                           state, option.memory, option.objects),
                       atol=1e-07)


def test_create_option_learner():
    """Tests for create_option_learner()."""
    utils.reset_config({
        "env": "blocks",
        "approach": "nsrt_learning",
        "num_train_tasks": 3,
        "option_learner": "not a real option learner"
    })
    env = create_new_env("blocks")
    with pytest.raises(NotImplementedError):
        create_option_learner(env.action_space)


def test_create_rl_option_learner():
    """Tests for create_rl_option_learner()."""
    utils.reset_config({
        "env": "cover_multistep_options",
        "approach": "nsrt_rl",
        "num_train_tasks": 3,
        "nsrt_rl_option_learner": "not a real option learner"
    })
    with pytest.raises(NotImplementedError):
        create_rl_option_learner()


@longrun
def test_option_learning_approach_multistep_cover():
    """A long test to identify any regressions in option learning."""
    utils.reset_config({
        "env": "cover_multistep_options",
        "approach": "nsrt_learning",
        "option_learner": "direct_bc",
        "segmenter": "atom_changes",
        "sampler_learner": "oracle",
        "num_train_tasks": 10,
        "num_test_tasks": 10,
    })
    env = create_new_env("cover_multistep_options")
    train_tasks = env.get_train_tasks()
    approach = create_approach("nsrt_learning", env.predicates, env.options,
                               env.types, env.action_space, train_tasks)
    dataset = create_dataset(env, train_tasks, known_options=set())
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    num_test_successes = 0
    for task in env.get_test_tasks():
        try:
            policy = approach.solve(task, timeout=CFG.timeout)
            traj = utils.run_policy_with_simulator(policy,
                                                   env.simulate,
                                                   task.init,
                                                   task.goal_holds,
                                                   max_num_steps=CFG.horizon)
            if task.goal_holds(traj.states[-1]):
                num_test_successes += 1
        except (ApproachFailure, ApproachTimeout):
            continue
    # This number is expected to be relatively low because the number of train
    # tasks is pretty limiting. But if it goes lower than this, that could
    # be a performance regression that we should investigate.
    assert num_test_successes >= 3


@longrun
def test_implicit_bc_option_learning_touch_point():
    """A long test to identify regressions in implicit BC option learning."""
    utils.reset_config({
        "env": "touch_point",
        "approach": "nsrt_learning",
        "option_learner": "implicit_bc",
        "segmenter": "atom_changes",
        "num_test_tasks": 10,
    })
    env = create_new_env("touch_point")
    train_tasks = env.get_train_tasks()
    approach = create_approach("nsrt_learning", env.predicates, env.options,
                               env.types, env.action_space, train_tasks)
    dataset = create_dataset(env, train_tasks, known_options=set())
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    num_test_successes = 0
    for task in env.get_test_tasks():
        try:
            policy = approach.solve(task, timeout=CFG.timeout)
            traj = utils.run_policy_with_simulator(policy,
                                                   env.simulate,
                                                   task.init,
                                                   task.goal_holds,
                                                   max_num_steps=CFG.horizon)
            if task.goal_holds(traj.states[-1]):
                num_test_successes += 1
        except (ApproachFailure, ApproachTimeout):
            continue
    assert num_test_successes == 10


class _ReverseOrderPadActionConverter(_ActionConverter):
    """Reverses the order of the actions and adds/removes padding."""

    def env_to_reduced(self, env_action_arr):
        reversed_action = list(reversed(env_action_arr))
        padded_action = reversed_action + [0.0, 0.0]
        return np.array(padded_action, dtype=np.float32)

    def reduced_to_env(self, reduced_action_arr):
        unpadded_action = reduced_action_arr[:-2]
        unreversed_action = list(reversed(unpadded_action))
        return np.array(unreversed_action, dtype=np.float32)


def test_action_conversion():
    """Tests for _ActionConverter() subclasses."""
    utils.reset_config({
        "env": "touch_point",
        "approach": "nsrt_learning",
        "option_learner": "direct_bc",
        "strips_learner": "oracle",
        "sampler_learner": "oracle",
        "mlp_regressor_max_itr": 10,
        "segmenter": "atom_changes",
        "num_train_tasks": 5,
        "num_test_tasks": 1,
    })
    with patch(f"{_MODULE_PATH}.create_action_converter") as mocker:
        mocker.return_value = _ReverseOrderPadActionConverter()
        env = create_new_env("touch_point")
        train_tasks = env.get_train_tasks()
        approach = create_approach("nsrt_learning", env.predicates, set(),
                                   env.types, env.action_space, train_tasks)
        dataset = create_dataset(env, train_tasks, known_options=set())
        approach.learn_from_offline_dataset(dataset)
        task = env.get_test_tasks()[0]
        robot, target = sorted(list(task.init))
        param_option = next(iter(approach._get_current_nsrts())).option  # pylint: disable=protected-access
        params = param_option.params_space.sample()
        option = param_option.ground([robot, target], params)
        assert option.initiable(task.init)
        action = option.policy(task.init)
        assert env.action_space.contains(action.arr)


def test_create_action_converter():
    """Tests for create_action_converter():"""
    # Cover case with unknown action space converter.
    utils.reset_config({
        "option_learning_action_converter":
        "not a real converter",
    })
    with pytest.raises(NotImplementedError):
        create_action_converter()


def test_kinematic_action_conversion():
    """Tests for _KinematicActionConverter() subclasses."""
    utils.reset_config({
        "pybullet_robot": "panda",
        "option_learning_action_converter": "kinematic",
    })
    physics_client_id = p.connect(p.DIRECT)
    robot = create_single_arm_pybullet_robot("panda", physics_client_id)
    robot.go_home()
    env_action_arr = np.array(robot.get_joints())
    converter = create_action_converter()
    reduced_action_arr = converter.env_to_reduced(env_action_arr)
    assert reduced_action_arr.shape == (4, )
    env_action_arr2 = converter.reduced_to_env(reduced_action_arr)
    reduced_action_arr2 = converter.env_to_reduced(env_action_arr2)
    assert np.allclose(reduced_action_arr, reduced_action_arr2)
    # Test case where IK fails.
    with pytest.raises(utils.OptionExecutionFailure) as e:
        converter.reduced_to_env([100, 100, 100, 0])
    assert "IK failure in action conversion" in str(e)
    # Test pickling and loading when the physics client disconnects.
    p.disconnect(physics_client_id)
    pkl_file = tempfile.NamedTemporaryFile(delete=False).name
    with open(pkl_file, "wb") as f:
        pkl.dump(converter, f)
    with open(pkl_file, "rb") as f:
        loaded_converter = pkl.load(f)
    reduced_action_arr3 = loaded_converter.env_to_reduced(env_action_arr2)
    assert np.allclose(reduced_action_arr, reduced_action_arr3)
