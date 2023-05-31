"""Test cases for the repeated NextTo environment."""

from pathlib import Path

import numpy as np
import pytest

from predicators import utils
from predicators.envs.spot_env import SpotBikeEnv, SpotGroceryEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options


def test_spot_grocery_env():
    """Tests for SpotGroceryEnv class."""
    utils.reset_config({
        "env": "spot_grocery_env",
        "num_train_tasks": 1,
        "num_test_tasks": 1
    })
    env = SpotGroceryEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert {pred.name for pred in env.predicates} == \
        {"ReachableSurface", "HoldingCan", "On", "HandEmpty", "ReachableCan"}
    assert {pred.name for pred in env.goal_predicates} == {"On"}
    options = get_gt_options(env.get_name())
    assert len(options) == 4
    assert len(env.types) == 3
    assert len(env.strips_operators) == 4
    with pytest.raises(NotImplementedError):
        env.render_state_plt(task.init, task)
    with pytest.raises(NotImplementedError):
        env.simulate(task.init, [])


def test_spot_bike_env():
    """Tests for SpotBikeEnv class."""
    utils.reset_config({
        "env": "spot_bike_env",
        "num_train_tasks": 1,
        "num_test_tasks": 1
    })
    env = SpotBikeEnv()
    assert {pred.name
            for pred in env.goal_predicates
            } == {pred.name
                  for pred in env.predicates}


def test_spot_env_oracle_nsrts():
    """Tests for the oracle policies and samplers."""

    utils.reset_config()
    rng = np.random.default_rng(123)
    env = SpotGroceryEnv()
    options = get_gt_options(env.get_name())
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    MoveToSurface = [o for o in nsrts if o.name == "MoveToSurface"][0]
    MoveToCan = [o for o in nsrts if o.name == "MoveToCan"][0]
    GraspCan = [o for o in nsrts if o.name == "GraspCan"][0]
    PlaceCanOnTop = [o for o in nsrts if o.name == "PlaceCanOntop"][0]

    state = env.reset("train", 0)
    state_copy = state.copy()
    assert state.allclose(state_copy)
    state_copy.simulator_state["atoms"].clear()
    assert not state.allclose(state_copy)
    assert len(state.simulator_state["atoms"]) == 2
    assert "On(soda_can:soda_can, counter:flat_surface)" in str(state)
    assert "HandEmpty(spot:robot)" in str(state)
    objs = list(state)
    counter = [obj for obj in objs if obj.name == "counter"][0]
    soda_can = [obj for obj in objs if obj.name == "soda_can"][0]
    spot = [obj for obj in objs if obj.name == "spot"][0]

    ground_nsrt = MoveToSurface.ground([spot, counter])
    ground_option = ground_nsrt.sample_option(state, set(), rng)
    act = ground_option.policy(state)
    assert env.action_space.contains(act.arr)
    name, objs, params = env._parse_action(state, act)  # pylint:disable=protected-access
    assert name == "navigate"
    assert objs == [spot, counter]
    assert np.allclose(params, ground_option.params)
    # Test simulate with valid action.
    next_state = env._get_next_simulator_state(state, action)  # pylint:disable=protected-access
    assert not next_state.allclose(state)
    
    ground_nsrt = MoveToCan.ground([spot, soda_can])
    ground_option = ground_nsrt.sample_option(state, set(), rng)
    act = ground_option.policy(state)
    assert env.action_space.contains(act.arr)
    name, objs, params = env._parse_action(state, act)  # pylint:disable=protected-access
    assert name == "navigate"
    assert objs == [spot, soda_can]
    assert np.allclose(params, ground_option.params)

    ground_nsrt = GraspCan.ground([spot, soda_can, counter])
    ground_option = ground_nsrt.sample_option(state, set(), rng)
    act = ground_option.policy(state)
    assert env.action_space.contains(act.arr)
    name, objs, params = env._parse_action(state, act)  # pylint:disable=protected-access
    assert name == "grasp"
    assert objs == [spot, soda_can, counter]
    assert np.allclose(params, ground_option.params)

    ground_nsrt = PlaceCanOnTop.ground([spot, soda_can, counter])
    ground_option = ground_nsrt.sample_option(state, set(), rng)
    act = ground_option.policy(state)
    assert env.action_space.contains(act.arr)
    name, objs, params = env._parse_action(state, act)  # pylint:disable=protected-access
    assert name == "placeOnTop"
    assert objs == [spot, soda_can, counter]
    assert np.allclose(params, ground_option.params)
    # Test simulate with invalid action.
    next_state = env._get_next_simulator_state(state, action)  # pylint:disable=protected-access
    assert next_state.allclose(state)


def test_natural_language_goal_prompt_prefix():
    """Test the prompt prefix creation function."""
    env = SpotGroceryEnv()
    object_names = {"spot", "counter", "snack_table", "soda_can"}
    prompt = env._get_language_goal_prompt_prefix(object_names)  # pylint: disable=W0212
    assert prompt == '# The available predicates are: On\n# The available ' +\
    'objects are: counter, snack_table, soda_can, spot\n# Use the ' +\
    'available predicates and objects to convert natural language ' +\
    'goals into JSON goals.\n        \n# Hey spot, can you move the ' +\
    'soda can to the snack table?\n{"On": [["soda_can", "snack_table"]]}\n'

    env = SpotBikeEnv()
    object_names = {"spot", "hammer", "toolbag", "low_wall_rack"}
    prompt = env._get_language_goal_prompt_prefix(object_names)  # pylint: disable=W0212
    assert "Will you put the bag onto the low rack, please?" in prompt


def test_json_loading():
    """Test JSON loading from a specially-created test JSON file."""
    env = SpotGroceryEnv()
    output_task = env._load_task_from_json(  # pylint: disable=W0212
        Path('predicators/spot_utils/json_tasks/grocery/test.json'))
    assert str(
        output_task
    ) == "EnvironmentTask(init_obs=_PartialPerceptionState(data={counter:flat_surface: array([], dtype=float32), snack_table:flat_surface: array([], dtype=float32), soda_can:soda_can: array([], dtype=float32), spot:robot: array([], dtype=float32)}, simulator_state={'predicates': {ReachableSurface, On, HoldingCan, HandEmpty, ReachableCan}, 'atoms': {On(soda_can:soda_can, counter:flat_surface), HandEmpty(spot:robot)}}), goal_description={On(soda_can:soda_can, snack_table:flat_surface)})"  # pylint:disable=line-too-long
