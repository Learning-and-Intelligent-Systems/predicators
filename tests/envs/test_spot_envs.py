"""Test cases for the repeated NextTo environment."""

from pathlib import Path

import numpy as np
import pytest

from predicators import utils
from predicators.envs.spot_env import SpotBikeEnv, SpotGroceryEnv
from predicators.ground_truth_models import get_gt_options


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


def test_spot_env_simulate():
    """Tests for the simulate() function."""
    utils.reset_config({
        "env": "repeated_nextto",
        "approach": "nsrt_learning",
    })
    env = SpotGroceryEnv()
    options = get_gt_options(env.get_name())
    MoveToSurface = [o for o in options if o.name == "MoveToSurface"][0]
    MoveToCan = [o for o in options if o.name == "MoveToCan"][0]
    GraspCan = [o for o in options if o.name == "GraspCan"][0]
    PlaceCanOnTop = [o for o in options if o.name == "PlaceCanOntop"][0]

    task = env.get_train_tasks()[0]
    state = task.init
    assert len(state.simulator_state) == 2
    assert "On(soda_can:soda_can, counter:flat_surface)" in str(state)
    assert "HandEmpty(spot:robot)" in str(state)
    objs = list(task.init)
    counter = [obj for obj in objs if obj.name == "counter"][0]
    soda_can = [obj for obj in objs if obj.name == "soda_can"][0]
    spot = [obj for obj in objs if obj.name == "spot"][0]
    # Try moving to the counter.
    act = MoveToSurface.ground([spot, counter],
                               np.array([0.0, 0.0, 0.0])).policy(task.init)
    counter_state = env.simulate(task.init, act)
    assert "ReachableSurface(spot:robot, counter:flat_surface)" in str(
        counter_state.simulator_state)
    # Try moving to the can.
    act = MoveToCan.ground([spot, soda_can], np.array([0.0, 0.0,
                                                       0.0])).policy(task.init)
    can_state = env.simulate(task.init, act)
    assert "ReachableCan(spot:robot, soda_can:soda_can)" in str(
        can_state.simulator_state)
    # Try grasping the can when it is reachable.
    act = GraspCan.ground([spot, soda_can, counter],
                          np.array([0.0])).policy(can_state)
    can_held_state = env.simulate(can_state, act)
    assert "HoldingCan(spot:robot, soda_can:soda_can)}" in str(
        can_held_state.simulator_state)
    # Try grasping the can when it's not reachable.
    act = GraspCan.ground([spot, soda_can, counter],
                          np.array([0.0])).policy(task.init)
    init_state = env.simulate(task.init, act)
    assert task.init.allclose(init_state)
    # Try placing the can after it has been held.
    act = MoveToSurface.ground([spot, counter],
                               np.array([0.0, 0.0,
                                         0.0])).policy(can_held_state)
    counter_holding_state = env.simulate(can_held_state, act)
    act = PlaceCanOnTop.ground([spot, soda_can, counter],
                               np.array([0.0])).policy(counter_holding_state)
    can_putdown_state = env.simulate(counter_holding_state, act)
    assert can_putdown_state.allclose(counter_state)
    # Try placing the can when it isn't held.
    act = PlaceCanOnTop.ground([spot, soda_can, counter],
                               np.array([0.0])).policy(task.init)
    init_state = env.simulate(task.init, act)
    assert task.init.allclose(init_state)


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
    ) == "EnvironmentTask(init_obs=_PDDLEnvState(data={counter:flat_" +\
        "surface: {}, snack_table:flat_surface: {}, soda_can:soda_can: {}, " +\
        "spot:robot: {}}, simulator_state={On(soda_can:soda_can, " +\
        "counter:flat_surface), HandEmpty(spot:robot)}), " +\
        "goal_description={On(soda_can:soda_can, snack_table:flat_surface)})"
