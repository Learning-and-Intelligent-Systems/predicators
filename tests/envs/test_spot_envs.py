"""Test cases for the repeated NextTo environment."""

from pathlib import Path

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
