"""Test cases for the Kitchen environment."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.kitchen_perceiver import KitchenPerceiver
from predicators.structs import Object

longrun = pytest.mark.skipif("not config.getoption('longrun')")


@longrun
def test_kitchen():
    """Tests for kitchen env.

    Since the gym environment can be slow to initialize, we group all
    tests together.
    """
    utils.reset_config({
        "env": "kitchen",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
    })
    env = KitchenEnv()
    perceiver = KitchenPerceiver()
    assert env.get_name() == "kitchen"
    assert perceiver.get_name() == "kitchen"
    for env_task in env.get_train_tasks():
        task = perceiver.reset(env_task)
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for env_task in env.get_test_tasks():
        task = perceiver.reset(env_task)
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 3
    At, OnTop, TurnedOn = sorted(env.predicates)
    assert At.name == "At"
    assert OnTop.name == "OnTop"
    assert TurnedOn.name == "TurnedOn"
    assert env.goal_predicates == {At, OnTop, TurnedOn}
    options = get_gt_options(env.get_name())
    assert len(options) == 3
    moveto_option, pushobjonobjforward_option, pushobjturnonright_option = \
        sorted(options)
    assert moveto_option.name == "moveto_option"
    assert pushobjonobjforward_option.name == "pushobjonobjforward_option"
    assert pushobjturnonright_option.name == "pushobjturnonright_option"
    assert len(env.types) == 2
    gripper_type, object_type = env.types
    assert gripper_type.name == "gripper"
    assert object_type.name == "obj"
    assert env.action_space.shape == (29, )
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 3
    env_train_tasks = env.get_train_tasks()
    assert len(env_train_tasks) == 1
    env_test_tasks = env.get_test_tasks()
    assert len(env_test_tasks) == 2
    env_task = env_test_tasks[1]
    obs = env.reset("test", 1)
    assert all(
        np.allclose(m1, m2)
        for m1, m2 in zip(obs['state_info'].values(),
                          env_task.init_obs['state_info'].values()))
    imgs = env.render()
    assert len(imgs) == 1
    task = perceiver.reset(env_task)
    state = task.init

    # Cover not implemented methods.
    with pytest.raises(NotImplementedError) as e:
        env.render_state_plt(obs, task)
    assert "This env does not use Matplotlib" in str(e)
    with pytest.raises(NotImplementedError) as e:
        env.render_state(obs, task)
    assert "A gym environment cannot render arbitrary states." in str(e)
    with pytest.raises(NotImplementedError) as e:
        env.simulate(obs, env.action_space.sample())
    assert "Simulate not implemented for gym envs." in str(e)
