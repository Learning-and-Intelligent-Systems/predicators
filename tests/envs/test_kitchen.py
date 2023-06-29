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
    At, On, OnTop = sorted(env.predicates)
    assert At.name == "At"
    assert On.name == "On"
    assert OnTop.name == "OnTop"
    assert env.goal_predicates == {At, On, OnTop}
    options = get_gt_options(env.get_name())
    assert len(options) == 12
    Angled_x_y_grasp, Close_gripper, Drop, Lift, Move_backward, \
    Move_delta_ee_pose, Move_forward, Move_left, Move_right, Open_gripper, \
    Rotate_about_x_axis, Rotate_about_y_axis = sorted(
        options)
    assert Angled_x_y_grasp.name == "Angled_x_y_grasp"
    assert Close_gripper.name == "Close_gripper"
    assert Drop.name == "Drop"
    assert Lift.name == "Lift"
    assert Move_backward.name == "Move_backward"
    assert Move_delta_ee_pose.name == "Move_delta_ee_pose"
    assert Move_forward.name == "Move_forward"
    assert Move_left.name == "Move_left"
    assert Move_right.name == "Move_right"
    assert Open_gripper.name == "Open_gripper"
    assert Rotate_about_x_axis.name == "Rotate_about_x_axis"
    assert Rotate_about_y_axis.name == "Rotate_about_y_axis"
    assert len(env.types) == 2
    gripper_type, object_type = env.types
    assert gripper_type.name == "gripper"
    assert object_type.name == "obj"
    assert env.action_space.shape == (29, )
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 1
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
    atoms = utils.abstract(state, env.predicates)

    # Hardcode a sequence of actions.
    plan = ["Open_gripper", "Move_delta_ee_pose"]
    option_names = {o.name: o for o in options}
    for name in plan:
        param_option = option_names[name]
        gripper = Object("gripper", gripper_type)
        obj = Object("knob1", object_type)
        if name == "Move_delta_ee_pose":
            state = env.state_info_to_state(recovered_obs["state_info"])
            ox = state.get(obj, "x")
            oy = state.get(obj, "y")
            oz = state.get(obj, "z")
            option = param_option.ground([gripper, obj],
                                         np.array([ox, oy, oz, 0.0]))
        elif name == "Open_gripper":
            option = param_option.ground([],
                                         np.array([0.5]).astype(np.float32))
        assert option.initiable(state)
        option_done = False
        while not option_done:
            action = option.policy(state)
            obs = env.step(action)
            state = perceiver.step(obs)
            option_done = option.terminal(state)
        recovered_obs = env.get_observation()
        assert (np.allclose(m1, m2) for m1, m2 in zip(
            obs['state_info'].values(), recovered_obs['state_info'].values()))
    assert env.goal_reached()
    atoms = utils.abstract(state, env.predicates)
    # Now one of the goals should be covered.
    assert len({a for a in atoms if a.predicate == At}) == 1
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
