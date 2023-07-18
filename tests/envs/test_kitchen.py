"""Test cases for the Kitchen environment."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.kitchen_perceiver import KitchenPerceiver

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
        "kitchen_use_perfect_samplers": True,
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
    assert len(env.predicates) == 4
    At, CanTurnDial, OnTop, TurnedOn = sorted(env.predicates)
    assert At.name == "At"
    assert CanTurnDial.name == "CanTurnDial"
    assert OnTop.name == "OnTop"
    assert TurnedOn.name == "TurnedOn"
    assert env.goal_predicates == {At, CanTurnDial, OnTop, TurnedOn}
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

    # Test NSRTs.
    MoveTo, PushObjOnObjForward, PushObjTurnOnRight = sorted(nsrts)
    assert MoveTo.name == "MoveTo"
    assert PushObjOnObjForward.name == "PushObjOnObjForward"
    assert PushObjTurnOnRight.name == "PushObjTurnOnRight"

    obs = env_test_tasks[0].init_obs
    init_state = env.state_info_to_state(obs["state_info"])
    rng = np.random.default_rng(123)

    # For reference.
    """
    ######################## STATE #######################
    type: gripper            x         y        z
    ---------------  ---------  --------  -------
    gripper          -0.431879  0.118886  2.02457

    type: obj            x         y        z        angle
    -----------  ---------  --------  -------  -----------
    burner1      -0.3       0.3       0.629    0
    burner2      -0.3       0.8       0.629    0
    burner3       0.204     0.8       0.629    0
    burner4       0.206     0.3       0.629    0
    hinge1       -0.682831  0.580057  2.6      0
    hinge2       -0.526226  0.582535  2.6      0
    kettle       -0.272806  0.348292  1.77842  0
    knob1        -0.133     0.6399    2.22643  0
    knob2        -0.256     0.6399    2.22643  3.12877e-05
    knob3        -0.133     0.6399    2.34043  6.28065e-05
    knob4        -0.256     0.6399    2.34043  0
    light        -0.368465  0.615715  2.28     0
    microhandle  -0.641919  0.492627  1.792    0
    slide        -0.108466  0.607     2.6      0
    ######################################################
    """
    obj_name_to_obj = {o.name: o for o in init_state}
    gripper = obj_name_to_obj["gripper"]
    knob3 = obj_name_to_obj["knob3"]
    kettle = obj_name_to_obj["kettle"]
    burner2 = obj_name_to_obj["burner2"]

    # Test moving to and pushing knob3.
    # TODO

    # Test moving to and pushing the kettle on top of burner2.
    move_to_kettle_nsrt = MoveTo.ground([gripper, kettle])
    for atom in move_to_kettle_nsrt.preconditions:
        assert atom.holds(init_state)
    # This sampler should always succeed.
    move_to_kettle_option = move_to_kettle_nsrt.sample_option(
        init_state, set(), rng)
    assert move_to_kettle_option.initiable(init_state)
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    for _ in range(100):
        act = move_to_kettle_option.policy(state)
        obs = env.step(act)
        state = env.state_info_to_state(obs["state_info"])
        if move_to_kettle_option.terminal(state):
            break
    for atom in move_to_kettle_nsrt.add_effects:
        assert atom.holds(state)
    for atom in move_to_kettle_nsrt.delete_effects:
        assert not atom.holds(state)
    # TODO add pushing the kettle on tpo of burner2.
