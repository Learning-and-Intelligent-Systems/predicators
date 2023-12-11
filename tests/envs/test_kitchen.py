"""Test cases for the Kitchen environment."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.kitchen_perceiver import KitchenPerceiver

longrun = pytest.mark.skipif("not config.getoption('longrun')")
USE_GUI = True


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
    env = KitchenEnv(use_gui=USE_GUI)
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
    assert len(env.predicates) == 10

    AtPrePullKettle, AtPrePushOnTop, AtPreTurnOff, AtPreTurnOn, Closed, \
        NotOnTop, OnTop, Open, TurnedOff, TurnedOn = sorted(env.predicates)
    assert AtPrePullKettle.name == "AtPrePullKettle"
    assert AtPrePushOnTop.name == "AtPrePushOnTop"
    assert AtPreTurnOff.name == "AtPreTurnOff"
    assert AtPreTurnOn.name == "AtPreTurnOn"
    assert Closed.name == "Closed"
    assert NotOnTop.name == "NotOnTop"
    assert OnTop.name == "OnTop"
    assert Open.name == "Open"
    assert TurnedOff.name == "TurnedOff"
    assert TurnedOn.name == "TurnedOn"
    assert env.goal_predicates == {OnTop, TurnedOn, Open}
    options = get_gt_options(env.get_name())
    assert len(env.types) == 8
    assert env.action_space.shape == (7, )
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 12
    assert len(options) == len(nsrts)
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
    MoveToPrePullKettle, MoveToPrePushOnTop, MoveToPreTurnOff, \
        MoveToPreTurnOn, PullKettle, PushCloseHingeDoor, \
        PushObjOnObjForward, PushOpenHingeDoor, TurnOffKnob, \
        TurnOffSwitch, TurnOnKnob, TurnOnSwitch = sorted(nsrts)
    assert MoveToPrePushOnTop.name == "MoveToPrePushOnTop"
    assert MoveToPrePullKettle.name == "MoveToPrePullKettle"
    assert MoveToPreTurnOff.name == "MoveToPreTurnOff"
    assert MoveToPreTurnOn.name == "MoveToPreTurnOn"
    assert PullKettle.name == "PullKettle"
    assert PushCloseHingeDoor.name == "PushCloseHingeDoor"
    assert PushObjOnObjForward.name == "PushObjOnObjForward"
    assert PushOpenHingeDoor.name == "PushOpenHingeDoor"
    assert TurnOffSwitch.name == "TurnOffSwitch"
    assert TurnOnKnob.name == "TurnOnKnob"
    assert TurnOffKnob.name == "TurnOffKnob"
    assert TurnOnSwitch.name == "TurnOnSwitch"

    obs = env_test_tasks[0].init_obs
    init_state = env.state_info_to_state(obs["state_info"])
    rng = np.random.default_rng(123)

    obj_name_to_obj = {o.name: o for o in init_state}
    gripper = obj_name_to_obj["gripper"]
    knob4 = obj_name_to_obj["knob4"]
    kettle = obj_name_to_obj["kettle"]
    burner2 = obj_name_to_obj["burner2"]
    burner4 = obj_name_to_obj["burner4"]
    light = obj_name_to_obj["light"]
    microhandle = obj_name_to_obj["microhandle"]
    hinge1 = obj_name_to_obj["hinge1"]
    hinge2 = obj_name_to_obj["hinge2"]
    slide = obj_name_to_obj["slide"]

    def _run_ground_nsrt(ground_nsrt,
                         state,
                         override_params=None,
                         assert_add_effects=True,
                         assert_delete_effects=True):
        obs_to_state = lambda obs: env.state_info_to_state(obs["state_info"])
        return utils.run_ground_nsrt_with_assertions(
            ground_nsrt,
            state,
            env,
            rng,
            obs_to_state,
            override_params,
            assert_add_effects=assert_add_effects,
            assert_delete_effects=assert_delete_effects)

    # Set up all the NSRTs for the following tests.
    move_to_light_pre_on_nsrt = MoveToPreTurnOn.ground([gripper, light])
    turn_on_light_nsrt = TurnOnSwitch.ground([gripper, light])
    move_to_light_pre_off_nsrt = MoveToPreTurnOff.ground([gripper, light])
    turn_off_light_nsrt = TurnOffSwitch.ground([gripper, light])
    move_to_knob4_pre_on_nsrt = MoveToPreTurnOn.ground([gripper, knob4])
    turn_on_knob4_nsrt = TurnOnKnob.ground([gripper, knob4])
    move_to_knob4_pre_off_nsrt = MoveToPreTurnOff.ground([gripper, knob4])
    turn_off_knob4_nsrt = TurnOffKnob.ground([gripper, knob4])
    move_to_kettle_pre_push_nsrt = MoveToPrePushOnTop.ground([gripper, kettle])
    push_kettle_on_burner4_nsrt = PushObjOnObjForward.ground(
        [gripper, kettle, burner4])
    move_to_kettle_pre_pull_nsrt = MoveToPrePullKettle.ground(
        [gripper, kettle])
    pull_kettle_on_burner2_nsrt = PullKettle.ground([gripper, kettle, burner2])

    move_to_microhandle_pre_on_nsrt = MoveToPreTurnOn.ground(
        [gripper, microhandle])
    push_open_microhandle_nsrt = PushOpenHingeDoor.ground(
        [gripper, microhandle])
    move_to_microhandle_pre_off_nsrt = MoveToPreTurnOff.ground(
        [gripper, microhandle])
    push_closed_microhandle_nsrt = PushCloseHingeDoor.ground(
        [gripper, microhandle])

    move_to_slide_pre_on_nsrt = MoveToPreTurnOn.ground([gripper, slide])
    push_open_slide_nsrt = PushOpenHingeDoor.ground([gripper, slide])
    move_to_slide_pre_off_nsrt = MoveToPreTurnOff.ground([gripper, slide])
    push_closed_slide_nsrt = PushCloseHingeDoor.ground([gripper, slide])

    move_to_hinge1_pre_on_nsrt = MoveToPreTurnOn.ground([gripper, hinge1])
    push_open_hinge1_nsrt = PushOpenHingeDoor.ground([gripper, hinge1])
    move_to_hinge1_pre_off_nsrt = MoveToPreTurnOff.ground([gripper, hinge1])
    push_closed_hinge1_nsrt = PushCloseHingeDoor.ground([gripper, hinge1])

    move_to_hinge2_pre_on_nsrt = MoveToPreTurnOn.ground([gripper, hinge2])
    push_open_hinge2_nsrt = PushOpenHingeDoor.ground([gripper, hinge2])
    move_to_hinge2_pre_off_nsrt = MoveToPreTurnOff.ground([gripper, hinge2])
    push_closed_hinge2_nsrt = PushCloseHingeDoor.ground([gripper, hinge2])

    # Test pushing the microwave open and then closing it
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_microhandle_pre_on_nsrt, state)
    state = _run_ground_nsrt(push_open_microhandle_nsrt, state)
    assert Open([microhandle]).holds(state)
    state = _run_ground_nsrt(move_to_microhandle_pre_off_nsrt, state)
    state = _run_ground_nsrt(push_closed_microhandle_nsrt, state)
    assert Closed([microhandle]).holds(state)

    # Test pushing the hinge1 open and then closing it
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_hinge1_pre_on_nsrt, state)
    state = _run_ground_nsrt(push_open_hinge1_nsrt, state)
    assert Open([hinge1]).holds(state)
    state = _run_ground_nsrt(move_to_hinge1_pre_off_nsrt, state)
    state = _run_ground_nsrt(push_closed_hinge1_nsrt, state)
    assert Closed([hinge1]).holds(state)

    # Test pushing the hinge2 open and then closing it
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_hinge2_pre_on_nsrt, state)
    state = _run_ground_nsrt(push_open_hinge2_nsrt, state)
    assert Open([hinge2]).holds(state)
    state = _run_ground_nsrt(move_to_hinge2_pre_off_nsrt, state)
    state = _run_ground_nsrt(push_closed_hinge2_nsrt, state)
    assert Closed([hinge2]).holds(state)

    # Test pushing the slide open and then closing it
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_slide_pre_on_nsrt, state)
    state = _run_ground_nsrt(push_open_slide_nsrt, state)
    assert Open([slide]).holds(state)
    state = _run_ground_nsrt(move_to_slide_pre_off_nsrt, state)
    state = _run_ground_nsrt(push_closed_slide_nsrt, state)
    assert Closed([slide]).holds(state)

    # Test pushing the kettle forward and then bringing it back. Twice!
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_kettle_pre_push_nsrt, state)
    state = _run_ground_nsrt(push_kettle_on_burner4_nsrt, state)
    assert OnTop([kettle, burner4]).holds(state)
    state = _run_ground_nsrt(move_to_kettle_pre_pull_nsrt, state)
    state = _run_ground_nsrt(pull_kettle_on_burner2_nsrt, state)
    assert OnTop([kettle, burner2]).holds(state)

    state = _run_ground_nsrt(move_to_kettle_pre_push_nsrt, state)
    state = _run_ground_nsrt(push_kettle_on_burner4_nsrt, state)
    assert OnTop([kettle, burner4]).holds(state)
    state = _run_ground_nsrt(move_to_kettle_pre_pull_nsrt, state)
    state = _run_ground_nsrt(pull_kettle_on_burner2_nsrt, state)
    assert OnTop([kettle, burner2]).holds(state)
    # Test moving to and turning the light on and off.
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_light_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_light_nsrt, state)
    assert TurnedOn([light]).holds(state)
    state = _run_ground_nsrt(move_to_light_pre_off_nsrt, state)
    state = _run_ground_nsrt(turn_off_light_nsrt, state)
    assert TurnedOff([light]).holds(state)

    # Test moving to and turning knob4 on and off (two times.)
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_knob4_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_knob4_nsrt, state)
    assert TurnedOn([knob4]).holds(state)
    state = _run_ground_nsrt(move_to_knob4_pre_off_nsrt, state)
    state = _run_ground_nsrt(turn_off_knob4_nsrt, state)
    assert TurnedOff([knob4]).holds(state)
    state = _run_ground_nsrt(move_to_knob4_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_knob4_nsrt, state)
    assert TurnedOn([knob4]).holds(state)
    state = _run_ground_nsrt(move_to_knob4_pre_off_nsrt, state)
    state = _run_ground_nsrt(turn_off_knob4_nsrt, state)
    assert TurnedOff([knob4]).holds(state)

    # Test moving to and pushing knob4, then moving to and pushing the kettle.
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_knob4_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_knob4_nsrt, state)
    state = _run_ground_nsrt(move_to_kettle_pre_push_nsrt, state)
    state = _run_ground_nsrt(push_kettle_on_burner4_nsrt, state)
    assert OnTop([kettle, burner4]).holds(state)
    assert TurnedOn([knob4]).holds(state)

    # Test reverse order: moving to and pushing the kettle, then moving to and
    # pushing knob4.
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_kettle_pre_push_nsrt, state)
    state = _run_ground_nsrt(push_kettle_on_burner4_nsrt, state)
    state = _run_ground_nsrt(move_to_knob4_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_knob4_nsrt, state)
    assert OnTop([kettle, burner4]).holds(state)
    assert TurnedOn([knob4]).holds(state)

    # Test light, kettle, then knob.
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_light_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_light_nsrt, state)
    state = _run_ground_nsrt(move_to_kettle_pre_push_nsrt, state)
    state = _run_ground_nsrt(push_kettle_on_burner4_nsrt, state)
    state = _run_ground_nsrt(move_to_knob4_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_knob4_nsrt, state)
    assert OnTop([kettle, burner4]).holds(state)
    assert TurnedOn([knob4]).holds(state)
    assert TurnedOn([light]).holds(state)

    # Test knob, light, kettle.
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_knob4_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_knob4_nsrt, state)
    state = _run_ground_nsrt(move_to_light_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_light_nsrt, state)
    state = _run_ground_nsrt(move_to_kettle_pre_push_nsrt, state)
    state = _run_ground_nsrt(push_kettle_on_burner4_nsrt, state)
    assert OnTop([kettle, burner4]).holds(state)
    assert TurnedOn([knob4]).holds(state)
    assert TurnedOn([light]).holds(state)

    # Test that we can't push the knob twice in a row, even if the first push
    # failed to turn on the burner.
    obs = env.reset("test", 0)
    state = env.state_info_to_state(obs["state_info"])
    assert state.allclose(init_state)
    state = _run_ground_nsrt(move_to_knob4_pre_on_nsrt, state)
    state = _run_ground_nsrt(turn_on_knob4_nsrt,
                             state,
                             override_params=np.array([-np.pi / 6]),
                             assert_add_effects=False,
                             assert_delete_effects=False)
    assert not TurnedOn([knob4]).holds(state)
    assert not all(p.holds(state) for p in turn_on_knob4_nsrt.preconditions)
