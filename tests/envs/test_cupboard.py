import tkinter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from predicators import utils
from predicators.envs.cupboard import CupboardEnv

matplotlib.use('TkAgg')


def test_init_sampler():
    utils.reset_config({"env": "cupboard", 'cupboard_against_wall': True})
    env = CupboardEnv()
    # t = env._get_tasks(1, [200], env._train_rng)
    for _ in range(10):
        t = env._get_tasks(1, [6], [0], env._train_rng)
        state = t[0].init
        env.render_state_plt(state, t[0])
        plt.show()
        plt.close()

def test_navigate_cupboard_action():
    utils.reset_config({"env": "cupboard", 'cupboard_against_wall': True})
    env = CupboardEnv()
    t = env._get_tasks(1, [0], [0], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    navigate_objects = [env._robot, env._cupboard]
    navigate_params = np.array([-0.4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='left')
    plt.show()

    navigate_params = np.array([1.4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    plt.show()

    navigate_params = np.array([0.5, -1], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='bottom')
    plt.show()

    navigate_params = np.array([0.5, 2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='top')
    plt.show()

    navigate_params = np.array([1.5, 2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([-0.5, 2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([1.5, -1], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([-0.5, -1], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([0, 0], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='fail')
    plt.show()


def test_navigate_cup_action():
    utils.reset_config({"env": "cupboard", 'cupboard_against_wall': True})
    env = CupboardEnv()
    t = env._get_tasks(1, [1], [0], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    cup = [b for b in state if b.is_instance(env._cup_type)][0]
    navigate_objects = [env._robot, cup]
    navigate_params = np.array([-4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='left')
    plt.show()

    navigate_params = np.array([5, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    plt.show()

    navigate_params = np.array([0.5, -2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='bottom')
    plt.show()

    navigate_params = np.array([0.5, 3], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='top')
    plt.show()

    navigate_params = np.array([5, 3], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([-4, 3], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([5, -2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([-4, -2], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='corner')
    plt.show()

    navigate_params = np.array([0, 0], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='fail')
    plt.show()


def test_navigate_pick_place():
    utils.reset_config({"env": "cupboard", 'cupboard_against_wall': True})
    env = CupboardEnv()
    t = env._get_tasks(1, [1], [0], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    cup = [b for b in state if b.is_instance(env._cup_type)][0]
    navigate_objects = [env._robot, cup]
    navigate_params = np.array([-4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='left')
    plt.show()
    assert env._CanReach_holds(s, [cup, env._robot])

    state = s
    pick_objects = [env._robot, cup]
    pick_params = np.array([1., 0.0], dtype=np.float32)
    a = env._PickCup_policy(state, {}, pick_objects, pick_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='pick')
    plt.show()
    assert env._Holding_holds(s, [cup])

    state = s
    navigate_params = np.array([-0.4, 0.5], dtype=np.float32)
    navigate_objects = [env._robot, env._cupboard]
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    plt.show()
    assert env._CanReach_holds(s, [env._cupboard, env._robot])

    state = s
    place_params = np.array([1.0], dtype=np.float32)
    place_objects = [env._robot, cup, env._cupboard]
    a = env._PlaceCupOnCupboard_policy(state, {}, place_objects, place_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    plt.show()
    assert env._OnCupboard_holds(s, [cup, env._cupboard])

if __name__ == '__main__':
    test_init_sampler()
    # test_navigate_cupboard_action()
    # test_navigate_cup_action()
    # test_navigate_pick_place()
