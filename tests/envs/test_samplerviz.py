import tkinter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from predicators import utils
from predicators.envs.sampler_viz import SamplerVizEnv

matplotlib.use('TkAgg')


def test_init_sampler():
    utils.reset_config({"env": "samplerviz"})
    env = SamplerVizEnv()
    for _ in range(10):
        t = env._get_tasks(1, [0], env._train_rng)
        state = t[0].init
        env.render_state_plt(state, t[0])
        plt.show()
        plt.close()

def test_navigate_shelf_action():
    utils.reset_config({"env": "samplerviz"})
    env = SamplerVizEnv()
    t = env._get_tasks(1, [0], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    navigate_objects = [env._robot, env._shelf]
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

def test_navigate_pick_place():
    utils.reset_config({"env": "samplerviz"})
    env = SamplerVizEnv()
    t = env._get_tasks(1, [0], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    shelf = env._shelf
    navigate_objects = [env._robot, shelf]
    navigate_params = np.array([0.5, -1], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='bottom')
    assert env._CanReach_holds(s, [shelf, env._robot])
    plt.show()

    state = s
    pick_objects = [env._robot, shelf]
    pick_params = np.array([], dtype=np.float32)
    a = env._PickShelf_policy(state, {}, pick_objects, pick_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='pick')
    assert env._Holding_holds(s, [shelf])
    plt.show()

    state = s
    push_params = np.array([5], dtype=np.float32)
    push_objects = [env._robot, env._shelf]
    a = env._PushShelf_policy(state, {}, push_objects, push_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    assert env._OnGoal_holds(s, [env._shelf])
    plt.show()

if __name__ == '__main__':
    # test_init_sampler()
    # test_navigate_shelf_action()
    test_navigate_pick_place()
