import tkinter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from predicators import utils
from predicators.envs.bookshelf import BookshelfEnv

matplotlib.use('TkAgg')


def test_init_sampler():
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
    t = env._get_tasks(1, [200], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()


def test_navigate_bookshelf_action():
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
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


def test_navigate_book_action():
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
    t = env._get_tasks(1, [1], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    book = [b for b in state if b.is_instance(env._book_type)][0]
    navigate_objects = [env._robot, book]
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
    utils.reset_config({"env": "bookshelf"})
    env = BookshelfEnv()
    t = env._get_tasks(1, [1], env._train_rng)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    book = [b for b in state if b.is_instance(env._book_type)][0]
    navigate_objects = [env._robot, book]
    navigate_params = np.array([-4, 0.5], dtype=np.float32)
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='left')
    assert env._CanReach_holds(s, [book, env._robot])
    plt.show()

    state = s
    pick_objects = [env._robot, book]
    pick_params = np.array([0.25, 0.0], dtype=np.float32)
    a = env._PickBook_policy(state, {}, pick_objects, pick_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='pick')
    assert env._Holding_holds(s, [book])
    plt.show()

    state = s
    navigate_params = np.array([-0.4, 0.5], dtype=np.float32)
    navigate_objects = [env._robot, env._shelf]
    a = env._NavigateTo_policy(state, {}, navigate_objects, navigate_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    assert env._CanReach_holds(s, [env._shelf, env._robot])
    plt.show()

    state = s
    place_params = np.array([1.0], dtype=np.float32)
    place_objects = [env._robot, book, env._shelf]
    a = env._PlaceBookOnShelf_policy(state, {}, place_objects, place_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    assert env._OnShelf_holds(s, [book, env._shelf])
    plt.show()


if __name__ == '__main__':
    test_init_sampler()
    test_navigate_bookshelf_action()
    test_navigate_book_action()
    test_navigate_pick_place()
