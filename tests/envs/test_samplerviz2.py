import tkinter

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from predicators import utils
from predicators.envs.sampler_viz2 import SamplerViz2Env

matplotlib.use('TkAgg')


def test_init_sampler():
    utils.reset_config({"env": "sampler_viz2"})
    env = SamplerViz2Env()
    for _ in range(10):
        t = env._get_tasks(1)
        state = t[0].init
        env.render_state_plt(state, t[0])
        plt.show()
        plt.close()

def test_place_blocks_action():
    utils.reset_config({"env": "sampler_viz2"})
    env = SamplerViz2Env()
    t = env._get_tasks(1)
    state = t[0].init
    env.render_state_plt(state, t[0])
    plt.show()

    place_objects = [env._block_a, env._container]
    # place_params = np.array([0.1, 0.1, np.pi/3], dtype=np.float32)
    place_params = np.array([0.5480968, 0.01418366, 0.9155914], dtype=np.float32)
    a = env._PlaceBlock_policy(state, {}, place_objects, place_params)
    s = env.simulate(state, a)
    env.render_state_plt(s, t[0], caption='right')
    plt.show()
    assert env._InContainer_holds(s, place_objects)

    place_objects = [env._block_b, env._container]
    place_params = np.array([0.3, 0.1, -np.pi/4], dtype=np.float32)
    a = env._PlaceBlock_policy(s, {}, place_objects, place_params)
    s = env.simulate(s, a)
    env.render_state_plt(s, t[0], caption='right')
    plt.show()
    assert env._InContainer_holds(s, place_objects)

    assert t[0].goal_holds(s)


if __name__ == "__main__":
    # test_init_sampler()
    test_place_blocks_action()