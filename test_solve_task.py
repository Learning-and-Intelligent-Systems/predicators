#!/usr/bin/env python3

import argparse
from minigrid.wrappers import *
from mini_behavior.window import Window
from mini_behavior.utils.save import get_step, save_demo
from mini_behavior.grid import GridDimension
import numpy as np
from PIL import Image
from mini_behavior.states import *

# Size in pixels of a tile in the full-scale human view
TILE_PIXELS = 32
show_furniture = False


def redraw(img):
    if not args.agent_view:
        img = env.render()
    window.no_closeup()
    window.set_inventory(env)
    window.show_img(img)
    image_path = "output_image.jpeg"
    window.save_img(image_path)

def render_furniture():
    global show_furniture
    show_furniture = not show_furniture

    if show_furniture:
        img = np.copy(env.furniture_view)

        # i, j = env.agent.cur_pos
        i, j = env.agent_pos
        ymin = j * TILE_PIXELS
        ymax = (j + 1) * TILE_PIXELS
        xmin = i * TILE_PIXELS
        xmax = (i + 1) * TILE_PIXELS

        img[ymin:ymax, xmin:xmax, :] = GridDimension.render_agent(
            img[ymin:ymax, xmin:xmax, :], env.agent_dir)
        img = env.render_furniture_states(img)

        window.show_img(img)
    else:
        obs = env.gen_obs()
        redraw(obs)


def show_states():
    imgs = env.render_states()
    window.show_closeup(imgs)


def reset():
    if args.seed != -1:
        env.seed(args.seed)

    obs = env.reset()

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)


def load():
    if args.seed != -1:
        env.seed(args.seed)

    env.reset()
    obs = env.load_state(args.load)

    if hasattr(env, 'mission'):
        print('Mission: %s' % env.mission)
        window.set_caption(env.mission)

    redraw(obs)

def get_lifted_state(env):
    objs = env.objs
    obj_instances = {}
    for obj_type, obj_list in objs.items():
        for obj in obj_list:
            obj_instances[obj.name] = obj
    ground_atoms = []
    try:
        for k, o in obj_instances.items():
            for pred_name, pred in o.states.items():
                if isinstance(o.states[pred_name], AbsoluteObjectState):
                    if o.states[pred_name].get_value(env):
                        ground_atoms.append(pred_name+'('+k+')')
                elif isinstance(o.states[pred_name], AbilityState):
                    if o.states[pred_name].get_value(env):
                        ground_atoms.append(pred_name+'('+k+')')
                elif isinstance(o.states[pred_name], ObjectProperty):
                    if o.states[pred_name].get_value(env):
                        ground_atoms.append(pred_name+'('+k+')')
                elif isinstance(o.states[pred_name], RelativeObjectState):
                    for k2, o2 in obj_instances.items():
                        if o.check_rel_state(env, o2, pred_name):
                            ground_atoms.append(pred_name+'('+k+','+k2+')')
    except:
        import ipdb; ipdb.set_trace()
    return ground_atoms
                         


def step(action):
    prev_obs = env.gen_obs()

    prev_state = get_lifted_state(env)
    obs, reward, done, terminated, info = env.step(action)
    state = get_lifted_state(env)

    print('step=%s, reward=%.2f' % (env.step_count, reward))
    for atom in state:
        print(atom)

    if args.save:
        all_steps[env.step_count] = (prev_obs, prev_state, action, obs, state)

    if done:
        print('done!')
        if args.save:
            save_demo(all_steps, args.env, env.episode)
        reset()
    else:
        redraw(obs)


def switch_dim(dim):
    env.switch_dim(dim)
    print(f'switching to dim: {env.render_dim}')
    obs = env.gen_obs()
    redraw(obs)


def key_handler_cartesian(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'backspace':
        reset()
        return
    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return
    # Spacebar
    if event.key == ' ':
        render_furniture()
        return
    if event.key == 'pageup':
        step('choose')
        return
    if event.key == 'enter':
        env.save_state()
        return
    if event.key == 'pagedown':
        show_states()
        return
    if event.key == '0':
        switch_dim(None)
        return
    if event.key == '1':
        switch_dim(0)
        return
    if event.key == '2':
        switch_dim(1)
        return
    if event.key == '3':
        switch_dim(2)
        return

def key_handler_primitive(event):
    print('pressed', event.key)
    if event.key == 'escape':
        window.close()
        return
    if event.key == 'left':
        step(env.actions.left)
        return
    if event.key == 'right':
        step(env.actions.right)
        return
    if event.key == 'up':
        step(env.actions.forward)
        return
    if event.key == '0':
        step(env.actions.pickup_0)
        return
    if event.key == '1':
        step(env.actions.pickup_1)
        return
    if event.key == '2':
        step(env.actions.pickup_2)
        return
    if event.key == '3':
        step(env.actions.drop_0)
        return
    if event.key == '4':
        step(env.actions.drop_1)
        return
    if event.key == '5':
        step(env.actions.drop_2)
        return
    if event.key == 't':
        step(env.actions.toggle)
        return
    if event.key == 'o':
        step(env.actions.open)
        return
    if event.key == 'c':
        step(env.actions.close)
        return
    if event.key == 'k':
        step(env.actions.cook)
        return
    if event.key == '6':
        step(env.actions.slice)
        return
    if event.key == 'i':
        step(env.actions.drop_in)
        return
    if event.key == 'pagedown':
        show_states()
        return


parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    help="gym environment to load",
    default='MiniGrid-InstallingAPrinter-8x8-N2-v0'
)
parser.add_argument(
    "--seed",
    type=int,
    help="random seed to generate the environment with",
    default=-1
)
parser.add_argument(
    "--tile_size",
    type=int,
    help="size at which to render tiles",
    default=32
)
parser.add_argument(
    '--agent_view',
    default=False,
    help="draw the agent sees (partially observable view)",
    action='store_true'
)
# NEW
parser.add_argument(
    "--save",
    default=False,
    help="whether or not to save the demo_16"
)
# NEW
parser.add_argument(
    "--load",
    default=None,
    help="path to load state from"
)

args = parser.parse_args()
# ###
# all_envs = [env_id for env_id in gym.envs.registry.keys() if "MiniGrid-" in env_id]
# print(args)
# print(all_envs)
# quit()
# ###

env = gym.make(args.env)
env.teleop_mode()
if args.save:
    # We do not support save for cartesian action space
    assert env.mode == "primitive"

all_steps = {}

if args.agent_view:
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

window = Window('mini_behavior - ' + args.env)
if env.mode == "cartesian":
    window.reg_key_handler(key_handler_cartesian)
elif env.mode == "primitive":
    window.reg_key_handler(key_handler_primitive)

if args.load is None:
    reset()
else:
    load()

# Blocking event loop
window.show(block=True)