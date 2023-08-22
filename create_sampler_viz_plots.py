import torch
import numpy as np

from predicators import utils
from predicators.envs.sampler_viz import SamplerVizEnv
from predicators.envs.sampler_viz2 import SamplerViz2Env
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


# utils.reset_config({"env": "sampler_viz", "render_state_dpi": 150})
# env = SamplerVizEnv()

# # fig, ax = plt.subplots(1, 4)
# t = env._get_tasks(1, [0], env._train_rng)
# state = t[0].init
# state.set(env._goal, "x", 10)
# state.set(env._goal, "y", 10)
# fig = env.render_state_plt(state, t, None)
# # plt.title("Start state")
# plt.savefig("results_viz/00sampler_viz.svg")
# # ax[0].imshow(img_0)
# # Nav
# navigate_objects = [env._robot, env._shelf]
# action = env._NavigateTo_policy(state, {}, navigate_objects, np.array([0.5, -1], dtype=np.float32))
# state = env.simulate(state, action)
# assert env._CanReach_holds(state, [env._shelf, env._robot])
# fig = env.render_state_plt(state, t, action)
# # plt.title("Shelf reachable")
# plt.savefig("results_viz/01sampler_viz.svg")
# # ax[1].imshow(img_1)
# # Pick
# pick_objects = navigate_objects
# action = env._PickShelf_policy(state, {}, pick_objects, np.array([], dtype=np.float32))
# state = env.simulate(state, action)
# assert env._Holding_holds(state, [env._shelf])
# fig = env.render_state_plt(state, t, action)
# # plt.title("Shelf held")
# plt.savefig("results_viz/02sampler_viz.svg")
# # ax[2].imshow(img_2)
# # Push
# push_objects = navigate_objects
# action = env._PushShelf_policy(state, {}, push_objects, np.array([5], dtype=np.float32))
# state = env.simulate(state, action)
# # assert env._OnGoal_holds(state, [env._shelf])
# fig = env.render_state_plt(state, t, action)
# # plt.title("On goal")
# plt.savefig("results_viz/03sampler_viz.svg")
# # ax[3].imshow(img_3)
# # plt.show()
# exit()


# for idx, prefix in enumerate(['', 'singlestep_', 'learned_', 'learned_singlestep_']):
#     nav_data = torch.load(f'data_sampler_viz/{prefix}NavigateTo_full_obs_5ktasks.pt')
#     t = env._get_tasks(1, [0], env._train_rng)
#     img_list = []
#     params_list = []
#     next_state_list = []
#     for state, params in zip(nav_data['state_dicts'][:100], nav_data['actions'][:100]):
#         navigate_objects = [env._robot, env._shelf]
#         action = env._NavigateTo_policy(state, {}, navigate_objects, params)
#         next_state = env.simulate(state, action)
#         next_state_list.append(next_state)
#         params_list.append(params)
#     fig = env.render_overlying_states_plt(next_state_list)
#     plt.savefig("results_viz/" + (prefix if prefix != '' else 'multistep_') + "sampler_viz.pdf")

utils.reset_config({"env": "sampler_viz2", "render_state_dpi": 150})
env = SamplerViz2Env()

# fig, ax = plt.subplots(1, 4)
t = env._get_tasks(1)
state = t[0].init
state.set(env._block_a, "pose_x", 0.9)
state.set(env._block_a, "pose_y", 1.2)
state.set(env._block_a, "yaw", np.pi / 2)
state.set(env._block_b, "pose_x", 1.1)
state.set(env._block_b, "pose_y", 0.5)
fig = env.render_state_plt(state, t, None)
# plt.title("Start state")
plt.savefig("results_viz/00sampler_viz2.svg")
# ax[0].imshow(img_0)
# Block a
place_objects = [env._block_a, env._container]
action = env._PlaceBlock_policy(state, {}, place_objects, np.array([0.2, 0.1, np.pi / 9], dtype=np.float32))
state = env.simulate(state, action)
# assert env._CanReach_holds(state, [env._shelf, env._robot])
fig = env.render_state_plt(state, t, action)
# plt.title("Shelf reachable")
plt.savefig("results_viz/01sampler_viz2.svg")
# ax[1].imshow(img_1)
# Block b
place_objects = [env._block_b, env._container]
action = env._PlaceBlock_policy(state, {}, place_objects, np.array([0.1, 0.6, -np.pi / 96], dtype=np.float32))
state = env.simulate(state, action)
# assert env._Holding_holds(state, [env._shelf])
fig = env.render_state_plt(state, t, action)
# plt.title("Shelf held")a
plt.savefig("results_viz/02sampler_viz2.svg")
# ax[2].imshow(img_2)
exit()

# for idx, prefix in enumerate(['', 'singlestep_', 'learned_', 'learned_singlestep_']):
#     place_data = torch.load(f'data_sampler_viz2/{prefix}PlaceBlock_full_obs_5ktasks.pt')
#     print(len(place_data['state_dicts']), place_data['actions'].shape, place_data['states'].shape)
#     t = env._get_tasks(1)
#     img_list = []
#     params_list = []
#     next_state_list = []
#     for state, params, state_features in zip(place_data['state_dicts'][:200], place_data['actions'][:200], place_data['states'][:200]):
#         if state_features[8] == env.block_a_width and state_features[9] == env.block_a_height:
#             place_objects = [env._block_a, env._container]
#             action = env._PlaceBlock_policy(state, {}, place_objects, params)
#             next_state = env.simulate(state, action)
#             next_state_list.append(next_state)
#             params_list.append(params)
#         else:
#             assert state_features[8] == env.block_b_width and state_features[9] == env.block_b_height
#     fig = env.render_overlying_states_plt(next_state_list)
#     plt.savefig("results_viz/" + (prefix if prefix != '' else 'multistep_') + "sampler_viz2.pdf")
