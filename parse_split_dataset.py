import torch
import numpy as np

nav = []
pick = []
place = []
push = []

# for i in range(50):
#     nav.append(torch.load(f'data_cupboard/data/50ksplit/{i:02}_NavigateTo_full_obs_50ktasks.pt'))
#     pick.append(torch.load(f'data_cupboard/data/50ksplit/{i:02}_PickCup_full_obs_50ktasks.pt'))
#     place.append(torch.load(f'data_cupboard/data/50ksplit/{i:02}_PlaceCupOnCupboard_full_obs_50ktasks.pt'))

# nav = {k: np.concatenate([d[k] for d in nav]) for k in nav[0] if k != 'next_states'}
# pick = {k: np.concatenate([d[k] for d in pick]) for k in pick[0] if k != 'next_states'}
# place = {k: np.concatenate([d[k] for d in place]) for k in place[0] if k != 'next_states'}
# torch.save(nav, 'data_cupboard/data/NavigateTo_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(pick, 'data_cupboard/data/PickCup_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(place, 'data_cupboard/data/PlaceCupOnCupboard_full_obs_50ktasks.pt', pickle_protocol=4)

# for i in range(50):
#     nav.append(torch.load(f'data_ballbin/data/50ksplit/{i:02}_NavigateTo_full_obs_50ktasks.pt'))
#     pick.append(torch.load(f'data_ballbin/data/50ksplit/{i:02}_PickBall_full_obs_50ktasks.pt'))
#     place.append(torch.load(f'data_ballbin/data/50ksplit/{i:02}_PlaceBallOnBin_full_obs_50ktasks.pt'))

# nav = {k: np.concatenate([d[k] for d in nav]) for k in nav[0] if k != 'next_states'}
# pick = {k: np.concatenate([d[k] for d in pick]) for k in pick[0] if k != 'next_states'}
# place = {k: np.concatenate([d[k] for d in place]) for k in place[0] if k != 'next_states'}
# torch.save(nav, 'data_ballbin/data/NavigateTo_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(pick, 'data_ballbin/data/PickBall_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(place, 'data_ballbin/data/PlaceBallOnBin_full_obs_50ktasks.pt', pickle_protocol=4)

# for i in range(50):
#     nav.append(torch.load(f'data_stickbasket/data/50ksplit/{i:02}_NavigateTo_full_obs_50ktasks.pt'))
#     pick.append(torch.load(f'data_stickbasket/data/50ksplit/{i:02}_PickStick_full_obs_50ktasks.pt'))
#     place.append(torch.load(f'data_stickbasket/data/50ksplit/{i:02}_PlaceStickOnBasket_full_obs_50ktasks.pt'))

# nav = {k: np.concatenate([d[k] for d in nav]) for k in nav[0] if k != 'next_states'}
# pick = {k: np.concatenate([d[k] for d in pick]) for k in pick[0] if k != 'next_states'}
# place = {k: np.concatenate([d[k] for d in place]) for k in place[0] if k != 'next_states'}
# torch.save(nav, 'data_stickbasket/data/NavigateTo_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(pick, 'data_stickbasket/data/PickStick_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(place, 'data_stickbasket/data/PlaceStickOnBasket_full_obs_50ktasks.pt', pickle_protocol=4)

# for i in range(50):
#     nav.append(torch.load(f'data_boxtray/data/50ksplit/{i:02}_NavigateTo_full_obs_50ktasks.pt'))
#     pick.append(torch.load(f'data_boxtray/data/50ksplit/{i:02}_PickBox_full_obs_50ktasks.pt'))
#     place.append(torch.load(f'data_boxtray/data/50ksplit/{i:02}_PlaceBoxOnTray_full_obs_50ktasks.pt'))

# nav = {k: np.concatenate([d[k] for d in nav]) for k in nav[0] if k != 'next_states'}
# pick = {k: np.concatenate([d[k] for d in pick]) for k in pick[0] if k != 'next_states'}
# place = {k: np.concatenate([d[k] for d in place]) for k in place[0] if k != 'next_states'}
# torch.save(nav, 'data_boxtray/data/NavigateTo_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(pick, 'data_boxtray/data/PickBox_full_obs_50ktasks.pt', pickle_protocol=4)
# torch.save(place, 'data_boxtray/data/PlaceBoxOnTray_full_obs_50ktasks.pt', pickle_protocol=4)

# for i in range(20):
#     # nav.append(torch.load(f'data_sampler_viz/5ksplit/{i:02}_singlestep_NavigateTo_full_obs_5ktasks.pt'))
#     # nav.append(torch.load(f'data_sampler_viz/5ksplit/{i:02}_NavigateTo_full_obs_5ktasks.pt'))
#     tmp = torch.load(f'data_sampler_viz/5ksplit/{i:02}_NavigateTo_full_obs_5ktasks.pt')
#     tmp = {k: np.array(v) for k, v in tmp.items()}
#     mask = tmp['actions'][:, 1] < 0
#     tmp2 = {k: v[mask] for k, v in tmp.items() if k != 'state_dicts'}
#     tmp2['state_dicts'] = [tmp['state_dicts'][i] for i in range(len(tmp['state_dicts'])) if tmp['actions'][i, 1] < 0]
#     nav.append(tmp2)
#     pick.append(torch.load(f'data_sampler_viz/5ksplit/{i:02}_PickShelf_full_obs_5ktasks.pt'))
#     push.append(torch.load(f'data_sampler_viz/5ksplit/{i:02}_PushShelf_full_obs_5ktasks.pt'))

# nav_tmp = {k: np.concatenate([d[k] for d in nav]) for k in nav[0] if k != 'next_states' and k!= 'state_dicts'}
# pick_tmp = {k: np.concatenate([d[k] for d in pick]) for k in pick[0] if k != 'next_states' and k!= 'state_dicts'}
# push_tmp = {k: np.concatenate([d[k] for d in push]) for k in push[0] if k != 'next_states' and k!= 'state_dicts'}

# nav_tmp['state_dicts'] = []
# for split in nav:
#     nav_tmp['state_dicts'] += split['state_dicts']
# pick_tmp['state_dicts'] = []
# for split in pick:
#     pick_tmp['state_dicts'] += split['state_dicts']
# push_tmp['state_dicts'] = []
# for split in push:
#     push_tmp['state_dicts'] += split['state_dicts']

# # torch.save(nav_tmp, 'data_sampler_viz/singlestep_NavigateTo_full_obs_5ktasks.pt', pickle_protocol=4)
# torch.save(nav_tmp, 'data_sampler_viz/NavigateTo_full_obs_5ktasks.pt', pickle_protocol=4)
# torch.save(pick_tmp, 'data_sampler_viz/PickShelf_full_obs_5ktasks.pt', pickle_protocol=4)
# torch.save(push_tmp, 'data_sampler_viz/PushShelf_full_obs_5ktasks.pt', pickle_protocol=4)

# place = []
# for i in range(20):
#     # place.append(torch.load(f'data_sampler_viz2/5ksplit/{i:02}_PlaceBlock_full_obs_5ktasks.pt'))
#     place.append(torch.load(f'data_sampler_viz2/5ksplit/{i:02}_singlestep_PlaceBlock_full_obs_5ktasks.pt'))
# place_tmp = {k: np.concatenate([d[k] for d in place]) for k in place[0] if k != 'next_states' and k != 'state_dicts'}
# place_tmp['state_dicts'] = []
# for split in place:
#     place_tmp['state_dicts'] += split['state_dicts']
# # torch.save(place_tmp, 'data_sampler_viz2/PlaceBlock_full_obs_5ktasks.pt', pickle_protocol=4)
# torch.save(place_tmp, 'data_sampler_viz2/singlestep_PlaceBlock_full_obs_5ktasks.pt', pickle_protocol=4)


# for i in range(50):
#     print(i)
#     # Navigation
#     try:
#         tmp_d = torch.load(f'data_img/data_obs/5ksplit/{i:02}_NavigateTo_full_obs_5ktasks_random.pt')
#         del tmp_d
#         continue
#     except:
#         print(i)
#         continue
#     del tmp_d['next_states']
#     if i == 0:
#         nav = {k: np.empty((tmp_d[k].shape[0] * 21, *tmp_d[k].shape[1:])) for k in tmp_d}
#         cnt = {k: 0 for k in tmp_d}
#     for k in tmp_d:
#         nav[k][cnt[k] : cnt[k] + tmp_d[k].shape[0]] = tmp_d[k]
#         cnt[k] += tmp_d[k].shape[0]
#     del tmp_d


#     # # Picking
#     # tmp_d = torch.load(f'data_img/data_obs/5ksplit/{i:02}_PickBook_full_obs_5ktasks.pt')
#     # del tmp_d['next_states']
#     # if i == 0:
#     #     pick = {k: np.empty((tmp_d[k].shape[0] * 21, *tmp_d[k].shape[1:])) for k in tmp_d}
#     #     cnt = {k: 0 for k in tmp_d}
#     # for k in tmp_d:
#     #     pick[k][cnt[k] : cnt[k] + tmp_d[k].shape[0]] = tmp_d[k]
#     #     cnt[k] += tmp_d[k].shape[0]


#     # Placing
#     # tmp_d = torch.load(f'data_img/data_obs/5ksplit/{i:02}_PlaceBookOnShelf_full_obs_5ktasks.pt')
#     # del tmp_d['next_states']
#     # place_datasets.append(tmp_d)

# for k in nav:
#     nav[k] = nav[k][:cnt[k]]
# print('truncated data, saving...')
# torch.save(nav, 'data_img/data_obs/NavigateTo_full_obs_5ktasks.pt', pickle_protocol=4)
# # for k in pick:
# #     pick[k] = pick[k][:cnt[k]]
# # pick = {k: np.concatenate([d[k] for d in pick_datasets]) for k in pick_datasets[0] if k != 'next_states'}
# # torch.save(pick, 'data_obs/PickBook_full_obs_50ktasks_random.pt', pickle_protocol=4)
# # place = {k: np.concatenate([d[k] for d in place_datasets]) for k in place_datasets[0] if k != 'next_states'}
# # torch.save(place, 'data_obs/PlaceBookOnShelf_full_obs_50ktasks_random.pt', pickle_protocol=4)

# env = 'bookshelf'
# for env in ['ballbin', 'boxtray', 'cupboard', 'stickbasket']:
for env in ['boxtray']:
    trajectories = []
    task_idx = []
    for i in range(50):
        print(i)
        tmp_d = torch.load(f'data_{env}/data/50ksplit/{i:02}_50ktasks_trajectories.pt')
        trajectories += tmp_d['trajectories']
        task_idx += tmp_d['train_task_idx']
    torch.save({'trajectories': trajectories, 'train_task_idx': np.array(task_idx)}, f'data_{env}/data/50ktasks_trajectories.pt')
