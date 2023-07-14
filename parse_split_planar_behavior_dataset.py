import torch
import numpy as np

objects = ['Book', 'Shelf', 'Cup', 'Cupboard', 'Stick', 'Basket', 'Box', 'Tray', 'Ball', 'Bin']
assert len(objects) == 10
pickables = ['Book', 'Cup', 'Stick', 'Box', 'Ball']
assert len(pickables) == 5
placeables = ['Shelf', 'Cupboard', 'Basket', 'Tray', 'Bin']

nav = []
for obj in objects:
    print(f'loading NavigateTo{obj} data')
    nav.append(torch.load(f'data_planar_behavior/data/NavigateTo{obj}_full_obs_50ktasks.pt'))
nav = {k: np.concatenate([d[k] for d in nav]) for k in nav[0] if k != 'next_states'}
torch.save(nav, 'data_planar_behavior/data/NavigateTo_full_obs_50ktasks.pt')
print(f'saving NavigateTo data')

pick = []
for obj in pickables:
    print(f'lodaing Pick{obj} data')
    pick.append(torch.load(f'data_planar_behavior/data/Pick{obj}_full_obs_50ktasks.pt'))
pick = {k: np.concatenate([d[k] for d in pick]) for k in pick[0] if k != 'next_states'}
torch.save(pick, 'data_planar_behavior/data/Pick_full_obs_50ktasks.pt')
print(f'saving Pick data')

place = []
for obj1, obj2 in zip(pickables, placeables):
    print(f'loading Place{obj1}On{obj2} data')
    place.append(torch.load(f'data_planar_behavior/data/Place{obj1}On{obj2}_full_obs_50ktasks.pt'))
place = {k: np.concatenate([d[k] for d in place]) for k in place[0] if k != 'next_states'}
torch.save(place, 'data_planar_behavior/data/Place_full_obs_50ktasks.pt')
print(f'saving Place data')