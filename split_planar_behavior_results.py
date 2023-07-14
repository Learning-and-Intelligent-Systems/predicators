import pandas as pd
import numpy as np
import os
import pickle
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

seeds = list(range(10))
sampler_choice_list = ['random']#['oracle']#['generic', 'uniform', 'proportional', 'reconstruction', 'classifier', 'geometry', 'geometry+']
mix_choices = {'uniform', 'proportional', 'reconstruction', 'classifier', 'geometry', 'geometry+'}
nonlearned_choices = {'random', 'oracle'}

num_train_tasks_list = [50, 500, 5000, 50000]

args_list = list(product(seeds, num_train_tasks_list, sampler_choice_list))

### Joint domain
env = 'planar_behavior'
for seed, num_train_tasks, sampler_choice in args_list:
    try:
        if sampler_choice not in nonlearned_choices:
            fname = f'results_mix_notimeout/{sampler_choice}_{num_train_tasks}/{env}__sampler_learning{"_mix" if sampler_choice in mix_choices else ""}__{seed}________0.pkl'
        else:
            fname = f'results_mix_notimeout/{sampler_choice}_{num_train_tasks}/{env}__oracle__{seed}________None.pkl'
        with open(fname, 'rb') as f:
            r = pickle.load(f)
    except:
        print(seed, num_train_tasks, sampler_choice)
        continue
    new_r = {}
    new_r['config'] = r['config']
    new_r['config'].pybullet_robot_ee_orns = None
    new_r['config'].get_arg_specific_settings = None
    new_r['git_commit_hash'] = r['git_commit_hash']
    new_r['results'] = {}
    for subenv in ['bookshelf', 'cupboard', 'ballbin', 'boxtray', 'stickbasket']:
        for k, v in r['results'].items():
            if k.startswith(subenv):
                new_r['results'][k.replace(subenv + '_', '')] = v
        os.makedirs(f'results_mix_notimeout/joint_{sampler_choice}_{num_train_tasks}', exist_ok=True)
        with open(f'results_mix_notimeout/joint_{sampler_choice}_{num_train_tasks}/{subenv}__sampler_learning{"_mix" if sampler_choice in mix_choices else ""}__{seed}________0.pkl', 'wb') as f:
            pickle.dump(new_r, f)