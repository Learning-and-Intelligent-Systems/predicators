import pandas as pd
import numpy as np
import pickle
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

seeds = list(range(10))
envs = ['bookshelf']#['boxtray']
diffusion_steps = [2, 4, 8, 16, 32, 64, 128]
results = ['samples', 'time']

idx = pd.MultiIndex.from_product((seeds, ['Mixture', 'Distilled', 'Uniform']), names=('seed', 'Sampler choice'))
col = pd.MultiIndex.from_product((diffusion_steps, results), names=['# steps', 'result'])
for e in envs:
    results_df = pd.DataFrame(index=idx, columns=col)
    # Mixture
    for s in seeds:
        for steps in diffusion_steps:
            with open(f'results_timing/diffusion_steps_{steps}/{e}__sampler_learning_mix__{s}________0.pkl', 'rb') as f:
                r = pickle.load(f)
            solved = r['results']['num_solved']
            samples_solved = r['results']['avg_num_samples']
            samples_unsolved = r['config'].sesame_max_samples_total
            unsolved = r['results']['num_total'] - solved
            time = r['results']['avg_suc_time']

            results_df.loc[(s, 'Mixture'), (steps, 'samples')] = (samples_solved * solved + samples_unsolved * unsolved) / (solved + unsolved)
            results_df.loc[(s, 'Mixture'), (steps, 'time')] = time
    # Distilled
    for s in seeds:
        for steps in diffusion_steps:
            with open(f'results_timing/distilled_diffusion_steps_{steps//2}/{e}__sampler_learning_mix__{s}________0.pkl', 'rb') as f:
                r = pickle.load(f)
            solved = r['results']['num_solved']
            samples_solved = r['results']['avg_num_samples']
            samples_unsolved = r['config'].sesame_max_samples_total
            unsolved = r['results']['num_total'] - solved
            time = r['results']['avg_suc_time']

            results_df.loc[(s, 'Distilled'), (steps, 'samples')] = (samples_solved * solved + samples_unsolved * unsolved) / (solved + unsolved)
            results_df.loc[(s, 'Distilled'), (steps, 'time')] = time

        with open(f'results_timing/{e}__oracle__{s}________None.pkl', 'rb') as f:
            r = pickle.load(f)
        solved = r['results']['num_solved']
        samples_solved = r['results']['avg_num_samples']
        samples_unsolved = r['config'].sesame_max_samples_total
        unsolved = r['results']['num_total'] - solved
        time = r['results']['avg_suc_time']

        results_df.loc[(s, 'Uniform'), pd.IndexSlice[:, 'samples']] = (samples_solved * solved + samples_unsolved * unsolved) / (solved + unsolved)
        results_df.loc[(s, 'Uniform'), pd.IndexSlice[:, 'time']] = time

    results_df_mean = results_df.groupby(['Sampler choice']).mean()
    results_df_std = results_df.groupby(['Sampler choice']).std() / np.sqrt(len(seeds))

    table_df = results_df_mean.applymap(lambda x: f"{x:.2f}±") + results_df_std.applymap(lambda x: f"{x:.2f}")
    print(table_df.to_markdown())