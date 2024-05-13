import pickle
from typing import Tuple
import os
import pandas as pd
import itertools
from glob import glob
import numpy as np
import matplotlib.pyplot as plt

def load_perf(
    env: str,
    approach: str,
    sampler_regressor_model: str,
    seed: int,
    num_train_tasks: int,
    env_size: int,
    results_dir: str,
) -> Tuple[int, float]:
    assert env in {'shelves2d', 'statue', 'donuts', 'wbox', 'pybullet_packing'}
    assert approach in {'nsrt_learning',
                        'search_pruning', 'gnn_action_policy', 'oracle'}
    assert sampler_regressor_model in {'neural_gaussian', 'diffusion'}
    assert seed >= 0
    assert num_train_tasks > 0

    results_prefix = os.path.join(results_dir, f'{env}-{approach}-{sampler_regressor_model}-{seed}-{num_train_tasks}-{env_size}')
    results_file = glob(os.path.join(results_prefix, '*.pkl'))[0]

    results = pickle.load(open(results_file, 'rb'))['results']
    return results['num_solved']/50*100, results['learning_time'], results['avg_num_samples']

# pd.DataFrame(columns=['env', 'approach', 'seed', 'num_solved', 'learning_time'])
data = []

for idx, ((env, env_size), (approach, sampler_regressor_model, name), seed) in enumerate(itertools.product(
    [('shelves2d', 5), ('statue', 4), ('donuts', 3)],
    [('search_pruning', 'diffusion', 'Feasibility\nClassification'), ('nsrt_learning', 'neural_gaussian', 'Myopic\nGaussian\nSamplers'), ('nsrt_learning', 'diffusion', 'Myopic\nDiffusion\nSamplers'), ('gnn_action_policy', 'diffusion', 'GNN\nPolicy')],
    range(8)
)):
    general_results_dir = 'experiment-results'
    if env == 'statue' and approach == 'search_pruning':
        general_results_dir = os.path.join(general_results_dir, 'old2')
    elif env == 'statue':
        general_results_dir = os.path.join(general_results_dir, 'old6')
    num_solved, learning_time, avg_num_samples = load_perf(
        env, approach, sampler_regressor_model, seed, 2000, env_size, general_results_dir
    )
    data.append([
        env, name, seed, num_solved, learning_time, avg_num_samples
    ])

df = pd.DataFrame(data, columns=['env', 'approach', 'seed', 'num_solved', 'learning_time', 'avg_num_samples'])

## CHATGPT CODE
def pr25(x):
    return np.percentile(x, 25)

def pr75(x):
    return np.percentile(x, 75)

approach_order = ['Myopic\nGaussian\nSamplers', 'Myopic\nDiffusion\nSamplers', 'GNN\nPolicy', 'Feasibility\nClassification']
# approach_order = ['Myopic\nGaussian\nSamplers', 'Myopic\nDiffusion\nSamplers', 'Feasibility\nClassification']

df = df.loc[df['approach'].isin(approach_order)]

df['approach'] = pd.Categorical(df['approach'],
    categories = approach_order,
    ordered=True
)

# Aggregate the data by environment and approach
col = 'num_solved'
# col = 'avg_num_samples'
# col = 'learning_time'
aggregated_df = df
print(df)
aggregated_df = aggregated_df.loc[aggregated_df[col] != 0]
aggregated_df = aggregated_df.loc[np.logical_not(np.isnan(df[col]))]
aggregated_df = aggregated_df.loc[np.logical_not(np.isinf(df[col]))]
aggregated_df = aggregated_df.groupby(
    ['env', 'approach'], as_index=False
)[col].agg(
    ['mean', 'std', pr25, pr75, 'min', 'max']
)
print(aggregated_df)

approach_colors = dict(zip(approach_order, 'ygrb'))
# approach_colors = dict(zip(approach_order, 'ygb'))

def generate_bar_graph(ax, env):
    for idx, ((key,), grp) in enumerate(aggregated_df.loc[aggregated_df['env'] == env].groupby(['approach'])):
        grp['mean'][np.isnan(grp['mean'])] = 0
        grp['pr25'][np.isnan(grp['pr25'])] = 0
        grp['pr75'][np.isnan(grp['pr75'])] = 0
        ax.barh(idx, grp['mean'], color=approach_colors[key], alpha=0.6)
        # print(grp['mean'])
        # print()
        # print(idx)
        # print()
        ax.errorbar(grp['mean'], idx, xerr=[grp['mean'] - grp['pr25'], grp['pr75'] - grp['mean']], color='black', markersize=8, lw=15, alpha=0.4)
        # ax.errorbar(grp['mean'], idx, xerr=np.array([grp['mean'] - grp['pr25'], grp['pr75'] - grp['mean']]), color='black', markersize=8, lw=22, alpha=0.7)
        ax.errorbar(grp['mean'], idx, xerr=[grp['mean'] - grp['min'], grp['max'] - grp['mean']], color='black', markersize=8)
        # ax.text(grp['mean'] - 1, idx, key, ha='right', va='center', fontsize=10)
    ax.set_yticks([])
    ax.set_title(env.title())
    # ax.set_xscale('log')

# Plotting
fig, axs = plt.subplots(figsize=(10, 4), ncols=3)

generate_bar_graph(axs[0], 'shelves2d')
generate_bar_graph(axs[1], 'statue')
generate_bar_graph(axs[2], 'donuts')

axs[1].set_xlabel('Average number of solved tasks (Higher is better)')
# axs[1].set_xlabel('Average number of samples (Lower is better)')

# Set y-axis ticks and labels
axs[0].set_yticks(range(len(approach_order)))
axs[0].set_yticklabels(approach_order)

plt.tight_layout()

plt.savefig('gathered-data')