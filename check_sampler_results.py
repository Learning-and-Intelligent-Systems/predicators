import pandas as pd
import numpy as np
import pickle
from itertools import product
import matplotlib.pyplot as plt


seeds = list(range(1))
full_state = [True, False]
# skeleton_state = [True, False]
# horizon = list(range(1, 6))
myopic = [True, False]
ebm = [True]
againstwall = [True, False]

# args_list = list(product(seeds, full_state, skeleton_state, horizon, ebm))
args_list = list(product(seeds, full_state, myopic, againstwall))

num_learning_iters = 100
columns = list(range(num_learning_iters))

# idx = pd.MultiIndex.from_product((ebm,full_state,skeleton_state,horizon,seeds), names=('ebm','full_state','skeleton_state','horizon','seed'))
idx = pd.MultiIndex.from_product((seeds, full_state, myopic, againstwall), names=('seeds', 'full_state', 'myopic', 'againstwall'))
solved_df = pd.DataFrame(index=idx, columns=columns)
samples_df = pd.DataFrame(index=idx, columns=columns)

# for seed, use_full_state, use_skeleton_state, sampler_horizon, use_ebm in args_list:
for seed, use_full_state, single_step, shelf_wall in args_list:
    for i in range(num_learning_iters):
        try:
            with open(f'results_sql/obs_{use_full_state}_myopic_{single_step}_againstwall_{shelf_wall}/bookshelf__sampler_learning__{seed}________{i}.pkl', 'rb') as f:
                r = pickle.load(f)
            solved_df.loc[(seed, use_full_state, single_step, shelf_wall), i] = r['results']['num_solved']
            samples_df.loc[(seed, use_full_state, single_step, shelf_wall), i] = r['results']['avg_num_samples']
        except:
            pass

print(solved_df.shape)
solved_df.dropna(axis=0, how='all', inplace=True)
print(solved_df.shape)
solved_df.dropna(axis=1, how='all', inplace=True)
print(solved_df.shape)
solved_df = solved_df.groupby(['full_state', 'myopic', 'againstwall']).mean()
samples_df.dropna(axis=0, how='all', inplace=True)
# samples_df.dropna(axis=1, how='all', inplace=True)
samples_df = samples_df.groupby(['full_state', 'myopic', 'againstwall']).mean()

maxs = solved_df.max(axis=1)
print(maxs)
sorted_idx = maxs.argsort()
# best_idx = sorted_idx[-5:][::-1]
best_idx = sorted_idx[::-1]

df_best_solved = solved_df.iloc[best_idx]
df_best_samples = samples_df.iloc[best_idx]

df_best_solved.transpose().plot()
plt.savefig('sampler_results_solved.pdf')

df_best_samples.transpose().plot()
plt.savefig('sampler_results_samples.pdf')