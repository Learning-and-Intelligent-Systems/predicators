import pandas as pd
import numpy as np
import pickle
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

num_seeds = 1#0
num_iters = 50
burnin = 50
n_tasks = 50
lifelong_approaches = ['retrain-balanced', 'finetune']
stages = ['train', 'test']
model = 'geometry+'
total_tasks = '_500'

model_map = {'geometry+': 'Mixture'}
approach_map = {'finetune': 'Finetune', 'retrain-balanced': 'Replay'}
stage_map = {'train': 'before training', 'test': 'after training'}
color_map = {
    ('train', 'retrain-balanced'): 'C0',
    ('test', 'retrain-balanced'): 'C0',
    ('train', 'finetune'): 'C1',
    ('test', 'finetune'): 'C1',
}

successes = {}
samples = {}
diff_successes = {}
diff_samples = {}
for approach in lifelong_approaches:
    for stage in stages:
        successes[stage, approach] = np.full((num_seeds, num_iters), np.nan)
        samples[stage, approach] = np.full((num_seeds, num_iters), np.nan)

        final_iter = num_iters
        for seed in range(num_seeds):
            for n_iter in range(num_iters):
                if stage == 'train':
                    fname = f'results_lifelong/{approach}_{model}_{n_tasks}{total_tasks}/planar_behavior__lifelong_sampler_learning_mix__{seed}________{n_iter}.pkl'
                else:
                    fname = f'results_forgetting/{approach}_{model}_{n_tasks}{total_tasks}/planar_behavior__lifelong_sampler_learning_mix__{seed}________{n_iter}.pkl'
                try: 
                    with open(fname, 'rb') as f:
                        r = pickle.load(f)
                    solved = r['results']['num_solved']
                    unsolved = r['results']['num_unsolved']
                    samples_solved = r['results']['avg_num_samples'] * solved if solved > 0 else 0
                    samples_unsolved = r['results']['avg_num_samples_failed'] * unsolved if unsolved > 0 else 0
                    successes[stage, approach][seed, n_iter] = solved
                    samples[stage, approach][seed, n_iter] = samples_solved + samples_unsolved
                except FileNotFoundError:
                    if n_iter < final_iter:
                        final_iter = n_iter
                    break
        print(f'{approach}, {stage}: {final_iter} iters done')
        successes[stage, approach][:, final_iter:] = np.nan
        samples[stage, approach][:, final_iter:] = np.nan

    diff_successes[approach] = successes['train', approach] - successes['test', approach]
    diff_samples[approach] = samples['train', approach] - samples['test', approach]

for approach in lifelong_approaches:
    for stage in stages:
        style = '-' if stage == 'train' else '--'
        model_legend = model_map[model]
        approach_legend = approach_map[approach] 
        stage_legend = stage_map[stage]
        color = color_map[stage, approach]
        plt.figure(0)
        plt.plot(successes[stage,approach].mean(axis=0), label=f'{model_legend}+{approach_legend}--{stage_legend}', linewidth=4, linestyle=style, color=color)
        plt.figure(1)
        plt.plot(samples[stage,approach].mean(axis=0).cumsum(axis=0), successes[stage,approach].mean(axis=0), label=f'{model_legend}+{approach_legend}--{stage_legend}', linewidth=4, linestyle=style, color=color)
        plt.figure(2)
        plt.plot(samples[stage,approach].mean(axis=0).cumsum(axis=0), successes[stage,approach].mean(axis=0).cumsum(axis=0), label=f'{model_legend}+{approach_legend}--{stage_legend}', linewidth=4, linestyle=style, color=color)
        # plt.errorbar(samples[stage, approach].mean(axis=0).cumsum(axis=0), successes[stage, approach].mean(axis=0).cumsum(axis=0), yerr=successes[stage, approach].cumsum(axis=1).std(axis=0) / np.sqrt(num_seeds))
        plt.figure(3)
        div = np.full(samples[stage, approach].shape[1], n_tasks)
        div[0] = burnin
        plt.plot(samples[stage,approach].mean(axis=0).cumsum(axis=0), samples[stage,approach].mean(axis=0) / div, label=f'{model_legend}+{approach_legend}--{stage_legend}', linewidth=4, linestyle=style, color=color)
        plt.figure(4)
        div = np.full(samples[stage, approach].shape[1], n_tasks)
        div[0] = burnin
        plt.plot(samples[stage,approach].mean(axis=0) / div, label=f'{model_legend}+{approach_legend}--{stage_legend}', linewidth=4, linestyle=style, color=color)
        plt.figure(5, figsize=(18,6))
        plt.plot(np.array([0]), np.array([0]), label=f'{model_legend}+{approach_legend} -- {stage_legend}', linewidth=4, linestyle=style, color=color)

    plt.figure(6)
    plt.plot(diff_successes[approach].mean(axis=0), label=f'{model_legend}+{approach_legend}', linewidth=4)
    plt.figure(7)
    plt.plot(diff_samples[approach].mean(axis=0) / div, label=f'{model_legend}+{approach_legend}', linewidth=4)


plt.figure(0)
# plt.legend()
plt.xlabel('# training iters')
plt.ylabel('# solved tasks per iter')
plt.tight_layout()
plt.savefig(f'results_forgetting/avg_solved_per_iter_{n_tasks}_rebuttal.pdf')
plt.close()
plt.figure(1)
# plt.legend()
plt.xlabel('# compute units')
plt.ylabel('# solved tasks per iter')
plt.tight_layout()
plt.savefig(f'results_forgetting/avg_solved_per_sample_{n_tasks}_rebuttal.pdf')
plt.close()
plt.figure(2)
# plt.legend(fontsize=12)
plt.xlabel('# samples')
plt.ylabel('# cumulative solved')
plt.tight_layout()
plt.savefig(f'results_forgetting/total_solved_{n_tasks}_rebuttal.pdf')
plt.close()
plt.figure(3)
# plt.legend()
plt.xlabel('# compute units')
plt.ylabel('# samples per attempted task')
plt.savefig(f'results_forgetting/avg_samples_{n_tasks}_rebuttal.pdf')
plt.tight_layout()
plt.close()
plt.figure(4)
# plt.legend()
plt.xlabel('# training iters')
plt.ylabel('# samples per attempted task')
plt.savefig(f'results_forgetting/avg_samples_per_iter_{n_tasks}_rebuttal.pdf')
plt.tight_layout()
plt.close()
plt.figure(5, figsize=(18,6))
plt.legend(ncol=2)
plt.savefig('results_forgetting/legend_forgetting_rebuttal.pdf')
plt.close()
plt.figure(6)
plt.xlabel('# training iters')
plt.ylabel('diff solved tasks per iter')
plt.tight_layout()
plt.savefig(f'results_forgetting/diff_solved_per_iter_{n_tasks}_rebuttal.pdf')
plt.close()
plt.figure(7)
plt.xlabel('# training iters')
plt.ylabel('diff samples tasks per iter')
plt.tight_layout()
plt.savefig(f'results_forgetting/diff_samples_per_iter_{n_tasks}_rebuttal.pdf')
plt.close()

print({stage_approach: samples[stage_approach][:, 1:].mean() / n_tasks for stage_approach in samples})
print({stage_approach: successes[stage_approach][:, 1:].mean() / n_tasks for stage_approach in successes})
for approach in lifelong_approaches:
    nan_mask = np.isnan(samples['test', approach])
    first_nan = nan_mask.argmax() if nan_mask.any() else nan_mask.shape[1]
    samples_test = samples['test', approach][:, :first_nan].sum()
    samples_train = samples['train', approach][:, :first_nan].sum()
    print(f'Sample increase ({approach}): {(samples_test - samples_train) / samples_train * 100:.2f}%')

    successes_test = successes['test', approach][:, :first_nan].sum()
    successes_train = successes['train', approach][:, :first_nan].sum()
    print(f'Success decrease ({approach}): {(successes_train - successes_test) / successes_train * 100:.2f}%')
    print()
