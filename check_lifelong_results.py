import pandas as pd
import numpy as np
import pickle
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

num_seeds = 10
num_iters = 50
burnin = 50
num_tasks_per_cycle_list = [50]
# lifelong_approaches = ['distill','retrain-scratch', 'finetune', 'retrain-balanced']
lifelong_approaches = ['retrain-balanced', 'retrain-scratch', 'finetune']
model_types = ['geometry+']#, 'specialized', 'generic']
total_tasks = '_500'

model_map = {'geometry+': 'Mixture', 'specialized': 'Specialized', 'generic': 'Generic'}
approach_map = {'finetune': 'Finetune', 'retrain-balanced': 'Replay', 'retrain-scratch': 'Retrain'}

for n_tasks  in num_tasks_per_cycle_list:
    successes = {}
    samples = {}
    for model in model_types:
        for approach in lifelong_approaches:
            successes[model, approach] = np.full((num_seeds, num_iters), np.nan)
            samples[model, approach] = np.full((num_seeds, num_iters), np.nan)

            final_iter = num_iters
            for seed in range(num_seeds):
                for n_iter in range(num_iters):
                    if model in ['specialized', 'generic']:
                        fname = f'results_lifelong/{approach}_{model}_{n_tasks}{total_tasks}/planar_behavior__lifelong_sampler_learning__{seed}________{n_iter}.pkl'
                    else:
                        fname = f'results_lifelong/{approach}_{model}_{n_tasks}{total_tasks}/planar_behavior__lifelong_sampler_learning_mix__{seed}________{n_iter}.pkl'
                    try: 
                        with open(fname, 'rb') as f:
                            r = pickle.load(f)
                        solved = r['results']['num_solved']
                        unsolved = r['results']['num_unsolved']
                        samples_solved = r['results']['avg_num_samples'] * solved if solved > 0 else 0
                        samples_unsolved = r['results']['avg_num_samples_failed'] * unsolved if unsolved > 0 else 0
                        successes[model, approach][seed, n_iter] = solved
                        samples[model, approach][seed, n_iter] = samples_solved + samples_unsolved
                    except FileNotFoundError:
                        if n_iter < final_iter:
                            final_iter = n_iter
                        break
            print(f'{approach}, {model}: {final_iter} iters done')
            successes[model, approach][:, final_iter:] = np.nan
            samples[model, approach][:, final_iter:] = np.nan

    for model in model_types:
        for approach in lifelong_approaches:
            # if model == 'geometry+' and approach == 'retrain-scratch':
            #     continue
            if model != 'geometry+' and approach == 'retrain-balanced':
                continue
            model_legend = model_map[model]
            approach_legend = approach_map[approach]
            plt.figure(0)
            plt.plot(successes[model,approach].mean(axis=0), label=f'{model_legend}+{approach_legend}', linewidth=4)
            plt.figure(1)
            plt.plot(samples[model,approach].mean(axis=0).cumsum(axis=0), successes[model,approach].mean(axis=0), label=f'{model_legend}+{approach_legend}', linewidth=4)
            plt.figure(2)
            plt.plot(samples[model,approach].mean(axis=0).cumsum(axis=0), successes[model,approach].mean(axis=0).cumsum(axis=0), label=f'{model_legend}+{approach_legend}', linewidth=4)
            # plt.errorbar(samples[model, approach].mean(axis=0).cumsum(axis=0), successes[model, approach].mean(axis=0).cumsum(axis=0), yerr=successes[model, approach].cumsum(axis=1).std(axis=0) / np.sqrt(num_seeds))
            plt.figure(3)
            div = np.full(samples[model, approach].shape[1], n_tasks)
            div[0] = burnin
            plt.plot(samples[model,approach].mean(axis=0).cumsum(axis=0), samples[model,approach].mean(axis=0) / div, label=f'{model_legend}+{approach_legend}', linewidth=4)
            plt.figure(4, figsize=(18,6))
            plt.plot(np.array([0]), np.array([0]), label=f'{model_legend}+{approach_legend}', linewidth=4)

    plt.figure(0)
    plt.legend()
    plt.xlabel('# training iters')
    plt.ylabel('# solved tasks per iter')
    plt.tight_layout()
    plt.savefig(f'results_lifelong/avg_solved_per_iter_{n_tasks}_ablation.pdf')
    plt.close()
    plt.figure(1)
    plt.legend()
    plt.xlabel('# compute units')
    plt.ylabel('# solved tasks per iter')
    plt.tight_layout()
    plt.savefig(f'results_lifelong/avg_solved_per_sample_{n_tasks}_ablation.pdf')
    plt.close()
    plt.figure(2)
    # plt.legend(fontsize=12)
    plt.xlabel('# samples')
    plt.ylabel('# cumulative solved')
    plt.tight_layout()
    plt.savefig(f'results_lifelong/total_solved_{n_tasks}_ablation.pdf')
    plt.close()
    plt.figure(3)
    plt.legend()
    plt.xlabel('# compute units')
    plt.ylabel('# samples per attempted task')
    plt.savefig(f'results_lifelong/avg_samples_{n_tasks}_ablation.pdf')
    plt.tight_layout()
    plt.close()
    plt.figure(4, figsize=(18,6))
    plt.legend(ncol=3)
    plt.savefig('results_lifelong/legend_lifelong_ablation.pdf')
    plt.close()
    print({model_approach: samples[model_approach][:, 1:].mean() / n_tasks for model_approach in samples})
