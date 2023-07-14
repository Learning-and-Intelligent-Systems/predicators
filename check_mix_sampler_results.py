import pandas as pd
import numpy as np
import pickle
from itertools import product
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})

ignore_errors = False

seeds = list(range(10))
# sampler_choice_list_stl = ['specialized', 'generic', 'uniform', 'proportional', 'reconstruction', 'classifier', 'geometry', 'geometry+']
# sampler_choice_list_stl = ['specialized', 'generic', 'geometry+']
sampler_choice_list_stl = []#['geometry+', 'geometry', 'reconstruction', 'proportional', 'classifier', 'uniform']
# sampler_choice_list_mtl = ['joint_generic', 'joint_uniform', 'joint_proportional', 'joint_reconstruction', 'joint_classifier', 'joint_geometry', 'joint_geometry+']
# sampler_choice_list_mtl = ['joint_generic', 'joint_geometry+']
sampler_choice_list_mtl = ['joint_geometry+', 'joint_geometry', 'joint_reconstruction', 'joint_proportional', 'joint_classifier', 'joint_uniform']
sampler_choice_list_nolearning = []# ['joint_oracle']#, 'joint_random']
sampler_choice_list = sampler_choice_list_stl + sampler_choice_list_mtl + sampler_choice_list_nolearning
mix_choices = {'uniform', 'proportional', 'reconstruction', 'classifier', 'geometry', 'geometry+'}
mix_choices |= {'joint_uniform', 'joint_proportional', 'joint_reconstruction', 'joint_classifier', 'joint_geometry', 'joint_geometry+'}

# choice_map = {'geometry+': 'Mixture','joint_geometry+': 'CD mixture', 'specialized': 'Specialized', 'generic': 'Generic',  'joint_generic': 'CD generic', 'joint_oracle': 'Uniform'}# 'Hand-crafted', 'joint_random': 'Uniform'}
choice_map = {'joint_geometry+': 'Geometry', 'joint_geometry': 'Distance', 'joint_reconstruction': 'Reconstruction', 'joint_proportional': 'Proportional (X)', 'joint_classifier': 'Classifier (X)', 'joint_uniform': 'Uniform'}

num_train_tasks_list = [50, 500, 5000, 50000]
results = ['solved', 'samples']
# custom_sort_dict = {elem: i for i, elem in enumerate(sampler_choice_list)}
custom_sort_dict = {elem: i for i, elem in enumerate(choice_map.values())}

idx = pd.MultiIndex.from_product((seeds, [choice_map[s] for s in sampler_choice_list]), names=('seed', 'Sampler choice'))
col = pd.MultiIndex.from_product((num_train_tasks_list, results), names=['# train tasks', 'result'])

args_list = list(product(seeds, num_train_tasks_list, sampler_choice_list))

results_df_map = {}
for env in ['bookshelf', 'cupboard', 'ballbin', 'boxtray', 'stickbasket']:
    results_df = pd.DataFrame(index=idx, columns=col)
    for seed, num_train_tasks, sampler_choice in args_list:
        try:
            with open(f'results_mix_notimeout/{sampler_choice}_{num_train_tasks}/{env}__sampler_learning{"_mix" if sampler_choice in mix_choices else ""}__{seed}________0.pkl', 'rb') as f:
                r = pickle.load(f)
        except:
            print('incomplete:', env, sampler_choice, num_train_tasks, seed)
            if not ignore_errors:
                raise
            continue
        solved = r['results']['num_solved']
        samples_solved = r['results']['avg_num_samples']
        samples_unsolved = r['config'].sesame_max_samples_total
        unsolved = r['results']['num_total'] - solved

        results_df.loc[(seed, choice_map[sampler_choice]), (num_train_tasks, 'solved')] = solved
        results_df.loc[(seed, choice_map[sampler_choice]), (num_train_tasks, 'samples')] = (samples_solved * solved + samples_unsolved * unsolved) / (solved + unsolved)

    results_df_map[env] = results_df

    results_df_mean = results_df.groupby(['Sampler choice']).mean()
    results_df_std = results_df.groupby(['Sampler choice']).std() / np.sqrt(len(seeds))
    results_df = pd.concat({'mean': results_df_mean, 'stderr': results_df_std}, names=['Stat'])   # Add level to multiindex (https://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex)
    results_df = results_df.swaplevel().sort_index(key=lambda x: x.map(custom_sort_dict))

    print(env)
    print(results_df)
    print()
    plt.figure()
    ax = plt.gca()
    mask = results_df_mean.columns.get_level_values(1) == 'samples'
    samples_df_mean = results_df_mean.loc[:, mask]
    samples_df_mean.columns = samples_df_mean.columns.droplevel(1)
    # samples_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_stl]].plot(ax=ax, use_index=True, linewidth=4)
    samples_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_mtl]].plot(ax=ax, use_index=True, style='--', linewidth=4)
    # samples_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_nolearning]].plot(ax=ax, use_index=True, style=':', linewidth=4)#, color="black")
    plt.xlabel('# train tasks')
    plt.ylabel('avg # samples')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results_mix_notimeout/{env}_numsamples_ablation.pdf')
    plt.close()

    plt.figure()
    ax = plt.gca()
    mask = results_df_mean.columns.get_level_values(1) == 'solved'
    solved_df_mean = results_df_mean.loc[:, mask]
    solved_df_mean.columns = solved_df_mean.columns.droplevel(1)
    # solved_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_stl]].plot(ax=ax, use_index=True, linewidth=4)
    solved_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_mtl]].plot(ax=ax, use_index=True, style='--', linewidth=4)
    # solved_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_nolearning]].plot(ax=ax, use_index=True, style=':', linewidth=4)#, color="black")
    plt.xlabel('# train tasks')
    plt.ylabel('avg # solved')
    plt.xscale('log')
    # plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results_mix_notimeout/{env}_numsolved_ablation.pdf')
    plt.close()

cross_domain_df = pd.concat(results_df_map, names=['env'])
cross_domain_df_mean = cross_domain_df.groupby(['Sampler choice']).mean()
cross_domain_df_std = cross_domain_df.groupby(['Sampler choice', 'seed']).mean().groupby(['Sampler choice']).std() / np.sqrt(len(seeds))
cross_domain_df = pd.concat({'mean': cross_domain_df_mean, 'stderr': cross_domain_df_std}, names=['Stat'])   # Add level to multiindex (https://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex)
cross_domain_df = cross_domain_df.swaplevel().sort_index(key=lambda x: x.map(custom_sort_dict))
print("overall")
print(cross_domain_df)
print()

plt.figure()
ax = plt.gca()
mask = cross_domain_df_mean.columns.get_level_values(1) == 'samples'
samples_df_mean = cross_domain_df_mean.loc[:, mask]
samples_df_mean.columns = samples_df_mean.columns.droplevel(1)
samples_df_std = cross_domain_df_std.loc[:, mask]
samples_df_std.columns = samples_df_std.columns.droplevel(1)
# cols = [choice_map[s] for s in sampler_choice_list_stl]
# samples_df_mean.transpose()[cols].plot(ax=ax, use_index=True, legend=False, linewidth=4)
cols = [choice_map[s] for s in sampler_choice_list_mtl]
samples_df_mean.transpose()[cols].plot(ax=ax, use_index=True, legend=False, linestyle='--', linewidth=4)
# cols = [choice_map[s] for s in sampler_choice_list_nolearning]
# samples_df_mean.transpose()[cols].plot(ax=ax, use_index=True, legend=False, linestyle=':', linewidth=4, color="black")
plt.xlabel('# train tasks')
plt.ylabel('avg # samples')
plt.xscale('log')
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
# plt.show()
plt.savefig(f'results_mix_notimeout/avg_numsamples_ablation.pdf')
plt.close()

plt.figure()
ax = plt.gca()
mask = cross_domain_df_mean.columns.get_level_values(1) == 'solved'
solved_df_mean = cross_domain_df_mean.loc[:, mask]
solved_df_mean.columns = solved_df_mean.columns.droplevel(1)
solved_df_mean = solved_df_mean / 50 * 100
solved_df_std = cross_domain_df_std.loc[:, mask]
solved_df_std.columns = solved_df_std.columns.droplevel(1)
solved_df_std = solved_df_std / 50 * 100
# cols = [choice_map[s] for s in sampler_choice_list_stl]
# solved_df_mean.transpose()[cols].plot(ax=ax, use_index=True, legend=False, linewidth=4)
cols = [choice_map[s] for s in sampler_choice_list_mtl]
solved_df_mean.transpose()[cols].plot(ax=ax, use_index=True, linestyle='--', legend=False, linewidth=4)
# cols = [choice_map[s] for s in sampler_choice_list_nolearning]
# solved_df_mean.transpose()[cols].plot(ax=ax, use_index=True, linestyle=':', legend=False, linewidth=4, color="black")
plt.xlabel('# train tasks')
plt.ylabel('avg % solved')
plt.xscale('log')
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout()
# plt.show()
plt.savefig(f'results_mix_notimeout/avg_numsolved_ablation.pdf')
plt.close()

# ### Joint domain
# sampler_choice_list.pop(sampler_choice_list.index('specialized'))
# sampler_choice_list.pop(sampler_choice_list.index('classifier'))
# args_list = list(product(seeds, num_train_tasks_list, sampler_choice_list))
# idx = pd.MultiIndex.from_product((seeds, sampler_choice_list), names=('seed', 'Sampler choice'))
# env = 'planar_behavior'

# results_df = pd.DataFrame(index=idx, columns=col)
# for seed, num_train_tasks, sampler_choice in args_list:
#     with open(f'results_mix_notimeout/{sampler_choice}_{num_train_tasks}/{env}__sampler_learning{"_mix" if sampler_choice in mix_choices else ""}__{seed}________0.pkl', 'rb') as f:
#         r = pickle.load(f)
#     results_df.loc[(seed, sampler_choice), (num_train_tasks, 'solved')] = r['results']['num_solved']
#     results_df.loc[(seed, sampler_choice), (num_train_tasks, 'samples')] = r['results']['avg_num_samples']
#     results_df.loc[(seed, sampler_choice), (num_train_tasks, 'time')] = r['results']['avg_suc_time']

# results_df_map[env] = results_df

# results_df_mean = results_df.groupby(['Sampler choice']).mean()
# results_df_std = results_df.groupby(['Sampler choice']).std() / np.sqrt(len(seeds))
# results_df = pd.concat({'mean': results_df_mean, 'stderr': results_df_std}, names=['Stat'])   # Add level to multiindex (https://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex)
# results_df = results_df.swaplevel().sort_index(key=lambda x: x.map(custom_sort_dict))

# print(env)
# print(results_df)
# print()
# mask = results_df_mean.columns.get_level_values(1) == 'samples'
# samples_df_mean = results_df_mean.loc[:, mask]
# samples_df_mean.columns = samples_df_mean.columns.droplevel(1)
# samples_df_mean.transpose().plot(use_index=True)
# plt.xlabel('# train tasks')
# plt.ylabel('avg # samples')
# plt.xscale('log')
# # plt.show()
# plt.savefig(f'results_mix_notimeout/{env}_numsamples_ablation.pdf')

# mask = results_df_mean.columns.get_level_values(1) == 'solved'
# solved_df_mean = results_df_mean.loc[:, mask]
# solved_df_mean.columns = solved_df_mean.columns.droplevel(1)
# solved_df_mean.transpose().plot(use_index=True)
# plt.xlabel('# train tasks')
# plt.ylabel('avg # solved')
# plt.xscale('log')
# # plt.show()
# plt.savefig(f'results_mix_notimeout/{env}_numsolved_ablation.pdf')

# mask = results_df_mean.columns.get_level_values(1) == 'time'
# time_df_mean = results_df_mean.loc[:, mask]
# time_df_mean.columns = time_df_mean.columns.droplevel(1)
# time_df_mean.transpose().plot(use_index=True)
# plt.xlabel('# train tasks')
# plt.ylabel('avg solve time')
# plt.xscale('log')
# plt.savefig(f'results_mix_notimeout/{env}_time_ablation.pdf')

# plt.close()
