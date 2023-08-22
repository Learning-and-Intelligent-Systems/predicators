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
sampler_choice_list_stl = ['specialized', 'generic', 'geometry+']
# sampler_choice_list_stl = []#['geometry+', 'geometry', 'reconstruction', 'proportional', 'classifier', 'uniform']
# sampler_choice_list_mtl = ['joint_generic', 'joint_uniform', 'joint_proportional', 'joint_reconstruction', 'joint_classifier', 'joint_geometry', 'joint_geometry+']
sampler_choice_list_mtl = ['joint_generic', 'joint_geometry+']
# sampler_choice_list_mtl = ['joint_geometry+', 'joint_geometry', 'joint_reconstruction', 'joint_proportional', 'joint_classifier', 'joint_uniform']
sampler_choice_list_nolearning = ['joint_oracle']#, 'joint_random']
# sampler_choice_list_nolearning = []# ['joint_oracle']#, 'joint_random']
sampler_choice_list_baseline = ['nsrts']
sampler_choice_list = sampler_choice_list_stl + sampler_choice_list_mtl + sampler_choice_list_nolearning + sampler_choice_list_baseline
mix_choices = {'uniform', 'proportional', 'reconstruction', 'classifier', 'geometry', 'geometry+'}
mix_choices |= {'joint_uniform', 'joint_proportional', 'joint_reconstruction', 'joint_classifier', 'joint_geometry', 'joint_geometry+'}

choice_map = {'specialized': 'Specialized', 'generic': 'Generic',  'geometry+': 'Mixture', 'joint_generic': 'CD generic', 'joint_geometry+': 'CD mixture', 'joint_oracle': 'Uniform', 'nsrts': 'NSRTs [4]'}# 'Hand-crafted', 'joint_random': 'Uniform'}
# choice_map = {'joint_geometry+': 'Geometry', 'joint_geometry': 'Distance', 'joint_reconstruction': 'Reconstruction', 'joint_proportional': 'Proportional (X)', 'joint_classifier': 'Classifier (X)', 'joint_uniform': 'Uniform'}

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
            if sampler_choice == 'nsrts':
                fname = f'results_mix_notimeout/{sampler_choice}_{num_train_tasks}/{env}__nsrt_learning_sampler_only__{seed}________0.pkl'
            else:
                fname = f'results_mix_notimeout/{sampler_choice}_{num_train_tasks}/{env}__sampler_learning{"_mix" if sampler_choice in mix_choices else ""}__{seed}________0.pkl'
            with open(fname, 'rb') as f:
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

    plt.figure()
    ax = plt.gca()
    mask = results_df_mean.columns.get_level_values(1) == 'samples'
    samples_df_mean = results_df_mean.loc[:, mask]
    samples_df_mean.columns = samples_df_mean.columns.droplevel(1)
    samples_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_stl]].plot(ax=ax, use_index=True, linewidth=4)
    samples_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_mtl]].plot(ax=ax, use_index=True, style='--', linewidth=4)
    samples_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_nolearning]].plot(ax=ax, use_index=True, style=':', linewidth=4, color="black")
    samples_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_baseline]].plot(ax=ax, use_index=True, style='-.', linewidth=4)#, color="black")
    plt.xlabel('# train tasks')
    plt.ylabel('avg # samples')
    plt.xscale('log')
    plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results_mix_notimeout/{env}_numsamples_rebuttal.pdf')
    plt.close()

    plt.figure()
    ax = plt.gca()
    mask = results_df_mean.columns.get_level_values(1) == 'solved'
    solved_df_mean = results_df_mean.loc[:, mask]
    solved_df_mean.columns = solved_df_mean.columns.droplevel(1)
    solved_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_stl]].plot(ax=ax, use_index=True, linewidth=4)
    solved_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_mtl]].plot(ax=ax, use_index=True, style='--', linewidth=4)
    solved_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_nolearning]].plot(ax=ax, use_index=True, style=':', linewidth=4, color="black")
    solved_df_mean.transpose()[[choice_map[s] for s in sampler_choice_list_baseline]].plot(ax=ax, use_index=True, style='-.', linewidth=4)#, color="black")
    plt.xlabel('# train tasks')
    plt.ylabel('avg # solved')
    plt.xscale('log')
    # plt.legend(bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    # plt.show()
    plt.savefig(f'results_mix_notimeout/{env}_numsolved_rebuttal.pdf')
    plt.close()
    print(env)
    print(f"Samples: oracle -- {samples_df_mean.loc['Uniform',50000]}; mixture -- {samples_df_mean.loc['Mixture',50000]}, ({samples_df_mean.loc['Mixture',50000] / samples_df_mean.loc['Uniform',50000]})")
    print(f"Solved: oracle -- {solved_df_mean.loc['Uniform',50000]}; mixture -- {solved_df_mean.loc['Mixture',50000]}, ({solved_df_mean.loc['Mixture',50000] / solved_df_mean.loc['Uniform',50000]})")
    print(results_df)
    print()
    
exit()

cross_domain_df = pd.concat(results_df_map, names=['env'])
cross_domain_df_mean = cross_domain_df.groupby(['Sampler choice']).mean().sort_index(key=lambda x: x.map(custom_sort_dict))
cross_domain_df_std = cross_domain_df.groupby(['Sampler choice', 'seed']).mean().groupby(['Sampler choice']).std().sort_index(key=lambda x: x.map(custom_sort_dict)) / np.sqrt(len(seeds))
cross_domain_df = pd.concat({'mean': cross_domain_df_mean, 'stderr': cross_domain_df_std}, names=['Stat'])   # Add level to multiindex (https://stackoverflow.com/questions/14744068/prepend-a-level-to-a-pandas-multiindex)
cross_domain_df = cross_domain_df.swaplevel()
print("overall")
print(cross_domain_df)
print()

# fig = plt.figure()
# ax = plt.gca()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3]})
fig.subplots_adjust(hspace=0.05)  # adjust space between axes
mask = cross_domain_df_mean.columns.get_level_values(1) == 'samples'
samples_df_mean = cross_domain_df_mean.loc[:, mask]
samples_df_mean.columns = samples_df_mean.columns.droplevel(1)
samples_df_std = cross_domain_df_std.loc[:, mask]
samples_df_std.columns = samples_df_std.columns.droplevel(1)

latex_df = samples_df_mean.applymap(lambda x: f"${x:.2f}${{\\tiny$\\pm") + samples_df_std.applymap(lambda x: f"{x:.2f}$}}")
print(latex_df.to_latex(escape=False))

cols = [choice_map[s] for s in sampler_choice_list_stl]
samples_df_mean.transpose()[cols].plot(ax=ax1, use_index=True, legend=False, linewidth=4)
samples_df_mean.transpose()[cols].plot(ax=ax2, use_index=True, legend=False, linewidth=4)
# for choice in cols:
#     ax1.fill_between(samples_df_mean.columns, samples_df_mean.loc[choice] - samples_df_std.loc[choice], samples_df_mean.loc[choice] + samples_df_std.loc[choice], alpha=0.5)
#     ax2.fill_between(samples_df_mean.columns, samples_df_mean.loc[choice] - samples_df_std.loc[choice], samples_df_mean.loc[choice] + samples_df_std.loc[choice], alpha=0.5)
cols = [choice_map[s] for s in sampler_choice_list_mtl]
samples_df_mean.transpose()[cols].plot(ax=ax1, use_index=True, legend=False, linestyle='--', linewidth=4)
samples_df_mean.transpose()[cols].plot(ax=ax2, use_index=True, legend=False, linestyle='--', linewidth=4)
# for choice in cols:
#     ax1.fill_between(samples_df_mean.columns, samples_df_mean.loc[choice] - samples_df_std.loc[choice], samples_df_mean.loc[choice] + samples_df_std.loc[choice], alpha=0.5)
#     ax2.fill_between(samples_df_mean.columns, samples_df_mean.loc[choice] - samples_df_std.loc[choice], samples_df_mean.loc[choice] + samples_df_std.loc[choice], alpha=0.5)
cols = [choice_map[s] for s in sampler_choice_list_nolearning]
samples_df_mean.transpose()[cols].plot(ax=ax1, use_index=True, legend=False, linestyle=':', linewidth=4, color="black")
samples_df_mean.transpose()[cols].plot(ax=ax2, use_index=True, legend=False, linestyle=':', linewidth=4, color="black")
# for choice in cols:
#     ax1.fill_between(samples_df_mean.columns, samples_df_mean.loc[choice] - samples_df_std.loc[choice], samples_df_mean.loc[choice] + samples_df_std.loc[choice], alpha=0.5, color="black")
#     ax2.fill_between(samples_df_mean.columns, samples_df_mean.loc[choice] - samples_df_std.loc[choice], samples_df_mean.loc[choice] + samples_df_std.loc[choice], alpha=0.5, color="black")
cols = [choice_map[s] for s in sampler_choice_list_baseline]
bottom_ax2, top_ax2 = ax2.get_ylim()
samples_df_mean.transpose()[cols].plot(ax=ax1, use_index=True, legend=False, linestyle='-.', linewidth=4)#, color="black")
samples_df_mean.transpose()[cols].plot(ax=ax2, use_index=True, legend=False, linestyle='-.', linewidth=4)#, color="black")
# for choice in cols:
#     ax1.fill_between(samples_df_mean.columns, samples_df_mean.loc[choice] - samples_df_std.loc[choice], samples_df_mean.loc[choice] + samples_df_std.loc[choice], alpha=0.5)
#     ax2.fill_between(samples_df_mean.columns, samples_df_mean.loc[choice] - samples_df_std.loc[choice], samples_df_mean.loc[choice] + samples_df_std.loc[choice], alpha=0.5)
ax1.set_ylim(7500, 7500 + (top_ax2 - bottom_ax2) / 3)
ax1.set_yticks([8000])
ax2.set_ylim(bottom_ax2, top_ax2)
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.tick_params(axis='x', which='both',
                bottom=False) # turn off major & minor ticks on the bottom
ax2.xaxis.tick_bottom()
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
plt.xscale('log')
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel('# train tasks')
plt.ylabel('avg # samples', labelpad=12.0)
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout(h_pad=0.01)
# plt.show()
plt.savefig(f'results_mix_notimeout/avg_numsamples_rebuttal.pdf')
plt.close()

# plt.figure()
# ax = plt.gca()
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
fig.subplots_adjust(hspace=0.05)  # adjust space between axes
mask = cross_domain_df_mean.columns.get_level_values(1) == 'solved'
solved_df_mean = cross_domain_df_mean.loc[:, mask]
solved_df_mean.columns = solved_df_mean.columns.droplevel(1)
solved_df_mean = solved_df_mean / 50 * 100
solved_df_std = cross_domain_df_std.loc[:, mask]
solved_df_std.columns = solved_df_std.columns.droplevel(1)
solved_df_std = solved_df_std / 50 * 100

latex_df = solved_df_mean.applymap(lambda x: f"${x:.2f}${{\\tiny$\\pm") + solved_df_std.applymap(lambda x: f"{x:.2f}$}}")
print(latex_df.to_latex(escape=False))

cols = [choice_map[s] for s in sampler_choice_list_stl]
solved_df_mean.transpose()[cols].plot(ax=ax1, use_index=True, legend=False, linewidth=4)
solved_df_mean.transpose()[cols].plot(ax=ax2, use_index=True, legend=False, linewidth=4)
# for choice in cols:
#     ax1.fill_between(solved_df_mean.columns, solved_df_mean.loc[choice] - solved_df_std.loc[choice], solved_df_mean.loc[choice] + solved_df_std.loc[choice], alpha=0.5)
#     ax2.fill_between(solved_df_mean.columns, solved_df_mean.loc[choice] - solved_df_std.loc[choice], solved_df_mean.loc[choice] + solved_df_std.loc[choice], alpha=0.5)
cols = [choice_map[s] for s in sampler_choice_list_mtl]
solved_df_mean.transpose()[cols].plot(ax=ax1, use_index=True, linestyle='--', legend=False, linewidth=4)
solved_df_mean.transpose()[cols].plot(ax=ax2, use_index=True, linestyle='--', legend=False, linewidth=4)
# for choice in cols:
#     ax1.fill_between(solved_df_mean.columns, solved_df_mean.loc[choice] - solved_df_std.loc[choice], solved_df_mean.loc[choice] + solved_df_std.loc[choice], alpha=0.5)
#     ax2.fill_between(solved_df_mean.columns, solved_df_mean.loc[choice] - solved_df_std.loc[choice], solved_df_mean.loc[choice] + solved_df_std.loc[choice], alpha=0.5)
cols = [choice_map[s] for s in sampler_choice_list_nolearning]
solved_df_mean.transpose()[cols].plot(ax=ax1, use_index=True, linestyle=':', legend=False, linewidth=4, color="black")
solved_df_mean.transpose()[cols].plot(ax=ax2, use_index=True, linestyle=':', legend=False, linewidth=4, color="black")
# for choice in cols:
#     ax1.fill_between(solved_df_mean.columns, solved_df_mean.loc[choice] - solved_df_std.loc[choice], solved_df_mean.loc[choice] + solved_df_std.loc[choice], alpha=0.5, color="black")
#     ax2.fill_between(solved_df_mean.columns, solved_df_mean.loc[choice] - solved_df_std.loc[choice], solved_df_mean.loc[choice] + solved_df_std.loc[choice], alpha=0.5, color="black")
bottom_ax1, top_ax1 = ax1.get_ylim()
cols = [choice_map[s] for s in sampler_choice_list_baseline]
solved_df_mean.transpose()[cols].plot(ax=ax1, use_index=True, linestyle='-.', legend=False, linewidth=4)
solved_df_mean.transpose()[cols].plot(ax=ax2, use_index=True, linestyle='-.', legend=False, linewidth=4)
# for choice in cols:
#     ax1.fill_between(solved_df_mean.columns, solved_df_mean.loc[choice] - solved_df_std.loc[choice], solved_df_mean.loc[choice] + solved_df_std.loc[choice], alpha=0.5)
#     ax2.fill_between(solved_df_mean.columns, solved_df_mean.loc[choice] - solved_df_std.loc[choice], solved_df_mean.loc[choice] + solved_df_std.loc[choice], alpha=0.5)

ax1.set_ylim(bottom_ax1, top_ax1)
ax2.set_ylim(20, 20 + (top_ax1 - bottom_ax1) / 3)
ax2.set_yticks([20])
ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.tick_params(axis='x', which='both',
                bottom=False) # turn off major & minor ticks on the bottom
ax2.xaxis.tick_bottom()
d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)
plt.xscale('log')
fig.add_subplot(111, frameon=False)
plt.tick_params(labelcolor='none', which='both', top=False, bottom=False, left=False, right=False)
# plt.xlabel('# train tasks')
plt.ylabel('avg % solved', labelpad=-20.0)
# plt.legend(bbox_to_anchor=(1.0, 1.0))
plt.tight_layout(h_pad=0.01)
# plt.show()
plt.savefig(f'results_mix_notimeout/avg_numsolved_rebuttal.pdf')
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
# plt.savefig(f'results_mix_notimeout/{env}_numsamples_rebuttal.pdf')

# mask = results_df_mean.columns.get_level_values(1) == 'solved'
# solved_df_mean = results_df_mean.loc[:, mask]
# solved_df_mean.columns = solved_df_mean.columns.droplevel(1)
# solved_df_mean.transpose().plot(use_index=True)
# plt.xlabel('# train tasks')
# plt.ylabel('avg # solved')
# plt.xscale('log')
# # plt.show()
# plt.savefig(f'results_mix_notimeout/{env}_numsolved_rebuttal.pdf')

# mask = results_df_mean.columns.get_level_values(1) == 'time'
# time_df_mean = results_df_mean.loc[:, mask]
# time_df_mean.columns = time_df_mean.columns.droplevel(1)
# time_df_mean.transpose().plot(use_index=True)
# plt.xlabel('# train tasks')
# plt.ylabel('avg solve time')
# plt.xscale('log')
# plt.savefig(f'results_mix_notimeout/{env}_time_rebuttal.pdf')

# plt.close()
