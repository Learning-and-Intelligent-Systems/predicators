
# Data ingestion
from glob import glob
import os
from typing import Iterable, Optional, Tuple
from matplotlib import pyplot as plt
import numpy as np
import numpy.typing as npt
import pickle as pkl
import matplotlib
import matplotlib.ticker as mtick
matplotlib.use("tkagg")


# Constants
colors = ['#FF0000', '#FFE400', '#00FFD5', '#CD00FF', '#FF9C00', '#9EFF00', '#0B00FF']
bar_width = 10
bar_spacing = 5
group_spacing = 10
std_width = 2,
min_max_width = 7
fontsize = 15
fontsize = 15
fontsize = 15

num_test_tasks = 50

main_methods_colors = [
    ('search_pruning-diffusion', colors[0]),
    ('nsrt_learning-diffusion', colors[1]),
    ('nsrt_learning-neural_gaussian', colors[2]),
    ('gnn_action_policy-diffusion', colors[3]),
]

ablation_colors = [
    ('search_pruning-diffusion', colors[0]),
    ('search_pruning_no_backjumping-diffusion', colors[4]),
    ('search_pruning_last_action_negative-diffusion', colors[5]),
    ('search_pruning_all_actions_negative-diffusion', colors[6]),
]

envs_size_ranges = [
    ('shelves2d', 'Shelves', 'Number of Shelves', 5, 10),
    ('donuts', 'Donut', 'Number of Toppings', 3, 6),
    ('statue', 'Statue', 'Grid Size', 4, 8)
]

def parse_data(dirs) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if type(dirs) != list:
        file, = glob(os.path.join('experiment-results', dirs, '*.pkl'))
        data = pkl.load(open(file, 'rb'))['results']
        return data['num_solved']/num_test_tasks*100, np.array([
            data.get(f'PER_TASK_task{t}_solve_time', float('inf')) for t in range(num_test_tasks)
        ])
    num_solved, avg_solve_time = zip(*map(parse_data, dirs))
    return np.stack(list(num_solved)), np.stack(list(avg_solve_time))

def setup_xticks(ax: plt.Axes, name: str = ''):
    ax.tick_params(
        axis='x',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
    )
    if name:
        ax.set_xlabel("\n\n" + name, fontsize=fontsize, va='center')

def clear_yticks(ax: plt.Axes):
    ax.tick_params(
        axis='y',
        which='both',
        left=False,
        right=False,
        labelleft=False,
    )

def calculate_center_simple(i: int):
    return i*bar_width + (i-1) * bar_spacing

def calculate_center_complex(i: int, j: int, n: int):
    return calculate_center_simple(n) * j + group_spacing * (j-1) + calculate_center_simple(i)

def add_bar_graph(ax: plt.Axes, i: int, data: npt.NDArray[np.float32], color):
    center = calculate_center_simple(i)
    if len(data) == 0:
        ax.bar(center, 0, width=bar_width)
        return
    ax.bar(
        center,
        data.mean(),
        yerr=[[data.mean() - data.min()], [data.max() - data.mean()]],
        width=bar_width,
        color=color,
        error_kw = {'elinewidth': 1, 'capsize': min_max_width},
    )
    ax.bar(
        center,
        min(data.std(), 100 - data.mean() + data.std()/2),
        std_width,
        data.mean() - data.std()/2,
        color=(0, 0, 0, 0.7),
    )

# Generalization plot
fig, axs = plt.subplots(2, len(envs_size_ranges), figsize=(15, 10))
for ax_idx, (env_label, env_name, env_x_label, min_env_size, max_env_size), num_solved_ax, solve_time_ax in zip(range(len(envs_size_ranges)), envs_size_ranges, axs[0], axs[1], strict=True):
    num_solved_ax.set_xticks([calculate_center_simple(i) for i in range(0, max_env_size-min_env_size+1)], range(min_env_size, max_env_size+1))
    solve_time_ax.set_xticks([calculate_center_simple(i) for i in range(0, max_env_size-min_env_size+1)], range(min_env_size, max_env_size+1))

    num_solved_ax.tick_params(labelsize=fontsize)
    solve_time_ax.tick_params(labelsize=fontsize)

    solve_time_ax.sharex(num_solved_ax)
    num_solved_ax.set_title(env_name, fontsize=fontsize)
    solve_time_ax.set_xlabel(env_x_label, fontsize=fontsize)

    if ax_idx == 0:
        num_solved_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        solve_time_ax.set_yscale('log')
        solve_time_ax.yaxis.set_major_formatter(mtick.PercentFormatter(symbol='s'))

        num_solved_ax.set_ylabel('% Solved tasks (Higher is better)', fontsize=fontsize)
        solve_time_ax.set_ylabel('Solve Time (Lower is better)', fontsize=fontsize)
    else:
        num_solved_ax.sharey(axs[0, 0])
        solve_time_ax.sharey(axs[1, 0])
        clear_yticks(num_solved_ax)
        clear_yticks(solve_time_ax)


    data_num_solved, data_solve_time = parse_data([[[
        f"{env_label}-{method}-{seed}-2000-{env_size}"
    for seed in range(8)] for env_size in range(min_env_size, max_env_size+1)] for method, _ in main_methods_colors])

    for (method, method_color), num_solved_row, solve_time_row in zip(
        main_methods_colors, data_num_solved, data_solve_time, strict=True
    ):
        if (num_solved_row == 0).all():
            continue
        num_solved_ax.plot(
            [calculate_center_simple(i) for i in range(len(num_solved_row))],
            [num_solved[np.isfinite(num_solved)].mean() if len(np.isfinite(num_solved)) else 0 for num_solved in num_solved_row],
            color=method_color,
            alpha=0.5,
            lw=2,
        )
        for i, num_solved in enumerate(num_solved_row):
            num_solved_ax.bar(
                calculate_center_simple(i),
                0,
                1,
                num_solved[np.isfinite(num_solved)].mean(),
                yerr = [[num_solved[np.isfinite(num_solved)].std()/2]]*2,
                capsize=5,
                ecolor=method_color,
                color=(0, 0, 0, 0),
            )
        solve_time_ax.plot(
            [calculate_center_simple(i) for i in range(len(solve_time_row))],
            [solve_time[np.isfinite(solve_time)].mean() if len(np.isfinite(solve_time)) else 0 for solve_time in solve_time_row],
            color=method_color,
            alpha=0.7,
            lw=2,
        )
        for i, solve_time in enumerate(solve_time_row):
            solve_time_ax.bar(
                calculate_center_simple(i),
                0,
                1,
                solve_time[np.isfinite(solve_time)].mean(),
                yerr = [[solve_time[np.isfinite(solve_time)].std()/2]]*2,
                capsize=5,
                ecolor=method_color,
            )

plt.tight_layout()
fig.align_ylabels(axs[:, 0])
fig.savefig("generalization.png", dpi=300)

# Efficiency plot
fig, axs = plt.subplots(2, len(envs_size_ranges), figsize=(15, 10))
for ax_idx, (env_label, env_name, _, env_size, _), num_solved_ax, solve_time_ax in zip(range(len(envs_size_ranges)), envs_size_ranges, axs[0], axs[1], strict=True):
    num_solved_ax.set_xticks([calculate_center_simple(i) for i in range(4)], [500, 1000, 1500, 2000])
    solve_time_ax.set_xticks([calculate_center_simple(i) for i in range(4)], [500, 1000, 1500, 2000])

    num_solved_ax.tick_params(labelsize=fontsize)
    solve_time_ax.tick_params(labelsize=fontsize)

    solve_time_ax.sharex(num_solved_ax)
    num_solved_ax.set_title(env_name, fontsize=fontsize)
    solve_time_ax.set_xlabel('Number of Datapoints', fontsize=fontsize)

    if ax_idx == 0:
        num_solved_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        solve_time_ax.set_yscale('log')
        solve_time_ax.yaxis.set_major_formatter(mtick.PercentFormatter(symbol='s'))

        num_solved_ax.set_ylabel('% Solved tasks (Higher is better)', fontsize=fontsize)
        solve_time_ax.set_ylabel('Solve Time (Lower is better)', fontsize=fontsize)
    else:
        num_solved_ax.sharey(axs[0, 0])
        solve_time_ax.sharey(axs[1, 0])
        clear_yticks(num_solved_ax)
        clear_yticks(solve_time_ax)


    data_num_solved, data_solve_time = parse_data([[[
        f"{env_label}-{method}-{seed}-{num_datapoints}-{env_size}"
    for seed in range(8)] for num_datapoints in [500, 1000, 1500, 2000]] for method, _ in main_methods_colors])

    for (method, method_color), num_solved_row, solve_time_row in zip(
        main_methods_colors, data_num_solved, data_solve_time, strict=True
    ):
        if (num_solved_row == 0).all():
            continue
        num_solved_ax.plot(
            [calculate_center_simple(i) for i in range(len(num_solved_row))],
            [num_solved[np.isfinite(num_solved)].mean() if len(np.isfinite(num_solved)) else 0 for num_solved in num_solved_row],
            color=method_color,
            alpha=0.5,
            lw=2,
        )
        for i, num_solved in enumerate(num_solved_row):
            num_solved_ax.bar(
                calculate_center_simple(i),
                0,
                1,
                num_solved[np.isfinite(num_solved)].mean(),
                yerr = [[num_solved[np.isfinite(num_solved)].std()/2]]*2,
                capsize=5,
                ecolor=method_color,
                color=(0, 0, 0, 0),
            )
        solve_time_ax.plot(
            [calculate_center_simple(i) for i in range(len(solve_time_row))],
            [solve_time[np.isfinite(solve_time)].mean() if len(np.isfinite(solve_time)) else 0 for solve_time in solve_time_row],
            color=method_color,
            alpha=0.7,
            lw=2,
        )
        for i, solve_time in enumerate(solve_time_row):
            solve_time_ax.bar(
                calculate_center_simple(i),
                0,
                1,
                solve_time[np.isfinite(solve_time)].mean() if len(solve_time[np.isfinite(solve_time)]) else 0,
                yerr = [[solve_time[np.isfinite(solve_time)].std()/2]]*2,
                capsize=5,
                ecolor=method_color,
            )

plt.tight_layout()
fig.align_ylabels(axs[:, 0])
fig.savefig("efficiency.png", dpi=300)

# PyBullet Env Plot
fig, (num_solved_ax, solve_time_ax) = plt.subplots(2, 1, figsize=(5, 10))
data_num_solved, data_solve_time = parse_data([[
    f"pybullet_packing-{method}-diffusion-{seed}-2000-5"
for seed in range(8)] for method in ['search_pruning', 'nsrt_learning']])
for i, (method, method_color), num_solved, solve_time in zip(
        range(2), [('search_pruning', colors[0]), ('nsrt_learning', colors[1])], data_num_solved, data_solve_time, strict=True
    ):
        add_bar_graph(num_solved_ax, i, num_solved[np.isfinite(num_solved)], color=method_color)
        add_bar_graph(solve_time_ax, i, solve_time[np.isfinite(solve_time)], color=method_color)

setup_xticks(num_solved_ax)
setup_xticks(solve_time_ax, '')

num_solved_ax.tick_params(labelsize=fontsize)
solve_time_ax.tick_params(labelsize=fontsize)

solve_time_ax.sharex(num_solved_ax)
num_solved_ax.set_title('Packing', fontsize=fontsize)

num_solved_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
solve_time_ax.set_yscale('log')
solve_time_ax.yaxis.set_major_formatter(mtick.PercentFormatter(symbol='s'))

# num_solved_ax.set_ylabel('% Solved tasks (Higher is better)', fontsize=fontsize)
# solve_time_ax.set_ylabel('Solve Time (Lower is better)', fontsize=fontsize)

plt.tight_layout()
#   fig.align_ylabels([num_solved_ax, solve_time_ax])
fig.savefig("pybullet.png", dpi=300)

# Ablations Plot
fig, (num_solved_ax, solve_time_ax) = plt.subplots(2, 1, figsize=(10, 10))
data_num_solved, data_solve_time = parse_data([[
    f"shelves2d-{method}-{seed}-{4000 if 'action' in method else 2000}-5"
for seed in range(8)] for method, _ in ablation_colors])
for i, (method, method_color), num_solved, solve_time in zip(
        range(len(ablation_colors)), ablation_colors, data_num_solved, data_solve_time, strict=True
    ):
        add_bar_graph(num_solved_ax, i, num_solved[np.isfinite(num_solved)], color=method_color)
        add_bar_graph(solve_time_ax, i, solve_time[np.isfinite(solve_time)], color=method_color)

setup_xticks(num_solved_ax)
setup_xticks(solve_time_ax, '')

num_solved_ax.tick_params(labelsize=fontsize)
solve_time_ax.tick_params(labelsize=fontsize)

solve_time_ax.sharex(num_solved_ax)
num_solved_ax.set_title('Shelves', fontsize=fontsize)

num_solved_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
solve_time_ax.set_yscale('log')
solve_time_ax.yaxis.set_major_formatter(mtick.PercentFormatter(symbol='s'))

# num_solved_ax.set_ylabel('% Solved tasks (Higher is better)', fontsize=fontsize)
# solve_time_ax.set_ylabel('Solve Time (Lower is better)', fontsize=fontsize)

plt.tight_layout()
# fig.align_ylabels([num_solved_ax, solve_time_ax])
fig.savefig("ablations.png", dpi=300)

# for
# for ax_idx, (env_label, env_name, _, env_size, _), num_solved_ax, solve_time_ax in zip(range(len(envs_size_ranges)), envs_size_ranges, axs[0], axs[1], strict=True):
#     num_solved_ax.set_xticks([calculate_center_simple(i) for i in range(4)], [500, 1000, 1500, 2000])
#     solve_time_ax.set_xticks([calculate_center_simple(i) for i in range(4)], [500, 1000, 1500, 2000])

#     num_solved_ax.tick_params(labelsize=fontsize)
#     solve_time_ax.tick_params(labelsize=fontsize)

#     solve_time_ax.sharex(num_solved_ax)
#     num_solved_ax.set_title(env_name, fontsize=fontsize)
#     solve_time_ax.set_xlabel('Number of Datapoints', fontsize=fontsize)

#     if ax_idx == 0:
#         num_solved_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
#         solve_time_ax.set_yscale('log')
solve_time_ax.yaxis.set_major_formatter(mtick.PercentFormatter(symbol='s'))

#         num_solved_ax.set_ylabel('% Solved tasks (Higher is better)', fontsize=fontsize)
#         solve_time_ax.set_ylabel('Solve Time (Lower is better)', fontsize=fontsize)
#     else:
#         num_solved_ax.sharey(axs[0, 0])
#         solve_time_ax.sharey(axs[1, 0])
#         clear_yticks(num_solved_ax)
#         clear_yticks(solve_time_ax)


#     generalization_data_num_solved, generalization_data_solve_time = parse_data([[[
#         f"{env_label}-{method}-{seed}-{num_datapoints}-{env_size}"
#     for seed in range(8)] for num_datapoints in [500, 1000, 1500, 2000]] for method, _ in main_methods_colors])

#     for (method, method_color), num_solved_row, solve_time_row in zip(
#         main_methods_colors, generalization_data_num_solved, generalization_data_solve_time, strict=True
#     ):
#         num_solved_ax.plot(
#             [calculate_center_simple(i) for i in range(len(num_solved_row))],
#             [num_solved[np.isfinite(num_solved)].mean() if len(np.isfinite(num_solved)) else 0 for num_solved in num_solved_row],
#             color=method_color,
#             alpha=0.5,
#             lw=2,
#         )
#         for i, num_solved in enumerate(num_solved_row):
#             num_solved_ax.bar(
#                 calculate_center_simple(i),
#                 0,
#                 1,
#                 num_solved[np.isfinite(num_solved)].mean(),
#                 yerr = [[num_solved[np.isfinite(num_solved)].std()/2]]*2,
#                 capsize=5,
#                 ecolor=method_color,
#                 color=(0, 0, 0, 0),
#             )
#         solve_time_ax.plot(
#             [calculate_center_simple(i) for i in range(len(solve_time_row))],
#             [solve_time[np.isfinite(solve_time)].mean() if len(np.isfinite(solve_time)) else 0 for solve_time in solve_time_row],
#             color=method_color,
#             alpha=0.7,
#             lw=2,
#         )
#         for i, solve_time in enumerate(solve_time_row):
#             solve_time_ax.bar(
#                 calculate_center_simple(i),
#                 0,
#                 1,
#                 solve_time[np.isfinite(solve_time)].mean(),
#                 yerr = [[solve_time[np.isfinite(solve_time)].std()/2]]*2,
#                 capsize=5,
#                 ecolor=method_color,
#             )



# fig, axs = plt.subplots(2, len(envs_size_ranges), figsize=(15, 10))
# for ax_idx, (env_label, env_name, env_size, _), num_solved_ax, solve_time_ax in zip(range(len(envs_size_ranges)), envs_size_ranges, axs[0], axs[1], strict=True):
#     setup_xticks(num_solved_ax, env_name)
#     setup_xticks(solve_time_ax, env_name)
#     if ax_idx == 0:
#         num_solved_ax.tick_params(labelsize=fontsize)
#         num_solved_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
#         solve_time_ax.tick_params(labelsize=fontsize)
#         solve_time_ax.set_yscale('log')
solve_time_ax.yaxis.set_major_formatter(mtick.PercentFormatter(symbol='s'))

#         num_solved_ax.set_ylabel('% Solved tasks (Higher is better)', fontsize=fontsize)
#         solve_time_ax.set_ylabel('Solve Time (Lower is better)', fontsize=fontsize)
#     else:
#         num_solved_ax.sharey(axs[0, 0])
#         solve_time_ax.sharey(axs[1, 0])
#         clear_yticks(num_solved_ax)
#         clear_yticks(solve_time_ax)

#     generalization_data_num_solved, generalization_data_solve_time = parse_data([[
#         f"{env_label}-{method}-{seed}-2000-{env_size}"
#     for seed in range(8)] for method, _ in main_methods_colors])

#     for i, (method, method_color), num_solved, solve_time in zip(
#         range(len(main_methods_colors)), main_methods_colors, generalization_data_num_solved, generalization_data_solve_time, strict=True
#     ):
#         add_bar_graph(num_solved_ax, i, num_solved[np.isfinite(num_solved)])
#         if not method.startswith('gnn_action_policy'):
#             print(solve_time)
#             add_bar_graph(solve_time_ax, i, solve_time[np.isfinite(solve_time)])
# fig.align_ylabels(axs[:, 0])
