
# Data ingestion
from glob import glob
import os
from typing import Iterable, List, Optional, Tuple
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
fontsize = 20

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

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

def parse_data(dirs) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    if type(dirs) != list:
        file, = glob(os.path.join('experiment-results', dirs, '*.pkl'))
        data = pkl.load(open(file, 'rb'))['results']
        return data['num_solved']/num_test_tasks*100, np.array([
            data.get(f'PER_TASK_task{t}_solve_time', float('inf')) for t in range(num_test_tasks)
        ])
    num_solved, avg_solve_time = zip(*map(parse_data, dirs))
    return np.stack(list(num_solved)), np.stack(list(avg_solve_time))

def create_subplot(
        *,
        name: str = '',
        x_label: str = '',
        x_ticks: Iterable = [],
        width_mult: int = 1,
    ) -> Tuple[matplotlib.figure.Figure, plt.Axes, plt.Axes]: # type: ignore
    x_ticks = list(x_ticks)
    fig, (num_solved_ax, solve_time_ax) = plt.subplots(2, 1, figsize=(5 * width_mult, 10))
    if name:
        num_solved_ax.set_title(name, fontsize=fontsize)

    num_solved_ax.tick_params(
        axis='x',
        which='both',
        bottom=bool(x_ticks),
        top=False,
        labelbottom=bool(x_ticks),
        labelsize=fontsize,
    )
    solve_time_ax.tick_params(
        axis='x',
        which='both',
        bottom=bool(x_ticks),
        top=False,
        labelbottom=bool(x_ticks),
        labelsize=fontsize,
    )
    solve_time_ax.set_ylim([0.1, 120])
    if x_ticks:
        num_solved_ax.set_xticks(list(map(calculate_center_simple, range(len(x_ticks)))), x_ticks)
        solve_time_ax.set_xticks(list(map(calculate_center_simple, range(len(x_ticks)))), x_ticks)
    if x_label:
        solve_time_ax.set_xlabel(x_label, fontsize=fontsize)
    solve_time_ax.sharex(num_solved_ax)

    num_solved_ax.tick_params(axis='y', labelsize=fontsize)
    solve_time_ax.tick_params(axis='y', labelsize=fontsize)
    num_solved_ax.set_ylabel('% Solved tasks (Higher is better)', fontsize=fontsize)
    solve_time_ax.set_ylabel('Solve Time (Lower is better)', fontsize=fontsize)
    num_solved_ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    solve_time_ax.set_yscale('log')
    solve_time_ax.yaxis.set_major_formatter(mtick.PercentFormatter(symbol='s'))

    fig.align_ylabels([num_solved_ax, solve_time_ax])
    fig.tight_layout()

    return fig, num_solved_ax, solve_time_ax

def calculate_center_simple(i: int):
    return i*bar_width + (i-1) * bar_spacing

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
for env_label, env_name, env_x_label, min_env_size, max_env_size in envs_size_ranges:
    fig, num_solved_ax, solve_time_ax = create_subplot(name=env_name, x_label=env_x_label, x_ticks=range(min_env_size, max_env_size+1))

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
            if num_solved[np.isfinite(num_solved)].size != 0:
                num_solved_ax.bar(
                    calculate_center_simple(i),
                    0,
                    1,
                    num_solved[np.isfinite(num_solved)].mean(),
                    yerr = [
                        [num_solved[np.isfinite(num_solved)].mean() - num_solved[np.isfinite(num_solved)].min()],
                        [num_solved[np.isfinite(num_solved)].max() - num_solved[np.isfinite(num_solved)].mean()]
                    ],
                    capsize=5,
                    ecolor=method_color,
                    error_kw={'alpha': 0.3}
                )
        solve_time_ax.plot(
            [calculate_center_simple(i) for i in range(len(solve_time_row))],
            [solve_time[np.isfinite(solve_time)].mean() if len(np.isfinite(solve_time)) else 0 for solve_time in solve_time_row],
            color=method_color,
            alpha=0.7,
            lw=2,
        )
        for i, solve_time in enumerate(solve_time_row):
            if solve_time[np.isfinite(solve_time)].size != 0:
                solve_time_ax.bar(
                    calculate_center_simple(i),
                    0,
                    1,
                    solve_time[np.isfinite(solve_time)].mean(),
                    yerr = [
                        [solve_time[np.isfinite(solve_time)].mean() - solve_time[np.isfinite(solve_time)].min()],
                        [solve_time[np.isfinite(solve_time)].max() - solve_time[np.isfinite(solve_time)].mean()]
                    ],
                    capsize=5,
                    ecolor=method_color,
                    error_kw={'alpha': 0.3}
                )
    fig.savefig(os.path.join(plots_dir, f"generalization_{env_name.lower()}.png"), dpi=300)

# Efficiency plot
fig, axs = plt.subplots(2, len(envs_size_ranges), figsize=(15, 10))
for ax_idx, (env_label, env_name, _, env_size, _), num_solved_ax, solve_time_ax in zip(range(len(envs_size_ranges)), envs_size_ranges, axs[0], axs[1], strict=True):
    fig, num_solved_ax, solve_time_ax = create_subplot(name=env_name, x_label='Number of Datapoints', x_ticks=[500, 1000, 1500, 2000])

    data_num_solved, data_solve_time = parse_data([[[
        f"{env_label}-{method}-{seed}-{num_datapoints}-{env_size}"
    for seed in range(8)] for num_datapoints in [500, 1000, 1500, 2000]] for method, _ in main_methods_colors])

    if env_name == "Statue":
        print(data_solve_time[1][np.isfinite(data_solve_time[1])].max())


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
            if num_solved[np.isfinite(num_solved)].size != 0:
                num_solved_ax.bar(
                    calculate_center_simple(i),
                    0,
                    1,
                    num_solved[np.isfinite(num_solved)].mean(),
                    yerr = [
                        [num_solved[np.isfinite(num_solved)].mean() - num_solved[np.isfinite(num_solved)].min()],
                        [num_solved[np.isfinite(num_solved)].max() - num_solved[np.isfinite(num_solved)].mean()]
                    ],
                    capsize=5,
                    ecolor=method_color,
                    error_kw={'alpha': 0.3}
                )
        solve_time_ax.plot(
            [calculate_center_simple(i) for i in range(len(solve_time_row))],
            [solve_time[np.isfinite(solve_time)].mean() if len(np.isfinite(solve_time)) else 0 for solve_time in solve_time_row],
            color=method_color,
            alpha=0.7,
            lw=2,
        )
        for i, solve_time in enumerate(solve_time_row):
            if solve_time[np.isfinite(solve_time)].size != 0:
                solve_time_ax.bar(
                    calculate_center_simple(i),
                    0,
                    1,
                    solve_time[np.isfinite(solve_time)].mean(),
                    yerr = [
                        [solve_time[np.isfinite(solve_time)].mean() - solve_time[np.isfinite(solve_time)].min()],
                        [solve_time[np.isfinite(solve_time)].max() - solve_time[np.isfinite(solve_time)].mean()]
                    ],
                    capsize=5,
                    ecolor=method_color,
                    error_kw={'alpha': 0.3}
                )
    fig.savefig(os.path.join(plots_dir, f"efficiency_{env_name.lower()}.png"), dpi=300)

# PyBullet Env Plot
fig, num_solved_ax, solve_time_ax = create_subplot(name='Packing')
data_num_solved, data_solve_time = parse_data([[
    f"pybullet_packing-{method}-diffusion-{seed}-2000-5"
for seed in range(8)] for method in ['search_pruning', 'nsrt_learning']])
for i, (method, method_color), num_solved, solve_time in zip(
        range(2), [('search_pruning', colors[0]), ('nsrt_learning', colors[1])], data_num_solved, data_solve_time, strict=True
    ):
        add_bar_graph(num_solved_ax, i, num_solved[np.isfinite(num_solved)], color=method_color)
        add_bar_graph(solve_time_ax, i, solve_time[np.isfinite(solve_time)], color=method_color)

fig.savefig(os.path.join(plots_dir, "pybullet.png"), dpi=300)

# Ablations Plot
fig, num_solved_ax, solve_time_ax = create_subplot(name='Shelves', width_mult=2)
data_num_solved, data_solve_time = parse_data([[
    f"shelves2d-{method}-{seed}-{4000 if 'action' in method else 2000}-5"
for seed in range(8)] for method, _ in ablation_colors])
for i, (method, method_color), num_solved, solve_time in zip(
        range(len(ablation_colors)), ablation_colors, data_num_solved, data_solve_time, strict=True
    ):
        add_bar_graph(num_solved_ax, i, num_solved[np.isfinite(num_solved)], color=method_color)
        add_bar_graph(solve_time_ax, i, solve_time[np.isfinite(solve_time)], color=method_color)

fig.savefig(os.path.join(plots_dir, "ablations.png"), dpi=300)