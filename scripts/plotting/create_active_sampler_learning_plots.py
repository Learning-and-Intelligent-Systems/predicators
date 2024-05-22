"""Create plots for active sampler learning."""

import os
from functools import partial
from typing import Callable, Dict, Sequence, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from scripts.analyze_results_directory import create_raw_dataframe, \
    get_df_for_entry

pd.set_option('chained_assignment', None)

############################ Change below here ################################

# Details about the plt figure.
DPI = 500
FONT_SIZE = 12

# All column names and keys to load into the pandas tables before plotting.
COLUMN_NAMES_AND_KEYS = [
    ("EXPERIMENT_ID", "experiment_id"),
    ("APPROACH", "approach"),
    ("SEED", "seed"),
    ("NUM_SOLVED", "num_solved"),
    ("PERC_SOLVED", "perc_solved"),
    ("NUM_OFFLINE_TRANSITIONS", "num_offline_transitions"),
    ("NUM_ONLINE_TRANSITIONS", "num_online_transitions"),
    ("POLICY_CALL_TIME", "policy_call_time"),
    ("NUM_OPTIONS_EXECUTED", "num_options_executed"),
]


def _derive_per_task_average(metric: str,
                             row: Dict[str, float],
                             unsolved_task_penalty: float = 100) -> float:
    """Add a large constant penalty for unsolved tasks."""
    total_tasks = int(row["num_test_tasks"])
    total_value = 0.0
    for test_task_idx in range(total_tasks):
        key = f"PER_TASK_task{test_task_idx}_{metric}"
        # Add penalty for tasks that raised an approach failure.
        if key not in row:
            total_value += unsolved_task_penalty
        else:
            total_value += row[key]
    # Add penalty for unsolved tasks.
    num_unsolved_tasks = total_tasks - row["num_solved"]
    total_value += unsolved_task_penalty * num_unsolved_tasks
    # Get average.
    return total_value / total_tasks


DERIVED_KEYS: Sequence[Tuple[str, Callable[[Dict[str, float]], float]]] = [
    ("perc_solved", lambda r: 100 * r["num_solved"] / r["num_test_tasks"]),
    ("policy_call_time", partial(_derive_per_task_average, "exec_time")),
    ("num_options_executed",
     partial(_derive_per_task_average, "options_executed")),
]

# The first element is the name of the metric that will be plotted on the
# x axis. See COLUMN_NAMES_AND_KEYS for all available metrics. The second
# element is used to label the x axis.
X_KEY_AND_LABEL = [
    ("NUM_ONLINE_TRANSITIONS", "Number of Online Transitions"),
]

# Same as above, but for the y axis.
Y_KEY_AND_LABEL = [
    ("PERC_SOLVED", "% Evaluation Tasks Solved"),
    ("POLICY_CALL_TIME", "Policy Call Time (s)"),
    ("NUM_OPTIONS_EXECUTED", "# Skills Executed"),
]

# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are plot titles.
# The keys of the inner dict are (legend label, marker, df selector).
PLOT_GROUPS = {
    "Grid 1D Environment": [
        ("Planning Progress", "green", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "grid_row-planning_progress_explore" in v)),
        ("Task Repeat", "orange", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "grid_row-task_repeat_explore" in v)),
        ("Competence Gradient", "yellow", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "grid_row-competence_gradient" in v)),
        ("Fail Focus", "red", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "grid_row-success_rate_explore_ucb" in v)),
        ("Task-Relevant", "purple", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "grid_row-random_score_explore" in v)),
        ("Random Skills", "blue", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "grid_row-random_nsrts_explore" in v)),
        ("Skill Diversity", "pink", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "grid_row-skill_diversity" in v)),
        ("MAPLE-Q", "silver", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "grid_row-maple_q" in v)),
    ],
    "Ball and Cup Sticky Table": [
        ("Planning Progress", "green", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "sticky_table-planning_progress_explore" in v)),
        ("Task Repeat", "orange", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "sticky_table-task_repeat_explore" in v)),
        ("Competence Gradient", "yellow", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "sticky_table-competence_gradient" in v)),
        ("Fail Focus", "red", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "sticky_table-success_rate_explore_ucb" in v)),
        ("Task-Relevant", "purple", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "sticky_table-random_score_explore" in v)),
        ("Random Skills", "blue", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "sticky_table-random_nsrts_explore" in v)),
        ("Skill Diversity", "pink", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "sticky_table-skill_diversity" in v)),
        ("MAPLE-Q", "silver", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "sticky_table-maple_q" in v)),
    ],
    "Cleanup Playroom": [
        ("Planning Progress", "green", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "spot_sweeping_sim-planning_progress_explore" in v)),
        ("Task Repeat", "orange", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "spot_sweeping_sim-task_repeat_explore" in v)),
        ("Competence Gradient", "yellow", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "spot_sweeping_sim-competence_gradient" in v)),
        ("Fail Focus", "red", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "spot_sweeping_sim-success_rate_explore_ucb" in v)),
        ("Task-Relevant", "purple", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "spot_sweeping_sim-random_score_explore" in v)),
        ("Random Skills", "blue", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "spot_sweeping_sim-random_nsrts_explore" in v)),
        ("Skill Diversity", "pink", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "spot_sweeping_sim-skill_diversity" in v)),
        ("MAPLE-Q", "silver", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "spot_sweeping_sim-maple_q" in v)),
    ],
}

# If True, add (0, 0) to every plot.
ADD_ZERO_POINT = False

# Plot type.
PLOT_TYPE = "single_lines"  # single_lines or seed_lines

# Line transparency for seed line plots.
SEED_LINE_ALPHA = 0.5

# Fill between transparency for single line plots.
FILL_BETWEEN_ALPHA = 0.25

# Number of interpolation x ticks for the single line plots.
NUM_INTERP_POINTS = 10

#################### Should not need to change below here #####################


def _create_seed_line_plot(ax: plt.Axes, df: pd.DataFrame,
                           plot_group: Sequence[Tuple[str, str, Callable]],
                           x_key: str, y_key: str) -> bool:
    plot_has_data = False
    for label, color, selector in plot_group:
        entry_df = get_df_for_entry(x_key, df, selector)
        if entry_df.size == 0:
            print(f"No results found for {label}, skipping")
            continue
        plot_has_data = True
        # Draw one line per seed.
        for seed in entry_df["SEED"].unique():
            seed_df = entry_df[entry_df["SEED"] == seed]
            xs = seed_df[x_key].tolist()
            ys = seed_df[y_key].tolist()
            if ADD_ZERO_POINT:
                xs = [0] + xs
                ys = [0] + ys
            ax.plot(xs, ys, color=color, label=label, alpha=SEED_LINE_ALPHA)
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    return plot_has_data


def _create_single_line_plot(ax: plt.Axes, df: pd.DataFrame,
                             plot_group: Sequence[Tuple[str, str, Callable]],
                             x_key: str, y_key: str) -> bool:
    plot_has_data = False
    for label, color, selector in plot_group:
        entry_df = get_df_for_entry(x_key, df, selector)
        if entry_df.size == 0:
            print(f"No results found for {label}, skipping")
            continue
        plot_has_data = True
        # Draw one line total. To make sure the x ticks are aligned, we will
        # interpolate.
        all_xs, all_ys = [], []
        for seed in entry_df["SEED"].unique():
            seed_df = entry_df[entry_df["SEED"] == seed]
            xs = seed_df[x_key].tolist()
            ys = seed_df[y_key].tolist()
            if ADD_ZERO_POINT:
                xs = [0] + xs
                ys = [0] + ys
            all_xs.append(xs)
            all_ys.append(ys)
        # The max/min pattern here is so that we never have to extrapolate,
        # we only ever interpolate.
        min_x = max(min(seed_x) for seed_x in all_xs)
        max_x = min(max(seed_x) for seed_x in all_xs)
        # Create one consistent set of x ticks.
        new_xs = np.linspace(min_x, max_x, NUM_INTERP_POINTS)
        # Create the interpolated y data.
        all_interp_ys = []
        for xs, ys in zip(all_xs, all_ys):
            f = interpolate.interp1d(xs, ys)
            interp_ys = f(new_xs)
            all_interp_ys.append(interp_ys)
        # Get means and standard errors.
        mean_ys = np.mean(all_interp_ys, axis=0)
        n = np.size(all_interp_ys, axis=0)
        std_ys = np.std(all_interp_ys, ddof=1, axis=0) / np.sqrt(n)
        assert len(mean_ys) == len(std_ys) == len(new_xs)
        ax.plot(new_xs, mean_ys, label=label, color=color)
        ax.fill_between(new_xs,
                        mean_ys - std_ys,
                        mean_ys + std_ys,
                        color=color,
                        alpha=FILL_BETWEEN_ALPHA)
    # Add a legend.
    plt.legend()
    return plot_has_data


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': FONT_SIZE})
    df = create_raw_dataframe(COLUMN_NAMES_AND_KEYS, DERIVED_KEYS)
    for x_key, x_label in X_KEY_AND_LABEL:
        for y_key, y_label in Y_KEY_AND_LABEL:
            for plot_title, d in PLOT_GROUPS.items():
                _, ax = plt.subplots()
                has_res = False
                if PLOT_TYPE == "seed_lines":
                    has_res = _create_seed_line_plot(ax, df, d, x_key, y_key)
                elif PLOT_TYPE == "single_lines":
                    has_res = _create_single_line_plot(ax, df, d, x_key, y_key)
                else:
                    raise ValueError(f"Unknown PLOT_TYPE: {PLOT_TYPE}.")
                if not has_res:
                    print("No results found for whole plot, skipping")
                    continue
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                if y_key.startswith("PERC"):
                    ax.set_ylim((-5, 105))
                plt.title(plot_title)
                plt.tight_layout()
                filename = f"{plot_title}_{x_key}_{y_key}.png"
                filename = filename.replace(" ", "_").lower()
                outfile = os.path.join(outdir, filename)
                plt.savefig(outfile, dpi=DPI)
                print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
