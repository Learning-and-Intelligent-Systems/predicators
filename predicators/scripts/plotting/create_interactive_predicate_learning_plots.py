"""Create plots for interactive predicate learning."""

import os
from typing import Callable, Sequence, Tuple

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
    ("SEED", "seed"),
    ("NUM_SOLVED", "num_solved"),
    ("PERC_SOLVED", "perc_solved"),
    ("NUM_OFFLINE_TRANSITIONS", "num_offline_transitions"),
    ("NUM_ONLINE_TRANSITIONS", "num_online_transitions"),
    ("QUERY_COST", "query_cost"),
    ("PERC_EXEC_FAIL", "perc_exec_fail"),
    ("PERC_PLAN_FAIL", "perc_plan_fail"),
]

DERIVED_KEYS = [
    ("perc_solved", lambda r: 100 * r["num_solved"] / r["num_test_tasks"]),
    ("perc_exec_fail",
     lambda r: 100 * r["num_execution_failures"] / r["num_test_tasks"]),
    ("perc_plan_fail",
     lambda r: 100 * r["num_solve_failures"] / r["num_test_tasks"]),
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
    ("QUERY_COST", "Cumulative Query Cost"),
    ("PERC_EXEC_FAIL", "% Execution Failures"),
    ("PERC_PLAN_FAIL", "% Planning Failures"),
]

# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are plot titles.
# The keys of the inner dict are (legend label, marker, df selector).
PLOT_GROUPS = {
    "Main Approaches in CoverEnv Excluding Covers,Holding": [
        ("Main (Ensemble)", "blue",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: v == "main")),
        ("Main (MLP)", "orange",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: v == "main_mlp")),
    ],
    "Query Baselines in CoverEnv Excluding Covers,Holding": [
        ("Main (Entropy)", "blue",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: v == "main")),
        ("Ask All", "green",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: "section_kid" in v)),
        ("Ask None", "red",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: "silent_kid" in v)),
        ("Ask Randomly", "purple",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: "random_kid" in v)),
    ],
    "Action Baselines in CoverEnv Excluding Covers,Holding": [
        ("Main (Greedy Lookahead)", "blue",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: v == "main")),
        ("GLIB", "turquoise",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: "glib" in v)),
        ("Random Actions", "blueviolet",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: "random_actions" in v)
         ),
        ("No Actions", "gold",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: "no_actions" in v)),
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
                           x_key: str, y_key: str) -> None:
    for label, color, selector in plot_group:
        entry_df = get_df_for_entry(x_key, df, selector)
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


def _create_single_line_plot(ax: plt.Axes, df: pd.DataFrame,
                             plot_group: Sequence[Tuple[str, str, Callable]],
                             x_key: str, y_key: str) -> None:
    for label, color, selector in plot_group:
        entry_df = get_df_for_entry(x_key, df, selector)
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
        # Get means and stds.
        mean_ys = np.mean(all_interp_ys, axis=0)
        std_ys = np.std(all_interp_ys, axis=0)
        assert len(mean_ys) == len(std_ys) == len(new_xs)
        # Create the line.
        ax.plot(new_xs, mean_ys, label=label, color=color)
        ax.fill_between(new_xs,
                        mean_ys - std_ys,
                        mean_ys + std_ys,
                        color=color,
                        alpha=FILL_BETWEEN_ALPHA)
    # Add a legend.
    plt.legend()


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
                if PLOT_TYPE == "seed_lines":
                    _create_seed_line_plot(ax, df, d, x_key, y_key)
                elif PLOT_TYPE == "single_lines":
                    _create_single_line_plot(ax, df, d, x_key, y_key)
                else:
                    raise ValueError(f"Unknown PLOT_TYPE: {PLOT_TYPE}.")
                ax.set_title(plot_title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                if y_key.startswith("PERC"):
                    ax.set_ylim((-5, 105))
                plt.tight_layout()
                filename = f"{plot_title}_{x_key}_{y_key}.png"
                filename = filename.replace(" ", "_").lower()
                outfile = os.path.join(outdir, filename)
                plt.savefig(outfile, dpi=DPI)
                print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
