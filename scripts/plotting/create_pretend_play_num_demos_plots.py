"""Create plots for learning from varying numbers of demonstrations."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from predicators.scripts.analyze_results_directory import create_dataframes, \
    get_df_for_entry

pd.options.mode.chained_assignment = None  # default='warn'
# plt.rcParams["font.family"] = "CMU Serif"

############################ Change below here ################################

# Details about the plt figure.
DPI = 500
FONT_SIZE = 18

# Groups over which to take mean/std.
GROUPS = [
    "ENV", "APPROACH", "EXCLUDED_PREDICATES", "EXPERIMENT_ID",
    "NUM_TRAIN_TASKS", "CYCLE", "NUM_DEMOS"
]

# All column names and keys to load into the pandas tables before plotting.
COLUMN_NAMES_AND_KEYS = [
    ("ENV", "env"),
    ("APPROACH", "approach"),
    ("EXCLUDED_PREDICATES", "excluded_predicates"),
    ("EXPERIMENT_ID", "experiment_id"),
    ("SEED", "seed"),
    ("NUM_TRAIN_TASKS", "num_train_tasks"),
    ("CYCLE", "cycle"),
    ("NUM_SOLVED", "num_solved"),
    ("AVG_NUM_PREDS", "avg_num_preds"),
    ("AVG_TEST_TIME", "avg_suc_time"),
    ("AVG_NODES_CREATED", "avg_num_nodes_created"),
    ("LEARNING_TIME", "learning_time"),
    ("PERC_SOLVED", "perc_solved"),
    ("NUM_DEMOS", "max_initial_demos"),
]

DERIVED_KEYS = [("perc_solved",
                 lambda r: 100 * r["num_solved"] / r["num_test_tasks"])]

# The first element is the name of the metric that will be plotted on the
# x axis. See COLUMN_NAMES_AND_KEYS for all available metrics. The second
# element is used to label the x axis.
X_KEY_AND_LABEL = [
    ("NUM_DEMOS", "Number of Demonstrations"),
]

# Same as above, but for the y axis.
Y_KEY_AND_LABEL = [
    ("PERC_SOLVED", "% Evaluation Tasks Solved"),
]

# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are plot titles.
# The keys of the inner dict are (legend label, marker, df selector).
PLOT_GROUPS = {
    "Blocks": [
        ("Blocks", "o",
         lambda df: df["EXPERIMENT_ID"].apply(lambda v: "nsrt_learning_" in v)
         ),
    ],
}

# If True, add (0, 0) to every plot
ADD_ZERO_POINT = True

Y_LIM = (-5, 110)

#################### Should not need to change below here #####################


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': FONT_SIZE})
    grouped_means, grouped_stds, _ = create_dataframes(COLUMN_NAMES_AND_KEYS,
                                                       GROUPS, DERIVED_KEYS)
    means = grouped_means.reset_index()
    stds = grouped_stds.reset_index()
    for x_key, x_label in X_KEY_AND_LABEL:
        for y_key, y_label in Y_KEY_AND_LABEL:
            for plot_title, d in PLOT_GROUPS.items():
                _, ax = plt.subplots()
                for label, marker, selector in d:
                    exp_means = get_df_for_entry(x_key, means, selector)
                    exp_stds = get_df_for_entry(x_key, stds, selector)
                    xs = exp_means[x_key].tolist()
                    ys = exp_means[y_key].tolist()
                    y_stds = exp_stds[y_key].tolist()
                    if ADD_ZERO_POINT:
                        xs = [0] + xs
                        ys = [0] + ys
                        y_stds = [0] + y_stds
                    ax.errorbar(xs,
                                ys,
                                yerr=y_stds,
                                label=label,
                                marker=marker)
                # Automatically make x ticks integers for certain X KEYS.
                if x_key in ("CYCLE", "NUM_TRANSITIONS"):
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                elif x_key == "NUM_TRAIN_TASKS":
                    ax.set_xticks(xs)
                ax.set_title(plot_title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                ax.set_ylim(Y_LIM)
                # plt.legend()
                plt.tight_layout()
                filename = f"{plot_title}_{x_key}_{y_key}.png"
                filename = filename.replace(" ", "_").lower()
                outfile = os.path.join(outdir, filename)
                plt.savefig(outfile, dpi=DPI)
                print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
