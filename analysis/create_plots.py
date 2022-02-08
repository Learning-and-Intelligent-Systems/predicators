"""Create plots for online learning."""

from typing import Any
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from predicators.analysis.analyze_results_directory import create_dataframes

pd.options.mode.chained_assignment = None  # default='warn'

############################ Change below here ################################

# Groups over which to take mean/std.
GROUPS = [
    "ENV", "APPROACH", "EXCLUDED_PREDICATES", "EXPERIMENT_ID",
    "NUM_TRAIN_TASKS", "CYCLE"
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
    # ("AVG_SKELETONS", "avg_num_skeletons_optimized"),
    # ("MIN_SKELETONS", "min_skeletons_optimized"),
    # ("MAX_SKELETONS", "max_skeletons_optimized"),
    # ("AVG_NODES_EXPANDED", "avg_num_nodes_expanded"),
    # ("AVG_NUM_NSRTS", "avg_num_nsrts"),
    # ("AVG_DISCOVERED_FAILURES", "avg_num_failures_discovered"),
    # ("AVG_PLAN_LEN", "avg_plan_length"),
    # ("AVG_EXECUTION_FAILURES", "avg_execution_failures"),
    # ("NUM_TRANSITIONS", "num_transitions"),
]

DERIVED_KEYS = [("perc_solved",
                 lambda r: 100 * r["num_solved"] / r["num_test_tasks"])]

# The first element is the name of the metric that will be plotted on the
# x axis. See COLUMN_NAMES_AND_KEYS for all available metrics. The second
# element is used to label the x axis.
X_KEY_AND_LABEL = [
    ("NUM_TRAIN_TASKS", "Num demos"),
    # ("NUM_TRANSITIONS", "Num transitions"),
    # ("LEARNING_TIME", "Learning time in seconds"),
]

# Same as above, but for the y axis.
Y_KEY_AND_LABEL = [
    ("PERC_SOLVED", "% test tasks solved"),
    # ("AVG_NODES_CREATED", "Averaged nodes created"),
]

# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are plot titles.
# The keys of the inner dict are (df key, df value), and the dict values are
# labels for the legend. The df key/value are used to select a subset from
# the overall pandas dataframe.
PLOT_GROUPS = {
    "Learning from Few Demonstrations": {
        ("ENV", "cover_regrasp"): "PickPlace1D",
        ("ENV", "blocks"): "Blocks",
        ("ENV", "painting"): "Painting",
        ("ENV", "tools"): "Tools",
    },
}

# If True, add (0, 0) to every plot
ADD_ZERO_POINT = True

#################### Should not need to change below here #####################


def _get_df_for_plot_line(x_key: str, df: pd.DataFrame, select_key: str,
                          select_value: Any) -> pd.DataFrame:
    df = df[df[select_key] == select_value]
    # Handle CYCLE as a special case, since the offline learning phase is
    # logged as None. Note that we shift everything by 1 so the first data
    # point is 0, meaning 0 online learning cycles have happened so far.
    if "CYCLE" in df:
        df["CYCLE"].replace("None", "-1", inplace=True)
        df["CYCLE"] = df["CYCLE"].map(pd.to_numeric) + 1
    df = df.sort_values(x_key)
    df[x_key] = df[x_key].map(pd.to_numeric)
    return df


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 16})
    grouped_means, grouped_stds, _ = create_dataframes(COLUMN_NAMES_AND_KEYS,
                                                       GROUPS, DERIVED_KEYS)
    means = grouped_means.reset_index()
    stds = grouped_stds.reset_index()
    for x_key, x_label in X_KEY_AND_LABEL:
        for y_key, y_label in Y_KEY_AND_LABEL:
            for plot_title, d in PLOT_GROUPS.items():
                _, ax = plt.subplots()
                for (select_key, select_value), label in d.items():
                    exp_means = _get_df_for_plot_line(x_key, means, select_key,
                                                      select_value)
                    exp_stds = _get_df_for_plot_line(x_key, stds, select_key,
                                                     select_value)
                    xs = exp_means[x_key].tolist()
                    ys = exp_means[y_key].tolist()
                    y_stds = exp_stds[y_key].tolist()
                    if ADD_ZERO_POINT:
                        xs = [0] + xs
                        ys = [0] + ys
                        y_stds = [0] + y_stds
                    ax.errorbar(xs, ys, yerr=y_stds, label=label)
                # Automatically make x ticks integers for certain X KEYS.
                if x_key in ("CYCLE", "NUM_TRANSITIONS"):
                    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                elif x_key == "NUM_TRAIN_TASKS":
                    ax.set_xticks(xs)
                ax.set_title(plot_title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                plt.legend()
                plt.tight_layout()
                filename = f"{plot_title}_{x_key}_{y_key}.png"
                filename = filename.replace(" ", "_").lower()
                outfile = os.path.join(outdir, filename)
                plt.savefig(outfile)
                print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    _main()
