"""Create plots for interactive predicate learning."""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from predicators.scripts.analyze_results_directory import \
    create_raw_dataframe, get_df_for_entry

pd.options.mode.chained_assignment = None  # default='warn'

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
    ("NUM_TRANSITIONS", "num_transitions"),
    ("QUERY_COST", "query_cost"),
]

DERIVED_KEYS = [("perc_solved",
                 lambda r: 100 * r["num_solved"] / r["num_test_tasks"])]

# The first element is the name of the metric that will be plotted on the
# x axis. See COLUMN_NAMES_AND_KEYS for all available metrics. The second
# element is used to label the x axis.
X_KEY_AND_LABEL = [
    ("NUM_TRANSITIONS", "Number of Transitions"),
]

# Same as above, but for the y axis.
Y_KEY_AND_LABEL = [
    ("PERC_SOLVED", "% Evaluation Tasks Solved"),
    ("QUERY_COST", "Cumulative Query Cost"),
]

# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are plot titles.
# The keys of the inner dict are (legend label, marker, df selector).
PLOT_GROUPS = {
    "CoverEnv Excluding Covers,Holding": [
        ("Section Kid", "blue", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "excludeall_section_kid" in v)),
        ("Entropy", "orange", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "excludeall_entropy_0.1" in v)),
        ("BALD", "green", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "excludeall_BALD_0.01" in v)),
        ("Silent Kid", "red", lambda df: df["EXPERIMENT_ID"].apply(
            lambda v: "excludeall_silent_kid" in v)),
    ],
}

# If True, add (0, 0) to every plot
ADD_ZERO_POINT = True

# Line transparency.
ALPHA = 0.5

#################### Should not need to change below here #####################


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
                for label, color, selector in d:
                    entry_df = get_df_for_entry(x_key, df, selector)
                    # Draw one line per seed.
                    for seed in entry_df["SEED"].unique():
                        seed_df = entry_df[entry_df["SEED"] == seed]
                        xs = seed_df[x_key]
                        ys = seed_df[y_key]
                        if ADD_ZERO_POINT:
                            xs = [0] + xs
                            ys = [0] + ys
                        ax.plot(xs, ys, color=color, label=label, alpha=ALPHA)
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_title(plot_title)
                ax.set_xlabel(x_label)
                ax.set_ylabel(y_label)
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys())
                plt.tight_layout()
                filename = f"{plot_title}_{x_key}_{y_key}.png"
                filename = filename.replace(" ", "_").lower()
                outfile = os.path.join(outdir, filename)
                plt.savefig(outfile, dpi=DPI)
                print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
