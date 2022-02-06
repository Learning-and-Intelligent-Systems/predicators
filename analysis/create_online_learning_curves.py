"""Create plots for online learning."""

import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from predicators.analysis.analyze_results_directory import create_dataframes

pd.options.mode.chained_assignment = None  # default='warn'

############################ Change below here ################################

# X_KEY is the name of the metric that will be plotted on the x axis. See
# analyze_results_directory.py for all available metrics.
# X_LABEL is used to label the x axis.
X_KEY = "CYCLE"  # common: CYCLE, NUM_TRANSITIONS, LEARNING_TIME
X_LABEL = "Num online learning episodes"

# Y_KEY is the name of the metric that will be plotted on the y axes. See
# analyze_results_directory.py for all available metrics.
# Y_LABEL is used to label the y axis.
Y_KEY = "AVG_NODES_CREATED"  # common: NUM_SOLVED, AVG_NODES_CREATED
Y_LABEL = "Average nodes created"  #"Test tasks solved"

# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are are plot titles.
# The keys of the inner dict are experiment IDs, and the values are labels
# for the legend.
PLOT_GROUPS = {
    "Cover (Regrasp)": {
        "cover_regrasp_naive_allexclude": "Naive (All Excluded)",
        "cover_regrasp_targeted_allexclude": "Targeted (All Excluded)",
    },
    "Blocks": {
        "blocks_naive_allexclude": "Naive (All Excluded)",
        "blocks_targeted_allexclude": "Targeted (All Excluded)",
    },
}

#################### Should not need to change below here #####################


def _get_df_for_experiment_id(df: pd.DataFrame,
                              experiment_id: str) -> pd.DataFrame:
    df = df[df["EXPERIMENT_ID"] == experiment_id]
    # Handle CYCLE as a special case, since the offline learning phase is
    # logged as None. Note that we shift everything by 1 so the first data
    # point is 0, meaning 0 online learning cycles have happened so far.
    df["CYCLE"].replace("None", "-1", inplace=True)
    df["CYCLE"] = df["CYCLE"].map(pd.to_numeric) + 1
    # Always sort by cycle. We're assuming the other metrics of interest,
    # like num transitions or num queries, will be monotonically increasing
    # as the number of cycles increases.
    df = df.sort_values("CYCLE")
    df[X_KEY] = df[X_KEY].map(pd.to_numeric)
    return df


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 16})
    grouped_means, grouped_stds, _ = create_dataframes()
    means = grouped_means.reset_index()
    stds = grouped_stds.reset_index()
    for plot_title, experiment_id_dict in PLOT_GROUPS.items():
        _, ax = plt.subplots()
        for experiment_id, label in experiment_id_dict.items():
            exp_means = _get_df_for_experiment_id(means, experiment_id)
            exp_stds = _get_df_for_experiment_id(stds, experiment_id)
            ax.errorbar(exp_means[X_KEY],
                        exp_means[Y_KEY],
                        yerr=exp_stds[Y_KEY],
                        label=label)
        # Automatically make x ticks integers for certain X KEYS.
        if X_KEY in ("CYCLE", "NUM_TRANSITIONS"):
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(plot_title)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        plt.legend()
        plt.tight_layout()
        filename = f"{plot_title}_{X_KEY}_{Y_KEY}.png"
        filename = filename.replace(" ", "_").lower()
        outfile = os.path.join(outdir, filename)
        plt.savefig(outfile)
        print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    _main()
