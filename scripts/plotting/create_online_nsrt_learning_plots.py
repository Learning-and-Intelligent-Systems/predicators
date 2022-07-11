"""Create plots for online NSRT learning."""

import os
from functools import partial

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from predicators.scripts.analyze_results_directory import create_raw_dataframe
from predicators.scripts.plotting.create_interactive_learning_plots import \
    _create_seed_line_plot, _create_single_line_plot

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
    ("NUM_OFFLINE_TRANSITIONS", "num_offline_transitions"),
    ("NUM_ONLINE_TRANSITIONS", "num_online_transitions"),
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
    ("PERC_EXEC_FAIL", "% Execution Failures"),
    ("PERC_PLAN_FAIL", "% Planning Failures"),
]

ENVS = ["cover", "blocks", "painting", "tools"]

EXPLORERS = [
    "random_options",
    "no_explore",
    "exploit_planning",
]

EXPLORER_TO_COLOR = {
    "random_options": "red",
    "no_explore": "black",
    "exploit_planning": "blue",
}


def _select_data(env: str, explorer: str, df: pd.DataFrame) -> pd.DataFrame:
    return df["EXPERIMENT_ID"].apply(lambda v: explorer in v and env in v)


# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are plot titles.
# The keys of the inner dict are (legend label, marker, df selector).
PLOT_GROUPS = {
    env: [(explorer, EXPLORER_TO_COLOR[explorer],
           partial(_select_data, env, explorer)) for explorer in EXPLORERS]
    for env in ENVS
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
