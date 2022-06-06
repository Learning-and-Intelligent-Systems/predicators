"""Create plots for option learning."""

import os
from functools import partial
from typing import Callable, List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate

from predicators.scripts.analyze_results_directory import \
    create_raw_dataframe, get_df_for_entry

pd.options.mode.chained_assignment = None  # default='warn'
# plt.rcParams["font.family"] = "CMU Serif"

############################ Change below here ################################

# Details about the plt figure.
DPI = 500
FONT_SIZE = 18

# Groups over which to take mean/std.
GROUPS = [
    "ENV", "APPROACH", "EXCLUDED_PREDICATES", "EXPERIMENT_ID",
    "NUM_TRAIN_TASKS", "CYCLE"
]

SEEDS = list(range(456, 458))  #list(range(456, 466))
TASK_IDS = list(range(3))  #list(range(50))
FILL_BETWEEN_ALPHA = 0.25

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
] + [(f"TASK{i}_PLANNING_TIME", f"task{i}_planning_time") for i in TASK_IDS]

DERIVED_KEYS = [("perc_solved",
                 lambda r: 100 * r["num_solved"] / r["num_test_tasks"])]

# PLOT_GROUPS is a nested dict where each outer dict corresponds to one plot,
# and each inner entry corresponds to one line on the plot.
# The keys of the outer dict are plot titles.
# The keys of the inner dict are (legend label, marker, df selector).
TITLE_ENVS = [
    # ("Cover", "cover_multistep_options"),
    # ("Stick Button", "stick_button"),
    # ("Doors", "doors"),
    ("Coffee", "coffee"),
]

NUM_INTERP_POINTS = 50
TIMEOUT = 300


def _select_data(env: str, approach: str, df: pd.DataFrame) -> pd.DataFrame:
    return df["EXPERIMENT_ID"].apply(
        lambda v: v.startswith(f"{env}_{approach}_"))


PLOT_GROUPS = {
    title: [
        ("BPNS (Ours)", "darkgreen", "o", partial(_select_data, env, "main")),
        # ("Max Skeletons=1", "gray", "o",
        #  partial(_select_data, env, "direct_bc_max_skel1")),
        # ("Max Samples=1", "brown", "o",
        #  partial(_select_data, env, "direct_bc_max_samp1")),
    ]
    for (title, env) in TITLE_ENVS
}

X_LABEL = "Planning Time Elapsed (s)"
Y_LABEL = "% Evaluation Tasks Solved"

Y_LIM = (-5, 110)

#################### Should not need to change below here #####################


def _load_results_for_seed(df: pd.DataFrame, selector: Callable[[pd.DataFrame],
                                                                pd.DataFrame],
                           seed: int) -> Tuple[List[float], List[float]]:
    entry_df = get_df_for_entry("SEED", df, selector)
    seed_df = entry_df[entry_df["SEED"] == seed]
    all_time_results = []
    for test_task_idx in TASK_IDS:
        x_key = f"TASK{test_task_idx}_PLANNING_TIME"
        result_lst = seed_df[x_key].tolist()
        assert len(result_lst) == 1
        result = result_lst[0]
        if np.isnan(result):
            continue  # timeout
        all_time_results.append(result)
    assert len(all_time_results) > 0
    seed_xs = [0] + sorted(all_time_results)
    seed_ys = list(range(len(seed_xs)))
    assert seed_xs[-1] < TIMEOUT
    seed_xs.append(TIMEOUT)
    seed_ys.append(seed_ys[-1])
    return seed_xs, seed_ys


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': FONT_SIZE})
    df = create_raw_dataframe(COLUMN_NAMES_AND_KEYS, DERIVED_KEYS)
    for plot_title, d in PLOT_GROUPS.items():
        _, ax = plt.subplots(figsize=(10, 5))
        for label, color, marker, selector in d:
            all_xs, all_ys = [], []
            for seed in SEEDS:
                seed_xs, seed_ys = _load_results_for_seed(df, selector, seed)
                all_xs.append(seed_xs)
                all_ys.append(seed_ys)
            # Interpolate.
            # The max/min pattern here is so that we never have to extrapolate,
            # we only ever interpolate.
            min_x = max(min(seed_x) for seed_x in all_xs)
            max_x = min(max(seed_x) for seed_x in all_xs)
            assert min_x == 0
            assert max_x == TIMEOUT
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
        ax.set_title(plot_title)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.set_ylim(Y_LIM)
        plt.legend(loc='center left',
                   bbox_to_anchor=(1, 0.5),
                   prop={'size': 12})
        plt.tight_layout()
        filename = f"{plot_title}_timing_results.png"
        filename = filename.replace(" ", "_").lower()
        outfile = os.path.join(outdir, filename)
        plt.savefig(outfile, dpi=DPI)
        print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
