"""Create bar plots.

For example, https://arxiv.org/abs/2203.09634 Figure 3
"""

import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from scripts.analyze_results_directory import combine_selectors, \
    create_dataframes, get_df_for_entry, pd_create_equal_selector

plt.style.use('ggplot')
pd.set_option('chained_assignment', None)
# plt.rcParams["font.family"] = "CMU Serif"

############################ Change below here ################################

# Details about the plt figure.
DPI = 500
FONT_SIZE = 18
X_LIM = (-5, 110)

# Color configuration
USE_DIFFERENT_COLORS = True  # Set to False to use same color for all bars
SINGLE_BAR_COLOR = 'green'  # Color to use when USE_DIFFERENT_COLORS is False

# Color palette for different approaches
BAR_COLORS = [
    '#1f77b4',  # blue
    '#ff7f0e',  # orange
    '#2ca02c',  # green
    '#d62728',  # red
    '#9467bd',  # purple
    '#8c564b',  # brown
    '#e377c2',  # pink
    '#7f7f7f',  # gray
    '#bcbd22',  # olive
    '#17becf',  # cyan
]

# Groups over which to take mean/std.
GROUPS = [
    "ENV", "APPROACH", "EXCLUDED_PREDICATES", "EXPERIMENT_ID",
    "ONLINE_LEARNING_CYCLE"
]

# All column names and keys to load into the pandas tables.
COLUMN_NAMES_AND_KEYS = [
    ("ENV", "env"),
    ("APPROACH", "approach"),
    ("EXCLUDED_PREDICATES", "excluded_predicates"),
    ("EXPERIMENT_ID", "experiment_id"),
    ("SEED", "seed"),
    ("AVG_TEST_TIME", "avg_suc_time"),
    ("AVG_NODES_CREATED", "avg_num_nodes_created"),
    ("LEARNING_TIME", "learning_time"),
    ("PERC_SOLVED", "perc_solved"),
    ("ONLINE_LEARNING_CYCLE",
     "cycle"),  # add to select model at specific cycle
    ("AVG_NUM_FAILED_PLAN", "avg_num_skeletons_optimized"),
]

DERIVED_KEYS = [("perc_solved",
                 lambda r: 100 * r["num_solved"] / r["num_test_tasks"])]

KEYS = [
    "PERC_SOLVED",
]

# The keys of the dict are (df key, df value), and the dict values are
# labels for the legend. The df key/value are used to select a subset from
# the overall pandas dataframe.
PLOT_GROUPS = [
    ("Boil", pd_create_equal_selector("ENV", "pybullet_boil")),
    ("Coffee", pd_create_equal_selector("ENV", "pybullet_coffee")),
    ("Grow", pd_create_equal_selector("ENV", "pybullet_grow")),
    ("Fan", pd_create_equal_selector("ENV", "pybullet_fan")),
    ("Domino", pd_create_equal_selector("ENV", "pybullet_domino_grid")),
]

# See PLOT_GROUPS comment.
BAR_GROUPS = [
    ("Oracle", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "oracle" in v)),
    ("Ours",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "predicate_invention" in v)
     ),
    ("VisPred", lambda df: df["EXPERIMENT_ID"].apply(
        lambda v: "online_nsrt_learning" in v)),
    ("MAPLE", lambda df:
     (df["EXPERIMENT_ID"].apply(lambda v: "maple_q" in v))),
    ("ViLa(zs)",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "vlm_plan_zero_shot" in v)
     ),
    ("ViLa(fs)",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "vlm_plan_few_shot" in v)),
    ("No Bayes",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "ablate_bayes" in v)),
    ("No LLM",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "ablate_llm" in v)),
    ("Oracle-VI",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "no_param_learn" in v)),
    ("No invent",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "no_invent" in v)),
]

# Allow no result group
NO_RESULT_GROUP = ["No LLM", "VisPred"]

keep_max_cycle_only = True
#################### Should not need to change below here #####################


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results", "planning")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': FONT_SIZE})

    # When keeping max cycle only, don't group by cycle since
    # we've filtered to highest cycle
    groups_to_use = GROUPS.copy()
    if keep_max_cycle_only and "ONLINE_LEARNING_CYCLE" in groups_to_use:
        groups_to_use.remove("ONLINE_LEARNING_CYCLE")

    grouped_means, grouped_stds, _ = create_dataframes(
        COLUMN_NAMES_AND_KEYS,
        groups_to_use,
        DERIVED_KEYS,
        keep_max_cycle_only=keep_max_cycle_only)
    means = grouped_means.reset_index()
    stds = grouped_stds.reset_index()

    for key in KEYS:
        for plot_title, plot_selector in PLOT_GROUPS:
            _, ax = plt.subplots()
            plot_labels = []
            plot_means = []
            plot_stds = []
            plot_colors = []
            for i, (label, bar_selector) in enumerate(BAR_GROUPS):
                selector = combine_selectors([plot_selector, bar_selector])
                exp_means = get_df_for_entry(key, means, selector)
                exp_stds = get_df_for_entry(key, stds, selector)
                mean = exp_means[key].tolist()
                std = exp_stds[key].tolist()
                try:
                    assert len(mean) == len(std) == 1
                except Exception:  # pylint: disable=broad-except
                    if label in NO_RESULT_GROUP:
                        print(
                            f"No results for {label} {plot_title} {key} which"
                            f" is in the NO_RESULT_GROUP, setting mean/std to 0"
                        )
                        mean = [0]
                        std = [0]
                    else:
                        print(f"Error for {label} {plot_title} "
                              f"{key}, mean: {mean}, std: {std}")
                        raise
                plot_labels.append(label)
                plot_means.append(mean[0])
                plot_stds.append(std[0])
                if USE_DIFFERENT_COLORS:
                    plot_colors.append(BAR_COLORS[i % len(BAR_COLORS)])
                else:
                    plot_colors.append(SINGLE_BAR_COLOR)
            ax.barh(plot_labels,
                    plot_means,
                    xerr=plot_stds,
                    color=plot_colors,
                    capsize=5,
                    ecolor='black',
                    error_kw={'linewidth': 2})
            ax.set_xlim(X_LIM)
            ax.tick_params(axis='y', colors='black')
            ax.set_title(plot_title)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            filename = f"{plot_title}_{key}.png"
            filename = filename.replace(" ", "_").lower()
            outfile = os.path.join(outdir, filename)
            plt.savefig(outfile, dpi=DPI)
            print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
