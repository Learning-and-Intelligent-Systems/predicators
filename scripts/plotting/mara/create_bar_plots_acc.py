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

# Groups over which to take mean/std.
GROUPS = [
    "ENV",
    "APPROACH",
    # "EXCLUDED_PREDICATES",
    "EXPERIMENT_ID",
]

env_names = [
    "cover",
    "blocks",
    "coffee",
    "balance",
    "grow",
    "circuit",
    "float",
    "domino",
    "laser",
    "ants",
    "fan",
]

# All column names and keys to load into the pandas tables.
COLUMN_NAMES_AND_KEYS = [
    ("ENV", "env"),
    ("APPROACH", "approach"),
    # ("EXCLUDED_PREDICATES", "excluded_predicates"),
    ("EXPERIMENT_ID", "experiment_id"),
    ("SEED", "seed"),
    ("OVERALL_ACCURACY", "perc_overall_accuracy"),
] + [(f"{env}_ACCURACY", f"perc_{env}_accuracy") for env in env_names]

DERIVED_KEYS = [
    ("perc_overall_accuracy", lambda r: 100 * r["avg_accuracy"]),
    # In Python, when you create a lambda function inside a loop, it captures
    # the variable by reference, not by value.
    # When the lambda functions are actually called later, they all use the
    # final value of env from the loop.
    # use default arguments in the lambda to capture the current value:
    *[(f"perc_{env}_accuracy", lambda r, env=env: 100 * r[f"{env}_accuracy"])
      for env in env_names],
]

KEYS = [
    "OVERALL_ACCURACY",
    *[(f"{env}_ACCURACY") for env in env_names],
]

# The keys of the dict are (df key, df value), and the dict values are
# labels for the legend. The df key/value are used to select a subset from
# the overall pandas dataframe.
PLOT_GROUPS = [
    ("", pd_create_equal_selector("ENV", "all_tasks")),
    # ("Cover", pd_create_equal_selector(
    #     "ENV", "pybullet_cover_typed_options")),
    # ("Coffee", pd_create_equal_selector(
    #     "ENV", "pybullet_coffee")),
    # ("Cover Heavy", pd_create_equal_selector(
    #     "ENV", "pybullet_cover_weighted")),
    # ("Balance", pd_create_equal_selector("ENV", "pybullet_balance")),
]

# See PLOT_GROUPS comment.
BAR_GROUPS = [
    # ("Ours", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "nsp-nl" in v)),
    ("Human", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "human" in v)),
    ("VLM", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "vlm_clf" in v)),
    ("DINO-dtw",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "dino_sim_dtw" in v)),
    ("DINO-chf",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "dino_sim_chamfer" in v)),
]

#################### Should not need to change below here #####################


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results", "classification")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': FONT_SIZE})

    grouped_means, grouped_stds, _ = create_dataframes(
        COLUMN_NAMES_AND_KEYS, GROUPS, DERIVED_KEYS)  # type: ignore[arg-type]
    means = grouped_means.reset_index()
    stds = grouped_stds.reset_index()

    for key in KEYS:
        for plot_title, plot_selector in PLOT_GROUPS:
            _, ax = plt.subplots()
            plot_labels = []
            plot_means = []
            plot_stds = []
            for label, bar_selector in BAR_GROUPS:
                selector = combine_selectors([plot_selector, bar_selector])
                exp_means = get_df_for_entry(key, means, selector)
                exp_stds = get_df_for_entry(key, stds, selector)
                mean = exp_means[key].tolist()
                std = exp_stds[key].tolist()
                assert len(mean) == len(std) == 1
                plot_labels.append(label)
                plot_means.append(mean[0])
                plot_stds.append(std[0])
            ax.barh(plot_labels, plot_means, xerr=plot_stds, color='green')
            ax.set_xlim(X_LIM)
            ax.tick_params(axis='y', colors='black')
            key_name = key.lower().replace("_", " ")
            ax.set_title(key_name + plot_title)
            plt.gca().invert_yaxis()
            plt.tight_layout()
            filename = f"{plot_title}{key}.png"
            filename = filename.replace(" ", "_").lower()
            outfile = os.path.join(outdir, filename)
            plt.savefig(outfile, dpi=DPI)
            print(f"Wrote out to {outfile}")


if __name__ == "__main__":
    _main()
