"""Create LaTeX tables for papers."""

from operator import gt, lt
import numpy as np
import pandas as pd
from predicators.analysis.analyze_results_directory import create_dataframes, \
    pd_create_equal_selector, combine_selectors, get_df_for_entry

pd.options.mode.chained_assignment = None  # default='warn'

############################ Change below here ################################

# Groups over which to take mean/std.
GROUPS = [
    "ENV",
    "APPROACH",
    "EXCLUDED_PREDICATES",
    "EXPERIMENT_ID",
    # "NUM_TRAIN_TASKS", "CYCLE"
]

# All column names and keys to load into the pandas tables.
COLUMN_NAMES_AND_KEYS = [
    ("ENV", "env"),
    ("APPROACH", "approach"),
    ("EXCLUDED_PREDICATES", "excluded_predicates"),
    ("EXPERIMENT_ID", "experiment_id"),
    ("SEED", "seed"),
    # ("NUM_TRAIN_TASKS", "num_train_tasks"),
    # ("CYCLE", "cycle"),
    # ("NUM_SOLVED", "num_solved"),
    # ("AVG_NUM_PREDS", "avg_num_preds"),
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

TOP_ROW_LABEL = "\\bf{Environment}"

# The keys of the dict are (df key, df value), and the dict values are
# labels for the legend. The df key/value are used to select a subset from
# the overall pandas dataframe.
ROW_GROUPS = [
    ("PickPlace1D", pd_create_equal_selector("ENV", "cover")),
    ("Blocks", pd_create_equal_selector("ENV", "blocks")),
    ("Painting", pd_create_equal_selector("ENV", "painting")),
    ("Tools", pd_create_equal_selector("ENV", "tools")),
]

# See ROW_GROUPS comment.
OUTER_HEADER_GROUPS = [
    ("Ours", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "_main_200" in v)),
    ("Manual", lambda df: df["EXPERIMENT_ID"].apply(
        lambda v: "_noinventnoexclude_200" in v)),
    ("Down Learn",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "_downrefscore_200" in v)),
    ("Down Eval",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "_downrefeval_200" in v)),
    ("No Invent", lambda df: df["EXPERIMENT_ID"].apply(
        lambda v: "_noinventallexclude_200" in v)),
    ("Bisimulation",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "_prederror_200" in v)),
    ("Branching",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "_branchfac_200" in v)),
    ("Boltzmann",
     lambda df: df["EXPERIMENT_ID"].apply(lambda v: "_energy_200" in v)),
    ("Random", pd_create_equal_selector("APPROACH", "random_options")),
]

DOUBLE_LINES_AFTER = ["Ours", "Manual", "No Invent"]

# For bolding, how many stds to use.
BOLD_NUM_STDS = 2
DO_BOLDING = False

# If less than this, entry will be red.
RED_MIN_SIZE = 10

#### Main results ###

# See COLUMN_NAMES_AND_KEYS for all available metrics. The third entry is
# whether higher or lower is better.
INNER_HEADER_GROUPS = [
    ("Succ", "PERC_SOLVED", "higher"),
    # ("Node", "AVG_NODES_CREATED", "lower"),
    ("Time", "AVG_TEST_TIME", "lower"),
]

CAPTION = "TODO"
TABLE_LABEL = "tab:mainresults"

# #### Timing results ###

# See COLUMN_NAMES_AND_KEYS for all available metrics. The third entry is
# whether higher or lower is better.
# INNER_HEADER_GROUPS = [
#     ("Learn", "LEARNING_TIME", "lower"),
#     ("Plan", "AVG_TEST_TIME", "lower"),
# ]

# CAPTION = "TODO"
# TABLE_LABEL = "tab:timeresults"

#### Heuristic results ###

# DOUBLE_LINES_AFTER = []

# TOP_ROW_LABEL = "\\bf{Heuristic}"

# INNER_HEADER_GROUPS = [
#     ("Succ", "PERC_SOLVED", "higher"),
#     ("Time", "AVG_TEST_TIME", "lower"),
#     ("Node", "AVG_NODES_CREATED", "lower"),
# ]

# CAPTION = "TODO"
# TABLE_LABEL = "tab:haddresults"

# OUTER_HEADER_GROUPS = [
#     ("Ours", lambda df: df["EXPERIMENT_ID"].apply(
#         lambda v: "_main_200" in v or "mainhadd_200" in v)),
#     ("Manual", lambda df: df["EXPERIMENT_ID"].apply(
#         lambda v: "_noinventnoexclude_200" in v or \
#                   "_noinventnoexcludehadd_200" in v)),
# ]

# ROW_GROUPS = [
#     ("LMCut", lambda df: df["EXPERIMENT_ID"].apply(
#        lambda v: "blocks_main_" in v or "blocks_noinventnoexclude_" in v)),
#     ("hAdd", lambda df: df["EXPERIMENT_ID"].apply(lambda v: "hadd" in v)),
# ]

#################### Should not need to change below here #####################


def _main() -> None:
    grouped_means, grouped_stds, grouped_sizes = create_dataframes(
        COLUMN_NAMES_AND_KEYS, GROUPS, DERIVED_KEYS)
    means = grouped_means.reset_index()
    stds = grouped_stds.reset_index()
    sizes = grouped_sizes.reset_index().rename(columns={0: "SIZE"})

    num_inner_headers = len(INNER_HEADER_GROUPS)
    num_outer_headers = len(OUTER_HEADER_GROUPS)

    outer_labels = [label for label, _ in OUTER_HEADER_GROUPS]
    outer_lines = [
        "||" if l in DOUBLE_LINES_AFTER else "|" for l in outer_labels
    ]
    inner_lines = ["|" for _ in range(num_inner_headers * len(outer_labels))]
    for i, outer_line in enumerate(outer_lines):
        inner_lines[1 + num_inner_headers * i] = outer_line

    preamble = """\\begin{table*}
\t\\centering
\t\\scriptsize
\t\\begin{tabular}{| l | """ + \
    "".join("p{0.52cm} " + inner_lines[i] + " "
            for i in range(num_inner_headers * num_outer_headers)) + \
    """}
\t\\hline
\t\\multicolumn{1}{|c|}{} &""" + \
    "\n\t".join("\\multicolumn{" + str(num_inner_headers) + \
                "}{c" + outer_lines[i] + "}{\\bf{" + outer_label + "}} " + (
    "&" if i != num_outer_headers-1 else "\\\\")
    for i, (outer_label, _) in enumerate(OUTER_HEADER_GROUPS)) + \
"""
\t\\hline
\t""" + TOP_ROW_LABEL + """ &
""" + \
    "\t" + \
    "\n\t".join(" & ".join("{\\scriptsize " + inner_label + "}"
                for (inner_label, _, _) in INNER_HEADER_GROUPS) + (
    "&" if i != num_outer_headers-1 else "\\\\")
                for i in range(num_outer_headers)) + \
    "\n\t\\hline"

    body = ""

    for (row_label, row_selector) in ROW_GROUPS:
        # Extract entries.
        entry_to_mean_std_size = {}
        for (outer_label, header_selector) in OUTER_HEADER_GROUPS:
            for (inner_label, inner_header_key, _) in INNER_HEADER_GROUPS:
                entry = (outer_label, inner_label)
                selector = combine_selectors([row_selector, header_selector])
                mean_df = get_df_for_entry(inner_header_key, means, selector)
                std_df = get_df_for_entry(inner_header_key, stds, selector)
                size_df = get_df_for_entry("SIZE", sizes, selector)
                if len(mean_df) == 0:
                    entry_to_mean_std_size[entry] = (np.nan, np.nan, 0)
                    continue
                assert len(mean_df) == len(std_df) == len(size_df) == 1
                mean = mean_df[inner_header_key].item()
                std = std_df[inner_header_key].item()
                size = size_df["SIZE"].item()
                entry_to_mean_std_size[entry] = (mean, std, size)
        # Determine which should be bolded.
        bolded = set()
        inner_label_to_best_mean_std = {}
        inner_label_to_comp = {}
        for inner_label, _, higher_or_lower in INNER_HEADER_GROUPS:
            if higher_or_lower == "higher":
                inner_label_to_comp[inner_label] = gt
            else:
                assert higher_or_lower == "lower"
                inner_label_to_comp[inner_label] = lt
        for (outer_label, inner_label), (mean, std,
                                         _) in entry_to_mean_std_size.items():
            # Special case: exclude Manual
            if outer_label == "Manual":
                continue
            if inner_label not in inner_label_to_best_mean_std:
                inner_label_to_best_mean_std[inner_label] = (mean, std)
            else:
                comp = inner_label_to_comp[inner_label]
                if comp(mean, inner_label_to_best_mean_std[inner_label][0]):
                    inner_label_to_best_mean_std[inner_label] = (mean, std)
        for (outer_label, inner_label), (mean, _,
                                         _) in entry_to_mean_std_size.items():
            # Special case: exclude Manual
            if outer_label == "Manual":
                continue
            best_mean, best_std = inner_label_to_best_mean_std[inner_label]
            if abs(mean - best_mean) <= BOLD_NUM_STDS * best_std:
                bolded.add((outer_label, inner_label))
        # Determine which should be red.
        red = set()
        for entry, (mean, std, size) in entry_to_mean_std_size.items():
            if size < RED_MIN_SIZE:
                red.add(entry)
        # Create table entry.
        body += "\n\t" + row_label
        for (outer_label, _) in OUTER_HEADER_GROUPS:
            for (inner_label, _, _) in INNER_HEADER_GROUPS:
                m, _, _ = entry_to_mean_std_size[(outer_label, inner_label)]
                pre, end = "", ""
                if (outer_label, inner_label) in red:
                    pre = "{\\textcolor{red}" + pre
                    end = end + "}"
                if DO_BOLDING and (outer_label, inner_label) in bolded:
                    pre = "\\bf{" + pre
                    end = end + "}"
                # Special case random options / node expansions
                if np.isnan(m) or (outer_label == "Random" and \
                    inner_label == "Node"):
                    formatted_m = "\\;\\;\\;--"
                elif inner_label == "Time":
                    formatted_m = f"{m:.3f}"
                else:
                    formatted_m = f"{m:.1f}"
                body += " & " + pre + formatted_m + end
        body += " \\\\"

    footer = """\\hline
\t\\end{tabular}
\t\\caption{""" + CAPTION + """}
\t\\label{""" + TABLE_LABEL + """}
\\end{table*}
"""

    final = preamble + body + footer
    print()
    print(final)


if __name__ == "__main__":
    _main()
