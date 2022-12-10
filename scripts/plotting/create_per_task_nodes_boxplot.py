"""Create histograms for per-task metrics that start with "PER_TASK_".

Assumes that files in the results/ directory can be grouped by
experiment ID alone.
"""

import glob
import os
import re
from collections import defaultdict

import dill as pkl
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from predicators.settings import CFG

DPI = 500
sns.set_theme(style="ticks")
approaches = ["pred_error", "cluster_and_intersect", "cluster_and_search"]


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    expanded_data = defaultdict(list)
    created_data = defaultdict(list)
    for filepath in sorted(glob.glob(f"{CFG.results_dir}/*")):
        # Make sure that this results file corresponds to one
        # of the approaches that we're hoping to plot.
        for approach_name in approaches:
            if approach_name in filepath:
                break
        else:
            continue
        with open(filepath, "rb") as f:
            outdata = pkl.load(f)
        run_data_defaultdict = outdata["results"]
        run_data = dict(
            run_data_defaultdict)  # want to crash if key not found!
        for key in run_data:
            if not key.startswith("PER_TASK_"):
                continue
            match = re.match(r"PER_TASK_task\d+_nodes_(created|expanded)", key)
            if match is None:
                continue
            created_or_expanded = match.groups()[0]
            if created_or_expanded == "created":
                created_data[approach_name].append(run_data[key])
            else:
                expanded_data[approach_name].append(run_data[key])
        # Failed tasks do not have a nodes created/expanded, so we need to automatically
        # populate this with an 'inf'.
        num_test_tasks = outdata["config"].num_test_tasks
        if len(created_data[approach_name]) < num_test_tasks:
            for _ in range(num_test_tasks - len(created_data[approach_name])):
                created_data[approach_name].append(float('inf'))
        if len(expanded_data[approach_name]) < num_test_tasks:
            for _ in range(num_test_tasks - len(expanded_data[approach_name])):
                expanded_data[approach_name].append(float('inf'))
    if not created_data and not expanded_data:
        raise ValueError(f"No per-task node data found in {CFG.results_dir}/")
        
    # Convert to DataFrames
    nodes_created_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in created_data.items()]))
    nodes_created_df = nodes_created_df.fillna(float('inf'))
    nodes_expanded_df = pd.DataFrame(dict([(k,pd.Series(v)) for k,v in expanded_data.items()]))
    nodes_expanded_df = nodes_created_df.fillna(float('inf'))

    # Replace all infs with the max possible value in the df
    max_value = np.nanmax(nodes_created_df[nodes_created_df != np.inf])
    nodes_created_df.replace([np.inf, -np.inf], max_value, inplace=True)


    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    ax.set_xscale("log")
    sns.boxplot(data=nodes_created_df, orient='h', whis=[0, 100], width=0.6, palette="vlag")
    # sns.stripplot(data=nodes_created_df, size=4, color=".3", linewidth=0)
    plt.show()
    # import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    _main()
