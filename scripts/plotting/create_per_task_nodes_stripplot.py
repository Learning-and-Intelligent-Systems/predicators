"""Create histograms for per-task metrics that start with "PER_TASK_".

Assumes that files in the results/ directory can be grouped by
experiment ID alone.
"""
import glob
import os
import re
from collections import defaultdict
from typing import Dict, List

import dill as pkl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from predicators.settings import CFG

sns.set_theme(style="ticks")
# Names of particular approaches we want to plot.
approaches = [
    "pnad_search", "cluster_and_intersect_sideline_prederror",
    "cluster_and_intersect", "cluster_and_search"
]
# Mapping from approach names to names we actually want to use
# in the plot.
approach_rename = {
    "cluster_and_intersect_sideline_prederror": "CI + QE",
    "pnad_search": "Ours",
    "cluster_and_intersect": "CI",
    "cluster_and_search": "LOFT"
}
# Names of envs we want to plot.
behavior_envs = [
    "collecting_aluminum_cans", "sorting_books", "locking_every_window",
    "opening_presents"
]
envs = [
    "repeated_nextto_single_option", "repeated_nextto_painting", "painting",
    "satellites", "satellites_simple", "screws"
] + behavior_envs
env_rename = {
    "repeated_nextto_single_option": "Cluttered 1D",
    "repeated_nextto_painting": "Cluttered Painting",
    "painting": "Painting",
    "satellites": "Satellites",
    "satellites_simple": "Satellites Simple",
    "screws": "Screws",
    "collecting_aluminum_cans": "Collecting Cans",
    "sorting_books": "Sorting Books",
    "locking_every_window": "Locking Windows",
    "opening_presents": "Opening Presents"
}
# This script only analyzes nodes created/expanded for a fixed amount of
# training data. This below variable controls this amount.
num_demos_to_consider = 50
num_demos_to_consider_behavior = 10
DPI = 500
sns.set(font_scale=1.5)


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    expanded_data: Dict[str,
                        Dict[str,
                             List]] = defaultdict(lambda: defaultdict(list))
    created_data: Dict[str,
                       Dict[str,
                            List]] = defaultdict(lambda: defaultdict(list))
    for filepath in sorted(glob.glob(f"{CFG.results_dir}/*")):
        with open(filepath, "rb") as f:
            outdata = pkl.load(f)
        # Make sure that this results file corresponds to one
        # of the approaches that we're hoping to plot.
        for approach_name in approaches:
            if approach_name == outdata["config"].strips_learner and outdata[
                    "config"].approach == "nsrt_learning":
                break
        else:
            continue
        if approach_rename.get(approach_name, None) is not None:
            approach_name = approach_rename[approach_name]
        # Make sure that this results file corresponds to
        # one of the envs that we're hoping to include in the
        # plot.
        correct_env_name = None
        for env_name in envs:
            if outdata["config"].env == "behavior":
                assert len(outdata["config"].behavior_task_list) == 1
                if outdata["config"].behavior_task_list[0] in behavior_envs:
                    correct_env_name = outdata["config"].behavior_task_list[0]
                    break
            elif env_name == outdata["config"].env:
                correct_env_name = env_name
                break
        else:
            continue
        # Skip this result file if it didn't use the correct amount
        # of training data, but only for non-BEHAVIOR domains.
        if outdata["config"].env != "behavior" and \
            outdata["config"].num_train_tasks != num_demos_to_consider or \
            outdata["config"].env == "behavior" and \
            outdata["config"].num_train_tasks != \
                num_demos_to_consider_behavior:
            continue

        run_data_defaultdict = outdata["results"]
        run_data = dict(run_data_defaultdict)
        probs_solved_in_this_file = 0
        for key, val in run_data.items():
            if not key.startswith("PER_TASK_"):
                continue
            match = re.match(r"PER_TASK_task\d+_nodes_(created|expanded)", key)
            if match is None or val == 0.0:
                continue
            created_or_expanded = match.groups()[0]
            if created_or_expanded == "created":
                # There will be the same number of per task entries for
                # nodes created and expanded, so we increment the
                # probs_solved variable only here.
                probs_solved_in_this_file += 1
                created_data[approach_name]["values"].append(val)
                created_data[approach_name]["env_names"].append(
                    correct_env_name)
            else:
                expanded_data[approach_name]["values"].append(val)
                expanded_data[approach_name]["env_names"].append(
                    correct_env_name)
        # Failed tasks do not have a nodes created/expanded, so we need to
        # automatically populate this with a random high value (with some
        # jitter to make it appear distinct in the plot).
        num_test_tasks = outdata["config"].num_test_tasks
        if probs_solved_in_this_file < num_test_tasks:
            for _ in range(num_test_tasks - probs_solved_in_this_file):
                if correct_env_name not in behavior_envs:
                    value = 1e06
                else:
                    value = 1e03
                created_data[approach_name]["values"].append(value)
                created_data[approach_name]["env_names"].append(
                    correct_env_name)
                expanded_data[approach_name]["values"].append(value)
                expanded_data[approach_name]["env_names"].append(
                    correct_env_name)
    if not created_data and not expanded_data:
        raise ValueError(f"No per-task node data found in {CFG.results_dir}/")

    # Create a list of dataframes corresponding to nodes created by each
    # of the different methods.
    nodes_created_dfs_list = []
    for key, val in created_data.items():
        nodes_created_df = pd.DataFrame.from_dict(val)
        nodes_created_dfs_list.append(nodes_created_df)
    # Repeat for nodes expanded.
    nodes_expanded_dfs_list = []
    for key, val in expanded_data.items():
        nodes_expanded_df = pd.DataFrame.from_dict(val)
        nodes_expanded_dfs_list.append(nodes_expanded_df)

    # Concatenate these dataframes into one that uses the approach name as
    # an index, then convert the approach name into a column for plotting.
    all_methods_created_dfs = pd.concat(nodes_created_dfs_list,
                                        keys=list(created_data.keys()))
    all_methods_created_dfs = all_methods_created_dfs.reset_index()
    all_methods_expanded_dfs = pd.concat(nodes_expanded_dfs_list,
                                         keys=list(expanded_data.keys()))
    all_methods_expanded_dfs = all_methods_expanded_dfs.reset_index()

    for env_name in envs:
        if env_name in behavior_envs:
            correct_num_demos = num_demos_to_consider_behavior
        else:
            correct_num_demos = num_demos_to_consider
        curr_created_df = all_methods_created_dfs.loc[
            all_methods_created_dfs["env_names"] == env_name]
        curr_expanded_df = all_methods_expanded_dfs.loc[
            all_methods_expanded_dfs["env_names"] == env_name]

        # Initialize the figure with a logarithmic x axis
        f0, ax0 = plt.subplots()
        f1, ax1 = plt.subplots()
        ax0.set_xscale("log")
        ax1.set_xscale("log")
        # Create the plot for nodes_created. Here, "level_0" will correspond
        # to the approach name because of how the reset_index() command above
        # works.
        sns.stripplot(
            data=curr_created_df,
            x="values",
            y="level_0",
            orient='h',
            size=2.5,
            alpha=0.6,
            ax=ax0,
            marker="D",
            palette=sns.color_palette("hls", 6),
            # NOTE: This below line might need to get
            # changed if different approaches are used.
            order=["Ours", "CI", "LOFT", "CI + QE"],
            jitter=True)
        sns.violinplot(data=curr_created_df,
                       x="values",
                       y="level_0",
                       orient="h",
                       order=["Ours", "CI", "LOFT", "CI + QE"],
                       ax=ax0)
        ax0.set(xlabel="Nodes Created",
                ylabel="Learning Approach",
                title=f"Nodes Created for {env_rename[env_name]} with " +
                f"{correct_num_demos} Demos")
        # Create the plot for nodes_expanded.
        sns.stripplot(
            data=curr_expanded_df,
            x="values",
            y="level_0",
            # hue="env_names",
            orient='h',
            size=2.5,
            alpha=0.6,
            ax=ax1,
            marker="D",
            palette=sns.color_palette("hls", 6),
            # NOTE: This below line might need to get
            # changed if different approaches are used.
            order=["Ours", "CI", "LOFT", "CI + QE"],
            jitter=True)
        sns.violinplot(data=curr_expanded_df,
                       x="values",
                       y="level_0",
                       orient="h",
                       order=["Ours", "CI", "LOFT", "CI + QE"],
                       ax=ax1)
        ax1.set(xlabel="Nodes Expanded",
                ylabel="Learning Approach",
                title=f"Nodes Expanded for {env_rename[env_name]} with " +
                f"{correct_num_demos} Demos")

        # Save figures
        outfile = os.path.join(outdir, f"nodes_created_{env_name}.png")
        f0.set_size_inches(10, 4)
        f0.savefig(outfile, bbox_inches='tight')
        outfile = os.path.join(outdir, f"nodes_expanded_{env_name}.png")
        f1.set_size_inches(10, 4)
        f1.savefig(outfile, bbox_inches='tight')


if __name__ == "__main__":
    _main()
