"""Script to analyze experiments resulting from running the script
analysis/run_supercloud_experiments.sh."""

from typing import Tuple
import glob
import dill as pkl
import numpy as np
import pandas as pd
from predicators.src.settings import CFG


def create_dataframes() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Returns means standard deviations, and sizes.
    all_data = []
    column_names = [
        "ENV", "APPROACH", "EXCLUDED_PREDICATES", "EXPERIMENT_ID", "SEED",
        "CYCLE", "NUM_SOLVED", "AVG_TEST_TIME", "AVG_SKELETONS",
        "MIN_SKELETONS", "MAX_SKELETONS", "AVG_NODES_EXPANDED",
        "AVG_NODES_CREATED", "AVG_NUM_NSRTS", "AVG_DISCOVERED_FAILURES",
        "AVG_PLAN_LEN", "AVG_EXECUTION_FAILURES", "AVG_NUM_PREDS",
        "LEARNING_TIME", "NUM_TRANSITIONS"
    ]
    groups = [
        "ENV", "APPROACH", "EXCLUDED_PREDICATES", "EXPERIMENT_ID", "CYCLE"
    ]
    some_nonempty_experiment_id = False
    for filepath in sorted(glob.glob(f"{CFG.results_dir}/*")):
        with open(filepath, "rb") as f:
            run_data_defaultdict = pkl.load(f)
        (env, approach, seed, excluded_predicates, experiment_id,
         online_learning_cycle) = filepath[8:-4].split("__")
        if not excluded_predicates:
            excluded_predicates = "none"
        if experiment_id:
            some_nonempty_experiment_id = True
        run_data = dict(
            run_data_defaultdict)  # want to crash if key not found!
        data = [
            env,
            approach,
            excluded_predicates,
            experiment_id,
            seed,
            online_learning_cycle,
            run_data["num_solved"],
            run_data["avg_suc_time"],
            run_data["avg_num_skeletons_optimized"],
            run_data["min_skeletons_optimized"],
            run_data["max_skeletons_optimized"],
            run_data["avg_num_nodes_expanded"],
            run_data["avg_num_nodes_created"],
            run_data["avg_num_nsrts"],
            run_data["avg_num_failures_discovered"],
            run_data["avg_plan_length"],
            run_data["avg_execution_failures"],
            run_data["avg_num_preds"],
            run_data["learning_time"],
            run_data["num_transitions"],
        ]
        assert len(data) == len(column_names)
        all_data.append(data)
    if not all_data:
        print(f"No data found in {CFG.results_dir}/, terminating")
        return
    if some_nonempty_experiment_id:
        assert column_names[0] == groups[0] == "ENV"
        assert column_names[1] == groups[1] == "APPROACH"
        assert column_names[2] == groups[2] == "EXCLUDED_PREDICATES"
        for _ in range(3):
            for data in all_data:
                del data[0]
            del column_names[0]
            del groups[0]
    else:
        assert column_names[3] == groups[3] == "EXPERIMENT_ID"
        for data in all_data:
            del data[3]
        del column_names[3]
        del groups[3]
    # Group & aggregate data.
    pd.set_option("display.max_rows", 999999)
    df = pd.DataFrame(all_data)
    df.columns = column_names
    print("RAW DATA:")
    print(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    grouped = df.groupby(groups)
    means = grouped.mean()
    stds = grouped.std(ddof=0)
    sizes = grouped.size()
    return means, stds, sizes


def _main() -> None:
    means, stds, sizes = create_dataframes()
    # Add standard deviations to the printout.
    for col in means:
        for row in means[col].keys():
            mean = means.loc[row, col]
            std = stds.loc[row, col]
            means.loc[row, col] = f"{mean:.2f} ({std:.2f})"
    means["NUM_SEEDS"] = sizes
    print("\n\nAGGREGATED DATA OVER SEEDS:")
    print(means)
    means.to_csv("supercloud_analysis.csv")
    print("\n\nWrote out table to supercloud_analysis.csv")


if __name__ == "__main__":
    _main()
