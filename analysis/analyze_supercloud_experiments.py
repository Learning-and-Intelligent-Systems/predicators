"""Script to analyze experiments resulting from running the script
analysis/run_supercloud_experiments.sh."""

import glob
import dill as pkl
import pandas as pd
from predicators.src.settings import CFG


def _main() -> None:
    # Gather data.
    all_data = []
    column_names = [
        "ENV", "APPROACH", "EXCLUDED_PREDICATES", "EXPERIMENT_ID", "SEED",
        "NUM_SOLVED", "NUM_TOTAL", "AVG_TEST_TIME", "AVG_NUM_NODES",
        "AVG_PLAN_LEN", "LEARNING_TIME"
    ]
    for filepath in sorted(glob.glob(f"{CFG.results_dir}/*")):
        with open(filepath, "rb") as f:
            run_data = pkl.load(f)
        env, approach, seed, excluded_predicates, experiment_id = filepath[
            8:-4].split("__")
        if not excluded_predicates:
            excluded_predicates = "none"
        data = [
            env, approach, excluded_predicates, experiment_id, seed,
            run_data["num_solved"], run_data["num_total"],
            run_data["avg_suc_time"], run_data["avg_nodes_expanded"],
            run_data["avg_plan_length"], run_data["learning_time"]
        ]
        assert len(data) == len(column_names)
        all_data.append(data)
    if not all_data:
        print(f"No data found in {CFG.results_dir}/, terminating")
        return
    # Group & aggregate data by env name and approach name.
    pd.set_option("display.max_rows", 999999)
    df = pd.DataFrame(all_data)
    df.columns = column_names
    print("RAW DATA:")
    print(df)
    grouped = df.groupby(
        ["ENV", "APPROACH", "EXCLUDED_PREDICATES", "EXPERIMENT_ID"])
    means = grouped.mean()
    stds = grouped.std()
    sizes = grouped.size()
    # Add standard deviations to the printout.
    for col in means:
        for row in means[col].keys():
            mean = means.loc[row, col]
            std = stds.loc[row, col]
            means.loc[row, col] = f"{mean:.2f} ({std:.2f})"
    means["NUM_SEEDS"] = sizes
    print("\n\nAGGREGATED DATA OVER SEEDS:")
    print(means)


if __name__ == "__main__":
    _main()
