"""Script to analyze experiments resulting from running the script
analysis/run_supercloud_experiments.sh.
"""

import pickle as pkl
import glob
import pandas as pd


def _main() -> None:
    # Gather data.
    all_data = []
    column_names = ["ENV", "APPROACH", "SEED", "TEST_TASKS_SOLVED",
                    "TEST_TASKS_TOTAL", "TOTAL_TEST_TIME", "TOTAL_TIME"]
    for filepath in sorted(glob.glob("results/*")):
        with open(filepath, "rb") as f:
            run_data = pkl.load(f)
        env, approach, seed = filepath[8:-4].split("__")
        data = [env, approach, seed, run_data["test_tasks_solved"],
                run_data["test_tasks_total"], run_data["total_test_time"],
                run_data["total_time"]]
        assert len(data) == len(column_names)
        all_data.append(data)
    if not all_data:
        print("No data found in results/, terminating")
        return
    # Group & aggregate data by env name and approach name.
    df = pd.DataFrame(all_data)
    df.columns = column_names
    means = df.groupby(["ENV", "APPROACH"]).mean()
    stds = df.groupby(["ENV", "APPROACH"]).std()
    # Add standard deviations to the printout.
    for col in means:
        for row in means[col].keys():
            mean = means.loc[row, col]
            std = stds.loc[row, col]
            means.loc[row, col] = f"{mean:.2f} ({std:.2f})"
    print(means)


if __name__ == "__main__":
    _main()
