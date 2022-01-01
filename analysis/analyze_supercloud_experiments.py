"""Script to analyze experiments resulting from running the script
run_supercloud_experiments.sh.
"""

import pickle as pkl
import glob
import pandas as pd


def main() -> None:
    """Script entry point.
    """
    # Gather data.
    all_data = []
    for filepath in sorted(glob.glob("results/*")):
        with open(filepath, "rb") as f:
            run_data = pkl.load(f)
        env, approach, seed = filepath[8:-4].split("__")
        data = [env, approach, seed, run_data["test_tasks_solved"],
                run_data["test_tasks_total"], run_data["total_test_time"],
                run_data["total_time"]]
        all_data.append(data)
    # Group & aggregate data by seed.
    df = pd.DataFrame(all_data)
    df.columns = ["ENV", "APPROACH", "SEED", "TEST_TASKS_SOLVED",
                  "TEST_TASKS_TOTAL", "TOTAL_TEST_TIME", "TOTAL_TIME"]
    means = df.groupby(["ENV", "APPROACH"]).mean()
    stds = df.groupby(["ENV", "APPROACH"]).std()
    # Add standard deviations to the printout.
    for col in means:
        for row in means[col].keys():
            mean = means.loc[row, col]
            std = stds.loc[row, col]
            means.loc[row, col] = f"{mean:.3f} ({std:.3f})"
    print(means)


if __name__ == "__main__":
    main()
