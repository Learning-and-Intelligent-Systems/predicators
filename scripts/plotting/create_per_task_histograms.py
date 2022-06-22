"""Create histograms for per-task metrics that start with "PER_TASK_".

Assumes that files in the results/ directory can be grouped by
experiment ID alone.
"""

import glob
import os
import re
from collections import defaultdict

import dill as pkl
import matplotlib.pyplot as plt

from predicators.src.settings import CFG

DPI = 500


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    experiment_ids = set()
    solve_data = defaultdict(list)
    exec_data = defaultdict(list)
    for filepath in sorted(glob.glob(f"{CFG.results_dir}/*")):
        with open(filepath, "rb") as f:
            outdata = pkl.load(f)
        config = outdata["config"].__dict__.copy()
        run_data_defaultdict = outdata["results"]
        assert not set(config.keys()) & set(run_data_defaultdict.keys())
        run_data_defaultdict.update(config)
        _, _, _, _, _, experiment_id, _ = filepath[8:-4].split("__")
        experiment_ids.add(experiment_id)
        run_data = dict(
            run_data_defaultdict)  # want to crash if key not found!
        run_data.update({"experiment_id": experiment_id})
        for key in run_data:
            if not key.startswith("PER_TASK_"):
                continue
            match = re.match(r"PER_TASK_task\d+_(solve|exec)_time", key)
            assert match is not None
            solve_or_exec = match.groups()[0]
            if solve_or_exec == "solve":
                solve_data[experiment_id].append(run_data[key])
            else:
                exec_data[experiment_id].append(run_data[key])
    if not solve_data and not exec_data:
        raise ValueError(f"No per-task data found in {CFG.results_dir}/")
    print("Found the following experiment IDs:")
    for experiment_id in experiment_ids:
        print(experiment_id)
        _, (ax1, ax2) = plt.subplots(1, 2)
        ax1.hist(solve_data[experiment_id])
        ax2.hist(exec_data[experiment_id])
        ax1.set_title("Per-task solve time histogram")
        ax2.set_title("Per-task execution time histogram")
        outfile = os.path.join(outdir, f"{experiment_id}__per_task.png")
        plt.savefig(outfile, dpi=DPI)
        print(f"\tFound {len(solve_data[experiment_id])} task solve times and "
              f"{len(exec_data[experiment_id])} task execution times")
        print(f"\tWrote out to {outfile}")


if __name__ == "__main__":
    _main()
