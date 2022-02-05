"""Create plots with online learning cycle on the x axis, evaluation
performance on the y axis."""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from predicators.analysis.analyze_results_directory import create_dataframes
pd.options.mode.chained_assignment = None  # default='warn'


PLOT_GROUPS = {
    ("cover_regrasp", "Cover (Regrasp)"): 
    {"cover_regrasp_naive": "Naive",
     "cover_regrasp_targeted": "Targeted"},
}

Y_KEY = "NUM_SOLVED"
Y_LABEL = "Test tasks solved"


def _main() -> None:
    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)
    matplotlib.rcParams.update({'font.size': 16})
    grouped_means, grouped_stds, _ = create_dataframes()
    means = grouped_means.reset_index()
    stds = grouped_stds.reset_index()
    for (group_name, plot_title), experiment_id_dict in PLOT_GROUPS.items():
        # TODO: add error bars
        _, ax = plt.subplots()
        for experiment_id, label in experiment_id_dict.items():
            df = means[means["EXPERIMENT_ID"] == experiment_id]
            df["CYCLE"].replace("None", "-1", inplace=True)
            df["CYCLE"] = df["CYCLE"].map(pd.to_numeric) + 1
            df = df.sort_values("CYCLE")
            df.plot(x="CYCLE", y=Y_KEY, label=label, ax=ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(plot_title)
        ax.set_xlabel("Num online learning episodes")
        ax.set_ylabel(Y_LABEL)
        plt.tight_layout()
        outfile = os.path.join(outdir, f"{group_name}.png")
        plt.savefig(outfile)
        print(f"Wrote out to {outfile}.")


if __name__ == "__main__":
    _main()
