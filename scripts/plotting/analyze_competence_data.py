"""Create plots analyzing competence data from active sampler learning."""

import os
from pathlib import Path
from typing import Dict, Optional

import dill as pkl
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from predicators import utils
from predicators.structs import Array


def _main() -> None:
    # Parse & validate args
    args = utils.parse_args()
    utils.update_config(args)
    # Load data.
    approach_load_path = utils.get_approach_save_path_str()
    load_path_pattern = "_".join([
        approach_load_path,
        "*",  # operator[arguments]
        "*.competence"  # online learning cycle
    ])
    operator_str_results: Dict[str, Dict[Optional[int]]] = {}
    for load_path in Path(".").glob(load_path_pattern):
        with open(load_path, "rb") as f:
            x, y = pkl.load(f)
        _, operator_str, tail = load_path.name.rsplit("_", maxsplit=2)
        if operator_str not in operator_str_results:
            operator_str_results[operator_str] = {}
        online_learning_cycle_str, _ = tail.split(".")
        if online_learning_cycle_str == "None":
            continue
        online_learning_cycle = int(online_learning_cycle_str)
        operator_str_results[operator_str][online_learning_cycle] = (x, y)
    # Only plot last cycle.
    for operator_str, operator_results in operator_str_results.items():
        last = max(operator_results)
        x, y = operator_results[last]
        _make_plot(x, y, operator_str, online_learning_cycle)


def _make_plot(x: Array, y: Array, operator_str: str,
               online_learning_cycle: int) -> None:
    fig = plt.figure()
    plt.title(f"{operator_str}; Cycle={online_learning_cycle}")
    plt.plot(x, y, marker="o")
    plt.ylim(-0.1, 1.1)
    ax = fig.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlabel("Num data")
    plt.ylabel("Competence")
    plt.tight_layout()
    outdir = Path(__file__).parent / "results" / "competence_analysis"
    os.makedirs(outdir, exist_ok=True)
    outfile = outdir / f"{operator_str}_{online_learning_cycle}.png"
    plt.savefig(outfile)
    print(f"Wrote out to {outfile}")
    plt.close()


if __name__ == "__main__":
    _main()
