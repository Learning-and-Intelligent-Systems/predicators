"""Create plots analyzing competence data from active sampler learning."""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import dill as pkl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from predicators import utils
from predicators.competence_models import SkillCompetenceModel
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
    operator_str_results: Dict[str, Dict[int, Tuple[Array, Array]]] = {}
    for load_path in Path(".").glob(load_path_pattern):
        with open(load_path, "rb") as f:
            competence_model = pkl.load(f)
        x, y = _competence_model_to_regression_data(competence_model)
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
        _make_plot(x, y, operator_str, last)


def _competence_model_to_regression_data(
        competence_model: SkillCompetenceModel) -> Tuple[Array, Array]:
    # Replay the data in the competence model and create dataset where inputs
    # are number of data seen so far and outputs are estimated competence.
    new_model = competence_model.__class__("replay")
    all_observations = competence_model._cycle_observations  # pylint: disable=protected-access
    del competence_model
    x_lst: List[float] = []
    y_lst: List[float] = []
    num_data = 0
    for observations in all_observations:
        for obs in observations:
            new_model.observe(obs)
            num_data += 1
        x_lst.append(num_data)
        y_lst.append(new_model.get_current_competence())
        new_model.advance_cycle()
    x_arr = np.array(x_lst, dtype=np.float32)
    y_arr = np.array(y_lst, dtype=np.float32)
    return (x_arr, y_arr)


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
