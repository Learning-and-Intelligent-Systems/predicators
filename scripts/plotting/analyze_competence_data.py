"""Create plots analyzing competence data from active sampler learning."""

import os
from pathlib import Path
from typing import Dict, List

import dill as pkl
import numpy as np
from matplotlib import pyplot as plt

from predicators import utils
from predicators.competence_models import LatentVariableSkillCompetenceModel


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
    operator_str_results: Dict[str,
                               Dict[int,
                                    LatentVariableSkillCompetenceModel]] = {}
    for load_path in Path(".").glob(load_path_pattern):
        with open(load_path, "rb") as f:
            competence_model = pkl.load(f)
        _, operator_str, tail = load_path.name.rsplit("_", maxsplit=2)
        if operator_str not in operator_str_results:
            operator_str_results[operator_str] = {}
        online_learning_cycle_str, _ = tail.split(".")
        if online_learning_cycle_str == "None":
            continue
        online_learning_cycle = int(online_learning_cycle_str)
        operator_str_results[operator_str][
            online_learning_cycle] = competence_model
    # Only plot last cycle.
    for operator_str, operator_results in operator_str_results.items():
        last = max(operator_results)
        competence_model = operator_results[last]
        _make_plot(competence_model, last)


def _make_plot(model: LatentVariableSkillCompetenceModel,
               online_learning_cycle: int) -> None:

    # pylint: disable=protected-access
    cp_inputs = model._get_regressor_inputs()
    regressor = model._competence_regressor
    skill_name = model._skill_name
    history = model._cycle_observations
    posterior_competences = model._posterior_competences
    num_data = model._get_current_num_data()

    title = skill_name
    model_outputs = None
    if regressor is not None:
        params = regressor.get_transformed_params()
        title += f" [{params[0]:.3f} {params[1]:.3f} {params[2]:.3f}]"
        model_outputs = [regressor.predict_beta(n).mean() for n in cp_inputs]
    success_rate = [np.nan if not o else np.mean(o) for o in history]

    plt.figure()
    plt.title(title)
    plt.xlabel("Skill Trial")
    plt.ylabel("Competence / Outcome")
    plt.xlim((-1, num_data + 1))
    plt.ylim((-0.25, 1.25))
    plt.yticks(np.linspace(0.0, 1.0, 5, endpoint=True))
    # Mark learning cycles.
    for i, x in enumerate(cp_inputs):
        label = "Learning Cycle" if i == 0 else None
        plt.plot((x, x), (-1.1, 2.1),
                 linestyle="--",
                 color="gray",
                 label=label)
    # Plot observation data.
    observations = [o for co in history for o in co]
    timesteps = np.arange(len(observations))
    plt.scatter(timesteps,
                observations,
                marker="o",
                color="red",
                label="Outcomes")
    # Plot competence progress model outputs (betas).
    means: List[float] = []
    stds: List[float] = []
    if model_outputs:
        for rv in model_outputs:
            mean = rv.mean()
            std = rv.std()
            means.append(mean)
            stds.append(std)
        plt.plot(cp_inputs, means, color="blue", marker="+", label="CP Model")
        lb = np.subtract(means, stds)
        plt.plot(cp_inputs, lb, color="blue", linestyle="--")
        ub = np.add(means, stds)
        plt.plot(cp_inputs, ub, color="blue", linestyle="--")
    # Plot MAP competences.
    for cycle, rv in enumerate(posterior_competences):
        label = "MAP Competence" if cycle == 0 else None
        x_start = cp_inputs[cycle]
        if cycle == len(posterior_competences) - 1:
            x_end = x_start  # just a point
        else:
            x_end = cp_inputs[cycle + 1]
        y = rv.mean()
        plt.plot((x_start, x_end), (y, y),
                 color="green",
                 marker="*",
                 label=label)
    # Plot success rates within cycles.
    for cycle, rate in enumerate(success_rate):
        label = "Cycle Success Rate" if cycle == 0 else None
        x_start = cp_inputs[cycle]
        if cycle == len(posterior_competences) - 1:
            x_end = x_start  # just a point
        else:
            x_end = cp_inputs[cycle + 1]
        y = rate
        plt.plot((x_start, x_end), (y, y), color="orange", label=label)
    # Finish figure.
    plt.legend(loc="center right", framealpha=1.0)
    outdir = Path(__file__).parent / "results" / "competence_analysis"
    os.makedirs(outdir, exist_ok=True)
    outfile = outdir / f"{skill_name}_{online_learning_cycle}.png"
    plt.savefig(outfile, dpi=350)
    print(f"Wrote out to {outfile}")
    plt.close()


if __name__ == "__main__":
    _main()
