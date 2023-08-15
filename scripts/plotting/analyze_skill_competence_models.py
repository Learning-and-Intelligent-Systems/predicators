"""Create plots analyzing skill competence models."""

import os
from pathlib import Path
from typing import Dict, List

import dill as pkl
import numpy as np
from matplotlib import pyplot as plt

from predicators import utils
from predicators.competence_models import SkillCompetenceModel


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
    operator_str_results: Dict[str, Dict[int, SkillCompetenceModel]] = {}
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


def _make_plot(model: SkillCompetenceModel,
               online_learning_cycle: int) -> None:

    # pylint: disable=protected-access
    skill_name = model._skill_name
    history = model._cycle_observations

    title = skill_name
    cycle_lengths = [len(h) for h in history]
    num_data = sum(cycle_lengths)
    num_data_before_cycle = [0] + list(np.cumsum(cycle_lengths))
    success_rates = [np.nan if not o else np.mean(o) for o in history]

    # Reconstruct competences and predictions.
    lookahead = 1
    replay_model = model.__class__("replay")
    map_competences = [replay_model.get_current_competence()]
    predictions = [replay_model.predict_competence(lookahead)]
    for cycle_obs in history:
        for obs in cycle_obs:
            replay_model.observe(obs)
        replay_model.advance_cycle()
        map_competences.append(replay_model.get_current_competence())
        predictions.append(replay_model.predict_competence(lookahead))

    plt.figure()
    plt.title(title)
    plt.xlabel("Skill Trial")
    plt.ylabel("Competence / Outcome")
    plt.xlim((-1, num_data + 1))
    plt.ylim((-0.25, 1.25))
    plt.yticks(np.linspace(0.0, 1.0, 5, endpoint=True))
    # Mark learning cycles.
    for i, x in enumerate(num_data_before_cycle):
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
    # Plot MAP competences, success rates, and predictions.
    for cycle, (comp, rate, pred) in enumerate(
            zip(map_competences, success_rates, predictions)):
        label = "MAP Competence" if cycle == 0 else None
        x_start = num_data_before_cycle[cycle]
        x_end = num_data_before_cycle[cycle + 1]
        plt.plot((x_start, x_end), (comp, comp),
                 color="green",
                 marker="*",
                 label=label)
        label = "Success Rate" if cycle == 0 else None
        plt.plot((x_start, x_end), (rate, rate), color="orange", label=label)
        label = "Predictions" if cycle == 0 else None
        # + 10 just for improved visibility
        plt.plot((x_end, x_end + 10), (comp, pred),
                 color="blue",
                 linestyle="--",
                 label=label)

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
