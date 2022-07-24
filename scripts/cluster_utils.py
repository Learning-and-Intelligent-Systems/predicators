"""Utility functions for interacting with clusters."""

import os
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List

import yaml


@dataclass(frozen=True)
class RunConfig:
    """Config for a single run."""
    experiment_id: str
    approach: str
    env: str
    seed: int
    branch: str  # e.g. master
    args: List[str]  # e.g. --make_test_videos
    flags: Dict[str, Any]  # e.g. --num_train_tasks 1


def generate_run_configs(config_filename: str) -> Iterator[RunConfig]:
    """Generate run configs from a (local path) config file."""
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    configs_dir = os.path.join(scripts_dir, "configs")
    config_filepath = os.path.join(configs_dir, config_filename)
    with open(config_filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Loop over seeds.
    start_seed = config["START_SEED"]
    num_seeds = config["NUM_SEEDS"]
    args = config["ARGS"]
    flags = config["FLAGS"]
    branch = config["BRANCH"]
    for seed in range(start_seed, start_seed + num_seeds):
        # Loop over approaches.
        for approach_exp_id, approach_config in config["APPROACHES"].items():
            approach = approach_config["NAME"]
            # Loop over envs.
            for env_exp_id, env_config in config["ENVS"].items():
                env = env_config["NAME"]
                # Create the experiment ID and flags.
                experiment_id = f"{env_exp_id}-{approach_exp_id}"
                run_flags = flags.copy()
                run_flags.update(approach_config["FLAGS"])
                run_flags.update(env_config["FLAGS"])
                # Finish the run config.
                run_config = RunConfig(experiment_id, approach, env, seed,
                                       branch, args, run_flags)
                yield run_config
