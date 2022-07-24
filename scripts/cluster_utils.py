"""Utility functions for interacting with clusters."""

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

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


def parse_config(config_filename: str) -> Dict[str, Any]:
    """Parse the YAML config file."""
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    configs_dir = os.path.join(scripts_dir, "configs")
    config_filepath = os.path.join(configs_dir, config_filename)
    with open(config_filepath, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def generate_run_configs(config_filename: str) -> Iterator[RunConfig]:
    """Generate run configs from a (local path) config file."""
    config = parse_config(config_filename)
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


def run_cmds_on_machine(
    cmds: List[str],
    user: str,
    machine: str,
    ssh_key: str = None,
    allowed_return_codes: Tuple[int, ...] = (0, )) -> None:
    """SSH into the machine, run the commands, then exit."""
    host = f"{user}@{machine}"
    ssh_cmd = f"ssh -tt -o StrictHostKeyChecking=no {host}"
    if ssh_key is not None:
        ssh_cmd += f" -i {ssh_key}"
    server_cmd_str = "\n".join(cmds + ["exit"])
    final_cmd = f"{ssh_cmd} << EOF\n{server_cmd_str}\nEOF"
    response = subprocess.run(final_cmd,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.STDOUT,
                              shell=True,
                              check=False)
    if response.returncode not in allowed_return_codes:
        raise RuntimeError(f"Command failed: {final_cmd}")
