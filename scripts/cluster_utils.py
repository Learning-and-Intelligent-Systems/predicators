"""Utility functions for interacting with clusters."""

import os
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Tuple

import yaml

SAVE_DIRS = ["results", "logs", "saved_datasets", "saved_approaches"]
SUPERCLOUD_IP = "txe1-login.mit.edu"
DEFAULT_BRANCH = "master"


@dataclass(frozen=True)
class RunConfig:
    """Config for a single run."""
    experiment_id: str
    approach: str
    env: str
    args: List[str]  # e.g. --make_test_videos
    flags: Dict[str, Any]  # e.g. --num_train_tasks 1
    use_gpu: bool  # e.g. --use_gpu True

    def __post_init__(self) -> None:
        # For simplicity, disallow overrides of the SAVE_DIRS.
        assert "results_dir" not in self.flags
        assert "log_dir" not in self.flags
        assert "approach_dir" not in self.flags
        assert "data_dir" not in self.flags


@dataclass(frozen=True)
class SingleSeedRunConfig(RunConfig):
    """Config for a single run with a single seed."""
    seed: int


@dataclass(frozen=True)
class BatchSeedRunConfig(RunConfig):
    """Config for a run where seeds are batched together."""
    start_seed: int
    num_seeds: int


def config_to_logfile(cfg: RunConfig, suffix: str = ".log") -> str:
    """Create a log file name from a run config."""
    if isinstance(cfg, SingleSeedRunConfig):
        seed = cfg.seed
    else:
        assert isinstance(cfg, BatchSeedRunConfig)
        seed = None
    return f"{cfg.env}__{cfg.approach}__{cfg.experiment_id}__{seed}" + suffix


def config_to_cmd_flags(cfg: RunConfig) -> str:
    """Create a string of command flags from a run config."""
    arg_str = " ".join(f"--{a}" for a in cfg.args)
    flag_str = " ".join(f"--{f} {v}" for f, v in cfg.flags.items())
    args_and_flags_str = (f"--env {cfg.env} "
                          f"--approach {cfg.approach} "
                          f"--experiment_id {cfg.experiment_id} "
                          f"{arg_str} "
                          f"{flag_str}")
    if isinstance(cfg, SingleSeedRunConfig):
        args_and_flags_str += f" --seed {cfg.seed}"
    return args_and_flags_str


def parse_configs(config_filename: str) -> Iterator[Dict[str, Any]]:
    """Parse the YAML config file."""
    scripts_dir = os.path.dirname(os.path.realpath(__file__))
    configs_dir = os.path.join(scripts_dir, "configs")
    config_filepath = os.path.join(configs_dir, config_filename)
    with open(config_filepath, "r", encoding="utf-8") as f:
        for config in yaml.safe_load_all(f):
            yield config


def generate_run_configs(config_filename: str,
                         batch_seeds: bool = False) -> Iterator[RunConfig]:
    """Generate run configs from a (local path) config file."""
    for config in parse_configs(config_filename):
        start_seed = config["START_SEED"]
        num_seeds = config["NUM_SEEDS"]
        args = config["ARGS"]
        flags = config["FLAGS"]
        if "USE_GPU" in config.keys():
            use_gpu = config["USE_GPU"]
        else:
            use_gpu = False
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
                # Loop or batch over seeds.
                if batch_seeds:
                    yield BatchSeedRunConfig(experiment_id, approach, env,
                                             args, run_flags.copy(), use_gpu,
                                             start_seed, num_seeds)
                else:
                    for seed in range(start_seed, start_seed + num_seeds):
                        yield SingleSeedRunConfig(experiment_id,
                                                  approach, env, args,
                                                  run_flags.copy(), use_gpu,
                                                  seed)


def get_cmds_to_prep_repo(branch: str) -> List[str]:
    """Get the commands that should be run while already in the repository but
    before launching the experiments."""
    return [
        "mkdir -p logs",
        "git stash",
        "git fetch --all",
        f"git checkout {branch}",
        "git pull",
        # Remove old results.
        "rm -f results/* logs/* saved_approaches/* saved_datasets/*",
    ]


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
