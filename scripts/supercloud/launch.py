"""Launch supercloud experiments defined by config files.

Usage example:

    python scripts/supercloud/launch.py --config example_basic.yaml --user tslvr
"""

import argparse
import sys

from predicators.scripts.cluster_utils import SUPERCLOUD_IP, \
    BatchSeedRunConfig, config_to_cmd_flags, config_to_logfile, \
    generate_run_configs, get_cmds_to_prep_repo, parse_configs, \
    run_cmds_on_machine
from predicators.scripts.supercloud.submit_supercloud_job import \
    submit_supercloud_job


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--user", required=True, type=str)
    # This flag is used internally by the script.
    parser.add_argument("--on_supercloud", action="store_true")
    args = parser.parse_args()
    # If we're not yet on supercloud, ssh in and prepare. Then, we will
    # run this file again, but with the --on_supercloud flag.
    if not args.on_supercloud:
        _launch_from_local(args.config, args.user)
        print("Launched experiments.")
    # If we're already on supercloud, launch the experiments.
    else:
        _launch_experiments(args.config)


def _launch_from_local(config_file: str, user: str) -> None:
    configs = list(parse_configs(config_file))
    assert configs
    branch = configs[0]["BRANCH"]
    assert all(c["BRANCH"] == branch for c in configs), \
        "Experiments defined in the same config must have the same branch."
    str_args = " ".join(sys.argv)
    # Enter the repo.
    server_cmds = ["predicate"]
    # Prepare the repo.
    server_cmds.extend(get_cmds_to_prep_repo(branch))
    # Run this file again, but with the on_supercloud flag.
    server_cmds.append(f"python {str_args} --on_supercloud")
    run_cmds_on_machine(server_cmds, user, SUPERCLOUD_IP)


def _launch_experiments(config_file: str) -> None:
    # Loop over run configs.
    for cfg in generate_run_configs(config_file, batch_seeds=True):
        assert isinstance(cfg, BatchSeedRunConfig)
        cmd_flags = config_to_cmd_flags(cfg)
        # The None is a placeholder for seed.
        log_dir = "logs"
        log_prefix = config_to_logfile(cfg, suffix="")
        # Launch a job for this experiment.
        submit_supercloud_job(cfg.experiment_id, log_dir, log_prefix,
                              cmd_flags, cfg.start_seed, cfg.num_seeds)


if __name__ == "__main__":
    _main()
