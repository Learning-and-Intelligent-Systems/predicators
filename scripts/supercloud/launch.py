"""Launch supercloud experiments defined by config files.

Usage example:

    python scripts/supercloud/launch.py --config example_basic.yaml --user tslvr

The default branch can be overridden with the --branch flag.
"""

import argparse
import sys

from scripts.cluster_utils import DEFAULT_BRANCH, SUPERCLOUD_IP, \
    BatchSeedRunConfig, config_to_cmd_flags, config_to_logfile, \
    generate_run_configs, get_cmds_to_prep_repo, run_cmds_on_machine
from scripts.supercloud.submit_supercloud_job import submit_supercloud_job


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--user", required=True, type=str)
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH)
    # This flag is used internally by the script.
    parser.add_argument("--on_supercloud", action="store_true")
    args = parser.parse_args()
    # If we're not yet on supercloud, ssh in and prepare. Then, we will
    # run this file again, but with the --on_supercloud flag.
    if not args.on_supercloud:
        _launch_from_local(
            args.branch,
            args.user,
        )
        print("Launched experiments.")
    # If we're already on supercloud, launch the experiments.
    else:
        _launch_experiments(args.config)


def _launch_from_local(branch: str, user: str) -> None:
    str_args = " ".join(sys.argv)
    # Enter the repo.
    server_cmds = ["predicate_behavior"]
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
        log_dir = "logs"
        log_prefix = config_to_logfile(cfg, suffix="")
        # Launch a job for this experiment.
        submit_supercloud_job(cfg.experiment_id, log_dir, log_prefix,
                              cmd_flags, cfg.start_seed, cfg.num_seeds,
                              cfg.use_gpu)


if __name__ == "__main__":
    _main()
