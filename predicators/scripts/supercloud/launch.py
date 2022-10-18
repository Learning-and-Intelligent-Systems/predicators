"""Launch supercloud experiments defined by config files.

Usage example:

    python scripts/supercloud/launch.py --config example_basic.yaml --user tslvr

Usage example with supercloud dir:

    python scripts/supercloud/launch.py --config example_basic.yaml \
        --user njk --transfer_local_data \
        --supercloud_dir ~/GitHub/research/predicators_behavior

The default branch can be overridden with the --branch flag.
"""

import argparse

from scripts.cluster_utils import DEFAULT_BRANCH, SUPERCLOUD_IP, \
    BatchSeedRunConfig, config_to_cmd_flags, config_to_logfile, \
    generate_run_configs, get_cmds_to_prep_repo, run_cmds_on_machine, \
    run_command_with_subprocess
from scripts.supercloud.submit_supercloud_job import submit_supercloud_job


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--user", required=True, type=str)
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH)
    # This flag is used internally by the script.
    parser.add_argument("--on_supercloud", action="store_true")
    parser.add_argument("--transfer_local_data", action="store_true")
    parser.add_argument("--supercloud_dir", default="~/predicators", type=str)
    args = parser.parse_args()
    # If we're not yet on supercloud, ssh in and prepare. Then, we will
    # run this file again, but with the --on_supercloud flag.
    if not args.on_supercloud:
        _launch_from_local(args.branch, args.user, args.transfer_local_data,
                           args.supercloud_dir, args.config)
        print("Launched experiments.")
    # If we're already on supercloud, launch the experiments.
    else:
        # If we're on supercloud, we can't transfer local
        # data.
        assert not args.transfer_local_data
        _launch_experiments(args.config)


def _launch_from_local(branch: str, user: str, transfer_local_data: bool,
                       supercloud_dir: str, config_file: str) -> None:
    if transfer_local_data:
        # Enter the repo and wipe saved data, approaches and behavior states.
        server_cmds = ["predicate_behavior"]
        server_cmds += [
            "rm -f results/* logs/* saved_approaches/* saved_datasets/*"
        ]
        server_cmds += ["rm -rf tmp_behavior_states/*"]
        run_cmds_on_machine(server_cmds, user, SUPERCLOUD_IP)
        server_cmds = []
        for folder in [
                "saved_approaches", "saved_datasets", "tmp_behavior_states"
        ]:
            cmd = "rsync -avzhe ssh " + \
              f"{folder}/* {user}@{SUPERCLOUD_IP}:{supercloud_dir}/{folder}"
            server_cmds.append(cmd)
        server_cmd_str = "\n".join(server_cmds + ["exit"])
        run_command_with_subprocess(server_cmd_str)

    str_args = " ".join([
        "scripts/supercloud/launch.py", f"--config {config_file}",
        f"--user {user}"
    ])
    # Enter the repo.
    server_cmds = ["predicate_behavior"]
    # Prepare the repo.
    server_cmds.extend(get_cmds_to_prep_repo(branch, transfer_local_data))
    # Finally, run this file again, but with the on_supercloud flag.
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
