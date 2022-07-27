"""Launch experiments defined in config files locally.

Run experiments sequentially, not in parallel.

    python scripts/local/launch.py --config example_basic.yaml
"""

import argparse
import os
import subprocess

from predicators.scripts.cluster_utils import config_file_to_branch, \
    config_to_cmd_flags, config_to_logfile, generate_run_configs, \
    get_cmds_to_prep_repo


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    branch = config_file_to_branch(args.config)
    # Prepare the repo.
    for cmd in get_cmds_to_prep_repo(branch):
        subprocess.run(cmd, shell=True, check=False)
    # Create the run commands.
    cmds = []
    for cfg in generate_run_configs(args.config):
        cmd_flags = config_to_cmd_flags(cfg)
        logfile = os.path.join("logs", config_to_logfile(cfg))
        cmd_flags = config_to_cmd_flags(cfg)
        cmd = f"python src/main.py {cmd_flags} > {logfile}"
        cmds.append(cmd)
    # Run the commands in order.
    num_cmds = len(cmds)
    for i, cmd in enumerate(cmds):
        print(f"********* RUNNING COMMAND {i+1} of {num_cmds} *********")
        subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    _main()
