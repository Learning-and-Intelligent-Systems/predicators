"""Run the code by taking in a YAML config file, in an interactive mode, as
opposed to submitting a slurm job."""
import argparse
import os
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
#                                                 '..')))
import shlex
import subprocess
import sys

from scripts.cluster_utils import config_to_cmd_flags, generate_run_configs


def _main():
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    # generate configs--will only take the first one
    cfg = next(generate_run_configs(args.config))
    cmd_str = config_to_cmd_flags(cfg)
    # cmd_flags = shlex.split(cmd_str)

    # run the command
    subprocess.run(f"python predicators/main.py {cmd_str}",
                   shell=True,
                   check=False)


if __name__ == "__main__":
    _main()
