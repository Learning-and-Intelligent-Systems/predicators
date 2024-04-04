"""
Run the code by taking in a YAML config file, in an interactive
mode, as opposed to submitting a slurm job.
"""
import sys
import shlex
import argparse
import subprocess

from scripts.cluster_utils import generate_run_configs, config_to_cmd_flags

def _main():
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()

    # generate configs--will only take the first one
    # breakpoint()
    cfg = next(generate_run_configs(args.config))
    cmd_str = config_to_cmd_flags(cfg)
    cmd_flags = shlex.split(cmd_str)
    
    # run the command
    subprocess.run(["python", "predicators/main.py"] + cmd_flags)

if __name__ == "__main__":
    _main()