"""Helper script for just printing out the commands for a config.

Unlike other scripts, doesn't change the branch or prepare the repo.
"""

import argparse

from scripts.cluster_utils import config_to_cmd_flags, generate_run_configs


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    args = parser.parse_args()
    for i, cfg in enumerate(generate_run_configs(args.config)):
        cmd_flags = config_to_cmd_flags(cfg)
        cmd = f"python predicators/main.py {cmd_flags}"
        print(f"\n************** COMMAND {i} **************")
        print(cmd)


if __name__ == "__main__":
    _main()
