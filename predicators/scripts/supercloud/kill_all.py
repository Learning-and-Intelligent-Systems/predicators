"""Script to kill all experiments running on supercloud for a given user.

Runs scancel -u <user> on supercloud.

Usage example:

    python scripts/supercloud/kill_all.py --user tslvr
"""

import argparse

from scripts.cluster_utils import SUPERCLOUD_IP, run_cmds_on_machine


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--user", required=True, type=str)
    args = parser.parse_args()
    print(f"Killing all jobs on supercloud for user {args.user}")
    kill_cmd = f"scancel -u {args.user}"
    run_cmds_on_machine([kill_cmd], args.user, SUPERCLOUD_IP)


if __name__ == "__main__":
    _main()
