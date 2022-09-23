"""Download results from supercloud experiments.

Usage example:

    python scripts/supercloud/download.py --dir "$PWD" --user tslvr

By default, we assume that the predicators directory on supercloud is located
at ~/predicators. Otherwise, use the --supercloud_dir flag. Example:

    python scripts/supercloud/download.py --dir "$PWD" --user njk \
        --supercloud_dir "~/GitHub/research/predicators"
"""

import argparse
import os

from scripts.cluster_utils import SAVE_DIRS, SUPERCLOUD_IP


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, type=str)
    parser.add_argument("--user", required=True, type=str)
    parser.add_argument("--supercloud_dir", default="~/predicators", type=str)
    args = parser.parse_args()
    # Create the download directory if it doesn't exist.
    os.makedirs(args.dir, exist_ok=True)
    # Download the results.
    print(f"Downloading results from supercloud for user {args.user}")
    host = f"{args.user}@{SUPERCLOUD_IP}"
    for save_dir in SAVE_DIRS:
        local_save_dir = os.path.join(args.dir, save_dir)
        os.makedirs(local_save_dir, exist_ok=True)
        cmd = "rsync -avzhe ssh " + \
              f"{host}:{args.supercloud_dir}/{save_dir}/* {local_save_dir}"
        retcode = os.system(cmd)
        if retcode != 0:
            print(f"WARNING: command failed: {cmd}")


if __name__ == "__main__":
    _main()
