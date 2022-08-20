"""Script for killing all active openstack predicator experiments.

Analogous to scancel on supercloud.

WARNING: any other python3.8 processes running on the machine will also be
killed (but there typically shouldn't be any).

See launch.py for information about the format of machines.txt.

Usage example:
    python scripts/openstack/kill_all.py --machines machines.txt \
        --sshkey ~/.ssh/cloud.key
"""

import argparse
import os

from scripts.cluster_utils import run_cmds_on_machine


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", required=True, type=str)
    parser.add_argument("--sshkey", required=True, type=str)
    args = parser.parse_args()
    openstack_dir = os.path.dirname(os.path.realpath(__file__))
    # Load the machine IPs.
    machine_file = os.path.join(openstack_dir, args.machines)
    with open(machine_file, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()
    # Make sure that the ssh key exists.
    assert os.path.exists(args.sshkey)
    # Loop through each machine and kill the python3.8 process.
    kill_cmd = "pkill -9 python3.8"
    for machine in machines:
        print(f"Killing machine {machine}")
        # Allow return code 1, meaning that no process was found to kill.
        run_cmds_on_machine([kill_cmd],
                            "ubuntu",
                            machine,
                            ssh_key=args.sshkey,
                            allowed_return_codes=(0, 1))


if __name__ == "__main__":
    _main()
