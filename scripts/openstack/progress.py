"""Script for getting status of active openstack predicator experiments.

    WARNING: Will return the status of any other python3.8 processes running on
        each machine in machines.txt

    See launch.py for information about the format of machines.txt.

    Usage example:
        python scripts/openstack/progress.py --machines machines.txt \
            --sshkey ~/.ssh/cloud.key
"""

import argparse
import os

from predicators.scripts.openstack.launch import run_cmds_on_machine


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
    # Loop through each machine and print those that have a python3.8 process.
    progress_cmd = "pgrep python3.8"
    print("The following machines are still running:\n")
    for machine in machines:
        # If return code is 0, print active machine.
        returncode = run_cmds_on_machine([progress_cmd],
                                         machine,
                                         args.sshkey,
                                         allowed_return_codes=(0, 1))

        if returncode != 1:
            print(machine)


if __name__ == "__main__":
    _main()
