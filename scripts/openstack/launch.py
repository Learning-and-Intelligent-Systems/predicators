"""Launch script for openstack experiments.

Requires a file that contains a list of IP addresses for instances that are:
    - Turned on
    - Accessible via ssh for the user of this file
    - Configured with a predicators image
    - Sufficient in number to run all of the experiments in the config file

Usage example:
    python scripts/openstack/launch.py --config example.yaml \
        --machines machines.txt --sshkey ~/.ssh/cloud.key
"""

import argparse
import os
import subprocess
from typing import Dict, Sequence, Tuple

import yaml


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--machines", required=True, type=str)
    parser.add_argument("--sshkey", required=True, type=str)
    args = parser.parse_args()
    openstack_dir = os.path.dirname(os.path.realpath(__file__))
    # Load the config file.
    config_file = os.path.join(openstack_dir, "configs", args.config)
    with open(config_file, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    # Load the machine IPs.
    machine_file = os.path.join(openstack_dir, args.machines)
    with open(machine_file, "r", encoding="utf-8") as f:
        machines = f.read().splitlines()
    # Make sure that the ssh key exists.
    assert os.path.exists(args.sshkey)
    # Validate that the number of machines is sufficiently high.
    num_machines = len(machines)
    num_seeds = config["NUM_SEEDS"]
    num_approaches = len(config["APPROACHES"])
    num_envs = len(config["ENVS"])
    assert num_machines >= num_seeds * num_approaches * num_envs
    # Loop over seeds.
    available_machines = list(machines)
    start_seed = config["START_SEED"]
    cmd_args = config["ARGS"]
    branch = config["BRANCH"]
    for seed in range(start_seed, start_seed + num_seeds):
        # Loop over approaches.
        for approach_exp_id, approach_config in config["APPROACHES"].items():
            approach = approach_config["NAME"]
            # Loop over envs.
            for env_exp_id, env_config in config["ENVS"].items():
                env = env_config["NAME"]
                # Create the experiment ID.
                experiment_id = f"{env_exp_id}-{approach_exp_id}"
                experiment_flags = config["FLAGS"].copy()
                experiment_flags.update(approach_config["FLAGS"])
                experiment_flags.update(env_config["FLAGS"])
                # Launch the experiment for this approach / env / seed.
                logfile = _create_logfile(experiment_id, approach, env, seed)
                cmd = _create_cmd(experiment_id, approach, env, seed, cmd_args,
                                  experiment_flags)
                # One experiment per machine.
                machine = available_machines.pop()
                _launch_experiment(cmd, machine, logfile, args.sshkey, branch)


def _create_logfile(experiment_id: str, approach: str, env: str,
                    seed: int) -> str:
    return f"logs/{env}__{approach}__{experiment_id}__{seed}.log"


def _create_cmd(experiment_id: str, approach: str, env: str, seed: int,
                args: Sequence[str], flags: Dict) -> str:
    arg_str = " ".join(f"--{a}" for a in args)
    flag_str = " ".join(f"--{f} {v}" for f, v in flags.items())
    cmd = f"python3.8 src/main.py --env {env} --approach {approach} " + \
              f"--seed {seed} --experiment_id {experiment_id} {arg_str} " + \
              f"{flag_str}"
    return cmd


def run_cmds_on_machine(
    cmds: Sequence[str],
    machine: str,
    ssh_key: str,
    allowed_return_codes: Tuple[int] = (0, )) -> None:
    host = f"ubuntu@{machine}"
    ssh_cmd = f"ssh -tt -i {ssh_key} -o StrictHostKeyChecking=no {host}"
    server_cmd_str = "\n".join(cmds + ["exit"])
    final_cmd = f"{ssh_cmd} << EOF\n{server_cmd_str}\nEOF"
    response = subprocess.run(final_cmd,
                              stdout=subprocess.DEVNULL,
                              stderr=subprocess.STDOUT,
                              shell=True,
                              check=False)
    if response.returncode not in allowed_return_codes:
        import ipdb
        ipdb.set_trace()
        raise RuntimeError(f"Command failed: {final_cmd}")


def _launch_experiment(cmd: str, machine: str, logfile: str, ssh_key: str,
                       branch: str) -> None:
    print(f"Launching on machine {machine}: {cmd}")
    server_cmds = [
        # Prepare the predicators directory.
        "cd ~/predicators",
        "mkdir -p logs",
        f"git checkout {branch}",
        "git pull",
        # Remove old results.
        "rm -f results/* logs/* saved_approaches/* saved_datasets/*",
        # Run the main command.
        f"{cmd} &> {logfile} &",
    ]
    run_cmds_on_machine(server_cmds, machine, ssh_key)


if __name__ == "__main__":
    _main()
