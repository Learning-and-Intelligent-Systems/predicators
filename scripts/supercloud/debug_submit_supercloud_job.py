"""Script for submitting jobs on supercloud."""

import os
import subprocess
import sys

START_SEED = 456
NUM_SEEDS = 100


def _run() -> None:
    mystr = (f"#!/bin/bash\npython debug_supercloud.py "
             f"--seed $SLURM_ARRAY_TASK_ID")
    temp_run_file = "temp_run_file.sh"
    assert not os.path.exists(temp_run_file)
    with open(temp_run_file, "w", encoding="utf-8") as f:
        f.write(mystr)
    cmd = ("sbatch -p normal --time=99:00:00 --partition=xeon-p8 "
           f"--nodes=1 --exclusive --job-name=debug "
           f"--array={START_SEED}-{START_SEED+NUM_SEEDS-1} "
           f"{temp_run_file}")
    print(f"Running command: {cmd}")
    output = subprocess.getoutput(cmd)
    if "command not found" in output:
        os.remove(temp_run_file)
        raise Exception("Are you logged into supercloud?")
    os.remove(temp_run_file)


if __name__ == "__main__":
    _run()
