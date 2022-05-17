"""Script for submitting jobs on supercloud."""

import os
import subprocess
import sys

from predicators.src import utils
from predicators.src.settings import CFG

START_SEED=456
NUM_SEEDS=10


def _run() -> None:
    args = utils.parse_args(seed_required=False)
    utils.update_config(args)
    assert CFG.seed is None
    job_name = CFG.experiment_id
    os.makedirs(CFG.log_dir, exist_ok=True)
    logfile_pattern = os.path.join(CFG.log_dir,
                                   f"{utils.get_config_path_str()}__%j.log")
    assert logfile_pattern.count("None") == 1
    logfile_pattern = logfile_pattern.replace("None", "%a")
    argsstr = " ".join(sys.argv[1:])
    mystr = (f"#!/bin/bash\npython src/main.py {argsstr} "
             f"--seed $SLURM_ARRAY_TASK_ID")
    temp_run_file = "temp_run_file.sh"
    assert not os.path.exists(temp_run_file)
    with open(temp_run_file, "w", encoding="utf-8") as f:
        f.write(mystr)
    cmd = ("sbatch -p normal --time=99:00:00 --partition=xeon-p8 "
           f"--nodes=1 --exclusive --job-name={job_name} "
           f"--array={START_SEED}-{START_SEED+NUM_SEEDS-1} "
           f"-o {logfile_pattern} {temp_run_file}")
    print(f"Running command: {cmd}")
    output = subprocess.getoutput(cmd)
    if "command not found" in output:
        os.remove(temp_run_file)
        raise Exception("Are you logged into supercloud?")
    os.remove(temp_run_file)


if __name__ == "__main__":
    _run()
