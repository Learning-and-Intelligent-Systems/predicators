"""Script for submitting jobs on supercloud."""

import sys
import subprocess
import os
from predicators.src import utils
from predicators.src.settings import CFG


def _run() -> None:
    args = utils.parse_args()
    utils.update_config(args)
    job_name = f"{CFG.experiment_id}_{CFG.seed}"
    argsstr = " ".join(sys.argv[1:])
    mystr = f"#!/bin/bash\npython src/main.py {argsstr}"
    temp_run_file = "temp_run_file.sh"
    assert not os.path.exists(temp_run_file)
    with open(temp_run_file, "w", encoding="utf-8") as f:
        f.write(mystr)
    cmd = ("sbatch -p normal --time=99:00:00 --partition=xeon-p8 "
           f"--nodes=1 --exclusive --job-name={job_name} "
           f"-o /tmp/%j_log.out {temp_run_file}")
    print(f"Running command: {cmd}")
    output = subprocess.getoutput(cmd)
    if "command not found" in output:
        os.remove(temp_run_file)
        raise Exception("Are you logged into supercloud?")
    os.remove(temp_run_file)
    logfile = os.path.join(CFG.log_dir, f"{utils.get_config_path_str()}.log")
    print(f"Started job, see log with:\ntail -n 10000 -F {logfile}")


if __name__ == "__main__":
    _run()
