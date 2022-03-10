"""Script for submitting jobs on supercloud."""

import sys
import subprocess
import os
from predicators.src.settings import CFG
from predicators.src.utils import get_run_id_from_argv


def _run() -> None:
    unique_run_ids = set()
    log_dir = CFG.log_dir
    os.makedirs(log_dir, exist_ok=True)
    run_id = get_run_id_from_argv()
    run_bash_file = f"run_{run_id}.sh"
    if run_id in unique_run_ids:
        print(f"\n\nWARNING!!!!!!!! Launching job with duplicate run id: "
              f"{run_id}\n\n")
    unique_run_ids.add(run_id)
    argsstr = " ".join(sys.argv[1:])
    mystr = f"#!/bin/bash\npython src/main.py {argsstr}"
    with open(run_bash_file, "w", encoding="utf-8") as f:
        f.write(mystr)
    cmd = ("sbatch -p normal --time=99:00:00 --partition=xeon-p8 "
           f"--nodes=1 --exclusive --job-name={run_id} "
           f"-o {log_dir}/%j_log.out {run_bash_file}")
    print(f"Running command: {cmd}")
    output = subprocess.getoutput(cmd)
    if "command not found" in output:
        os.remove(run_bash_file)
        raise Exception("Are you logged into supercloud?")
    print("Started job, see log with:\ntail -n 10000 "
          f"-F {log_dir}/{run_id}.log")
    os.remove(run_bash_file)


if __name__ == "__main__":
    _run()
