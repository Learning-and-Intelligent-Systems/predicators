"""Script for submitting jobs on supercloud."""

import sys
import subprocess
import os
import numpy as np

RUN_INDEX = np.random.randint(int(1e15))


def _run() -> None:
    log_dir = "supercloud_logs"
    os.makedirs(log_dir, exist_ok=True)
    argsstr = " ".join(sys.argv[1:])
    mystr = f"#!/bin/bash\npython src/main.py {argsstr}"
    with open(f"run_{RUN_INDEX}.sh", "w", encoding="utf-8") as f:
        f.write(mystr)
    cmd = ("sbatch -p normal --time=99:00:00 --partition=xeon-p8 "
           "--nodes=1 --exclusive --job-name=run.sh "
           f"-o {log_dir}/%j_log.out run_{RUN_INDEX}.sh")
    print(f"Running command: {cmd}")
    output = subprocess.getoutput(cmd)
    if "command not found" in output:
        os.remove(f"run_{RUN_INDEX}.sh")
        raise Exception("Are you logged into supercloud?")
    job_id = output.split()[-1]
    print("Started job, see log with:\ntail -n 10000 "
          f"-F {log_dir}/{job_id}_log.out")
    os.remove(f"run_{RUN_INDEX}.sh")


if __name__ == "__main__":
    _run()
