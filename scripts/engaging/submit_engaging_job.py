"""Script for submitting jobs on the MIT ORCD Engaging cluster."""

import os
import subprocess
import sys
from pathlib import Path

START_SEED = 456
NUM_SEEDS = 10

# parents[0] = scripts/engaging, parents[1] = scripts, parents[2] = repo root
_REPO_ROOT = Path(__file__).resolve().parents[2]

# The conda env this repo is set up in on Engaging (see repo setup docs).
_CONDA_ENV = "pred"
_MINIFORGE_MODULE = "miniforge/25.11.0-0"

# CPU jobs run on mit_normal (12hr limit); GPU jobs must use mit_normal_gpu
# (6hr limit). See `sinfo` for current partition limits.
_CPU_PARTITION = "mit_normal"
_GPU_PARTITION = "mit_normal_gpu"
_CPU_TIME = "12:00:00"
_GPU_TIME = "06:00:00"


def _run() -> None:
    # Imported here rather than at module level so that importers of
    # submit_engaging_job() (e.g. launch.py) do not pay for the full
    # predicators import chain, which pulls in gym, pybullet and genai.
    # pylint: disable=import-outside-toplevel
    from predicators import utils
    from predicators.settings import CFG

    args = utils.parse_args(seed_required=False)
    utils.update_config(args)
    assert CFG.seed is None, "Do not pass in a seed to this script!"
    job_name = CFG.experiment_id
    log_dir = CFG.log_dir
    logfile_prefix = utils.get_config_path_str()
    args_and_flags_str = " ".join(sys.argv[1:])
    submit_engaging_job("main.py", job_name, log_dir, logfile_prefix,
                        args_and_flags_str, START_SEED, NUM_SEEDS)


def submit_engaging_job(entry_point: str,
                        job_name: str,
                        log_dir: str,
                        logfile_prefix: str,
                        args_and_flags_str: str,
                        start_seed: int,
                        num_seeds: int,
                        use_gpu: bool = False,
                        use_mujoco: bool = False) -> None:
    """Launch one Slurm array job (one array task per seed) on Engaging."""
    del use_mujoco  # unused
    os.makedirs(log_dir, exist_ok=True)
    logfile_pattern = os.path.join(log_dir, f"{logfile_prefix}__%j.log")
    assert logfile_pattern.count("None") == 1
    logfile_pattern = logfile_pattern.replace("None", "%a")

    bash_strs = [
        "#!/bin/bash -l",  # -l => login shell, so /etc/profile.d (module) loads
        f"module load {_MINIFORGE_MODULE}",
        f"conda activate {_CONDA_ENV}",
        f"cd {_REPO_ROOT}",
        f"python predicators/{entry_point} "
        f"{args_and_flags_str} --seed $SLURM_ARRAY_TASK_ID",
    ]
    mystr = "\n".join(bash_strs)
    temp_run_file = "temp_run_file.sh"
    assert not os.path.exists(temp_run_file)
    with open(temp_run_file, "w", encoding="utf-8") as f:
        f.write(mystr)

    if use_gpu:
        partition = _GPU_PARTITION
        time_limit = _GPU_TIME
    else:
        partition = _CPU_PARTITION
        time_limit = _CPU_TIME

    cmd = f"sbatch --time={time_limit} --partition={partition} "
    if use_gpu:
        cmd += "--gres=gpu:1 "
    cmd += ("--nodes=1 "
            "--cpus-per-task=4 "
            "--mem=16G "
            f"--job-name={job_name} "
            f"--array={start_seed}-{start_seed + num_seeds - 1} "
            f"-o {logfile_pattern} {temp_run_file}")
    print(f"Running command: {cmd}")
    output = subprocess.getoutput(cmd)
    print(output)
    if "command not found" in output:
        os.remove(temp_run_file)
        raise Exception("Are you logged into the Engaging cluster?")
    os.remove(temp_run_file)


if __name__ == "__main__":
    _run()
