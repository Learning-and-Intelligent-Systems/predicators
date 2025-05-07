"""Script for submitting jobs on supercloud."""

import os
import subprocess
import sys

from predicators import utils
from predicators.settings import CFG
from scripts.cluster_utils import SAVE_DIRS

START_SEED = 456
NUM_SEEDS = 10


def _run() -> None:
    args = utils.parse_args(seed_required=False)
    utils.update_config(args)
    assert CFG.seed is None, "Do not pass in a seed to this script!"
    job_name = CFG.experiment_id
    log_dir = CFG.log_dir
    logfile_prefix = utils.get_config_path_str()
    args_and_flags_str = " ".join(sys.argv[1:])
    return submit_supercloud_job("main.py", job_name, log_dir, logfile_prefix,
                                 args_and_flags_str, START_SEED, NUM_SEEDS)


# Commands for using MuJoCo.
# Reference: https://github.com/openai/mujoco-py/issues/486
_OLD_DIRS = " ".join(f"{d}/*" for d in SAVE_DIRS)
_MUJOCO_PREP = f"""# Make temporary folder
mkdir -p /state/partition1/user/$USER

# Copy mujoco-py folder to locked part of cluster
rsync -av ~/mujoco-py /state/partition1/user/$USER/ --exclude .git
cd /state/partition1/user/$USER/mujoco-py

# Install it and import it to build
python setup.py install --user
python -c "import mujoco_py"

# Move code to this folder and mujoco-py into code
rsync -av ~/predicators /state/partition1/user/$USER/ \
    --exclude predicators/logs
cp -r mujoco_py ../predicators/

# Change directory to predicators
cd ../predicators

# Remove existing files
rm -f {_OLD_DIRS}

# Run the code
"""
_MUJOCO_FINISH = """# Copy this directory back to where it started
cd ../
rsync -av predicators ~/ --exclude mujoco_py

# Remove temporary folder
rm -rf /state/partition1/user/$USER
"""


def submit_openmind_job(entry_point: str,
                        job_name: str,
                        log_dir: str,
                        logfile_prefix: str,
                        args_and_flags_str: str,
                        start_seed: int,
                        num_seeds: int,
                        use_gpu: bool = False,
                        use_mujoco: bool = False) -> None:
    """Launch the openmind job."""
    assert entry_point in ("main.py", "train_refinement_estimator.py")
    os.makedirs(log_dir, exist_ok=True)
    logfile_pattern = os.path.join(log_dir, f"{logfile_prefix}__%j.log")
    assert logfile_pattern.count("None") == 1
    logfile_pattern = logfile_pattern.replace("None", "%a")
    bash_strs = [
        "#!/bin/bash",
        # Why doesn't it activate the conda environment here?
        "cd /om2/user/ycliang/predicators",
        f"python predicators/{entry_point} "
        f"{args_and_flags_str} --seed $SLURM_ARRAY_TASK_ID",
    ]
    mystr = "\n".join(bash_strs)
    temp_run_file = "temp_run_file.sh"
    assert not os.path.exists(temp_run_file)
    with open(temp_run_file, "w", encoding="utf-8") as f:
        f.write(mystr)

    # Should update these configs
    cmd = "sbatch --time=1-00:00 "
    if use_gpu:
        cmd += "--gres=gpu:1 "
    # else:
    #     cmd += "--partition=xeon-p8 "
    cmd += (
        "--nodes=1 "
        # "--mem=512G " # for blocks
        "--mem=128G "
        # "-p tenenbaum "
        f"--job-name={job_name} "
        f"--array={start_seed}-{start_seed+num_seeds-1} "
        f"-o {logfile_pattern} {temp_run_file}")
    print(f"Running command: {cmd}")
    output = subprocess.getoutput(cmd)
    if "command not found" in output:
        os.remove(temp_run_file)
        raise Exception("Are you logged into supercloud?")
    os.remove(temp_run_file)


if __name__ == "__main__":
    _run()
