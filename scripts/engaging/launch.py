"""Launch Engaging (ORCD) experiments defined by config files, adapted from
openmind/launch.py.

Each experiment (approach x env combo) is submitted as its own Slurm array
job, with one array task per seed, so all experiments run concurrently on
compute nodes rather than in the current terminal/login node.

Usage example:

    python scripts/engaging/launch.py -c predicatorv3/exp_domino.yaml
"""
import argparse
import sys
from pathlib import Path

# Add project root to sys.path so `scripts` is importable without PYTHONPATH=.
# parents[0] = scripts/engaging, parents[1] = scripts, parents[2] = repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from scripts.cluster_utils import BatchSeedRunConfig, config_to_cmd_flags, \
    config_to_logfile, generate_run_configs
from scripts.engaging.submit_engaging_job import submit_engaging_job


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    args = parser.parse_args()
    _launch_experiments(args.config)


def _launch_experiments(config_file: str) -> None:
    # Loop over run configs.
    for cfg in generate_run_configs(config_file, batch_seeds=True):
        assert isinstance(cfg, BatchSeedRunConfig)
        cmd_flags = config_to_cmd_flags(cfg)
        log_dir = "logs"
        log_prefix = config_to_logfile(cfg, suffix="")
        # Launch a job for this experiment.

        if "use_classification_problem_setting" in cfg.flags:
            use_classification_problem_setting = cfg.flags[
                'use_classification_problem_setting']
        else:
            use_classification_problem_setting = False

        if use_classification_problem_setting:
            entry_point = "main_classification.py"
        else:
            entry_point = "main.py"
        submit_engaging_job(entry_point, cfg.experiment_id, log_dir,
                            log_prefix, cmd_flags, cfg.start_seed,
                            cfg.num_seeds, cfg.use_gpu, cfg.use_mujoco)


if __name__ == "__main__":
    _main()
