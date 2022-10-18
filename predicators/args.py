"""Contains settings that vary per run.

All global, immutable settings should be in settings.py.
"""

import argparse
import logging


def create_arg_parser(env_required: bool = True,
                      approach_required: bool = True,
                      seed_required: bool = True) -> argparse.ArgumentParser:
    """Defines command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=env_required, type=str)
    parser.add_argument("--approach", required=approach_required, type=str)
    parser.add_argument("--excluded_predicates", default="", type=str)
    parser.add_argument("--included_options", default="", type=str)
    parser.add_argument("--seed", required=seed_required, type=int)
    parser.add_argument("--option_learner", type=str, default="no_learning")
    parser.add_argument("--explorer", type=str, default="no_explore")
    # NOTE: this timeout affects both data generation and evaluation.
    # If you want to change only the data generation timeout,
    # modify offline_data_planning_timeout.
    parser.add_argument("--timeout", default=10, type=float)
    parser.add_argument("--make_test_videos", action="store_true")
    parser.add_argument("--make_failure_videos", action="store_true")
    parser.add_argument("--make_interaction_videos", action="store_true")
    parser.add_argument("--make_demo_videos", action="store_true")
    parser.add_argument("--load_approach", action="store_true")
    parser.add_argument("--load_data", action="store_true")
    parser.add_argument("--load_atoms", action="store_true")
    parser.add_argument("--skip_until_cycle", default=-1, type=int)
    parser.add_argument("--experiment_id", default="", type=str)
    parser.add_argument("--load_experiment_id", default="", type=str)
    parser.add_argument("--log_file", default="", type=str)
    parser.add_argument('--debug',
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.INFO)
    parser.add_argument("--pg3_init_policy", default="", type=str)
    return parser
