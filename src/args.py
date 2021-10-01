"""Contains settings that vary per run.
All global, immutable settings should be in settings.py.
"""

import argparse


def parse_args():
    """Defines command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--approach", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    print_args(args)
    return vars(args)


def print_args(args):
    """Print all info for this experiment.
    """
    print(f"Seed: {args.seed}")
    print(f"Env: {args.env}")
    print(f"Approach: {args.approach}")
    print()
