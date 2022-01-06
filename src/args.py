"""Contains settings that vary per run.

All global, immutable settings should be in settings.py.
"""

import argparse


def create_arg_parser() -> argparse.ArgumentParser:
    """Defines command line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--approach", required=True, type=str)
    parser.add_argument("--excluded_predicates", default="", type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--timeout", default=10, type=float)
    parser.add_argument("--make_videos", action="store_true")
    parser.add_argument("--load", action="store_true")
    return parser
