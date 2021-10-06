"""Contains settings that vary per run.
All global, immutable settings should be in settings.py.
"""

import argparse
from typing import Dict, Any


def parse_args() -> Dict[str, Any]:
    """Defines command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, type=str)
    parser.add_argument("--approach", required=True, type=str)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--timeout", default=10, type=float)
    parser.add_argument("--make_videos", action="store_true")
    args, overrides = parser.parse_known_args()
    print_args(args)
    arg_dict = vars(args)
    if len(overrides) == 0:
        return arg_dict
    # Override global settings
    assert len(overrides) >= 2
    assert len(overrides) % 2 == 0
    for flag, value in zip(overrides[:-1:2], overrides[1::2]):
        assert flag.startswith("--")
        setting_name = flag[2:]
        if value.isdigit():
            value = eval(value)
        arg_dict[setting_name] = value
    return arg_dict


def print_args(args: argparse.Namespace) -> None:
    """Print all info for this experiment.
    """
    print(f"Seed: {args.seed}")
    print(f"Env: {args.env}")
    print(f"Approach: {args.approach}")
    print(f"Timeout: {args.timeout}")
    print()
