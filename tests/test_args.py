"""Tests for args.py."""

import sys

from predicators.src import utils


def test_args():
    """Tests for args.py."""
    sys.argv = [
        "dummy", "--env", "my_env", "--approach", "my_approach", "--seed",
        "123"
    ]
    args = utils.parse_args()
    assert args["env"] == "my_env"
    assert args["approach"] == "my_approach"
    assert args["seed"] == 123
