"""Test cases for the doors environment."""

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.envs.doors import DoorsEnv
from predicators.src.structs import Action, GroundAtom, Task


def test_doors():
    """Tests for DoorsEnv()."""
    utils.reset_config({
        "env": "doors",
    })
    import ipdb
    ipdb.set_trace()
