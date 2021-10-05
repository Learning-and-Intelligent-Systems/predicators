"""Tests for main.py.
"""

import sys
import pytest
from predicators.src.main import main


def test_main():
    """Tests for main.py.
    """
    sys.argv = ["dummy", "--env", "my_env", "--approach", "my_approach",
                "--seed", "123"]
    with pytest.raises(NotImplementedError):
        main()  # invalid env
    sys.argv = ["dummy", "--env", "cover", "--approach", "my_approach",
                "--seed", "123"]
    with pytest.raises(NotImplementedError):
        main()  # invalid approach
    sys.argv = ["dummy", "--env", "cover", "--approach", "random",
                "--seed", "123"]
    main()
    sys.argv = ["dummy", "--env", "cover", "--approach", "oracle",
                "--seed", "123"]
    main()
    sys.argv = ["dummy", "--env", "cover", "--approach", "trivial_learning",
                "--seed", "123"]
    main()
