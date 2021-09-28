"""Test cases for structs.
"""

from predicators.src import structs


def test_test_struct():
    """Test TestStruct.
    """
    struct = structs.TestStruct()
    assert struct.x == 3
