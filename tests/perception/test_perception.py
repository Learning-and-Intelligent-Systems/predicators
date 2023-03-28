"""Tests for perceivers."""

import pytest

from predicators.perception import create_perceiver
from predicators.perception.trivial_perceiver import TrivialPerceiver


def test_create_perceiver():
    """Tests for create_perceiver()."""
    perceiver = create_perceiver("trivial")
    assert isinstance(perceiver, TrivialPerceiver)

    with pytest.raises(NotImplementedError) as e:
        create_perceiver("not a real perceiver")
    assert "Unrecognized perceiver" in str(e)
