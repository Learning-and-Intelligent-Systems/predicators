"""Test cases for teacher."""

import pytest
from predicators.src.teacher import Teacher, GroundAtomHoldsQuery, \
    GroundAtomHoldsResponse
from predicators.src.envs import create_env
from predicators.src import utils


def test_GroundAtomHolds():
    """Tests for answering queries of type GroundAtomHoldsQuery."""
    utils.update_config({"env": "cover",
                         "approach": "unittest"})
    teacher = Teacher()
    env = create_env("cover")
    state = env.get_train_tasks()[0].init
    block_type = [t for t in env.types if t.name == "block"][0]
    target_type = [t for t in env.types if t.name == "target"][0]
    block = block_type("block0")
    target = target_type("target0")
    query = GroundAtomHoldsQuery("IsBlock", [target])
    with pytest.raises(AssertionError):  # wrong object types
        teacher.answer_query(state, query)
    query = GroundAtomHoldsQuery("IsBlock", [block])
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomHoldsResponse)
    assert response.query is query
    assert response.holds
    query = GroundAtomHoldsQuery("Covers", [block, target])
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomHoldsResponse)
    assert response.query is query
    assert not response.holds
