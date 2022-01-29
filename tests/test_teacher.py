"""Test cases for teacher."""

from predicators.src.teacher import Teacher, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse
from predicators.src.envs import create_env
from predicators.src.ground_truth_nsrts import _get_predicates_by_names
from predicators.src.structs import GroundAtom
from predicators.src import utils


def test_GroundAtomHolds():
    """Tests for answering queries of type GroundAtomsHoldQuery."""
    utils.update_config({"env": "cover", "approach": "unittest"})
    teacher = Teacher()
    env = create_env("cover")
    state = env.get_train_tasks()[0].init
    block_type = [t for t in env.types if t.name == "block"][0]
    target_type = [t for t in env.types if t.name == "target"][0]
    block = block_type("block0")
    target = target_type("target0")
    Covers, IsBlock = _get_predicates_by_names("cover", ["Covers", "IsBlock"])
    Covers = utils.strip_predicate(Covers)
    IsBlock = utils.strip_predicate(IsBlock)
    is_block_block = GroundAtom(IsBlock, [block])
    query = GroundAtomsHoldQuery({is_block_block})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert response.holds[is_block_block]
    covers_block_target = GroundAtom(Covers, [block, target])
    query = GroundAtomsHoldQuery({covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert not response.holds[covers_block_target]
    query = GroundAtomsHoldQuery({covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert not response.holds[covers_block_target]
    query = GroundAtomsHoldQuery({is_block_block, covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 2
    assert response.holds[is_block_block]
    assert not response.holds[covers_block_target]
