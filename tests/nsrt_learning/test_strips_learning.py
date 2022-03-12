"""Tests for STRIPS operator learning."""

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import \
    learn_strips_operators
from predicators.tests.nsrt_learning.test_segmentation import \
    test_segment_trajectory


def test_learn_strips_operators():
    """Tests for learn_strips_operators()."""
    utils.reset_config({"min_data_for_nsrt": 0})
    known_option_segments, unknown_option_segments = test_segment_trajectory()
    known_option_pnads = learn_strips_operators(known_option_segments)
    known_option_ops = [pnad.op for pnad in known_option_pnads]
    assert len(known_option_ops) == 1
    assert str((known_option_ops[0])) == """STRIPS-Op0:
    Parameters: [?x0:cup_type]
    Preconditions: []
    Add Effects: []
    Delete Effects: []
    Side Predicates: []"""
    unknown_option_pnads = learn_strips_operators(unknown_option_segments)
    unknown_option_ops = [pnad.op for pnad in unknown_option_pnads]
    assert len(unknown_option_ops) == 1
    assert str(unknown_option_ops[0]) == """STRIPS-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x0:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred1(?x2:cup_type, ?x2:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Side Predicates: []"""  # pylint: disable=line-too-long
