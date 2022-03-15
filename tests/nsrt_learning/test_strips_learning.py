"""Tests for STRIPS operator learning."""

import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.nsrt_learning.strips_learning import (
    learn_strips_operators, segment_trajectory)
from predicators.src.structs import (Action, LowLevelTrajectory,
                                     ParameterizedOption, Predicate, State,
                                     Type)


def test_segment_trajectory():
    """Tests for segment_trajectory()."""
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    pred1 = Predicate("Pred1", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    preds = {pred0, pred1, pred2}
    state0 = State({cup0: [0.4], cup1: [0.7], cup2: [0.1]})
    atoms0 = utils.abstract(state0, preds)
    state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    atoms1 = utils.abstract(state1, preds)
    # Tests with known options.
    param_option = ParameterizedOption("dummy", [cup_type], Box(0.1, 1, (1, )),
                                       lambda s, m, o, p: Action(p),
                                       utils.always_initiable,
                                       utils.onestep_terminal)
    option0 = param_option.ground([cup0], np.array([0.2]))
    assert option0.initiable(state0)
    action0 = option0.policy(state0)
    # Even though the option changes, the option spec stays the same, so we do
    # not want to segment. This is because we are segmenting based on symbolic
    # aspects only, because the strips operators can only depend on symbols.
    option1 = param_option.ground([cup0], np.array([0.1]))
    assert option1.initiable(state0)
    action1 = option1.policy(state0)
    option2 = param_option.ground([cup1], np.array([0.1]))
    assert option2.initiable(state0)
    action2 = option2.policy(state0)
    trajectory = (LowLevelTrajectory([state0.copy() for _ in range(5)],
                                     [action0, action1, action2, action0]),
                  [atoms0, atoms0, atoms0, atoms0, atoms0])
    known_option_segments = segment_trajectory(trajectory)
    assert len(known_option_segments) == 3
    assert len(known_option_segments[0].actions) == 2
    assert len(known_option_segments[1].actions) == 1
    assert len(known_option_segments[2].actions) == 1
    # Tests without known options.
    action0 = option0.policy(state0)
    action0.unset_option()
    action1 = option0.policy(state0)
    action1.unset_option()
    action2 = option1.policy(state0)
    action2.unset_option()
    trajectory = (LowLevelTrajectory([state0.copy() for _ in range(5)],
                                     [action0, action1, action2, action0]),
                  [atoms0, atoms0, atoms0, atoms0, atoms0])
    assert len(segment_trajectory(trajectory)) == 0
    trajectory = (LowLevelTrajectory(
        [state0.copy() for _ in range(5)] + [state1],
        [action0, action1, action2, action0, action1]),
                  [atoms0, atoms0, atoms0, atoms0, atoms0, atoms1])
    unknown_option_segments = segment_trajectory(trajectory)
    assert len(unknown_option_segments) == 1
    assert len(unknown_option_segments[0].actions) == 5
    return known_option_segments, unknown_option_segments


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
