"""Tests for NSRT learning."""

from gym.spaces import Box
import numpy as np
# We need this unused import to prevent cyclic import issues when running
# this file as a standalone test (pytest -s tests/test_nsrt_learning.py).
from predicators.src import approaches  # pylint:disable=unused-import
from predicators.src.nsrt_learning import learn_nsrts_from_data, \
    segment_trajectory, learn_strips_operators
from predicators.src.structs import Type, Predicate, State, Action, \
    ParameterizedOption, LowLevelTrajectory
from predicators.src import utils


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


def test_nsrt_learning_specific_nsrts():
    """Tests with a specific desired set of NSRTs."""
    utils.reset_config({
        "min_data_for_nsrt": 0,
        "sampler_mlp_classifier_max_itr": 1000,
        "neural_gaus_regressor_max_itr": 1000
    })
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    cup3 = cup_type("cup3")
    cup4 = cup_type("cup4")
    cup5 = cup_type("cup5")
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    pred1 = Predicate("Pred1", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    preds = {pred0, pred1, pred2}
    state1 = State({cup0: [0.4], cup1: [0.7], cup2: [0.1]})
    option1 = ParameterizedOption(
        "dummy", [], Box(0.1, 1, (1, )), lambda s, m, o, p: Action(p),
        utils.always_initiable, utils.onestep_terminal).ground([],
                                                               np.array([0.2]))
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    dataset = [LowLevelTrajectory([state1, next_state1], [action1])]
    nsrts = learn_nsrts_from_data(dataset, [], preds, sampler_learner="neural")
    assert len(nsrts) == 1
    nsrt = nsrts.pop()
    assert str(nsrt) == """NSRT-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x0:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred1(?x2:cup_type, ?x2:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Side Predicates: []
    Option Spec: dummy()"""
    # Test the learned samplers
    for _ in range(10):
        assert abs(
            nsrt.ground([cup0, cup1, cup2]).sample_option(
                state1, set(), np.random.default_rng(123)).params - 0.2) < 0.01
    # The following test was used to manually check that unify caches correctly.
    pred0 = Predicate("Pred0", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    pred1 = Predicate("Pred1", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred2 = Predicate("Pred2", [cup_type], lambda s, o: s[o[0]][0] > 0.5)
    preds = {pred0, pred1, pred2}
    state1 = State({cup0: [0.4], cup1: [0.7], cup2: [0.1]})
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    state2 = State({cup3: [0.4], cup4: [0.7], cup5: [0.1]})
    action2 = option1.policy(state2)
    action2.set_option(option1)
    next_state2 = State({cup3: [0.8], cup4: [0.3], cup5: [1.0]})
    dataset = [
        LowLevelTrajectory([state1, next_state1], [action1]),
        LowLevelTrajectory([state2, next_state2], [action2])
    ]
    nsrts = learn_nsrts_from_data(dataset, [], preds, sampler_learner="random")
    assert len(nsrts) == 1
    nsrt = nsrts.pop()
    assert str(nsrt) == """NSRT-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x0:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred1(?x2:cup_type, ?x2:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Side Predicates: []
    Option Spec: dummy()"""
    # The following two tests check edge cases of unification with respect to
    # the split between add and delete effects. Specifically, it's important
    # to unify both of them together, not separately, which requires changing
    # the predicates so that unification does not try to unify add ones with
    # delete ones.
    pred0 = Predicate("Pred0", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.7 and s[o[1]][0] < 0.3)
    preds = {pred0}
    state1 = State({cup0: [0.4], cup1: [0.8], cup2: [0.1]})
    option1 = ParameterizedOption(
        "dummy", [], Box(0.1, 0.5, (1, )), lambda s, m, o, p: Action(p),
        utils.always_initiable, utils.onestep_terminal).ground([],
                                                               np.array([0.3]))
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.9], cup1: [0.2], cup2: [0.5]})
    state2 = State({cup4: [0.9], cup5: [0.2], cup2: [0.5], cup3: [0.5]})
    option2 = ParameterizedOption(
        "dummy", [], Box(0.1, 0.5, (1, )), lambda s, m, o, p: Action(p),
        utils.always_initiable, utils.onestep_terminal).ground([],
                                                               np.array([0.5]))
    action2 = option2.policy(state2)
    action2.set_option(option2)
    next_state2 = State({cup4: [0.5], cup5: [0.5], cup2: [1.0], cup3: [0.1]})
    dataset = [
        LowLevelTrajectory([state1, next_state1], [action1]),
        LowLevelTrajectory([state2, next_state2], [action2])
    ]
    nsrts = learn_nsrts_from_data(dataset, [], preds, sampler_learner="random")
    assert len(nsrts) == 2
    expected = {
        "Op0":
        """NSRT-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Side Predicates: []
    Option Spec: dummy()""",
        "Op1":
        """NSRT-Op1:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type, ?x3:cup_type]
    Preconditions: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Side Predicates: []
    Option Spec: dummy()"""
    }
    pred0 = Predicate("Pred0", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.7 and s[o[1]][0] < 0.3)
    preds = {pred0}
    state1 = State({cup0: [0.5], cup1: [0.5]})
    action1 = option2.policy(state1)
    action1.set_option(option2)
    next_state1 = State({
        cup0: [0.9],
        cup1: [0.1],
    })
    state2 = State({cup4: [0.9], cup5: [0.1]})
    action2 = option2.policy(state2)
    action2.set_option(option2)
    next_state2 = State({cup4: [0.5], cup5: [0.5]})
    dataset = [
        LowLevelTrajectory([state1, next_state1], [action1]),
        LowLevelTrajectory([state2, next_state2], [action2])
    ]
    nsrts = learn_nsrts_from_data(dataset, [], preds, sampler_learner="random")
    assert len(nsrts) == 2
    expected = {
        "Op0":
        """NSRT-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: []
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: dummy()""",
        "Op1":
        """NSRT-Op1:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Add Effects: []
    Delete Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Side Predicates: []
    Option Spec: dummy()"""
    }
    for nsrt in nsrts:
        assert str(nsrt) == expected[nsrt.name]
    # Test minimum number of examples parameter
    utils.update_config({"min_data_for_nsrt": 3})
    nsrts = learn_nsrts_from_data(dataset, [], preds, sampler_learner="random")
    assert len(nsrts) == 0
    # Test max_rejection_sampling_tries = 0
    utils.update_config({
        "min_data_for_nsrt": 0,
        "max_rejection_sampling_tries": 0,
        "sampler_mlp_classifier_max_itr": 1,
        "neural_gaus_regressor_max_itr": 1
    })
    nsrts = learn_nsrts_from_data(dataset, [], preds, sampler_learner="neural")
    assert len(nsrts) == 2
    for nsrt in nsrts:
        for _ in range(10):
            sampled_params = nsrt.ground([cup0, cup1]).sample_option(
                state1, set(), np.random.default_rng(123)).params
            assert option1.parent.params_space.contains(sampled_params)
