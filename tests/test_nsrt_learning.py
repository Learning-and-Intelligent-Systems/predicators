"""Tests for NSRT learning.
"""

import time
from gym.spaces import Box
import numpy as np
from predicators.src.nsrt_learning import learn_nsrts_from_data
from predicators.src.structs import Type, Predicate, State, Action, \
    ParameterizedOption
from predicators.src import utils


def test_nsrt_learning_specific_nsrts():
    """Tests with a specific desired set of NSRTs.
    """
    utils.update_config({"min_data_for_nsrt": 0, "seed": 123,
                         "classifier_max_itr_sampler": 1000,
                         "regressor_max_itr": 1000})
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    cup1 = cup_type("cup1")
    cup2 = cup_type("cup2")
    cup3 = cup_type("cup3")
    cup4 = cup_type("cup4")
    cup5 = cup_type("cup5")
    pred0 = Predicate("Pred0", [cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred1 = Predicate("Pred1", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred2 = Predicate("Pred2", [cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    preds = {pred0, pred1, pred2}
    state1 = State({cup0: [0.4], cup1: [0.7], cup2: [0.1]})
    option1 = ParameterizedOption(
        "dummy", [], Box(0.1, 1, (1,)), lambda s, o, p: Action(p),
        lambda s, o, p: False, lambda s, o, p: False).ground(
            [], np.array([0.2]))
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    dataset = [([state1, next_state1], [action1])]
    nsrts = learn_nsrts_from_data(dataset, preds, do_sampler_learning=True)
    assert len(nsrts) == 1
    nsrt = nsrts.pop()
    assert str(nsrt) == """Operator0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []"""
    # Test the learned samplers
    for _ in range(10):
        assert abs(nsrt.ground([cup0, cup1, cup2]).sample_option(
            state1, np.random.default_rng(123)).params - 0.2) < 0.01
    # The following test was used to manually check that unify caches correctly.
    pred0 = Predicate("Pred0", [cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred1 = Predicate("Pred1", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    pred2 = Predicate("Pred2", [cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    preds = {pred0, pred1, pred2}
    state1 = State({cup0: [0.4], cup1: [0.7], cup2: [0.1]})
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    state2 = State({cup3: [0.4], cup4: [0.7], cup5: [0.1]})
    action2 = option1.policy(state2)
    action2.set_option(option1)
    next_state2 = State({cup3: [0.8], cup4: [0.3], cup5: [1.0]})
    dataset = [([state1, next_state1], [action1]),
               ([state2, next_state2], [action2])]
    nsrts = learn_nsrts_from_data(dataset, preds, do_sampler_learning=True)
    assert len(nsrts) == 1
    nsrt = nsrts.pop()
    assert str(nsrt) == """Operator0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []"""
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
        "dummy", [], Box(0.1, 1, (1,)), lambda s, o, p: Action(p),
        lambda s, o, p: False, lambda s, o, p: False).ground(
            [], np.array([0.3]))
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.9], cup1: [0.2], cup2: [0.5]})
    state2 = State({cup4: [0.9], cup5: [0.2], cup2: [0.5], cup3: [0.5]})
    option2 = ParameterizedOption(
        "dummy", [], Box(0.1, 1, (1,)), lambda s, o, p: Action(p),
        lambda s, o, p: False, lambda s, o, p: False).ground(
            [], np.array([0.7]))
    action2 = option2.policy(state2)
    action2.set_option(option2)
    next_state2 = State({cup4: [0.5], cup5: [0.5], cup2: [1.0], cup3: [0.1]})
    dataset = [([state1, next_state1], [action1]),
               ([state2, next_state2], [action2])]
    nsrts = learn_nsrts_from_data(dataset, preds, do_sampler_learning=True)
    assert len(nsrts) == 2
    expected = {"Operator0": """Operator0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []""", "Operator1": """Operator1:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type, ?x3:cup_type]
    Preconditions: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []"""}
    for nsrt in nsrts:
        assert str(nsrt) == expected[nsrt.name]
        # Test the learned samplers
        if nsrt.name == "Operator0":
            for _ in range(10):
                assert abs(nsrt.ground([cup0, cup1, cup2]).sample_option(
                    state1, np.random.default_rng(123)).params - 0.3) < 0.01
        if nsrt.name == "Operator1":
            for _ in range(10):
                assert abs(nsrt.ground([cup2, cup3, cup4, cup5]).sample_option(
                    state2, np.random.default_rng(123)).params - 0.7) < 0.01
    pred0 = Predicate("Pred0", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.7 and s[o[1]][0] < 0.3)
    preds = {pred0}
    state1 = State({cup0: [0.5], cup1: [0.5]})
    action1 = option2.policy(state1)
    action1.set_option(option2)
    next_state1 = State({cup0: [0.9], cup1: [0.1],})
    state2 = State({cup4: [0.9], cup5: [0.1]})
    action2 = option2.policy(state2)
    action2.set_option(option2)
    next_state2 = State({cup4: [0.5], cup5: [0.5]})
    dataset = [([state1, next_state1], [action1]),
               ([state2, next_state2], [action2])]
    nsrts = learn_nsrts_from_data(dataset, preds, do_sampler_learning=True)
    assert len(nsrts) == 2
    expected = {"Operator0": """Operator0:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: []
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: []
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []""", "Operator1": """Operator1:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Add Effects: []
    Delete Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []"""}
    for nsrt in nsrts:
        assert str(nsrt) == expected[nsrt.name]
    # Test minimum number of examples parameter
    utils.update_config({"min_data_for_nsrt": 3})
    nsrts = learn_nsrts_from_data(dataset, preds, do_sampler_learning=True)
    assert len(nsrts) == 0
    # Test sampler giving out-of-bounds outputs
    utils.update_config({"min_data_for_nsrt": 0, "seed": 123,
                         "classifier_max_itr_sampler": 1,
                         "regressor_max_itr": 1})
    nsrts = learn_nsrts_from_data(dataset, preds, do_sampler_learning=True)
    assert len(nsrts) == 2
    for nsrt in nsrts:
        for _ in range(10):
            assert option1.parent.params_space.contains(
                nsrt.ground([cup0, cup1]).sample_option(
                    state1, np.random.default_rng(123)).params)
    # Test max_rejection_sampling_tries = 0
    utils.update_config({"max_rejection_sampling_tries": 0, "seed": 1234})
    nsrts = learn_nsrts_from_data(dataset, preds, do_sampler_learning=True)
    assert len(nsrts) == 2
    for nsrt in nsrts:
        for _ in range(10):
            assert option1.parent.params_space.contains(
                nsrt.ground([cup0, cup1]).sample_option(
                    state1, np.random.default_rng(123)).params)
    # Test do_sampler_learning = False
    utils.update_config({"seed": 123, "classifier_max_itr_sampler": 100000,
                         "regressor_max_itr": 100000})
    start_time = time.time()
    nsrts = learn_nsrts_from_data(dataset, preds, do_sampler_learning=False)
    assert time.time()-start_time < 0.1  # should be lightning fast
    assert len(nsrts) == 2
    for nsrt in nsrts:
        for _ in range(10):
            # Will just return random parameters
            assert option1.parent.params_space.contains(
                nsrt.ground([cup0, cup1]).sample_option(
                    state1, np.random.default_rng(123)).params)
