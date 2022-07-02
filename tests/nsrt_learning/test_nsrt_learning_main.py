"""Tests for NSRT learning."""

import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.nsrt_learning.nsrt_learning_main import \
    learn_nsrts_from_data
from predicators.src.structs import Action, GroundAtom, LowLevelTrajectory, \
    Predicate, State, Task, Type


def test_nsrt_learning_specific_nsrts():
    """Tests with a specific desired set of NSRTs."""
    utils.reset_config({
        "min_data_for_nsrt": 0,
        "min_perc_data_for_nsrt": 0,
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
    action_space = Box(0, 1, (1, ))
    param_option1 = utils.SingletonParameterizedOption(
        "Dummy", lambda s, m, o, p: Action(p), params_space=Box(0.1, 1, (1, )))
    options = {param_option1}
    option1 = param_option1.ground([], np.array([0.2]))
    assert option1.initiable(state1)
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    dataset = [LowLevelTrajectory([state1, next_state1], [action1])]
    nsrts, _, _ = learn_nsrts_from_data(dataset, [],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="neural")
    assert len(nsrts) == 1
    nsrt = nsrts.pop()
    assert str(nsrt) == """NSRT-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x0:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred1(?x2:cup_type, ?x2:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Side Predicates: []
    Option Spec: Dummy()"""
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
    option1 = param_option1.ground([], np.array([0.2]))
    options = {param_option1}
    assert option1.initiable(state1)
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    state2 = State({cup3: [0.4], cup4: [0.7], cup5: [0.1]})
    option1 = param_option1.ground([], np.array([0.2]))
    assert option1.initiable(state2)
    action2 = option1.policy(state2)
    action2.set_option(option1)
    next_state2 = State({cup3: [0.8], cup4: [0.3], cup5: [1.0]})
    dataset = [
        LowLevelTrajectory([state1, next_state1], [action1]),
        LowLevelTrajectory([state2, next_state2], [action2])
    ]
    nsrts, _, _ = learn_nsrts_from_data(dataset, [],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="random")
    assert len(nsrts) == 1
    nsrt = nsrts.pop()
    assert str(nsrt) == """NSRT-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x0:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred1(?x2:cup_type, ?x2:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x1:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Side Predicates: []
    Option Spec: Dummy()"""
    # The following two tests check edge cases of unification with respect to
    # the split between add and delete effects. Specifically, it's important
    # to unify both of them together, not separately, which requires changing
    # the predicates so that unification does not try to unify add ones with
    # delete ones.
    pred0 = Predicate("Pred0", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.7 and s[o[1]][0] < 0.3)
    preds = {pred0}
    state1 = State({cup0: [0.4], cup1: [0.8], cup2: [0.1]})
    option1 = utils.SingletonParameterizedOption(
        "Dummy",
        lambda s, m, o, p: Action(p),
        params_space=Box(0.1, 0.5, (1, ))).ground([], np.array([0.3]))
    option1 = param_option1.ground([], np.array([0.2]))
    assert option1.initiable(state1)
    action1 = option1.policy(state1)
    action1.set_option(option1)
    next_state1 = State({cup0: [0.9], cup1: [0.2], cup2: [0.5]})
    state2 = State({cup4: [0.9], cup5: [0.2], cup2: [0.5], cup3: [0.5]})
    param_option2 = utils.SingletonParameterizedOption(
        "Dummy",
        lambda s, m, o, p: Action(p),
        params_space=Box(0.1, 0.5, (1, )))
    options = {param_option1, param_option2}
    option2 = param_option2.ground([], np.array([0.5]))
    assert option2.initiable(state2)
    action2 = option2.policy(state2)
    action2.set_option(option2)
    next_state2 = State({cup4: [0.5], cup5: [0.5], cup2: [1.0], cup3: [0.1]})
    dataset = [
        LowLevelTrajectory([state1, next_state1], [action1]),
        LowLevelTrajectory([state2, next_state2], [action2])
    ]
    nsrts, _, _ = learn_nsrts_from_data(dataset, [],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="random")
    assert len(nsrts) == 2
    expected = {
        "Op0":
        """NSRT-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Side Predicates: []
    Option Spec: Dummy()""",
        "Op1":
        """NSRT-Op1:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type, ?x3:cup_type]
    Preconditions: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Side Predicates: []
    Option Spec: Dummy()"""
    }
    pred0 = Predicate("Pred0", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.7 and s[o[1]][0] < 0.3)
    preds = {pred0}
    state1 = State({cup0: [0.5], cup1: [0.5]})
    option2 = param_option2.ground([], np.array([0.5]))
    assert option2.initiable(state1)
    action1 = option2.policy(state1)
    action1.set_option(option2)
    next_state1 = State({
        cup0: [0.9],
        cup1: [0.1],
    })
    state2 = State({cup4: [0.9], cup5: [0.1]})
    option2 = param_option2.ground([], np.array([0.5]))
    assert option2.initiable(state2)
    action2 = option2.policy(state2)
    action2.set_option(option2)
    next_state2 = State({cup4: [0.5], cup5: [0.5]})
    dataset = [
        LowLevelTrajectory([state1, next_state1], [action1]),
        LowLevelTrajectory([state2, next_state2], [action2])
    ]
    nsrts, _, _ = learn_nsrts_from_data(dataset, [],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="random")
    assert len(nsrts) == 2
    expected = {
        "Op0":
        """NSRT-Op0:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: []
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: []
    Side Predicates: []
    Option Spec: Dummy()""",
        "Op1":
        """NSRT-Op1:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Add Effects: []
    Delete Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Side Predicates: []
    Option Spec: Dummy()"""
    }
    for nsrt in nsrts:
        assert str(nsrt) == expected[nsrt.name]
    # Test minimum number of examples parameter
    utils.update_config({
        "min_data_for_nsrt": 3,
        "min_perc_data_for_nsrt": 0,
    })
    nsrts, _, _ = learn_nsrts_from_data(dataset, [],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="random")
    assert len(nsrts) == 0
    # Test minimum percent of examples parameter
    utils.update_config({
        "min_data_for_nsrt": 0,
        "min_perc_data_for_nsrt": 90,
    })
    nsrts, _, _ = learn_nsrts_from_data(dataset, [],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="random")
    assert len(nsrts) == 0
    # Test max_rejection_sampling_tries = 0
    utils.update_config({
        "min_data_for_nsrt": 0,
        "min_perc_data_for_nsrt": 0,
        "max_rejection_sampling_tries": 0,
        "sampler_mlp_classifier_max_itr": 1,
        "neural_gaus_regressor_max_itr": 1
    })
    nsrts, _, _ = learn_nsrts_from_data(dataset, [],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="neural")
    assert len(nsrts) == 2
    for nsrt in nsrts:
        for _ in range(10):
            sampled_params = nsrt.ground([cup0, cup1]).sample_option(
                state1, set(), np.random.default_rng(123)).params
            assert option1.parent.params_space.contains(sampled_params)

    ###
    # Weird Case
    # Agent features are loc: 0, 1, 2 [start, shelf1, shelf2]; holding: True or False whether an object is in hand
    agent_type = Type("agent_type", ["loc", "holding"])
    agent = agent_type("agent")
    # Hardback features are loc: -1, 0, 1, 2 [in_hand, start, shelf1, shelf2]
    hardback_type = Type("hardback_type", ["loc"]) # loc: -1, 0, 1, 2 [in_hand, start, shelf1, shelf2]
    hardback1 = hardback_type("hardback1")
    hardback2 = hardback_type("hardback2")
    # Shelf features are loc: 2 (only a shelf at location two)
    shelf_type = Type("shelf_type", ["loc"])
    shelf = shelf_type("shelf")
    # Predicates
    NextTo = Predicate("NextTo", [hardback_type], lambda s, o: s[o[0]][0] == -1 or s[o[0]][0] == s[agent][0] or (s[o[0]][0] in [1,2] and s[agent][0] in [1,2]))
    NextToShelf = Predicate("NextToShelf", [shelf_type], lambda s, o: s[agent][0] == 2)
    HandEmpty = Predicate("HandEmpty", [], lambda s, o: s[agent][1])
    Holding = Predicate("Holding", [hardback_type], lambda s, o: s[o[0]][0] == -1)
    OnTop = Predicate("OnTop", [hardback_type, shelf_type], lambda s, o: s[o[0]][0] == s[o[1]][0])
    preds = {NextTo, NextToShelf, HandEmpty, Holding, OnTop}

    # Agent not holding anything at location 0, hardbacks at loaction 1, and shelf at location 2
    state1 = State({agent: [0, False], hardback1: [1], hardback2: [1], shelf: [2]})
    # Agent moves to location 1
    state2 = State({agent: [1, False], hardback1: [1], hardback2: [1], shelf: [2]})
    # Agent grabs hardback1
    state3 = State({agent: [1, True], hardback1: [-1], hardback2: [1], shelf: [2]})
    # Agent moves to location 2
    state4 = State({agent: [2, True], hardback1: [-1], hardback2: [1], shelf: [2]})
    # Agent places hardback1
    state5 = State({agent: [2, False], hardback1: [2], hardback2: [1], shelf: [2]})
    # Agent moves to location 1
    state6 = State({agent: [1, False], hardback1: [2], hardback2: [1], shelf: [2]})
    # Agent grabs hardback2
    state7 = State({agent: [1, True], hardback1: [2], hardback2: [-1], shelf: [2]})
    # Agent moves to location 2
    state8 = State({agent: [2, True], hardback1: [2], hardback2: [-1], shelf: [2]})
    # Agent places hardback2
    state9 = State({agent: [2, False], hardback1: [2], hardback2: [2], shelf: [2]})
    action_space = Box(0, 1, (1, ))

    moveto_param_option = utils.ParameterizedOption(
        "MoveTo",  [hardback_type], policy=lambda s, m, o, p: Action(p), params_space=Box(0.1, 1, (1, )), initiable=lambda _1, _2, _3, _4 : True, terminal=lambda _1, _2, _3, _4 : True)
    moveto_option = moveto_param_option.ground([hardback1], np.array([0.5]))
    assert moveto_option.initiable(state1)
    moveto_hard2 = moveto_option.policy(state1)
    moveto_hard2.set_option(moveto_option)

    pick_param_option = utils.ParameterizedOption(
        "Pick",  [hardback_type], policy=lambda s, m, o, p: Action(p), params_space=Box(0.1, 1, (1, )), initiable=lambda _1, _2, _3, _4 : True, terminal=lambda _1, _2, _3, _4 : True)
    pick_option = pick_param_option.ground([hardback1], np.array([0.5]))
    assert pick_option.initiable(state2)
    pick_hard2 = pick_option.policy(state2)
    pick_hard2.set_option(pick_option)
    
    movetoshelf_param_option = utils.ParameterizedOption(
        "MoveToShelf",  [shelf_type], policy=lambda s, m, o, p: Action(p), params_space=Box(0.1, 1, (1, )), initiable=lambda _1, _2, _3, _4 : True, terminal=lambda _1, _2, _3, _4 : True)
    movetoshelf_option = movetoshelf_param_option.ground([shelf], np.array([0.5]))
    assert movetoshelf_option.initiable(state3)
    movetoshelf1 = movetoshelf_option.policy(state3)
    movetoshelf1.set_option(movetoshelf_option)

    place_param_option = utils.ParameterizedOption(
        "Place",  [hardback_type, shelf_type], policy=lambda s, m, o, p: Action(p), params_space=Box(0.1, 1, (1, )), initiable=lambda _1, _2, _3, _4 : True, terminal=lambda _1, _2, _3, _4 : True)
    place_option = place_param_option.ground([hardback1, shelf], np.array([0.5]))
    assert place_option.initiable(state4)
    place_hard2 = place_option.policy(state4)
    place_hard2.set_option(place_option)

    moveto_option_2 = moveto_param_option.ground([hardback2], np.array([0.5]))
    assert moveto_option_2.initiable(state5)
    moveto_hard1 = moveto_option_2.policy(state5)
    moveto_hard1.set_option(moveto_option_2)
    pick_option_2 = pick_param_option.ground([hardback2], np.array([0.5]))
    assert pick_option_2.initiable(state6)
    pick_hard1 = pick_option_2.policy(state6)
    pick_hard1.set_option(pick_option_2)
    movetoshelf_option_2 = movetoshelf_param_option.ground([shelf], np.array([0.5]))
    assert movetoshelf_option_2.initiable(state7)
    movetoshelf2 = movetoshelf_option_2.policy(state7)
    movetoshelf2.set_option(movetoshelf_option_2)
    place_option_2 = place_param_option.ground([hardback1, shelf], np.array([0.5]))
    assert place_option_2.initiable(state8)
    place_hard1 = place_option_2.policy(state8)
    place_hard1.set_option(place_option_2)
    moveto_option_3 = moveto_param_option.ground([hardback1], np.array([0.5]))
    assert moveto_option_3.initiable(state1)
    moveto_hard1_2 = moveto_option_3.policy(state1)
    moveto_hard1_2.set_option(moveto_option_3)
    options = {moveto_param_option, pick_param_option, movetoshelf_param_option, place_param_option}
    
    # Two Tasks: (1) place both objects on top shelf and (2) navigate next to hardback1
    task1 = Task(init=state1, goal=[GroundAtom(OnTop, [hardback1, shelf]), GroundAtom(OnTop, [hardback2, shelf])])
    task2 = Task(init=state1, goal=[GroundAtom(NextTo, [hardback1])])
    dataset = [
        LowLevelTrajectory([state1, state2, state3, state4, state5, state6, state7, state8, state9], 
            [moveto_hard2, pick_hard2, movetoshelf1, place_hard2, moveto_hard1, pick_hard1, movetoshelf2, place_hard1]),
        LowLevelTrajectory([state1, state2], [moveto_hard1_2])
    ]

    nsrts, _, _ = learn_nsrts_from_data(dataset, [task1, task2],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="random")
    dataset.reverse()
    reverse_order_nsrts, _, _ = learn_nsrts_from_data(dataset, [task2, task1],
                                        preds,
                                        options,
                                        action_space,
                                        sampler_learner="random")
                                        
    assert nsrts == reverse_order_nsrts