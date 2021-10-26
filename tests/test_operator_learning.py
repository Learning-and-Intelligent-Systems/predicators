"""Tests for operator learning.
"""

import time
from gym.spaces import Box
import numpy as np
from predicators.src.operator_learning import \
    learn_operators_from_data, _create_sampler_data
from predicators.src.structs import Type, Predicate, State, Action, \
    ParameterizedOption, LiftedAtom
from predicators.src import utils


def test_operator_learning_specific_operators():
    """Tests with a specific desired set of operators.
    """
    utils.update_config({"min_data_for_operator": 0, "seed": 123,
                         "classifier_max_itr": 1000,
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
    ops, _ = learn_operators_from_data(dataset, preds, do_sampler_learning=True)
    assert len(ops) == 1
    op = ops.pop()
    assert str(op) == """dummy0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x0:cup_type), Pred0(?x2:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []"""
    # Test the learned samplers
    for _ in range(10):
        assert abs(op.ground([cup0, cup1, cup2]).sample_option(
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
    ops, _ = learn_operators_from_data(dataset, preds, do_sampler_learning=True)
    assert len(ops) == 1
    op = ops.pop()
    assert str(op) == """dummy0:
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
    ops, _ = learn_operators_from_data(dataset, preds, do_sampler_learning=True)
    assert len(ops) == 2
    expected = {"dummy0": """dummy0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []""", "dummy1": """dummy1:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type, ?x3:cup_type]
    Preconditions: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []"""}
    for op in ops:
        assert str(op) == expected[op.name]
        # Test the learned samplers
        if op.name == "dummy0":
            for _ in range(10):
                assert abs(op.ground([cup0, cup1, cup2]).sample_option(
                    state1, np.random.default_rng(123)).params - 0.3) < 0.01
        if op.name == "dummy1":
            for _ in range(10):
                assert abs(op.ground([cup2, cup3, cup4, cup5]).sample_option(
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
    ops, _ = learn_operators_from_data(dataset, preds, do_sampler_learning=True)
    assert len(ops) == 2
    expected = {"dummy0": """dummy0:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: []
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: []
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []""", "dummy1": """dummy1:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Add Effects: []
    Delete Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Option: ParameterizedOption(name='dummy', types=[])
    Option Variables: []"""}
    for op in ops:
        assert str(op) == expected[op.name]
    # Test minimum number of examples parameter
    utils.update_config({"min_data_for_operator": 3})
    ops, _ = learn_operators_from_data(dataset, preds, do_sampler_learning=True)
    assert len(ops) == 0
    # Test sampler giving out-of-bounds outputs
    utils.update_config({"min_data_for_operator": 0, "seed": 123,
                         "classifier_max_itr": 1,
                         "regressor_max_itr": 1})
    ops, _ = learn_operators_from_data(dataset, preds, do_sampler_learning=True)
    assert len(ops) == 2
    for op in ops:
        for _ in range(10):
            assert option1.parent.params_space.contains(
                op.ground([cup0, cup1]).sample_option(
                    state1, np.random.default_rng(123)).params)
    # Test max_rejection_sampling_tries = 0
    utils.update_config({"max_rejection_sampling_tries": 0, "seed": 1234})
    ops, _ = learn_operators_from_data(dataset, preds, do_sampler_learning=True)
    assert len(ops) == 2
    for op in ops:
        for _ in range(10):
            assert option1.parent.params_space.contains(
                op.ground([cup0, cup1]).sample_option(
                    state1, np.random.default_rng(123)).params)
    # Test do_sampler_learning = False
    utils.update_config({"seed": 123, "classifier_max_itr": 100000,
                         "regressor_max_itr": 100000})
    start_time = time.time()
    ops, _ = learn_operators_from_data(
        dataset, preds, do_sampler_learning=False)
    assert time.time()-start_time < 0.1  # should be lightning fast
    assert len(ops) == 2
    for op in ops:
        for _ in range(10):
            # Will just return random parameters
            assert option1.parent.params_space.contains(
                op.ground([cup0, cup1]).sample_option(
                    state1, np.random.default_rng(123)).params)
    # The following test checks edge cases of unification with respect to
    # the split between effects and option variables. This is similar to
    # the tests above, but with options instead of delete effects.
    # This also tests the case where an operator parameter appears in
    # an option variable alone, rather than in effects or preconds.
    # The case is basically this:
    # Add set 1: P(x, y)
    # Option 1: A(y, z)
    # Add set 2: P(a, b)
    # Option 2: A(c, d)
    pred0 = Predicate("Pred0", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.7 and s[o[1]][0] < 0.3)
    preds = {pred0}
    # Nothing true
    state3 = State({cup0: [0.4], cup1: [0.2], cup2: [0.1]})
    # Option0(cup0, cup1)
    option3 = ParameterizedOption(
        "Option0", [cup_type, cup_type], Box(0.1, 1, (1,)),
        lambda s, o, p: Action(p),
        lambda s, o, p: True, lambda s, o, p: True).ground(
            [cup0, cup1], np.array([0.3]))
    action3 = option3.policy(state3)
    action3.set_option(option3)
    # Pred0(cup1, cup2) true
    next_state3 = State({cup0: [0.4], cup1: [0.8], cup2: [0.1]})
    # Nothing true
    state4 = State({cup4: [0.2], cup5: [0.2], cup2: [0.5], cup3: [0.5]})
    # Option0(cup2, cup3)
    option4 = ParameterizedOption(
        "Option0", [cup_type, cup_type], Box(0.1, 1, (1,)),
        lambda s, o, p: Action(p),
        lambda s, o, p: True, lambda s, o, p: True).ground(
            [cup2, cup3], np.array([0.7]))
    action4 = option4.policy(state4)
    action4.set_option(option4)
    # Pred0(cup4, cup5) True
    next_state4 = State({cup4: [0.8], cup5: [0.1], cup2: [0.5], cup3: [0.5]})
    dataset = [([state3, next_state3], [action3]),
               ([state4, next_state4], [action4])]
    ops, _ = learn_operators_from_data(
        dataset, preds, do_sampler_learning=False)
    assert len(ops) == 2


def test_create_sampler_data():
    """Tests for _create_sampler_data().
    """
    utils.update_config({"min_data_for_operator": 0, "seed": 123})
    # Create two partitions
    cup_type = Type("cup_type", ["feat1"])
    cup0 = cup_type("cup0")
    var_cup0 = cup_type("?cup0")
    pred0 = Predicate("Pred0", [cup_type],
                      lambda s, o: s[o[0]][0] > 0.5)
    predicates = {pred0}
    option = ParameterizedOption(
        "dummy", [], Box(0.1, 1, (1,)), lambda s, o, p: Action(p),
        lambda s, o, p: False, lambda s, o, p: False).ground(
            [], np.array([0.3]))

    # Transition 1: adds pred0(cup0)
    state = State({cup0: [0.4]})
    action = option.policy(state)
    action.set_option(option)
    next_state = State({cup0: [0.9]})
    atoms = utils.abstract(state, predicates)
    next_atoms = utils.abstract(next_state, predicates)
    add_effects = next_atoms - atoms
    delete_effects = atoms - next_atoms
    transition1 = (state, next_state, atoms, option, next_atoms,
                   add_effects, delete_effects)

    # Transition 2: does nothing
    state = State({cup0: [0.4]})
    action = option.policy(state)
    action.set_option(option)
    next_state = state
    atoms = utils.abstract(state, predicates)
    next_atoms = utils.abstract(next_state, predicates)
    add_effects = next_atoms - atoms
    delete_effects = atoms - next_atoms
    transition2 = (state, next_state, atoms, option, next_atoms,
                   add_effects, delete_effects)

    transitions = [[(transition1, {cup0: var_cup0})],
                   [(transition2, {})]]
    variables = [var_cup0]
    preconditions = set()
    add_effects = {LiftedAtom(pred0, [var_cup0])}
    delete_effects = set()
    param_option = option.parent
    partition_idx = 0

    positive_examples, negative_examples = _create_sampler_data(
        transitions, variables, preconditions, add_effects,
        delete_effects, param_option, partition_idx)
    assert len(positive_examples) == 1
    assert len(negative_examples) == 1

    # When building data for a partition with effects X, if we
    # encounter a transition with effects Y, and if Y is a superset
    # of X, then we do not want to include the transition as a
    # negative example, because if Y was achieved, then X was also
    # achieved. So for now, we just filter out such examples.
    #
    # In the example here, transition 1's effects are a superset
    # of transition 2's effects. So when creating the examples
    # for partition 2, we do not want to inclue transition 1
    # in the negative effects.
    variables = []
    add_effects = set()
    partition_idx = 1
    positive_examples, negative_examples = _create_sampler_data(
        transitions, variables, preconditions, add_effects,
        delete_effects, param_option, partition_idx)
    assert len(positive_examples) == 1
    assert len(negative_examples) == 0
