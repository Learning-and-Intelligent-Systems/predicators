"""Test cases for the operator learning approach.
"""

from gym.spaces import Box
import numpy as np
from predicators.src.envs import CoverEnv
from predicators.src.approaches import OperatorLearningApproach
from predicators.src.datasets import create_dataset
from predicators.src.settings import CFG
from predicators.src.structs import Type, Predicate, State, Action, \
    ParameterizedOption
from predicators.src import utils


def test_operator_learning_approach():
    """Tests for OperatorLearningApproach class.
    """
    utils.update_config({"env": "cover", "approach": "operator_learning",
                         "timeout": 10, "max_samples_per_step": 1000,
                         "seed": 0})
    env = CoverEnv()
    approach = OperatorLearningApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    dataset = create_dataset(env)
    assert approach.is_learning_based
    approach.learn_from_offline_dataset(dataset)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=CFG.timeout)
        assert utils.policy_solves_task(
            policy, task, env.simulate, env.predicates)


def test_operator_learning_specific_operators():
    """Tests with a specific desired set of operators.
    """
    option = ParameterizedOption(
        "dummy", Box(0, 1, (1,)), lambda s, p: Action(np.array([0.0])),
        lambda s, p: False, lambda s, p: False).ground(np.array([0.0]))
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
    action1 = option.policy(state1)
    action1.set_option((option, 0))
    next_state1 = State({cup0: [0.8], cup1: [0.3], cup2: [1.0]})
    dataset = [([state1, next_state1], [action1])]
    ops = OperatorLearningApproach.learn_operators_from_data(dataset, preds)
    assert len(ops) == 1
    op = ops.pop()
    assert str(op) == """dummy0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Add Effects: [Pred0(?x2:cup_type), Pred0(?x0:cup_type), Pred1(?x2:cup_type, ?x0:cup_type), Pred1(?x0:cup_type, ?x2:cup_type), Pred1(?x2:cup_type, ?x1:cup_type), Pred1(?x0:cup_type, ?x1:cup_type), Pred2(?x0:cup_type), Pred2(?x2:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type), Pred1(?x1:cup_type, ?x0:cup_type), Pred1(?x1:cup_type, ?x2:cup_type), Pred2(?x1:cup_type)]
    Option: ParameterizedOption(name='dummy')"""
    # The following two tests check edge cases of unification with respect to
    # the split between add and delete effects. Specifically, it's important
    # to unify both of them together, not separately, which requires changing
    # the predicates so that unification does not try to unify add ones with
    # delete ones.
    pred0 = Predicate("Pred0", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.7 and s[o[1]][0] < 0.3)
    preds = {pred0}
    state1 = State({cup0: [0.4], cup1: [0.8], cup2: [0.1]})
    action1 = option.policy(state1)
    action1.set_option((option, 0))
    next_state1 = State({cup0: [0.9], cup1: [0.2], cup2: [0.5]})
    state2 = State({cup4: [0.9], cup5: [0.2], cup2: [0.5], cup3: [0.5]})
    action2 = option.policy(state1)
    action2.set_option((option, 0))
    next_state2 = State({cup4: [0.5], cup5: [0.5], cup2: [1.0], cup3: [0.1]})
    dataset = [([state1, next_state1], [action1]),
               ([state2, next_state2], [action2])]
    ops = OperatorLearningApproach.learn_operators_from_data(dataset, preds)
    assert len(ops) == 2
    expected = {"dummy0": """dummy0:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type]
    Preconditions: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x1:cup_type, ?x2:cup_type)]
    Option: ParameterizedOption(name='dummy')""", "dummy1": """dummy1:
    Parameters: [?x0:cup_type, ?x1:cup_type, ?x2:cup_type, ?x3:cup_type]
    Preconditions: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: [Pred0(?x2:cup_type, ?x3:cup_type)]
    Option: ParameterizedOption(name='dummy')"""}
    for op in ops:
        assert str(op) == expected[op.name]
    pred0 = Predicate("Pred0", [cup_type, cup_type],
                      lambda s, o: s[o[0]][0] > 0.7 and s[o[1]][0] < 0.3)
    preds = {pred0}
    state1 = State({cup0: [0.5], cup1: [0.5]})
    action1 = option.policy(state1)
    action1.set_option((option, 0))
    next_state1 = State({cup0: [0.9], cup1: [0.1],})
    state2 = State({cup4: [0.9], cup5: [0.1]})
    action2 = option.policy(state1)
    action2.set_option((option, 0))
    next_state2 = State({cup4: [0.5], cup5: [0.5]})
    dataset = [([state1, next_state1], [action1]),
               ([state2, next_state2], [action2])]
    ops = OperatorLearningApproach.learn_operators_from_data(dataset, preds)
    assert len(ops) == 2
    expected = {"dummy0": """dummy0:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: []
    Add Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Delete Effects: []
    Option: ParameterizedOption(name='dummy')""", "dummy1": """dummy1:
    Parameters: [?x0:cup_type, ?x1:cup_type]
    Preconditions: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Add Effects: []
    Delete Effects: [Pred0(?x0:cup_type, ?x1:cup_type)]
    Option: ParameterizedOption(name='dummy')"""}
    for op in ops:
        assert str(op) == expected[op.name]
