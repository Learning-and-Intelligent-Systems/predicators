"""Test cases for the random options approach class."""

import pytest
from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.random_options_approach import \
    RandomOptionsApproach
from predicators.structs import Action, ParameterizedOption, Predicate, \
    State, Task, Type


def test_random_options_approach():
    """Tests for RandomOptionsApproach class."""
    utils.reset_config({"env": "cover"})
    cup_type = Type("cup_type", ["feat1"])
    cup = cup_type("cup")
    state = State({cup: [0.5]})

    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1, )
        ns[cup][0] += a.arr.item()
        return ns

    params_space = Box(0, 1, (1, ))

    def _policy(_1, _2, _3, p):
        return Action(p)

    def _initiable(_1, _2, _3, p):
        return p > 0.25

    def _terminal(s, _1, _2, _3):
        return s[cup][0] > 9.9

    parameterized_option = ParameterizedOption("Move", [], params_space,
                                               _policy, _initiable, _terminal)

    def _solved_classifier(s, o):
        return s[o[0]][0] > 7.5

    Solved = Predicate("Solved", [cup_type], _solved_classifier)
    task = Task(state, {Solved([cup])})
    approach = RandomOptionsApproach({Solved}, {parameterized_option},
                                     {cup_type}, params_space, [task])
    assert not approach.is_learning_based
    policy = approach.solve(task, 500)
    solved = False
    act_var = None
    for _ in range(25):
        act = policy(state)
        assert act.has_option()
        if act_var is None:
            act_var = act.arr.item()
        else:
            # RandomOptionsApproach should use the same option all the way
            # to the end of the execution when the task is solved, so the
            # parameter should always be the same.
            assert abs(act_var - act.arr.item()) < 1e-3
        state = _simulator(state, act)
        if task.goal.issubset(utils.abstract(state, {Solved})):
            solved = True
            break
    assert solved
    # Test what happens when there's no initializable option.
    parameterized_option2 = ParameterizedOption("Move", [], params_space,
                                                _policy,
                                                lambda _1, _2, _3, _4: False,
                                                _terminal)
    task = Task(state, {Solved([cup])})
    approach = RandomOptionsApproach({Solved}, {parameterized_option2},
                                     {cup_type}, params_space, task)
    policy = approach.solve(task, 500)
    with pytest.raises(ApproachFailure) as e:
        policy(state)
    assert "Random option sampling failed!" in str(e)
    # Test what happens when there's no object of the right type.
    dummy_type = Type("dummy_type", ["feat1"])
    parameterized_option3 = ParameterizedOption("Move", [dummy_type],
                                                params_space, _policy,
                                                lambda _1, _2, _3, _4: True,
                                                _terminal)
    task = Task(state, {Solved([cup])})
    approach = RandomOptionsApproach({Solved}, {parameterized_option3},
                                     {cup_type}, params_space, task)
    policy = approach.solve(task, 500)
    with pytest.raises(ApproachFailure) as e:
        policy(state)
    assert "Random option sampling failed!" in str(e)
    # Test what happens when the option is always terminal.
    parameterized_option4 = ParameterizedOption("Move", [], params_space,
                                                _policy, _initiable,
                                                lambda _1, _2, _3, _4: True)
    task = Task(state, {Solved([cup])})
    approach = RandomOptionsApproach({Solved}, {parameterized_option4},
                                     {cup_type}, params_space, [task])
    policy = approach.solve(task, 500)
    act_var = None
    actions = []
    for _ in range(10):
        act = policy(state)
        actions.append(act)
        assert act.has_option()
        if act_var is None:
            act_var = act.arr.item()
        else:
            # RandomOptionsApproach should use different options on each step.
            assert abs(act_var - act.arr.item()) > 1e-3
            act_var = act.arr.item()
        state = _simulator(state, act)
    # Test reproducibility
    assert str(actions) == "[Action(_arr=array([0.6823519], dtype=float32)), Action(_arr=array([0.8120945], dtype=float32)), Action(_arr=array([0.92334497], dtype=float32)), Action(_arr=array([0.2765744], dtype=float32)), Action(_arr=array([0.81975454], dtype=float32)), Action(_arr=array([0.8898927], dtype=float32)), Action(_arr=array([0.51297045], dtype=float32)), Action(_arr=array([0.8242416], dtype=float32)), Action(_arr=array([0.74146706], dtype=float32)), Action(_arr=array([0.6299402], dtype=float32))]"  # pylint: disable=line-too-long
