"""Test cases for the random options approach class."""

from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.random_options_approach import \
    RandomOptionsApproach
from predicators.src.structs import Action, DefaultState, \
    ParameterizedOption, Predicate, State, Task, Type


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
    for _ in range(10):
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
    act = policy(state)
    assert not act.has_option()  # should have fallen back to random action
    # Test what happens when there's no object of the right type.
    parameterized_option3 = ParameterizedOption("Move", [cup_type],
                                                params_space, _policy,
                                                lambda _1, _2, _3, _4: False,
                                                _terminal)
    task = Task(state, {Solved([cup])})
    approach = RandomOptionsApproach({Solved}, {parameterized_option3},
                                     {cup_type}, params_space, task)
    policy = approach.solve(task, 500)
    act = policy(DefaultState)
    assert not act.has_option()  # should have fallen back to random action
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
    assert str(actions) == "[Action(_arr=array([0.70787615], dtype=float32)), Action(_arr=array([0.3698764], dtype=float32)), Action(_arr=array([0.29010695], dtype=float32)), Action(_arr=array([0.9975787], dtype=float32)), Action(_arr=array([0.9942262], dtype=float32)), Action(_arr=array([0.98252517], dtype=float32)), Action(_arr=array([0.55868745], dtype=float32)), Action(_arr=array([0.68523175], dtype=float32)), Action(_arr=array([0.99104315], dtype=float32)), Action(_arr=array([0.8620031], dtype=float32))]"  # pylint: disable=line-too-long
