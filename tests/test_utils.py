"""Test cases for utils.
"""

import pytest
from gym.spaces import Box  # type: ignore
from predicators.src.structs import State, Type, ParameterizedOption
from predicators.src import utils


def test_option_to_trajectory():
    """Tests for option_to_trajectory().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})
    def _simulator(s, a):
        ns = s.copy()
        assert a.shape == (1,)
        ns[cup][0] += a.item()
        return ns
    params_space = Box(0, 1, (1,))
    def _policy(_, p):
        return p
    def _initiable(_1, p):
        return p > 0.25
    def _terminal(s, _):
        return s[cup][0] > 9.9
    parameterized_option = ParameterizedOption(
        "Move", params_space, _policy, _initiable, _terminal)
    params = [0.1]
    option = parameterized_option.ground(params)
    with pytest.raises(AssertionError):
        # option is not initiable from start state
        utils.option_to_trajectory(state, _simulator, option,
                                   max_num_steps=5)
    params = [0.5]
    option = parameterized_option.ground(params)
    states, actions = utils.option_to_trajectory(
        state, _simulator, option, max_num_steps=100)
    assert len(actions) == len(states)-1 == 19
    states, actions = utils.option_to_trajectory(
        state, _simulator, option, max_num_steps=10)
    assert len(actions) == len(states)-1 == 10
