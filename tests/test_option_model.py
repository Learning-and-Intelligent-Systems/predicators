"""Test cases for option models."""

from typing import Set
import pytest
from gym.spaces import Box
from predicators.src.structs import State, Action, Type, ParameterizedOption
from predicators.src.option_model import create_option_model, \
    _OracleOptionModel
from predicators.src import utils


def test_default_option_model():
    """Tests for the default option model."""

    # First test create_option_model with a real environment.
    utils.reset_config({"env": "cover"})
    model = create_option_model("oracle")
    assert isinstance(model, _OracleOptionModel)

    # Test with a mock environment.
    type1 = Type("type1", ["feat1", "feat2"])
    type2 = Type("type2", ["feat3", "feat4", "feat5"])
    obj3 = type1("obj3")
    obj7 = type1("obj7")
    obj1 = type2("obj1")
    obj4 = type2("obj4")
    obj9 = type2("obj9")

    class _MockEnv:

        @staticmethod
        def simulate(state: State, action: Action) -> State:
            """A mock simulate method."""
            next_state = state.copy()
            obj = list(state)[0]
            next_state.set(obj, "feat3",
                           next_state.get(obj, "feat3") + action.arr[1])
            return next_state

        @property
        def options(self) -> Set[ParameterizedOption]:
            """Mock options."""
            params_space = Box(-10, 10, (2, ))

            def policy(s, m, o, p):
                del s, m, o  # unused
                return Action(p * 2)

            def initiable(s, m, o, p):
                del o, p  # unused
                obj = list(s)[0]
                m["start_state"] = s
                return s[obj][0] < 10 or s[obj][0] > 60

            def terminal(s, m, o, p):
                del o, p  # unused
                obj = list(s)[0]
                return s[obj][0] > 50 and not s.allclose(m["start_state"])

            parameterized_option = ParameterizedOption("Pick", [],
                                                       params_space, policy,
                                                       initiable, terminal)

            return {parameterized_option}

    env = _MockEnv()
    parameterized_option = env.options.pop()

    params = [-5, 5]
    option1 = parameterized_option.ground([], params)
    params = [-7, 7]
    option2 = parameterized_option.ground([], params)
    params = [-8, 2]
    option3 = parameterized_option.ground([], params)
    state = State({
        obj3: [1, 2],
        obj7: [3, 4],
        obj1: [5, 6, 7],
        obj4: [8, 9, 10],
        obj9: [11, 12, 13]
    })
    model = _OracleOptionModel(env)
    next_state = model.get_next_state(state, option1)
    # Test that the option's memory has not been updated.
    assert "start_state" not in option1.memory
    # But after actually calling the option, it should be updated.
    option1.initiable(state)
    assert "start_state" in option1.memory
    assert abs(next_state.get(obj1, "feat3") - 55) < 1e-6
    with pytest.raises(AssertionError):  # option2 is not initiable
        model.get_next_state(next_state, option2)
    next_state = model.get_next_state(state, option2)
    assert abs(next_state.get(obj1, "feat3") - 61) < 1e-6
    next_next_state = model.get_next_state(next_state, option3)
    assert abs(next_state.get(obj1, "feat3") - 61) < 1e-6  # no change
    assert abs(next_next_state.get(obj1, "feat3") - 65) < 1e-6


def test_option_model_notimplemented():
    """Tests for various NotImplementedErrors."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
    })
    with pytest.raises(NotImplementedError):
        create_option_model("not a real option model")
