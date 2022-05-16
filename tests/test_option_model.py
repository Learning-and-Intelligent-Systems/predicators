"""Test cases for option models."""

import pytest
from gym.spaces import Box

from predicators.src import utils
from predicators.src.option_model import _BehaviorOptionModel, \
    _OracleOptionModel, create_option_model
from predicators.src.structs import Action, ParameterizedOption, State, Type


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

    class _MockEnv:

        @staticmethod
        def simulate(state, action):
            """A mock simulate method."""
            next_state = state.copy()
            obj = list(state)[0]
            next_state.set(obj, "feat3",
                           next_state.get(obj, "feat3") + action.arr[1])
            return next_state

        @property
        def options(self):
            """Mock options."""

            parameterized_option = ParameterizedOption("Pick", [],
                                                       params_space, policy,
                                                       initiable, terminal)

            return {parameterized_option}

    env = _MockEnv()
    parameterized_option = env.options.pop()

    params1 = [-5, 5]
    option1 = parameterized_option.ground([], params1)
    params2 = [-7, 7]
    option2 = parameterized_option.ground([], params2)
    params3 = [-8, 2]
    option3 = parameterized_option.ground([], params3)
    state = State({
        obj3: [1, 2],
        obj7: [3, 4],
        obj1: [5, 6, 7],
        obj4: [8, 9, 10],
        obj9: [11, 12, 13]
    })
    model = _OracleOptionModel(env)
    next_state, num_act = model.get_next_state_and_num_actions(state, option1)
    assert num_act == 5
    # Test that the option's memory has not been updated.
    assert "start_state" not in option1.memory
    # But after actually calling the option, it should be updated.
    option1.initiable(state)
    assert "start_state" in option1.memory
    assert abs(next_state.get(obj1, "feat3") - 55) < 1e-6
    with pytest.raises(AssertionError):  # option2 is not initiable
        model.get_next_state_and_num_actions(next_state, option2)
    next_state, num_act = model.get_next_state_and_num_actions(state, option2)
    remembered_next_state = next_state
    assert num_act == 4
    assert abs(next_state.get(obj1, "feat3") - 61) < 1e-6
    next_next_state, num_act = model.get_next_state_and_num_actions(
        next_state, option3)
    assert num_act == 1
    assert abs(next_state.get(obj1, "feat3") - 61) < 1e-6  # no change
    assert abs(next_next_state.get(obj1, "feat3") - 65) < 1e-6
    # Test calling the option model with a learned option.
    learned_param_opt = ParameterizedOption("MockOption", [], params_space,
                                            policy, initiable, terminal)
    learned_option = learned_param_opt.ground([], params2)
    with pytest.raises(AssertionError):  # "Learned" doesn't appear in the name
        model.get_next_state_and_num_actions(state, learned_option)
    learned_param_opt = ParameterizedOption("MockLearnedOption", [],
                                            params_space, policy, initiable,
                                            terminal)
    learned_option = learned_param_opt.ground([], params2)
    # We fixed the name; now it should work.
    next_state, num_act = model.get_next_state_and_num_actions(
        state, learned_option)
    assert num_act == 4
    assert remembered_next_state.allclose(next_state)

    # Test case where an option execution failure occurs. Expected behavior is
    # that the transition should function as a noop.
    def failing_policy(s, m, o, p):
        del s, m, o, p  # unused
        raise utils.OptionExecutionFailure("mock error")

    failing_param_opt = ParameterizedOption("FailingLearnedOption", [],
                                            params_space, failing_policy,
                                            initiable, terminal)
    failing_option = failing_param_opt.ground([], params2)
    next_state, num_act = model.get_next_state_and_num_actions(
        state, failing_option)
    assert num_act == 0
    assert state.allclose(next_state)

    # Test case where an option gets stuck in a state.
    never_terminate = lambda s, o, m, p: False
    infinite_param_opt = ParameterizedOption("InfiniteLearnedOption", [],
                                             params_space, policy, initiable,
                                             never_terminate)

    class _NoopMockEnv:

        @staticmethod
        def simulate(state, action):
            """A mock simulate method."""
            del action  # unused
            return state.copy()

        @property
        def options(self):
            """Mock options."""
            return {infinite_param_opt}

    infinite_option = infinite_param_opt.ground([], params1)
    env = _NoopMockEnv()
    model = _OracleOptionModel(env)
    next_state, num_act = model.get_next_state_and_num_actions(
        state, infinite_option)
    assert next_state.allclose(state)
    assert num_act == 0


def test_option_model_notimplemented():
    """Tests for various NotImplementedErrors."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
    })
    with pytest.raises(NotImplementedError):
        create_option_model("not a real option model")


def test_create_option_model():
    """Tests for create_option_model()."""
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
    })
    model = create_option_model("oracle")
    assert isinstance(model, _OracleOptionModel)
    utils.reset_config({
        "env": "pybullet_blocks",
        "approach": "nsrt_learning",
    })
    model = create_option_model("oracle_blocks")
    assert isinstance(model, _OracleOptionModel)
    model = create_option_model("oracle_behavior")
    assert isinstance(model, _BehaviorOptionModel)
