"""Test cases for the TAMP approach.
"""

import numpy as np
import pytest
from predicators.src.envs import CoverEnv
from predicators.src.approaches import TAMPApproach, ApproachFailure
from predicators.src.approaches.tamp_approach import option_plan_to_policy
from predicators.src.structs import Type, State, Action, Box, \
    ParameterizedOption
from predicators.src import utils


def test_tamp_approach():
    """Tests for TAMPApproach class.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnv()
    approach = TAMPApproach(
        env.simulate, env.predicates, env.options, env.types,
        env.action_space, env.get_train_tasks())
    for task in env.get_train_tasks():
        with pytest.raises(NotImplementedError):
            approach.solve(task, timeout=500)


def test_option_plan_to_policy():
    """Tests for option_plan_to_policy().
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})
    def _simulator(s, a):
        ns = s.copy()
        assert a.arr.shape == (1,)
        ns[cup][0] += a.arr.item()
        return ns
    params_space = Box(0, 1, (1,))
    def _policy(_1, _2, p):
        return Action(p)
    def _initiable(_1, _2, p):
        return p > 0.25
    def _terminal(s, _1, _2):
        return s[cup][0] > 9.9
    parameterized_option = ParameterizedOption(
        "Move", [], params_space, _policy, _initiable, _terminal)
    params = [0.1]
    option = parameterized_option.ground([], params)
    plan = [option]
    policy = option_plan_to_policy(plan)
    with pytest.raises(AssertionError):
        # option is not initiable from start state
        policy(state)
    params = [0.5]
    option = parameterized_option.ground([], params)
    plan = [option]
    policy = option_plan_to_policy(plan)
    expected_states, expected_actions = utils.option_to_trajectory(
        state, _simulator, option, max_num_steps=100)
    assert len(expected_actions) == len(expected_states)-1 == 19
    for t in range(19):
        assert not option.terminal(state)
        assert state.allclose(expected_states[t])
        action = policy(state)
        assert np.allclose(action.arr, expected_actions[t].arr)
        state = _simulator(state, action)
    assert option.terminal(state)
    with pytest.raises(ApproachFailure):
        # Ran out of options
        policy(state)
