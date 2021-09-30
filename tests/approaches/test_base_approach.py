"""Test cases for the base approach class.
"""

from typing import Callable
import pytest
from gym.spaces import Box  # type: ignore
import numpy as np
from numpy.typing import NDArray
from predicators.src.approaches import BaseApproach, create_approach
from predicators.src.envs import CoverEnv
from predicators.src.structs import State, Type, ParameterizedOption, \
    Predicate, Task

Array = NDArray[np.float32]


class _DummyApproach(BaseApproach):
    """Dummy approach for testing.
    """
    def _solve(self, task: Task, timeout: int) -> Callable[[State], Array]:
        # Just return some option's policy, ground with random parameters.
        parameterized_option = next(iter(self._initial_options))
        params = parameterized_option.params_space.sample()
        option = parameterized_option.ground(params)
        return option.policy


def test_base_approach():
    """Tests for BaseApproach class.
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    pred1 = Predicate("On", [cup_type, plate_type], _classifier=None)
    pred2 = Predicate("Is", [cup_type, plate_type, plate_type],
                      _classifier=None)
    cup = cup_type("cup")
    plate1 = plate_type("plate1")
    plate2 = plate_type("plate2")
    state = State({cup: [0.5], plate1: [1.0, 1.2], plate2: [-9.0, 1.0]})
    def _simulator(s, a):
        ns = s.copy()
        assert a.shape == (1,)
        ns[cup][0] += a.item()
        return ns
    action_space = Box(0, 0.5, (1,))
    params_space = Box(0, 1, (1,))
    def _policy(_, p):
        return np.clip(p, a_min=None, a_max=0.45)  # ensure in action_space
    predicates = {pred1, pred2}
    options = {ParameterizedOption(
        "Move", params_space, _policy, _initiable=None, _terminal=None)}
    approach = BaseApproach(_simulator, predicates, options, action_space)
    approach.seed(123)
    goal = {pred1([cup, plate1])}
    task = Task(state, goal)
    # Check that methods are abstract.
    with pytest.raises(NotImplementedError):
        approach.solve(task, 500)
    approach = _DummyApproach(_simulator, predicates, options, action_space)
    # Try solving with dummy approach.
    policy = approach.solve(task, 500)
    for _ in range(10):
        act = policy(state)
        assert action_space.contains(act)
        state = _simulator(state, act)


def test_create_approach():
    """Tests for create_approach.
    """
    env = CoverEnv()
    for name in ["Random Actions", "Oracle"]:
        approach = create_approach(name, env.simulate,
            env.predicates, env.options, env.action_space)
        assert isinstance(approach, BaseApproach)
    with pytest.raises(NotImplementedError):
        create_approach("Not a real approach",
            env.simulate, env.predicates, env.options, env.action_space)
