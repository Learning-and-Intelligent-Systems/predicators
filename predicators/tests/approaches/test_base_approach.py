"""Test cases for the base approach class."""

from typing import Callable

import numpy as np
import pytest
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches import BaseApproach, create_approach
from predicators.src.envs.cover import CoverEnv
from predicators.src.structs import Action, ParameterizedOption, Predicate, \
    State, Task, Type


class _DummyApproach(BaseApproach):
    """Dummy approach for testing."""

    @classmethod
    def get_name(cls) -> str:
        return "dummy"

    @property
    def is_learning_based(self):
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        # Just return some option's policy, ground with random parameters.
        parameterized_option = next(iter(self._initial_options))
        params = parameterized_option.params_space.sample()
        option = parameterized_option.ground([], params)
        return option.policy


def test_base_approach():
    """Tests for BaseApproach class."""
    utils.reset_config()
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
        assert a.arr.shape == (1, )
        ns[cup][0] += a.arr.item()
        return ns

    action_space = Box(0, 0.5, (1, ))
    params_space = Box(0, 1, (1, ))

    def policy(_1, _2, _3, p):
        return Action(np.clip(p, a_min=None, a_max=0.45))

    predicates = {pred1, pred2}
    types = {cup_type, plate_type}
    options = {
        ParameterizedOption("Move", [],
                            params_space,
                            policy,
                            initiable=None,
                            terminal=None)
    }
    goal = {pred1([cup, plate1])}
    task = Task(state, goal)
    train_tasks = [task]
    approach = _DummyApproach(predicates, options, types, action_space,
                              train_tasks)
    assert not approach.is_learning_based
    assert approach.learn_from_offline_dataset([]) is None
    # Try solving with dummy approach.
    policy = approach.solve(task, 500)
    for _ in range(10):
        act = policy(state)
        assert action_space.contains(act.arr)
        state = _simulator(state, act)


def test_create_approach():
    """Tests for create_approach."""
    env = CoverEnv()
    train_tasks = env.get_train_tasks()
    for name in [
            "random_actions",
            "random_options",
            "gnn_option_policy",
            "gnn_metacontroller",
            "oracle",
            "nsrt_learning",
            "interactive_learning",
    ]:
        utils.reset_config({
            "env": "cover",
            "approach": name,
        })
        approach = create_approach(name, env.predicates, env.options,
                                   env.types, env.action_space, train_tasks)
        assert isinstance(approach, BaseApproach)
    with pytest.raises(NotImplementedError):
        create_approach("Not a real approach", env.predicates, env.options,
                        env.types, env.action_space, train_tasks)
