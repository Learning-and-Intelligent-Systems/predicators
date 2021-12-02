"""Test cases for the base environment class.
"""

import pytest
from predicators.src.envs import BaseEnv, create_env, EnvironmentFailure
from predicators.src.structs import State, Type, Task


def test_base_env():
    """Tests for BaseEnv class.
    """
    cup_type = Type("cup_type", ["feat1"])
    plate_type = Type("plate_type", ["feat1", "feat2"])
    cup = cup_type("cup")
    plate = plate_type("plate")
    state = State({cup: [0.5], plate: [1.0, 1.2]})
    task = Task(state, set())
    env = BaseEnv()
    env.seed(123)
    # Check that methods are abstract.
    with pytest.raises(NotImplementedError):
        env.simulate(state, [1, 2])
    with pytest.raises(NotImplementedError):
        env.get_train_tasks()
    with pytest.raises(NotImplementedError):
        env.get_test_tasks()
    with pytest.raises(NotImplementedError):
        env.predicates()
    with pytest.raises(NotImplementedError):
        env.goal_predicates()
    with pytest.raises(NotImplementedError):
        env.options()
    with pytest.raises(NotImplementedError):
        env.types()
    with pytest.raises(NotImplementedError):
        env.action_space()
    with pytest.raises(NotImplementedError):
        env.render(state, task)


def test_create_env():
    """Tests for create_env.
    """
    for name in ["cover", "cover_typed_options", "cover_hierarchical_types",
                 "cluttered_table", "blocks", "playroom", "painting"]:
        env = create_env(name)
        assert isinstance(env, BaseEnv)
    with pytest.raises(NotImplementedError):
        create_env("Not a real env")


def test_env_failure():
    """Tests for EnvironmentFailure class.
    """
    cup_type = Type("cup_type", ["feat1"])
    cup = cup_type("cup")
    try:
        raise EnvironmentFailure("failure123", {cup})
    except EnvironmentFailure as e:
        assert str(e) == "EnvironmentFailure('failure123'): {cup:cup_type}"
        assert e.offending_objects == {cup}
