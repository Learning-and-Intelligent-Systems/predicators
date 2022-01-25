"""Test cases for the base environment class."""

import pytest
from predicators.src.envs import BaseEnv, create_env, EnvironmentFailure, \
    get_cached_env_instance
from predicators.src.structs import Type
from predicators.src import utils


def test_create_env():
    """Tests for create_env() and get_cached_env_instance()."""
    utils.update_config({"seed": 123})
    for name in [
            "cover",
            "cover_typed_options",
            "cover_hierarchical_types",
            "cluttered_table",
            "blocks",
            "playroom",
            "painting",
            "repeated_nextto",
            "cover_multistep_options",
            "cover_multistep_options_fixed_tasks",
    ]:
        env = create_env(name)
        assert isinstance(env, BaseEnv)
        other_env = get_cached_env_instance(name)
        assert env is other_env
    with pytest.raises(NotImplementedError):
        create_env("Not a real env")


def test_env_failure():
    """Tests for EnvironmentFailure class."""
    cup_type = Type("cup_type", ["feat1"])
    cup = cup_type("cup")
    try:
        raise EnvironmentFailure("failure123", {cup})
    except EnvironmentFailure as e:
        assert str(e) == "EnvironmentFailure('failure123'): {cup:cup_type}"
        assert e.offending_objects == {cup}
