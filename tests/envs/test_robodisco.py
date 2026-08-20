"""Tests for the RoboDisco gymnasium wrapper and env registration."""
# pylint: disable=redefined-outer-name

import gymnasium
import numpy as np
import pytest

from predicators import utils
from predicators.envs.gymnasium_wrapper import RoboDiscoEnv, get_all_env_ids, \
    make, register_all_environments
from predicators.structs import State


@pytest.fixture(scope="module")
def robodisco_env():
    """Create a single robodisco/Blocks-v0 env shared across the module."""
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})
    env = make("robodisco/Blocks-v0")
    yield env
    env.close()


@pytest.fixture(scope="module")
def rgb_env():
    """A robodisco/Blocks-v0 env with rgb_array rendering enabled."""
    utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})
    env = make("robodisco/Blocks-v0", render_mode="rgb_array")
    yield env
    env.close()


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_register_all_environments_count():
    """register_all_environments() registers all 15 envs."""
    register_all_environments()
    rd_ids = {
        eid
        for eid in gymnasium.registry if eid.startswith("robodisco/")
    }
    assert len(rd_ids) == 15


def test_get_all_env_ids_returns_15():
    """get_all_env_ids() returns exactly 15 ids."""
    assert len(get_all_env_ids()) == 15


def test_get_all_env_ids_prefix():
    """Every id returned by get_all_env_ids() starts with 'robodisco/'."""
    for eid in get_all_env_ids():
        assert eid.startswith("robodisco/"), \
            f"{eid} does not start with 'robodisco/'"


def test_register_is_idempotent():
    """register_all_environments() is safe to call multiple times."""
    register_all_environments()
    register_all_environments()
    assert len(get_all_env_ids()) == 15


# ---------------------------------------------------------------------------
# Environment creation
# ---------------------------------------------------------------------------


def test_make_creates_env(robodisco_env):
    """make('robodisco/Blocks-v0') creates a working gymnasium env."""
    assert robodisco_env is not None
    assert isinstance(robodisco_env.unwrapped, RoboDiscoEnv)


def test_observation_space(robodisco_env):
    """The env has a Box observation space with finite-shaped float32."""
    obs_space = robodisco_env.observation_space
    assert isinstance(obs_space, gymnasium.spaces.Box)
    assert len(obs_space.shape) == 1
    assert obs_space.shape[0] > 0
    assert obs_space.dtype == np.float32


def test_action_space(robodisco_env):
    """The env has a Box action space."""
    act_space = robodisco_env.action_space
    assert isinstance(act_space, gymnasium.spaces.Box)
    assert len(act_space.shape) >= 1
    assert act_space.shape[0] > 0


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_returns_tuple(robodisco_env):
    """reset() returns (obs, info) with correct shapes and types."""
    obs, info = robodisco_env.reset()
    assert isinstance(obs, np.ndarray)
    assert obs.shape == robodisco_env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# step()
# ---------------------------------------------------------------------------


def test_step_returns_five_tuple(robodisco_env):
    """step() returns the standard gymnasium 5-tuple."""
    robodisco_env.reset()
    action = robodisco_env.action_space.sample()
    obs, reward, terminated, truncated, info = robodisco_env.step(action)

    assert isinstance(obs, np.ndarray)
    assert obs.shape == robodisco_env.observation_space.shape
    assert obs.dtype == np.float32
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


# ---------------------------------------------------------------------------
# info dict
# ---------------------------------------------------------------------------


def test_reset_info_contains_state_and_goal_reached(robodisco_env):
    """info from reset() contains 'state' and 'goal_reached' keys."""
    _, info = robodisco_env.reset()
    assert "state" in info
    assert isinstance(info["state"], State)
    assert "goal_reached" in info
    assert isinstance(info["goal_reached"], bool)


def test_step_info_contains_state_and_goal_reached(robodisco_env):
    """info from step() also contains 'state' and 'goal_reached' keys."""
    robodisco_env.reset()
    action = robodisco_env.action_space.sample()
    _, _, _, _, info = robodisco_env.step(action)
    assert "state" in info
    assert isinstance(info["state"], State)
    assert "goal_reached" in info
    assert isinstance(info["goal_reached"], bool)


# ---------------------------------------------------------------------------
# render()
# ---------------------------------------------------------------------------


def test_render_returns_rgb_frame(rgb_env):
    """render() in rgb_array mode returns an HxWx3 uint8 ndarray."""
    rgb_env.reset()
    frame = rgb_env.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    assert frame.dtype == np.uint8
