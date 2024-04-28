"""Tests for VLM predicate invention environments."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.vlm_envs import IceTeaMakingEnv
from predicators.structs import Action, DefaultState, Object, State


def test_ice_tea_making():
    """Tests for the Iced Tea Making environment."""
    utils.reset_config({"num_train_tasks": 5, "num_test_tasks": 5})
    env = IceTeaMakingEnv()
    assert env.get_name() == "ice_tea_making"
    assert len(env.types) == 7
    assert len(env.predicates) == 1
    assert len(env.goal_predicates) == 1
    assert len(env.get_train_tasks()) == 5
    assert len(env.get_test_tasks()) == 5
    assert env.action_space.shape == (0, )
    with pytest.raises(ValueError):
        env.simulate(DefaultState, Action(np.zeros(0)))
    with pytest.raises(ValueError):
        env.render_state_plt(DefaultState, None, Action(np.zeros(0)))
    t_list = sorted(list(env.types))
    goal_type_list = [t for t in t_list if t.name == 'goal_object']
    goal_type = goal_type_list[0]
    goal_obj = Object("goal", goal_type)
    init_state = State({goal_obj: np.array([0.0])})
    goal_preds_list = list(env.goal_predicates)
    goal_pred = goal_preds_list[0]
    assert not goal_pred.holds(init_state, [goal_obj])
