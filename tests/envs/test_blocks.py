"""Test cases for the blocks environment.
"""

import pytest
import numpy as np
from predicators.src.envs import BlocksEnv
from predicators.src import utils
from predicators.src.structs import Action
from predicators.src.envs import EnvironmentFailure


def test_blocks():
    """Tests for BlocksEnv class.
    """
    utils.update_config({"env": "blocks"})
    env = BlocksEnv()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 5
    assert len(env.options) == 3
    assert len(env.types) == 2
    block_type = [t for t in env.types if t.name == "block"][0]
    assert env.action_space.shape == (4,)
    assert np.all(env.action_space.low <= min(BlocksEnv.x_lb, BlocksEnv.y_lb))
    assert np.all(env.action_space.high >= min(BlocksEnv.x_ub, BlocksEnv.y_ub))
    for i, task in enumerate(env.get_test_tasks()):
        state = task.init
        for block in state:
            if block.type != block_type:
                assert block.type.name == "robot"
                continue
            assert not (state.get(block, "held") and state.get(block, "clear"))
        act = Action(env.action_space.sample())
        if i == 0:
            with pytest.raises(NotImplementedError):
                env.render(state, task, act)
        env.simulate(state, act)
