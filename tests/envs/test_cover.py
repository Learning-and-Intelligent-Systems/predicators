"""Test cases for the cover environment.
"""

import numpy as np
from gym.spaces import Box
from predicators.src.envs import CoverEnv
from predicators.src import utils


def test_cover():
    """Tests for CoverEnv class.
    """
    env = CoverEnv()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {IsBlock, IsTarget, Covers, HandEmpty, Holding}.
    assert len(env.predicates) == 5
    # Options should be {PickPlace}.
    assert len(env.options) == 1
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 1-dimensional.
    assert env.action_space == Box(0, 1, (1,))
    # Run through a specific plan to test atoms.
    task = env.get_test_tasks()[2]
    assert len(task.goal) == 2  # harder goal
    state = task.init
    option = next(iter(env.options))
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    robot = [b for b in state if b.name == "robby"][0]
    # [pick block0 center, place on target0 center,
    #  pick block1 center, place on target1 center]
    option_sequence = [option.ground([state[block0][3]]),
                       option.ground([state[target0][3]]),
                       option.ground([state[block1][3]]),
                       option.ground([state[target1][3]])]
    plan = []
    state = task.init
    expected_lengths = [5, 5, 6, 6, 7]
    expected_hands = [state[block0][3], state[target0][3],
                      state[block1][3], state[target1][3]]
    for option in option_sequence:
        atoms = utils.abstract(state, env.predicates)
        assert not task.goal.issubset(atoms)
        assert len(atoms) == expected_lengths.pop(0)
        states, actions = utils.option_to_trajectory(
            state, env.simulate, option, max_num_steps=100)
        plan.extend(actions)
        assert len(actions) == 1
        assert len(states) == 2
        state = states[1]
        assert abs(state[robot][0]-expected_hands.pop(0)) < 1e-4
    assert not expected_hands
    atoms = utils.abstract(state, env.predicates)
    assert len(atoms) == expected_lengths.pop(0)
    assert not expected_lengths
    assert task.goal.issubset(atoms)  # goal achieved
    # Test being outside of a hand region. Should be a no-op.
    option = next(iter(env.options))
    states, _ = utils.option_to_trajectory(
        task.init, env.simulate, option.ground([0]),
        max_num_steps=100)
    assert all(np.all(states[0][obj] == states[1][obj]) for obj in states[0])
