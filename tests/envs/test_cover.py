"""Test cases for the cover environment.
"""

import numpy as np
from gym.spaces import Box
from predicators.src.envs import CoverEnv, CoverEnvTypedOptions, \
    CoverMultistepOptions
from predicators.src.structs import State, Action
from predicators.src import utils


def test_cover():
    """Tests for CoverEnv class.
    """
    utils.update_config({"env": "cover"})
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
    # Goal predicates should be {Covers}.
    assert {pred.name for pred in env.goal_predicates} == {"Covers"}
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
    option_sequence = [option.ground([], [state[block0][3]]),
                       option.ground([], [state[target0][3]]),
                       option.ground([], [state[block1][3]]),
                       option.ground([], [state[target1][3]])]
    plan = []
    state = task.init
    env.render(state, task)
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
        task.init, env.simulate, option.ground([], [0]),
        max_num_steps=100)
    assert len(states) == 2
    assert states[0].allclose(states[1])


def test_cover_typed_options():
    """Tests for CoverEnvTypedOptions class.
    """
    utils.update_config({"env": "cover"})
    env = CoverEnvTypedOptions()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {IsBlock, IsTarget, Covers, HandEmpty, Holding}.
    assert len(env.predicates) == 5
    # Options should be {Pick, Place}.
    assert len(env.options) == 2
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 1-dimensional.
    assert env.action_space == Box(0, 1, (1,))
    # Run through a specific plan to test atoms.
    task = env.get_test_tasks()[2]
    assert len(task.goal) == 2  # harder goal
    state = task.init
    pick_option = [o for o in env.options if o.name == "Pick"][0]
    place_option = [o for o in env.options if o.name == "Place"][0]
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    robot = [b for b in state if b.name == "robby"][0]
    # [pick relative position 0, place on target0 center,
    #  pick relative position 0, place on target1 center]
    option_sequence = [pick_option.ground([block0], [0.0]),
                       place_option.ground([target0], [state[target0][3]]),
                       pick_option.ground([block1], [0.0]),
                       place_option.ground([target1], [state[target1][3]])]
    plan = []
    state = task.init
    env.render(state, task)
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
        task.init, env.simulate, place_option.ground([target0], [0]),
        max_num_steps=100)
    assert len(states) == 2
    assert states[0].allclose(states[1])


def test_cover_multistep_options():
    """Tests for CoverMultistepOptions.
    """
    utils.update_config({"env": "cover_multistep_options"})
    env = CoverMultistepOptions()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {IsBlock, IsTarget, Covers, HandEmpty, Holding}.
    assert len(env.predicates) == 5
    # Options should be {Pick, Place}.
    assert len(env.options) == 2
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 3-dimensional.
    assert env.action_space == Box(-0.1, 0.1, (3,))
    # Run through a specific plan of low-level actions.
    task = env.get_test_tasks()[0]
    action_arrs = [
        # Move to above block0
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        # Move down to grasp and turn on magnet/vacuum
        np.array([0., -0.05, 0.05], dtype=np.float32),
        np.array([0., -0.05, 0.05], dtype=np.float32),
        np.array([0., -0.05, 0.05], dtype=np.float32),
        np.array([0., -0.05, 0.05], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.045, 0.], dtype=np.float32),
        # Move up to a safe height
        np.array([0., 0.05, 0.], dtype=np.float32),
        np.array([0., 0.05, 0.], dtype=np.float32),
        np.array([0., 0.05, 0.], dtype=np.float32),
        np.array([0., 0.05, 0.], dtype=np.float32),
        np.array([0., 0.05, 0.], dtype=np.float32),
        np.array([0., 0.05, 0.], dtype=np.float32),
        np.array([0., 0.05, 0.], dtype=np.float32),
        # Move to above target
        np.array([-0.05, 0., 0.], dtype=np.float32),
        np.array([-0.05, 0., 0.], dtype=np.float32),
        np.array([-0.05, 0., 0.], dtype=np.float32),
        np.array([-0.05, 0., 0.], dtype=np.float32),
        np.array([-0.05, 0., 0.], dtype=np.float32),
        np.array([-0.025, 0., 0.], dtype=np.float32),
        # Move down to prepare to place
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        # Ungrasp
        np.array([0., 0., -0.05], dtype=np.float32),
        np.array([0., 0., -0.05], dtype=np.float32),
        np.array([0., 0., -0.05], dtype=np.float32),
        np.array([0., 0., -0.05], dtype=np.float32),
    ]
    make_video = False  # Can toggle to true for debugging
    def policy(s: State) -> Action:
        del s  # unused
        return Action(action_arrs.pop(0))
    (states, _), video, _ = utils.run_policy_on_task(
        policy, task, env.simulate, env.predicates,
        len(action_arrs), make_video, env.render)
    if make_video:
        outfile = "hardcoded_actions_covers_options_multistep.mp4"
        utils.save_video(outfile, video)
    state = states[0]
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    block0 = [b for b in state if b.name == "block0"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    Covers = [p for p in env.predicates if p.name == "Covers"][0]
    init_atoms = utils.abstract(state, env.predicates)
    final_atoms = utils.abstract(states[-1], env.predicates)
    assert Covers([block0, target0]) not in init_atoms
    assert Covers([block0, target0]) in final_atoms
    # Run through a specific plan of options.
    pick_option = [o for o in env.options if o.name == "Pick"][0]
    place_option = [o for o in env.options if o.name == "Place"][0]
    option_sequence = [
        pick_option.ground([block1], [0.0]),
        place_option.ground([target1], [0.0]),
        pick_option.ground([block0], [0.0]),
        place_option.ground([target0], [0.0]),
    ]
    assert option_sequence[0].initiable(state)
    make_video = False  # Can toggle to true for debugging
    def option_policy(s: State) -> Action:
        current_option = option_sequence[0]
        if current_option.terminal(s):
            option_sequence.pop(0)
            assert len(option_sequence) > 0
            current_option = option_sequence[0]
            assert current_option.initiable(s)
        return current_option.policy(s)
    (states, _), video, _ = utils.run_policy_on_task(
        option_policy, task, env.simulate, env.predicates,
        100, make_video, env.render)
    if make_video:
        outfile = "hardcoded_options_covers_options_multistep.mp4"
        utils.save_video(outfile, video)
    final_atoms = utils.abstract(states[-1], env.predicates)
    assert Covers([block0, target0]) in final_atoms
    assert Covers([block1, target1]) in final_atoms
