"""Test cases for the cover environment."""
import matplotlib.pyplot as plt
import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.approaches.oracle_approach import OracleApproach
from predicators.envs import create_new_env
from predicators.envs.cover import CoverEnvRegrasp, CoverEnvTypedOptions, \
    CoverMultistepOptions
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import Action, EnvironmentTask


@pytest.mark.parametrize("env_name", ["cover", "cover_handempty"])
def test_cover(env_name):
    """Tests for CoverEnv class."""
    utils.reset_config({"env": env_name, "cover_initial_holding_prob": 0.0})
    env = create_new_env(env_name)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
            assert sum(
                task.init.get(obj, "grasp") != -1 for obj in task.init
                if obj.type.name == "block") == 0
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
            assert sum(
                task.init.get(obj, "grasp") != -1 for obj in task.init
                if obj.type.name == "block") == 0
    # Predicates should be {IsBlock, IsTarget, Covers, HandEmpty, Holding}.
    assert len(env.predicates) == 5
    # Goal predicates should be {Covers}.
    assert {pred.name for pred in env.goal_predicates} == {"Covers"}
    # Options should be {PickPlace}.
    assert len(get_gt_options(env.get_name())) == 1
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 1-dimensional.
    assert env.action_space == Box(0, 1, (1, ))
    # Run through a specific plan to test atoms.
    task = env.get_test_tasks()[2]
    assert len(task.goal) == 2  # harder goal
    state = task.init
    option = next(iter(get_gt_options(env.get_name())))
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    robot = [b for b in state if b.name == "robby"][0]
    # [pick block0 center, place on target0 center,
    #  pick block1 center, place on target1 center]
    option_sequence = [
        option.ground([], [state[block0][3]]),
        option.ground([], [state[target0][3]]),
        option.ground([], [state[block1][3]]),
        option.ground([], [state[target1][3]])
    ]
    plan = []
    state = task.init
    if env_name == "cover":
        env.render_state_plt(state, task)
        plt.close()
        env.reset("train", 0)
        env.render_plt()
        plt.close()
    expected_lengths = [5, 5, 6, 6, 7]
    expected_hands = [
        state[block0][3], state[target0][3], state[block1][3],
        state[target1][3]
    ]
    for option in option_sequence:
        atoms = utils.abstract(state, env.predicates)
        assert not task.goal.issubset(atoms)
        assert len(atoms) == expected_lengths.pop(0)
        assert option.initiable(state)
        traj = utils.run_policy_with_simulator(option.policy,
                                               env.simulate,
                                               state,
                                               option.terminal,
                                               max_num_steps=100)
        plan.extend(traj.actions)
        assert len(traj.actions) == 1
        assert len(traj.states) == 2
        state = traj.states[1]
        assert abs(state[robot][0] - expected_hands.pop(0)) < 1e-4
    assert not expected_hands
    atoms = utils.abstract(state, env.predicates)
    assert len(atoms) == expected_lengths.pop(0)
    assert not expected_lengths
    assert task.goal.issubset(atoms)  # goal achieved
    # Test being outside of a hand region. Should be a noop.
    option = next(iter(get_gt_options(env.get_name()))).ground([], [0])
    assert option.initiable(task.init)
    traj = utils.run_policy_with_simulator(option.policy,
                                           env.simulate,
                                           task.init,
                                           option.terminal,
                                           max_num_steps=100)
    assert len(traj.states) == 2
    assert traj.states[0].allclose(traj.states[1])
    # Test cover_initial_holding_prob.
    utils.update_config({"cover_initial_holding_prob": 1.0})
    env = create_new_env(env_name)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
            assert sum(
                task.init.get(obj, "grasp") != -1 for obj in task.init
                if obj.type.name == "block") == 1
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
            assert sum(
                task.init.get(obj, "grasp") != -1 for obj in task.init
                if obj.type.name == "block") == 1
    with pytest.raises(NotImplementedError) as e:
        env.get_event_to_action_fn()
    assert "did not implement an interface for human demonstrations" in str(e)
    with pytest.raises(NotImplementedError) as e:
        env._get_language_goal_prompt_prefix(set())  # pylint:disable=protected-access
    assert "did not implement an interface for language-based goals" in str(e)


def test_cover_typed_options():
    """Tests for CoverEnvTypedOptions class."""
    utils.reset_config({
        "env": "cover_typed_options",
        "cover_initial_holding_prob": 0.0
    })
    env = CoverEnvTypedOptions()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {IsBlock, IsTarget, Covers, HandEmpty, Holding}.
    assert len(env.predicates) == 5
    # Options should be {Pick, Place}.
    assert len(get_gt_options(env.get_name())) == 2
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 1-dimensional.
    assert env.action_space == Box(0, 1, (1, ))
    # Run through a specific plan to test atoms.
    task = env.get_test_tasks()[2]
    assert len(task.goal) == 2  # harder goal
    state = task.init
    pick_option = [
        o for o in get_gt_options(env.get_name()) if o.name == "Pick"
    ][0]
    place_option = [
        o for o in get_gt_options(env.get_name()) if o.name == "Place"
    ][0]
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    robot = [b for b in state if b.name == "robby"][0]
    # [pick relative position 0, place on target0 center,
    #  pick relative position 0, place on target1 center]
    option_sequence = [
        pick_option.ground([block0], [0.0]),
        place_option.ground([target0], [state[target0][3]]),
        pick_option.ground([block1], [0.0]),
        place_option.ground([target1], [state[target1][3]])
    ]
    plan = []
    state = task.init
    env.render_state(state, task, caption="caption")
    expected_lengths = [5, 5, 6, 6, 7]
    expected_hands = [
        state[block0][3], state[target0][3], state[block1][3],
        state[target1][3]
    ]
    for option in option_sequence:
        atoms = utils.abstract(state, env.predicates)
        assert not task.goal.issubset(atoms)
        assert len(atoms) == expected_lengths.pop(0)
        assert option.initiable(state)
        traj = utils.run_policy_with_simulator(option.policy,
                                               env.simulate,
                                               state,
                                               option.terminal,
                                               max_num_steps=100)
        plan.extend(traj.actions)
        assert len(traj.actions) == 1
        assert len(traj.states) == 2
        state = traj.states[1]
        assert abs(state[robot][0] - expected_hands.pop(0)) < 1e-4
    assert not expected_hands
    atoms = utils.abstract(state, env.predicates)
    assert len(atoms) == expected_lengths.pop(0)
    assert not expected_lengths
    assert task.goal.issubset(atoms)  # goal achieved
    # Test being outside of a hand region. Should be a noop.
    option = place_option.ground([target0], [0])
    assert option.initiable(task.init)
    traj = utils.run_policy_with_simulator(option.policy,
                                           env.simulate,
                                           task.init,
                                           option.terminal,
                                           max_num_steps=100)
    assert len(traj.states) == 2
    assert traj.states[0].allclose(traj.states[1])


def test_cover_regrasp():
    """Tests for CoverEnvRegrasp class."""
    utils.reset_config({"env": "cover_regrasp"})
    env = CoverEnvRegrasp()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be same as CoverEnv, plus Clear.
    assert len(env.predicates) == 6
    # Options should be {PickPlace}.
    assert len(get_gt_options(env.get_name())) == 1
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 1-dimensional.
    assert env.action_space == Box(0, 1, (1, ))
    # Tests for Clear.
    task = env.get_train_tasks()[0]
    Clear = [p for p in env.predicates if p.name == "Clear"][0]
    init_atoms = utils.abstract(task.init, {Clear})
    assert len(init_atoms) == 2
    # Clear should not be true after a place.
    state = task.init.copy()
    block0, _, _, target0, _ = list(state)
    state.set(block0, "pose", state.get(target0, "pose"))
    assert not Clear([target0]).holds(state)


def test_cover_multistep_options():
    """Tests for CoverMultistepOptions."""
    utils.reset_config({
        "env": "cover_multistep_options",
        "num_train_tasks": 10,
        "num_test_tasks": 10,
        "test_env_seed_offset": 0,
    })
    env = CoverMultistepOptions()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {IsBlock, IsTarget, Covers, HandEmpty, Holding}.
    assert len(env.predicates) == 5
    # Options should be {Pick, Place}.
    assert len(get_gt_options(env.get_name())) == 2
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 3-dimensional.
    assert len(env.action_space.low) == 3
    # Run through a specific plan of low-level actions.
    task = env.get_test_tasks()[0]
    state = task.init
    block0 = [b for b in state if b.name == "block0"][0]
    block1 = [b for b in state if b.name == "block1"][0]
    target0 = [b for b in state if b.name == "target0"][0]
    target1 = [b for b in state if b.name == "target1"][0]
    block0_hr = [b for b in state if b.name == "block0_hand_region"][0]
    block1_hr = [b for b in state if b.name == "block1_hand_region"][0]
    target0_hr = [b for b in state if b.name == "target0_hand_region"][0]
    target1_hr = [b for b in state if b.name == "target1_hand_region"][0]
    state.data[block0] = np.array([1., 0., 0.1, 0.43592563, -1., 0.1, 0.1])
    state.data[block1] = np.array([1., 0., 0.07, 0.8334956, -1., 0.1, 0.1])
    state.data[target0] = np.array([0., 1., 0.05, 0.17778981])
    state.data[target1] = np.array([0., 1., 0.03, 0.63629464])
    state.data[block0_hr] = np.array([-0.1 / 2, 0.1 / 2, 0])
    state.data[block1_hr] = np.array([-0.07 / 2, 0.07 / 2, 1])
    state.data[target0_hr] = np.array(
        [0.17778981 - 0.05 / 2, 0.17778981 + 0.05 / 2])
    state.data[target1_hr] = np.array(
        [0.63629464 - 0.03 / 2, 0.63629464 + 0.03 / 2])
    task = EnvironmentTask(state, task.goal)
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
        # Move down to grasp
        np.array([0., -0.05, 0], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.045, 0.1], dtype=np.float32),
        # Move up to a safe height
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        # Move to above target
        np.array([-0.05, 0., 0.1], dtype=np.float32),
        np.array([-0.05, 0., 0.1], dtype=np.float32),
        np.array([+0.05, 0., 0.1], dtype=np.float32),
        np.array([+0.05, 0., 0.1], dtype=np.float32),
        np.array([-0.05, 0., 0.1], dtype=np.float32),
        np.array([-0.05, 0., 0.1], dtype=np.float32),
        np.array([-0.05, 0., 0.1], dtype=np.float32),
        np.array([-0.05, 0., 0.1], dtype=np.float32),
        np.array([-0.05, 0., 0.1], dtype=np.float32),
        np.array([-0.025, 0., 0.1], dtype=np.float32),
        # Move down to prepare to place
        np.array([0., -0.05, 0.1], dtype=np.float32),
        np.array([0., -0.05, 0.1], dtype=np.float32),
        np.array([0., -0.05, 0.1], dtype=np.float32),
        np.array([0., -0.05, 0.1], dtype=np.float32),
        np.array([0., -0.05, 0.1], dtype=np.float32),
        np.array([0., -0.04, 0.1], dtype=np.float32),
        # Ungrasp
        np.array([0., 0., -0.1], dtype=np.float32),
    ]

    policy = utils.action_arrs_to_policy(action_arrs)

    # Here's an example of how to make a video within this test.
    # monitor = utils.SimulateVideoMonitor(task, env.render_state)
    # traj = utils.run_policy_with_simulator(policy,
    #                                        env.simulate,
    #                                        task.init,
    #                                        lambda _: False,
    #                                        max_num_steps=len(action_arrs),
    #                                        monitor=monitor)
    # video = monitor.get_video()
    # outfile = "hardcoded_actions_com.mp4"
    # utils.save_video(outfile, video)

    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    state = traj.states[0]
    env.render_state(state, task, caption="caption")
    # Render a state where we're grasping
    env.render_state(traj.states[20], task)
    Covers = [p for p in env.predicates if p.name == "Covers"][0]
    init_atoms = utils.abstract(state, env.predicates)
    final_atoms = utils.abstract(traj.states[-1], env.predicates)
    assert Covers([block0, target0]) not in init_atoms
    assert Covers([block0, target0]) in final_atoms

    # Test moving into a forbidden zone
    state = task.init
    for _ in range(10):
        act = Action(np.array([0., -0.05, 0], dtype=np.float32))
        state = env.simulate(state, act)

    # Check collision of the robot with a block. The expected behavior is that
    # the robot should be exactly at the block's height, because it snaps.
    action_arrs = [
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0., -0.05, 0], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.06, 0.0], dtype=np.float32),
    ]

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert np.allclose(traj.states[-1].get(robot, "y"),
                       traj.states[-1].get(block0, "y"))

    # Check collision of held block with a block via overlap.
    action_arrs = [
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0., -0.05, 0], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.045, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0.1, 0., 0.1], dtype=np.float32),
        np.array([0.1, 0., 0.1], dtype=np.float32),
        np.array([0.1, 0., 0.1], dtype=np.float32),
        np.array([0.1, 0., 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
    ]

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert np.allclose(traj.states[-1][robot], traj.states[-2][robot])

    # Check collision of held block with a block via translation intersection.
    action_arrs = [
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0., -0.05, 0], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.045, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0.1, 0., 0.1], dtype=np.float32),
        np.array([0.1, 0., 0.1], dtype=np.float32),
        np.array([0.08, 0., 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.05, 0.1], dtype=np.float32),
        np.array([0.1, 0.1, 0.1], dtype=np.float32),
    ]

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert np.allclose(traj.states[-1][robot], traj.states[-2][robot])

    # Check collision of held block with the floor. The expected behavior is
    # that the block will be exactly on the floor because it snaps.
    action_arrs = [
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0., -0.05, 0], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.045, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0.1, 0., 0.1], dtype=np.float32),
        np.array([0.1, 0., 0.1], dtype=np.float32),
        np.array([0.08, 0., 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.05, 0.1], dtype=np.float32),
        np.array([0., -0.07, 0.1], dtype=np.float32),
    ]

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert np.allclose(traj.states[-1].get(block0, "y"),
                       traj.states[-1].get(block0, "height"))
    assert np.allclose(traj.states[-1].get(block1, "y"),
                       traj.states[-1].get(block1, "height"))

    # Cover the case where a place is attempted outside of a hand region.
    action_arrs = [
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0., -0.05, 0], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.045, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0.05, 0., 0.1], dtype=np.float32),
        np.array([0.05, 0., 0.1], dtype=np.float32),
        np.array([0.05, 0., 0.1], dtype=np.float32),
        np.array([0.05, 0., 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., 0., -0.1], dtype=np.float32),
        np.array([0., 0.1, 0.], dtype=np.float32),
    ]

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert traj.states[-1].get(robot, "holding") > -1

    # Check max placement failure for target and block placement
    utils.reset_config({
        "env": "cover_multistep_options",
        "num_train_tasks": 10,
        "num_test_tasks": 10,
        "cover_block_widths": [0.25, 0.25],
        "cover_target_widths": [0.25, 0.25]
    })
    env = CoverMultistepOptions()
    with pytest.raises(RuntimeError):
        env.get_test_tasks()

    # Check max placement failure for hand region placement
    utils.reset_config({
        "env": "cover_multistep_options",
        "num_train_tasks": 10,
        "num_test_tasks": 10,
        "cover_multistep_max_tb_placements": 100,
        "cover_multistep_max_hr_placements": 2,
        "cover_multistep_thr_percent": 0.001,
        "cover_multistep_bhr_percent": 0.001
    })
    env = CoverMultistepOptions()
    with pytest.raises(RuntimeError):
        env.get_test_tasks()

    # Test that new _create_initial_state is working
    utils.reset_config({
        "env": "cover_multistep_options",
        "num_train_tasks": 10,
        "num_test_tasks": 10,
        "cover_multistep_thr_percent": 0.2,
        "cover_multistep_bhr_percent": 0.2,
        "test_env_seed_offset": 0
    })
    env = CoverMultistepOptions()
    task = env.get_test_tasks()[0]
    action_arrs = [
        np.array([0.88, 0., 0.], dtype=np.float32),
        np.array([0., -0.05, 0], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.05, 0.], dtype=np.float32),
        np.array([0., -0.045, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([0., 0.05, 0.1], dtype=np.float32),
        np.array([-0.14, 0., 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.1, 0.1], dtype=np.float32),
        np.array([0., -0.01, -0.1], dtype=np.float32),
    ]

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    Covers = [p for p in env.predicates if p.name == "Covers"][0]
    init_atoms = utils.abstract(state, env.predicates)
    final_atoms = utils.abstract(traj.states[-1], env.predicates)
    assert Covers([block0, target0]) not in init_atoms
    assert Covers([block0, target0]) in final_atoms

    # Test bimodal goal flag.
    utils.reset_config({
        "cover_multistep_bimodal_goal": True,
        "cover_num_blocks": 1,
        "cover_num_targets": 1
    })
    env = CoverMultistepOptions()
    task = env.get_test_tasks()[0]
    state = task.init
    goal = task.goal
    assert len(goal) == 1
    goal_atom = next(iter(goal))
    t = goal_atom.objects[1]
    tx, tw = state.get(t, "x"), state.get(t, "width")
    thr_found = False  # target hand region
    # Loop over objects in state to find target hand region,
    # whose center should overlap with the target.
    for obj in state.data:
        if obj.type.name == "target_hand_region":
            lb = state.get(obj, "lb")
            ub = state.get(obj, "ub")
            m = (lb + ub) / 2  # midpoint of hand region
            if tx - tw / 2 < m < tx + tw / 2:
                thr_found = True
                break
    assert thr_found
    # Assert off-center hand region
    assert abs(m - tx) > tw / 5


def test_regional_bumpy_cover_env():
    """Test coverage for RegionalBumpyCoverEnv."""
    env_name = "regional_bumpy_cover"
    utils.reset_config({
        "env": env_name,
        "bumpy_cover_init_bumpy_prob": 1.0,
        "bumpy_cover_bumpy_region_start": 0.5,
    })
    env = create_new_env(env_name)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be {IsBlock, IsTarget, Covers, HandEmpty, Holding,
    # Clear, InBumpyRegion, InSmoothRegion}.
    assert len(env.predicates) == 8
    # Goal predicates should be {Covers}.
    assert {pred.name for pred in env.goal_predicates} == {"Covers"}
    # Options should be {PickFromBumpy, PickFromSmooth, PlaceOnTarget,
    # PlaceOnBumpy}.
    options = get_gt_options(env.get_name())
    assert len(options) == 5
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 1-dimensional.
    assert env.action_space == Box(0, 1, (1, ))
    # Cover rendering.
    task = env.get_train_tasks()[0].task
    state = task.init
    env.render_state(state, task, caption="caption")
    # Create a state where a block is held.
    block_type = [t for t in env.types if t.name == "block"][0]
    block = [
        b for b in state.get_objects(block_type) if state.get(b, "bumpy") < 0.5
    ][0]
    block_pose = state.get(block, "pose")
    action = Action(np.array([block_pose], dtype=np.float32))
    held_state = env.simulate(state, action)
    # InSmoothRegion and InBumpyRegion should be false.
    pred_name_to_pred = {p.name: p for p in env.predicates}
    Holding = pred_name_to_pred["Holding"]
    InSmoothRegion = pred_name_to_pred["InSmoothRegion"]
    InBumpyRegion = pred_name_to_pred["InBumpyRegion"]
    assert Holding.holds(held_state, [block])
    assert not InSmoothRegion.holds(held_state, [block])
    assert not InBumpyRegion.holds(held_state, [block])
    # Test PlaceOnBumpy NSRT because it's never used in oracle planning.
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    PlaceOnBumpy = [n for n in nsrts if n.name == "PlaceOnBumpy"][0]
    ground_nsrt = PlaceOnBumpy.ground([block])
    rng = np.random.default_rng(123)
    option = ground_nsrt.sample_option(held_state, set(), rng)
    assert option.params[0] > 0.5

    # Now test making the blocks really big so that the sampler can't
    # find a good sample.
    held_block, other_block = state.get_objects(block_type)
    held_state.set(other_block, "pose", 0.8)
    held_state.set(other_block, "width", 0.2)
    held_state.set(held_block, "pose", 0.8)
    held_state.set(held_block, "width", 0.2)
    PlaceOnBumpy = [n for n in nsrts if n.name == "PlaceOnBumpy"][0]
    ground_nsrt = PlaceOnBumpy.ground([block])
    rng = np.random.default_rng(123)
    option = ground_nsrt.sample_option(held_state, set(), rng)
    assert option.params[0] > 0.5

    # Test that when the impossible NSRT is included, the planner tries to use
    # it, but then fails with an option execution error.
    utils.reset_config({
        "env": env_name,
        "regional_bumpy_cover_include_impossible_nsrt": True,
        "approach": "oracle",
        "bilevel_plan_without_sim": True,
        "num_train_tasks": 0,
        "num_test_tasks": 5,
    })
    env = create_new_env(env_name)
    options = get_gt_options(env.get_name())
    assert len(options) == 6
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 6
    test_tasks = [t.task for t in env.get_test_tasks()]
    approach = OracleApproach(env.predicates,
                              options,
                              env.types,
                              env.action_space,
                              train_tasks=[])
    for task in test_tasks:
        policy = approach.solve(task, 500)
        # Expected no-op.
        state = task.init.copy()
        act = policy(state)
        next_state = env.simulate(state, act)
        assert state.allclose(next_state)


def test_debug_atoms_vlm_str():
    """Tests the get_vlm_debug_atom_strs method."""
    utils.reset_config({
        "env": "cover",
        "excluded_predicates": "all",
        "num_train_tasks": 1
    })
    env = create_new_env("cover")
    tasks = env.get_train_tasks()
    debug_atoms_strs = env.get_vlm_debug_atom_strs(tasks)
    gt_debug_strs = [
        'Holding(block0)', 'IsBlock(block1)', 'IsTarget(target0)',
        'Holding(block1)', 'HandEmpty()', 'IsTarget(target1)',
        'IsBlock(block0)'
    ]
    gt_debug_strs = [[a] for a in gt_debug_strs]
    for debug_str in gt_debug_strs:
        assert debug_str in debug_atoms_strs
