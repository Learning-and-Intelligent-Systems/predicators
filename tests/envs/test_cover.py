"""Test cases for the cover environment."""

import numpy as np
from gym.spaces import Box
from predicators.src.envs import CoverEnv, CoverEnvTypedOptions, \
    CoverMultistepOptions, CoverMultistepOptionsFixedTasks, \
    CoverEnvRegrasp
from predicators.src.structs import State, Action, Task
from predicators.src import utils


def test_cover():
    """Tests for CoverEnv class."""
    utils.reset_config({"env": "cover", "cover_initial_holding_prob": 0.0})
    env = CoverEnv()
    env.seed(123)
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
    assert len(env.options) == 1
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 1-dimensional.
    assert env.action_space == Box(0, 1, (1, ))
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
    option_sequence = [
        option.ground([], [state[block0][3]]),
        option.ground([], [state[target0][3]]),
        option.ground([], [state[block1][3]]),
        option.ground([], [state[target1][3]])
    ]
    plan = []
    state = task.init
    env.render(state, task)
    expected_lengths = [5, 5, 6, 6, 7]
    expected_hands = [
        state[block0][3], state[target0][3], state[block1][3],
        state[target1][3]
    ]
    for option in option_sequence:
        atoms = utils.abstract(state, env.predicates)
        assert not task.goal.issubset(atoms)
        assert len(atoms) == expected_lengths.pop(0)
        traj = utils.option_to_trajectory(state,
                                          env.simulate,
                                          option,
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
    # Test being outside of a hand region. Should be a no-op.
    option = next(iter(env.options))
    traj = utils.option_to_trajectory(task.init,
                                      env.simulate,
                                      option.ground([], [0]),
                                      max_num_steps=100)
    assert len(traj.states) == 2
    assert traj.states[0].allclose(traj.states[1])
    # Test cover_initial_holding_prob.
    utils.update_config({"cover_initial_holding_prob": 1.0})
    env = CoverEnv()
    env.seed(123)
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


def test_cover_typed_options():
    """Tests for CoverEnvTypedOptions class."""
    utils.reset_config({"env": "cover", "cover_initial_holding_prob": 0.0})
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
    assert env.action_space == Box(0, 1, (1, ))
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
    option_sequence = [
        pick_option.ground([block0], [0.0]),
        place_option.ground([target0], [state[target0][3]]),
        pick_option.ground([block1], [0.0]),
        place_option.ground([target1], [state[target1][3]])
    ]
    plan = []
    state = task.init
    env.render(state, task)
    expected_lengths = [5, 5, 6, 6, 7]
    expected_hands = [
        state[block0][3], state[target0][3], state[block1][3],
        state[target1][3]
    ]
    for option in option_sequence:
        atoms = utils.abstract(state, env.predicates)
        assert not task.goal.issubset(atoms)
        assert len(atoms) == expected_lengths.pop(0)
        traj = utils.option_to_trajectory(state,
                                          env.simulate,
                                          option,
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
    # Test being outside of a hand region. Should be a no-op.
    option = next(iter(env.options))
    traj = utils.option_to_trajectory(task.init,
                                      env.simulate,
                                      place_option.ground([target0], [0]),
                                      max_num_steps=100)
    assert len(traj.states) == 2
    assert traj.states[0].allclose(traj.states[1])


def test_cover_regrasp():
    """Tests for CoverEnvRegrasp class."""
    utils.reset_config({"env": "cover_regrasp"})
    env = CoverEnvRegrasp()
    env.seed(123)
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    # Predicates should be same as CoverEnv, plus Clear.
    assert len(env.predicates) == 6
    # Options should be {PickPlace}.
    assert len(env.options) == 1
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
    block0, _, _, target0, _ = sorted(state)
    state.set(block0, "pose", state.get(target0, "pose"))
    assert not Clear([target0]).holds(state)


def test_cover_multistep_options():
    """Tests for CoverMultistepOptions."""
    utils.reset_config({
        "env": "cover_multistep_options",
        "num_train_tasks": 10,
        "num_test_tasks": 10
    })
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
    # Options should be {Pick, Place, LearnedEquivalentPick,
    #                    LearnedEquivalentPlace}.
    assert len(env.options) == 4
    # Types should be {block, target, robot}
    assert len(env.types) == 3
    # Action space should be 3-dimensional.
    assert len(env.action_space.low) == 3
    # Run through a specific plan of low-level actions.
    task = env.get_test_tasks()[0]
    state = task.init
    goal = task.goal
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
    state.data[block0_hr] = np.array([0.43592563 - 0.1/2, 0.43592563 + 0.1/2])
    state.data[block1_hr] = np.array([0.8334956 - 0.07/2, 0.8334956 + 0.07/2])
    state.data[target0_hr] = np.array([0.17778981 - 0.05/2, 0.17778981 + 0.05/2])
    state.data[target1_hr] = np.array([0.63629464 - 0.03/2, 0.63629464 + 0.03/2])
    task = Task(state, goal)
    env.render(state, task)
    break
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
    make_video = True  # Can toggle to true for debugging

    def policy(s: State) -> Action:
        del s  # unused
        return Action(action_arrs.pop(0))

    traj, video, _ = utils.run_policy_on_task(
        policy, task, env.simulate, len(action_arrs),
        env.render if make_video else None)
    if make_video:
        outfile = "hardcoded_actions_com.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    state = traj.states[0]
    env.render(state, task)
    # Render a state where we're grasping
    env.render(traj.states[20], task)
    Covers = [p for p in env.predicates if p.name == "Covers"][0]
    init_atoms = utils.abstract(state, env.predicates)
    final_atoms = utils.abstract(traj.states[-1], env.predicates)
    assert Covers([block0, target0]) not in init_atoms
    assert Covers([block0, target0]) in final_atoms

    # Run through a specific plan of options.
    pick_option = [o for o in env.options if o.name == "Pick"][0]
    place_option = [o for o in env.options if o.name == "Place"][0]
    plan = [
        pick_option.ground([block1], [0.0]),
        place_option.ground([target1], [0.0]),
        pick_option.ground([block0], [0.0]),
        place_option.ground([target0], [0.0]),
    ]
    assert plan[0].initiable(state)
    make_video = False  # Can toggle to true for debugging
    traj, video, _ = utils.run_policy_on_task(
        utils.option_plan_to_policy(plan), task, env.simulate, 100,
        env.render if make_video else None)
    if make_video:
        outfile = "hardcoded_options_com.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    final_atoms = utils.abstract(traj.states[-1], env.predicates)
    assert Covers([block0, target0]) in final_atoms
    assert Covers([block1, target1]) in final_atoms
    # Test moving into a forbidden zone
    state = task.init
    for _ in range(10):
        act = Action(np.array([0., -0.05, 0], dtype=np.float32))
        state = env.simulate(state, act)

    # Check collision of the robot with a block.
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
    make_video = False  # Can toggle to true for debugging
    traj, video, _ = utils.run_policy_on_task(
        policy, task, env.simulate, len(action_arrs),
        env.render if make_video else None)
    if make_video:
        outfile = "hardcoded_actions_robot_collision1.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert np.array_equal(traj.states[-1][robot], traj.states[-2][robot])

    # Check collision of the robot with the floor.
    action_arrs = [
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0.05, 0., 0.], dtype=np.float32),
        np.array([0., -0.1, 0], dtype=np.float32),
        np.array([0., -0.1, 0], dtype=np.float32),
        np.array([0., -0.1, 0], dtype=np.float32),
        np.array([0., -0.05, 0], dtype=np.float32),
        np.array([0., -0.1, 0], dtype=np.float32),
    ]
    make_video = False  # Can toggle to true for debugging
    traj, video, _ = utils.run_policy_on_task(
        policy, task, env.simulate, len(action_arrs),
        env.render if make_video else None)
    if make_video:
        outfile = "hardcoded_actions_robot_collision2.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert np.array_equal(traj.states[-1][robot], traj.states[-2][robot])

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
    make_video = False  # Can toggle to true for debugging
    traj, video, _ = utils.run_policy_on_task(
        policy, task, env.simulate, len(action_arrs),
        env.render if make_video else None)
    if make_video:
        outfile = "hardcoded_actions_block_collision1.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert np.array_equal(traj.states[-1][robot], traj.states[-2][robot])

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
    make_video = False  # Can toggle to true for debugging
    traj, video, _ = utils.run_policy_on_task(
        policy, task, env.simulate, len(action_arrs),
        env.render if make_video else None)
    if make_video:
        outfile = "hardcoded_actions_block_collision2.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert np.array_equal(traj.states[-1][robot], traj.states[-2][robot])

    # Check collision of held block with the floor.
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
    make_video = False  # Can toggle to true for debugging
    traj, video, _ = utils.run_policy_on_task(
        policy, task, env.simulate, len(action_arrs),
        env.render if make_video else None)
    if make_video:
        outfile = "hardcoded_actions_block_collision3.mp4"  # pragma: no cover
        utils.save_video(outfile, video)  # pragma: no cover
    robot = [r for r in traj.states[0] if r.name == "robby"][0]
    assert np.array_equal(traj.states[-1][robot], traj.states[-2][robot])


def test_cover_multistep_options_fixed_tasks():
    """Tests for CoverMultistepOptionsFixedTasks."""
    utils.reset_config({
        "env": "cover_multistep_options_fixed_tasks",
        "num_train_tasks": 10,
        "num_test_tasks": 10
    })
    env = CoverMultistepOptionsFixedTasks()
    env.seed(123)
    # This env is mostly the same as CoverMultistepOptions(), so we just test
    # that the tasks are indeed fixed.
    state = None
    all_goals = set()
    for task in env.get_train_tasks():
        if state is None:
            state = task.init
        assert state.allclose(task.init)
        all_goals.add(frozenset(task.goal))
    assert len(all_goals) == 3
    for task in env.get_test_tasks():
        assert state.allclose(task.init)
        assert frozenset(task.goal) in all_goals
