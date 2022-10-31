"""Test cases for the narrow_passage environment."""

import numpy as np

from predicators import utils
from predicators.envs.narrow_passage import NarrowPassageEnv
from predicators.structs import Action, GroundAtom, Task


def test_narrow_passage_properties():
    """Test env object initialization and properties."""
    utils.reset_config({"env": "narrow_passage"})
    env = NarrowPassageEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 2
    DoorIsOpen, TouchedGoal = sorted(env.predicates)
    assert DoorIsOpen.name == "DoorIsOpen"
    assert TouchedGoal.name == "TouchedGoal"
    assert env.goal_predicates == {TouchedGoal}
    assert len(env.options) == 2
    MoveAndOpenDoor, MoveToTarget = sorted(env.options)
    assert MoveAndOpenDoor.name == "MoveAndOpenDoor"
    assert MoveToTarget.name == "MoveToTarget"
    assert len(env.types) == 5
    door_type, door_sensor_type, robot_type, target_type, wall_type = sorted(
        env.types)
    assert door_type.name == "door"
    assert door_sensor_type.name == "door_sensor"
    assert robot_type.name == "robot"
    assert target_type.name == "target"
    assert wall_type.name == "wall"
    assert env.action_space.shape == (3, )


def test_narrow_passage_actions():
    """Test to check that basic actions and rendering works, especially door
    opening."""
    utils.reset_config({"env": "narrow_passage"})
    env = NarrowPassageEnv()
    DoorIsOpen, TouchedGoal = sorted(env.predicates)
    door_type, _, robot_type, target_type, _ = sorted(env.types)

    # Create task with fixed initial state
    sample_task = env.get_train_tasks()[0]
    state = sample_task.init.copy()
    goal = sample_task.goal
    door, = state.get_objects(door_type)
    robot, = state.get_objects(robot_type)
    state.set(robot, "x", 0.1)
    state.set(robot, "y", 0.8)
    target, = state.get_objects(target_type)
    state.set(target, "x", 0.5)
    state.set(target, "y", 0.2)
    task = Task(state, goal)
    # reset static geom cache so new target location has correct geom
    env._static_geom_cache = {}  # pylint: disable=protected-access

    # Fixed action sequences to test (each is a list of action arrays)
    # Move to within range of door and open it
    door_open_actions = [
        np.array([0.095, -0.05, 0]).astype(np.float32),
        np.array([0.05, -0.05, 0]).astype(np.float32),
        np.array([0, 0, 1.0]).astype(np.float32),
    ]
    # Move to goal
    move_target_actions = [
        np.array([0.09, 0, 0]).astype(np.float32),
        np.array([0.01, -0.1, 0]).astype(np.float32),
        np.array([0, -0.1, 0]).astype(np.float32),
        np.array([0.06, -0.1, 0]).astype(np.float32),
        np.array([0, -0.1, 0]).astype(np.float32),
        np.array([0.1, -0.05, 0]).astype(np.float32),
    ]
    all_action_arrs = door_open_actions + move_target_actions

    # Test that the open door action works
    s = state.copy()
    # check door is not open, not touching goal
    assert not GroundAtom(DoorIsOpen, [door]).holds(s)
    assert not GroundAtom(TouchedGoal, [robot, target]).holds(s)

    for action in door_open_actions:
        s = env.simulate(s, Action(action))
    # check door is open
    assert s.get(door, "open") == 1
    assert GroundAtom(DoorIsOpen, [door]).holds(s)

    for action in move_target_actions:
        s = env.simulate(s, Action(action))
    assert GroundAtom(TouchedGoal,
                      [robot, target]).holds(s)  # check touching goal
    assert task.goal_holds(s)  # check task goal reached

    # Test rendering entire plan
    policy = utils.action_arrs_to_policy(all_action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           task.goal_holds,
                                           max_num_steps=len(all_action_arrs))

    # Render a state before door opens
    env.render_state(traj.states[2], task)
    # Render a state after door opens
    env.render_state(traj.states[4], task, caption="caption")
    # Render state at end of trajectory
    env.render_state(traj.states[-1], task)


def test_narrow_passage_collisions():
    """Test that the robot can't go through walls or closed door."""
    # set up environment
    utils.reset_config({"env": "narrow_passage"})
    env = NarrowPassageEnv()
    door_type, _, robot_type, _, _ = sorted(env.types)

    # Test robot should not be able to walk through each wall nor the door
    sample_task = env.get_train_tasks()[0]
    test_robot_xs = [0.1, 0.35, 0.7, 0.9]
    y_midpoint = env.y_lb + (env.y_ub - env.y_lb) / 2
    down_action = Action(np.array([0, -0.08, 0]).astype(np.float32))
    for test_x in test_robot_xs:
        state = sample_task.init.copy()
        robot, = state.get_objects(robot_type)
        state.set(robot, "x", test_x)
        # try going down a lot
        for _ in range(50):
            state = env.simulate(state, down_action)
        # make sure robot is still above the wall
        robot_y = state.get(robot, "y")
        assert robot_y > y_midpoint, f"Robot did not collide at x={test_x}"

    # Test robot can walk through the doorway when the door is open
    state = sample_task.init.copy()
    robot, = state.get_objects(robot_type)
    door, = state.get_objects(door_type)
    state.set(robot, "x", 0.35)
    state.set(door, "open", 1)  # set the door to be open
    # try going down a lot
    for _ in range(50):
        state = env.simulate(state, down_action)
    # make sure robot is below the wall now but not out of bounds
    robot_y = state.get(robot, "y")
    assert robot_y < y_midpoint, "Robot wasn't able to pass through doorway"
    assert robot_y >= env.y_lb, "Robot went out of bounds"


def test_narrow_passage_options():
    """Test ParameterizedOptions work."""
    # Set up environment
    utils.reset_config({
        "env": "narrow_passage",
        "render_state_dpi": 150,
    })
    env = NarrowPassageEnv()
    DoorIsOpen, TouchedGoal = sorted(env.predicates)
    MoveAndOpenDoor, MoveToTarget = sorted(env.options)
    door_type, _, robot_type, target_type, _ = sorted(env.types)

    task = env.get_train_tasks()[0]
    state = task.init
    door, = state.get_objects(door_type)
    robot, = state.get_objects(robot_type)
    target, = state.get_objects(target_type)

    # Test MoveAndOpenDoor then MoveToTarget
    option_plan = [
        MoveAndOpenDoor.ground([robot, door], [0.3]),
        MoveToTarget.ground([robot, target], [0.5]),
    ]
    policy = utils.option_plan_to_policy(option_plan)
    monitor = utils.SimulateVideoMonitor(task, env.render_state)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure},
        monitor=monitor,
    )
    # Save video of run
    video = monitor.get_video()
    outfile = "hardcoded_door_options_narrow_passage.mp4"
    utils.save_video(outfile, video)
    final_state = traj.states[-1]
    assert GroundAtom(DoorIsOpen, [door]).holds(final_state)
    assert GroundAtom(TouchedGoal, [robot, target]).holds(final_state)
    assert task.goal_holds(final_state)

    # Test MoveToTarget directly without opening door
    option_plan = [
        MoveToTarget.ground([robot, target], [0.3]),
    ]
    policy = utils.option_plan_to_policy(option_plan)
    monitor = utils.SimulateVideoMonitor(task, env.render_state)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure},
        monitor=monitor,
    )
    # Save video of run
    video = monitor.get_video()
    outfile = "hardcoded_direct_options_narrow_passage.mp4"
    utils.save_video(outfile, video)
    final_state = traj.states[-1]
    assert not GroundAtom(DoorIsOpen, [door]).holds(final_state)
    assert GroundAtom(TouchedGoal, [robot, target]).holds(final_state)
    assert task.goal_holds(final_state)

    # Test MoveAndOpenDoor when door is already open
    option_plan = [
        MoveAndOpenDoor.ground([robot, door], [0.4]),
        MoveAndOpenDoor.ground([robot, door], [0.7]),
    ]
    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure},
    )
    final_state = traj.states[-1]
    assert GroundAtom(DoorIsOpen, [door]).holds(final_state)

    # Test MoveAndOpenDoor when robot already in range of door
    state = task.init.copy()
    goal = task.goal
    robot, = state.get_objects(robot_type)
    state.set(robot, "x", 0.25)
    state.set(robot, "y", 0.7)
    fixed_task = Task(state, goal)
    option_plan = [
        MoveAndOpenDoor.ground([robot, door], [0.1]),
    ]
    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        fixed_task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure},
    )
    final_state = traj.states[-1]
    assert GroundAtom(DoorIsOpen, [door]).holds(final_state)
    assert final_state.get(robot, "x") == 0.25
    assert final_state.get(robot, "y") == 0.7
