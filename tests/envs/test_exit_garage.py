"""Test cases for the exit_garage environment."""

import numpy as np

from predicators import utils
from predicators.envs.exit_garage import ExitGarageEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import Action, EnvironmentTask, GroundAtom


def test_exit_garage_properties():
    """Test env object initialization and properties."""
    utils.reset_config({"env": "exit_garage"})
    env = ExitGarageEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 5
    (CarHasExited, CarryingObstacle, NotCarryingObstacle, ObstacleCleared,
     ObstacleNotCleared) = sorted(env.predicates)
    assert CarHasExited.name == "CarHasExited"
    assert CarryingObstacle.name == "CarryingObstacle"
    assert NotCarryingObstacle.name == "NotCarryingObstacle"
    assert ObstacleCleared.name == "ObstacleCleared"
    assert ObstacleNotCleared.name == "ObstacleNotCleared"
    assert env.goal_predicates == {CarHasExited}
    assert len(get_gt_options(env.get_name())) == 3
    (DriveCarToExit, PickupObstacle,
     StoreObstacle) = sorted(get_gt_options(env.get_name()))
    assert DriveCarToExit.name == "DriveCarToExit"
    assert PickupObstacle.name == "PickupObstacle"
    assert StoreObstacle.name == "StoreObstacle"
    assert len(env.types) == 4
    car_type, obstacle_type, robot_type, storage_type = sorted(env.types)
    assert car_type.name == "car"
    assert obstacle_type.name == "obstacle"
    assert robot_type.name == "robot"
    assert storage_type.name == "storage"
    assert env.action_space.shape == (5, )


def test_exit_garage_actions():
    """Test to check that basic actions and rendering works as expected."""
    utils.reset_config({
        "env": "exit_garage",
        "exit_garage_min_num_obstacles": 1,
        "exit_garage_max_num_obstacles": 1,
        "num_train_tasks": 1,
    })
    env = ExitGarageEnv()
    (CarHasExited, CarryingObstacle, NotCarryingObstacle, ObstacleCleared,
     ObstacleNotCleared) = sorted(env.predicates)
    car_type, obstacle_type, robot_type, storage_type = sorted(env.types)

    # Create task with fixed initial state
    sample_task = env.get_train_tasks()[0].task
    state = sample_task.init.copy()
    goal = sample_task.goal
    car, = state.get_objects(car_type)
    state.set(car, "x", 0.15)
    state.set(car, "y", 0.3)
    robot, = state.get_objects(robot_type)
    state.set(robot, "x", 0.1)
    state.set(robot, "y", 0.8)
    obstacles = state.get_objects(obstacle_type)
    assert len(obstacles) == 1
    obstacle = obstacles[0]
    state.set(obstacle, "x", 0.5)
    state.set(obstacle, "y", 0.5)
    storage, = state.get_objects(storage_type)
    # Assert starting state predicates
    assert not GroundAtom(CarHasExited, [car]).holds(state)
    assert not GroundAtom(CarryingObstacle, [robot, obstacle]).holds(state)
    assert not GroundAtom(ObstacleCleared, [obstacle]).holds(state)
    assert GroundAtom(ObstacleNotCleared, [obstacle]).holds(state)
    assert GroundAtom(NotCarryingObstacle, [robot]).holds(state)
    task = EnvironmentTask(state, goal)

    # Fixed action sequences to test (each is a list of action arrays)
    # Move robot to obstacle and pickup
    bad_robot_action = np.array([0, 0, 0.1, 0.1, 1.0]).astype(np.float32)
    pickup_actions = [
        np.array([0, 0, 0.08, -0.095, 0]).astype(np.float32),
        np.array([0, 0, 0.095, -0.09, 0]).astype(np.float32),
        np.array([0, 0, 0.1, -0.1, 0]).astype(np.float32),
        np.array([0, 0, 0.1, 0.002, 0]).astype(np.float32),  # (0.475, 0.517)
        np.array([0, 0, 0, 0, 1.0]).astype(np.float32),  # pickup
    ]
    # Move robot to storage area and place obstacle
    store_actions = [
        np.array([0, 0, -0.05, 0.1, 0]).astype(np.float32),
        np.array([0, 0, -0.05, 0.095, 0]).astype(np.float32),
        np.array([0, 0, 0, 0.095, 0]).astype(np.float32),  # (0.465, 0.807)
        np.array([0, 0, 0, 0, 1.0]).astype(np.float32),  # place
    ]
    # Move car to exit
    drive_actions = [
        np.array([0.1, 0.1, 0, 0, 0]).astype(np.float32),
        np.array([0.095, -0.1, 0, 0, 0]).astype(np.float32),
        np.array([0.1, 0, 0, 0, 0]).astype(np.float32),
        np.array([0.099, 0, 0, 0, 0]).astype(np.float32),
        np.array([-0.01, 0, 0, 0, 0]).astype(np.float32),
        np.array([0.1, 0, 0, 0, 0]).astype(np.float32),
        np.array([0.1, 0, 0, 0, 0]).astype(np.float32),
        np.array([0.1, 0, 0, 0, 0]).astype(np.float32),
        np.array([0.1, 0, 0, 0, 0]).astype(np.float32),
    ]
    all_action_arrs = pickup_actions + store_actions + drive_actions

    # Test that picking up nothing does nothing
    s = state.copy()
    true_x = s.get(robot, "x")
    true_y = s.get(robot, "y")
    s = env.simulate(s, Action(bad_robot_action))
    # Robot shouldn't have moved since it was trying an action
    assert s.get(robot, "x") == true_x
    assert s.get(robot, "y") == true_y
    # Robot shouldn't have picked up anything since it wasn't on an obstacle
    assert not GroundAtom(CarryingObstacle, [robot, obstacle]).holds(s)
    assert GroundAtom(NotCarryingObstacle, [robot]).holds(s)

    # Test that going and picking up the obstacle works
    for action in pickup_actions:
        s = env.simulate(s, Action(action))
    assert GroundAtom(CarryingObstacle, [robot, obstacle]).holds(s)
    assert not GroundAtom(ObstacleNotCleared, [obstacle]).holds(s)
    assert not GroundAtom(NotCarryingObstacle, [robot]).holds(s)

    # Test that trying to place the obstacle outside storage does nothing
    true_x = s.get(robot, "x")
    true_y = s.get(robot, "y")
    s = env.simulate(s, Action(bad_robot_action))
    # Robot shouldn't have moved since it was trying an action
    assert s.get(robot, "x") == true_x
    assert s.get(robot, "y") == true_y
    # Robot should still be carrying obstacle
    assert GroundAtom(CarryingObstacle, [robot, obstacle]).holds(s)
    assert not GroundAtom(ObstacleNotCleared, [obstacle]).holds(s)
    assert not GroundAtom(NotCarryingObstacle, [robot]).holds(s)

    # Test that moving to storage and placing the obstacle works
    assert s.get(storage, "num_stored") == 0
    for action in store_actions:
        s = env.simulate(s, Action(action))
    # Check obstacle is placed
    assert not GroundAtom(CarryingObstacle, [robot, obstacle]).holds(s)
    assert GroundAtom(ObstacleCleared, [obstacle]).holds(s)
    assert GroundAtom(NotCarryingObstacle, [robot]).holds(s)
    # Check obstacle and robot are in storage area
    assert s.get(robot, "y") > 0.8
    assert s.get(obstacle, "y") > 0.8
    # Check number of stored items is now 1
    assert s.get(storage, "num_stored") == 1

    # Test that picking up in storage area does nothing
    s = env.simulate(s, Action(bad_robot_action))
    assert GroundAtom(NotCarryingObstacle, [robot]).holds(s)

    # Test moving car to exit
    for action in drive_actions:
        s = env.simulate(s, Action(action))
    # Check that car had moved up at some point (rotation)
    assert s.get(car, "y") > 0.3
    assert GroundAtom(CarHasExited, [car]).holds(s)
    assert task.task.goal_holds(s)  # check task goal reached

    # Test running entire plan
    policy = utils.action_arrs_to_policy(all_action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           task.task.goal_holds,
                                           max_num_steps=len(all_action_arrs))
    # Test rendering various states
    env.render_state(traj.states[0], task)  # at start
    env.render_state(traj.states[6], task, caption="caption")  # after pickup
    env.render_state(traj.states[10], task)  # after store
    env.render_state(traj.states[-1], task)  # state at end


def test_exit_garage_collisions():
    """Test that the car can't go through obstacles or storage area."""
    utils.reset_config({
        "env": "exit_garage",
        "exit_garage_min_num_obstacles": 1,
        "exit_garage_max_num_obstacles": 1,
        "num_train_tasks": 1,
    })
    env = ExitGarageEnv()
    car_type, obstacle_type, robot_type, _ = sorted(env.types)

    # Create sample state
    sample_task = env.get_train_tasks()[0].task
    state = sample_task.init.copy()

    # Car can't go out of bounds
    car, = state.get_objects(car_type)
    state.set(car, "x", 0.15)
    state.set(car, "y", 0.3)
    back_car_action = Action(np.array([-0.09, 0, 0, 0, 0]).astype(np.float32))
    for _ in range(10):
        state = env.simulate(state, back_car_action)
    # Make sure car is still in bounds
    assert 0.0 <= state.get(car, "x") <= 1.0

    # Robot can't go out of bounds
    robot, = state.get_objects(robot_type)
    state.set(robot, "x", 0.1)
    state.set(robot, "y", 0.8)
    robot_up_action = Action(np.array([0, 0, 0, 0.05, 0]).astype(np.float32))
    for _ in range(10):
        state = env.simulate(state, robot_up_action)
    # Make sure robot is still in bounds
    assert 0.0 <= state.get(robot, "y") <= 1.0

    # Car can't go through obstacle
    obstacle, = state.get_objects(obstacle_type)
    state.set(obstacle, "x", 0.5)
    state.set(obstacle, "y", 0.32)
    drive_car_action = Action(np.array([0.099, 0, 0, 0, 0]).astype(np.float32))
    for _ in range(50):
        state = env.simulate(state, drive_car_action)
    # Make sure car didn't go past obstacle
    assert state.get(car, "x") < 0.5

    # Test car can go past if obstacle is picked up
    state.set(obstacle, "carried", 1)
    for _ in range(50):
        state = env.simulate(state, drive_car_action)
    assert state.get(car, "x") > 0.5

    # Test car can't drive into storage area
    state.set(car, "x", 0.15)
    state.set(car, "y", 0.3)
    state.set(car, "theta", np.pi / 2.0)  # pointed up
    for _ in range(50):
        state = env.simulate(state, drive_car_action)
    # Make sure car stopped before storage area
    assert 0.5 <= state.get(car, "y") <= 0.8


def test_exit_garage_options():
    """Tests for exit garage parametrized options."""
    utils.reset_config({
        "env": "exit_garage",
        "exit_garage_pick_place_refine_penalty": 0,
        "exit_garage_min_num_obstacles": 2,
        "exit_garage_max_num_obstacles": 2,
        "exit_garage_rrt_num_control_samples": 15,
        "exit_garage_rrt_sample_goal_eps": 0.3,
        "num_train_tasks": 1,
    })
    env = ExitGarageEnv()
    (CarHasExited, CarryingObstacle, NotCarryingObstacle, ObstacleCleared,
     ObstacleNotCleared) = sorted(env.predicates)
    (DriveCarToExit, PickupObstacle,
     StoreObstacle) = sorted(get_gt_options(env.get_name()))
    car_type, obstacle_type, robot_type, _ = sorted(env.types)

    # Create task with fixed initial state
    sample_task = env.get_train_tasks()[0].task
    state = sample_task.init.copy()
    goal = sample_task.goal
    car, = state.get_objects(car_type)
    robot, = state.get_objects(robot_type)
    obstacle1, obstacle2 = state.get_objects(obstacle_type)
    state.set(obstacle1, "x", 0.5)
    state.set(obstacle1, "y", 0.3)
    state.set(obstacle2, "x", 0.8)
    state.set(obstacle2, "y", 0.1)
    task = EnvironmentTask(state, goal)

    # Test PickupObstacle, StoreObstacle, then DriveCarToExit
    option_plan = [
        PickupObstacle.ground([robot, obstacle2], [0.2]),
        StoreObstacle.ground([robot, obstacle2], [0.6]),
        DriveCarToExit.ground([car], [0.7]),
    ]
    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=5000,
        exceptions_to_break_on={utils.OptionExecutionFailure},
    )
    final_state = traj.states[-1]
    assert GroundAtom(CarHasExited, [car]).holds(final_state)
    assert not GroundAtom(CarryingObstacle,
                          [robot, obstacle2]).holds(final_state)
    assert GroundAtom(ObstacleCleared, [obstacle2]).holds(final_state)
    assert not GroundAtom(ObstacleNotCleared, [obstacle2]).holds(final_state)
    assert GroundAtom(NotCarryingObstacle, [robot]).holds(final_state)
    assert task.task.goal_holds(final_state)

    # Test scenarios where options shouldn't be initiable

    # Test StoreObstacle when robot isn't carrying anything
    assert GroundAtom(NotCarryingObstacle, [robot]).holds(state)
    store_obstacle = StoreObstacle.ground([robot, obstacle1], [0.3])
    assert not store_obstacle.initiable(state)

    # Test StoreObstacle when robot carrying different obstacle
    test_state = state.copy()
    test_state.set(robot, "carrying", 1)
    test_state.set(obstacle2, "carried", 1)
    assert not store_obstacle.initiable(test_state)

    # Test PickupObstacle when robot carrying obstacle already
    pickup_obstacle = PickupObstacle.ground([robot, obstacle1], [0.8])
    assert not pickup_obstacle.initiable(test_state)

    # Test PickupObstacle when obstacle already picked or stored
    pickup_obstacle = PickupObstacle.ground([robot, obstacle2], [0.4])
    assert not pickup_obstacle.initiable(test_state)  # already picked
    test_state = state.copy()
    test_state.set(obstacle2, "y", 0.9)  # obstacle2 already in storage
    assert not pickup_obstacle.initiable(test_state)


def test_exit_garage_failed_rrt():
    """Tests that exit garage parametrized options are correctly un-initiable
    if motion planning fails."""
    utils.reset_config({
        "env": "exit_garage",
        "exit_garage_pick_place_refine_penalty": 0,
        "exit_garage_min_num_obstacles": 6,
        "exit_garage_max_num_obstacles": 6,
        "exit_garage_rrt_num_attempts": 1,
        "exit_garage_rrt_num_control_samples": 1,
        "exit_garage_rrt_sample_goal_eps": 0,
        "num_train_tasks": 1,
    })
    env = ExitGarageEnv()
    DriveCarToExit, _, _ = sorted(get_gt_options(env.get_name()))
    car_type, obstacle_type, _, _ = sorted(env.types)

    # Create task with fixed initial state
    task = env.get_train_tasks()[0].task
    state = task.init.copy()
    car, = state.get_objects(car_type)
    # Block the car's drive using the obstacles to make motion planning fail
    for i, obstacle in enumerate(state.get_objects(obstacle_type)):
        state.set(obstacle, "x", 0.5)
        state.set(obstacle, "y", 0.15 * i - 0.075)

    # Test DriveCarToExit
    drive_car_to_exit = DriveCarToExit.ground([car], [0.34])
    assert not drive_car_to_exit.initiable(state)
