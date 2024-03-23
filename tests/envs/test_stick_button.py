"""Test cases for the stick button environment."""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from predicators import utils
from predicators.envs.stick_button import StickButtonEnv, \
    StickButtonMovementEnv
from predicators.envs.stick_button import StickButtonEnv, \
    StickButtonMovementEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.structs import Action, EnvironmentTask, GroundAtom


def test_stick_button():
    """Tests for StickButtonEnv()."""
    utils.reset_config({
        "env": "stick_button",
        "stick_button_num_buttons_train": [2],
        "stick_button_disable_angles": False,
        "stick_button_holder_scale": 0.001,
    })
    env = StickButtonEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 6
    assert len(env.goal_predicates) == 1
    AboveNoButton = [p for p in env.predicates if p.name == "AboveNoButton"][0]
    assert {pred.name for pred in env.goal_predicates} == {"Pressed"}
    assert len(get_gt_options(env.get_name())) == 4
    assert len(env.types) == 4
    button_type, holder_type, robot_type, stick_type = sorted(env.types)
    assert button_type.name == "button"
    assert holder_type.name == "holder"
    assert robot_type.name == "robot"
    assert stick_type.name == "stick"
    assert env.action_space.shape == (4, )
    # Create a custom initial state, with the robot in the middle, one button
    # reachable on the left, one button out of the reachable zone in the middle,
    # and the stick on the right at a 45 degree angle.
    state = env.get_train_tasks()[0].init.copy()
    robot, = state.get_objects(robot_type)
    stick, = state.get_objects(stick_type)
    buttons = state.get_objects(button_type)
    assert len(buttons) == 2
    robot_x = (env.rz_x_ub + env.rz_x_lb) / 2
    state.set(robot, "x", robot_x)
    state.set(robot, "y", (env.rz_y_ub + env.rz_y_lb) / 2)
    state.set(robot, "theta", np.pi / 2)
    reachable_button, unreachable_button = buttons
    reachable_x = (env.rz_x_ub + env.rz_x_lb) / 4
    state.set(reachable_button, "x", reachable_x)
    state.set(reachable_button, "y", (env.rz_y_ub + env.rz_y_lb) / 2)
    unreachable_x = robot_x
    state.set(unreachable_button, "x", unreachable_x)
    unreachable_y = 0.75 * env.y_ub
    assert not env.rz_y_lb <= unreachable_y <= env.rz_y_ub
    state.set(unreachable_button, "y", unreachable_y)
    stick_x = 3 * (env.rz_x_ub + env.rz_x_lb) / 4
    state.set(stick, "x", stick_x)
    state.set(stick, "y", (env.rz_y_ub + env.rz_y_lb) / 4)
    state.set(stick, "theta", np.pi / 4)
    task = EnvironmentTask(state, task.goal)
    env.render_state(state, task)
    assert GroundAtom(AboveNoButton, []).holds(state)

    ## Test simulate ##

    # Test for noops if the robot tries to leave the reachable zone.
    up_action = Action(np.array([0.0, -1.0, 0.0, 0.0], dtype=np.float32))
    s = state
    states = [s]
    for _ in range(20):
        s = env.simulate(s, up_action)
        states.append(s)
    assert states[-1].allclose(states[-2])

    # Test for going to press the reachable button.
    num_steps_to_left = int(np.ceil((robot_x - reachable_x) / env.max_speed))
    action_arrs = [
        np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(num_steps_to_left)
    ]

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-2].get(reachable_button, "pressed") < 0.5
    assert traj.states[-1].get(reachable_button, "pressed") > 0.5
    assert not GroundAtom(AboveNoButton, []).holds(traj.states[-1])

    # Test for going to pick up the stick.
    num_steps_to_right = 11
    action_arrs.extend([
        np.array([1.0, 0.0, 0.0, -1.0], dtype=np.float32)
        for _ in range(num_steps_to_right)
    ])
    # Figuring out these constants generally is a pain.
    action_arrs.append(np.array([0.2, 0.0, 0.0, 1.0], dtype=np.float32))

    # The stick should now be held.
    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-2].get(stick, "held") < 0.5
    assert traj.states[-1].get(stick, "held") > 0.5

    # Test for rotating the stick.
    assert env.max_angular_speed >= np.pi / 4
    action_arrs.append(np.array([0.0, 0.0, 1.0, -1.0], dtype=np.float32))

    # The stick should now be rotated.
    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert abs(traj.states[-2].get(stick, "theta") - np.pi / 4) < 1e-6
    assert traj.states[-1].get(stick, "theta") > np.pi / 4 + 1e-6

    # Test for moving and pressing the unreachable button with the stick.
    robot_x = traj.states[-1].get(robot, "x")
    num_steps_to_left = int(np.floor(
        (robot_x - unreachable_x) / env.max_speed))
    action_arrs.extend([
        np.array([-1.0, 0.0, 0.0, 1.0], dtype=np.float32)
        for _ in range(num_steps_to_left)
    ])
    # Move up and slightly left while pressing.
    action_arrs.extend([
        np.array([-0.2, 1.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float32),
    ])

    # The unreachable button should now be pressed.
    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           task.init,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-2].get(unreachable_button, "pressed") < 0.5
    assert traj.states[-1].get(unreachable_button, "pressed") > 0.5
    assert not GroundAtom(AboveNoButton, []).holds(traj.states[-1])

    # Uncomment for debugging.
    # policy = utils.action_arrs_to_policy(action_arrs)
    # monitor = utils.SimulateVideoMonitor(task, env.render_state)
    # traj = utils.run_policy_with_simulator(policy,
    #                                        env.simulate,
    #                                        task.init,
    #                                        lambda _: False,
    #                                        max_num_steps=len(action_arrs),
    #                                        monitor=monitor)
    # video = monitor.get_video()
    # outfile = "hardcoded_actions_stick_button.mp4"
    # utils.save_video(outfile, video)

    ## Test options ##

    options = get_gt_options(env.get_name())
    PickStick, PlaceStick, RobotPressButton, StickPressButton = sorted(options)
    assert PickStick.name == "PickStick"
    assert PlaceStick.name == "PlaceStick"
    assert RobotPressButton.name == "RobotPressButton"
    assert StickPressButton.name == "StickPressButton"

    # Test RobotPressButton.
    option = RobotPressButton.ground([robot, reachable_button, stick], [])
    option_plan = [option]

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(reachable_button, "pressed") < 0.5
    assert traj.states[-1].get(reachable_button, "pressed") > 0.5

    # Test PickStick.
    option = PickStick.ground([robot, stick], [0.1])
    option_plan.append(option)

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(stick, "held") < 0.5
    assert traj.states[-1].get(stick, "held") > 0.5

    # Test StickPressButton.
    option = StickPressButton.ground([robot, stick, unreachable_button], [])
    option_plan.append(option)

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(unreachable_button, "pressed") < 0.5
    assert traj.states[-1].get(unreachable_button, "pressed") > 0.5

    # Test PlaceStick.
    option = PlaceStick.ground([robot, stick], [-0.1])
    option_plan.append(option)

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(stick, "held") > 0.5
    assert traj.states[-1].get(stick, "held") < 0.5

    # Uncomment for debugging.
    # policy = utils.option_plan_to_policy(option_plan)
    # monitor = utils.SimulateVideoMonitor(task, env.render_state)
    # traj = utils.run_policy_with_simulator(
    #     policy,
    #     env.simulate,
    #     task.init,
    #     lambda _: False,
    #     max_num_steps=1000,
    #     exceptions_to_break_on={utils.OptionExecutionFailure},
    #     monitor=monitor)
    # video = monitor.get_video()
    # outfile = "hardcoded_options_stick_button.mp4"
    # utils.save_video(outfile, video)

    # Test that an EnviromentFailure is raised if the robot tries to pick
    # when colliding with the stick holder.
    utils.reset_config({
        "env": "stick_button",
        "stick_button_num_buttons_train": [1],
        "stick_button_disable_angles": True,
        "stick_button_holder_scale": 0.1,
    })
    env = StickButtonEnv()
    # Create a custom initial state, with the robot right on top of the stick
    # and stick holder.
    state = env.get_train_tasks()[0].init.copy()
    holder, = state.get_objects(holder_type)
    robot, = state.get_objects(robot_type)
    stick, = state.get_objects(stick_type)
    x = (env.rz_x_ub + env.rz_x_lb) / 2
    y = (env.rz_y_ub + env.rz_y_lb) / 2
    state.set(robot, "x", x)
    state.set(robot, "y", y)
    state.set(stick, "x", x)
    state.set(stick, "y", y)
    state.set(holder, "x", x - (env.holder_height - env.stick_height) / 2)
    state.set(holder, "y", y)
    # Press to pick up the stick.
    action = Action(np.array([0., 0., 0., 1.], dtype=np.float32))
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)  # should noop

    # Test interface for collecting human demonstrations.
    event_to_action = env.get_event_to_action_fn()
    fig = plt.figure()
    event = matplotlib.backend_bases.KeyEvent("test", fig.canvas, "down")
    assert isinstance(event_to_action(state, event), Action)
    event = matplotlib.backend_bases.MouseEvent("test",
                                                fig.canvas,
                                                x=1.0,
                                                y=2.0)
    with pytest.raises(AssertionError) as e2:
        event_to_action(state, event)
    assert "Out-of-bounds click" in str(e2)
    event.xdata = event.x
    event.ydata = event.y
    assert isinstance(event_to_action(state, event), Action)
    # Test quitting.
    event = matplotlib.backend_bases.KeyEvent("test", fig.canvas, "q")
    with pytest.raises(utils.HumanDemonstrationFailure) as e:
        event_to_action(state, event)
    assert "Human quit" in str(e)
    plt.close()

    # Special test for PlaceStick NSRT because it's not used by oracle.
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    nsrt = next(iter(n for n in nsrts if n.name == "PlaceStickFromNothing"))
    ground_nsrt = nsrt.ground([robot, stick])
    rng = np.random.default_rng(123)
    option = ground_nsrt.sample_option(state, set(), rng)
    assert -1 <= option.params[0] <= 1


def test_stick_button_move():
    """Tests for the movement variant of stick button."""
    utils.reset_config({
        "env": "stick_button_move",
        "stick_button_num_buttons_train": [2],
        "stick_button_disable_angles": False,
        "stick_button_holder_scale": 0.001,
    })
    env = StickButtonMovementEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 6
    assert len(env.goal_predicates) == 1
    AboveNoButton = [p for p in env.predicates if p.name == "AboveNoButton"][0]
    assert {pred.name for pred in env.goal_predicates} == {"Pressed"}
    assert len(get_gt_options(env.get_name())) == 6
    assert len(env.types) == 4
    button_type, holder_type, robot_type, stick_type = sorted(env.types)
    assert button_type.name == "button"
    assert holder_type.name == "holder"
    assert robot_type.name == "robot"
    assert stick_type.name == "stick"
    assert env.action_space.shape == (5, )
    # Create a custom initial state, with the robot in the middle, one button
    # reachable on the left, one button out of the reachable zone in the middle,
    # and the stick on the right at a 45 degree angle.
    state = env.get_train_tasks()[0].init.copy()
    robot, = state.get_objects(robot_type)
    stick, = state.get_objects(stick_type)
    holder, = state.get_objects(holder_type)
    buttons = state.get_objects(button_type)
    assert len(buttons) == 2
    robot_x = (env.rz_x_ub + env.rz_x_lb) / 2
    state.set(robot, "x", robot_x)
    state.set(robot, "y", (env.rz_y_ub + env.rz_y_lb) / 2)
    state.set(robot, "theta", np.pi / 2)
    state.set(robot, "fingers", 1.0)
    reachable_button, unreachable_button = buttons
    reachable_x = (env.rz_x_ub + env.rz_x_lb) / 4
    state.set(reachable_button, "x", reachable_x)
    state.set(reachable_button, "y", (env.rz_y_ub + env.rz_y_lb) / 2)
    unreachable_x = robot_x
    state.set(unreachable_button, "x", unreachable_x)
    unreachable_y = 0.75 * env.y_ub
    assert not env.rz_y_lb <= unreachable_y <= env.rz_y_ub
    state.set(unreachable_button, "y", unreachable_y)
    stick_x = 3 * (env.rz_x_ub + env.rz_x_lb) / 4
    state.set(stick, "x", stick_x)
    state.set(stick, "y", (env.rz_y_ub + env.rz_y_lb) / 4)
    state.set(stick, "theta", np.pi / 4)
    state.set(stick, "held", 0.0)

    task = EnvironmentTask(state, task.goal)
    env.render_state(state, task)
    assert GroundAtom(AboveNoButton, []).holds(state)
    ## Test options ##
    options = get_gt_options(env.get_name())
    PickStick, PlaceStick, RobotMoveToButton, RobotPressButton, \
        StickMoveToButton, StickPressButton = sorted(options)
    assert PickStick.name == "PickStick"
    assert PlaceStick.name == "PlaceStick"
    assert RobotPressButton.name == "RobotPressButton"
    assert StickPressButton.name == "StickPressButton"
    assert RobotMoveToButton.name == "RobotMoveToButton"
    assert StickMoveToButton.name == "StickMoveToButton"
    # Test PickStick.
    option = PickStick.ground([robot, stick], [0.1])
    option_plan = [option]

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(stick, "held") < 0.5
    assert traj.states[-2].get(robot, "fingers") > 0.5
    assert traj.states[-1].get(stick, "held") > 0.5
    assert traj.states[-1].get(robot, "fingers") <= 0.5

    # Test StickPressButton without moving first to show it doesn't work.
    option = StickPressButton.ground([robot, stick, unreachable_button], [])
    bad_option_plan = option_plan[:]
    bad_option_plan.append(option)

    policy = utils.option_plan_to_policy(bad_option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(unreachable_button, "pressed") < 0.5
    # assert pressing didn't work.
    assert traj.states[-1].get(unreachable_button, "pressed") < 0.5

    # Test StickPressButton properly with moving first.
    option = StickMoveToButton.ground([robot, unreachable_button, stick], [])
    option_plan.append(option)
    option = StickPressButton.ground([robot, stick, unreachable_button], [])
    option_plan.append(option)

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(unreachable_button, "pressed") < 0.5
    # assert pressing worked.
    assert traj.states[-1].get(unreachable_button, "pressed") > 0.5

    # Test PlaceStick
    utils.reset_config({
        "env": "stick_button_move",
        "stick_button_num_buttons_train": [2],
        "stick_button_disable_angles": True,
        "stick_button_holder_scale": 0.1,
    })
    env = StickButtonMovementEnv()
    state = env.get_train_tasks()[1].init.copy()
    task = EnvironmentTask(state, task.goal)
    robot, = state.get_objects(robot_type)
    stick, = state.get_objects(stick_type)
    holder, = state.get_objects(holder_type)
    buttons = state.get_objects(button_type)
    option_plan = [
        PickStick.ground([robot, stick], [0.3]),
        StickMoveToButton.ground([robot, buttons[0], stick], []),
        StickPressButton.ground([robot, stick, buttons[0]], []),
        PlaceStick.ground((robot, stick, holder), [0.4])
    ]
    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(stick, "held") > 0.5
    assert traj.states[-2].get(robot, "fingers") <= 0.5
    assert traj.states[-1].get(stick, "held") < 0.5
    assert traj.states[-1].get(robot, "fingers") > 0.5

    # Special test for PlaceStick NSRT because it's not used by oracle.
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    nsrt = next(iter(n for n in nsrts if n.name == "PlaceStickFromNothing"))
    ground_nsrt = nsrt.ground([robot, stick, holder])
    rng = np.random.default_rng(123)
    option = ground_nsrt.sample_option(state, set(), rng)
    assert -1 <= option.params[0] <= 1

    # Test that an EnviromentFailure is raised if the robot tries to pick
    # when colliding with the stick holder.
    utils.reset_config({
        "env": "stick_button",
        "stick_button_num_buttons_train": [1],
        "stick_button_disable_angles": True,
        "stick_button_holder_scale": 0.1,
    })
    env = StickButtonMovementEnv()
    # Create a custom initial state, with the robot right on top of the stick
    # and stick holder.
    state = env.get_train_tasks()[0].init.copy()
    holder, = state.get_objects(holder_type)
    robot, = state.get_objects(robot_type)
    stick, = state.get_objects(stick_type)
    x = (env.rz_x_ub + env.rz_x_lb) / 2
    y = (env.rz_y_ub + env.rz_y_lb) / 2
    state.set(robot, "x", x)
    state.set(robot, "y", y)
    state.set(stick, "x", x)
    state.set(stick, "y", y)
    state.set(holder, "x", x - (env.holder_height - env.stick_height) / 2)
    state.set(holder, "y", y)
    # Press to pick up the stick.
    action = Action(np.array([0.0, 0.0, 0.0, -1.0, 1.0], dtype=np.float32))
    next_state = env.simulate(state, action)
    assert state.allclose(next_state)  # should noop
