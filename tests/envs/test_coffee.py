"""Test cases for the coffee environment."""
import numpy as np

from predicators import utils
from predicators.envs.coffee import CoffeeEnv
from predicators.ground_truth_models import get_gt_options
from predicators.structs import Action, GroundAtom


def test_coffee():
    """Tests for CoffeeEnv()."""
    utils.reset_config({
        "env": "coffee",
        "coffee_num_cups_test": [4],  # used to assure 4 cups in custom state
        "video_fps": 10,  # for faster debugging videos
    })
    env = CoffeeEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 13
    assert len(env.goal_predicates) == 1
    pred_name_to_pred = {p.name: p for p in env.predicates}
    CupFilled = pred_name_to_pred["CupFilled"]
    JugInMachine = pred_name_to_pred["JugInMachine"]
    OnTable = pred_name_to_pred["OnTable"]
    NotAboveCup = pred_name_to_pred["NotAboveCup"]
    assert len(get_gt_options(env.get_name())) == 6
    option_name_to_option = {o.name: o for o in get_gt_options(env.get_name())}
    assert len(env.types) == 4
    type_name_to_type = {t.name: t for t in env.types}
    cup_type = type_name_to_type["cup"]
    jug_type = type_name_to_type["jug"]
    machine_type = type_name_to_type["machine"]
    robot_type = type_name_to_type["robot"]
    assert env.action_space.shape == (6, )
    # Create a custom initial state, with cups positions at the extremes of
    # their possible initial positions.
    state = env.get_test_tasks()[0].init.copy()
    robot, = state.get_objects(robot_type)
    jug, = state.get_objects(jug_type)
    machine, = state.get_objects(machine_type)
    cups = state.get_objects(cup_type)
    assert len(cups) == 4
    # Reposition the cups.
    cup_corners = [
        (env.cup_init_x_lb, env.cup_init_y_lb),
        (env.cup_init_x_lb, env.cup_init_y_ub),
        (env.cup_init_x_ub, env.cup_init_y_ub),
        (env.cup_init_x_ub, env.cup_init_y_lb),
    ]
    for cup, (x, y) in zip(cups, cup_corners):
        state.set(cup, "x", x)
        state.set(cup, "y", y)
    # Reposition the jug just so we know exactly where it is.
    state.set(jug, "x", env.jug_init_x_ub)
    state.set(jug, "y", env.jug_init_y_ub)
    state.set(jug, "rot", 0.0)
    env.render_state(state, task)

    ## Test simulate ##

    # Helper function to compute a sequence of actions that moves the robot
    # from an initial position to a target position in a straight line.
    def _get_position_action_arrs(init_x, init_y, init_z, final_x, final_y,
                                  final_z):
        action_arrs = []
        current_pos = np.array([init_x, init_y, init_z])
        delta = np.subtract((final_x, final_y, final_z), current_pos)
        pos_norm = float(np.linalg.norm(delta))
        if pos_norm > env.max_position_vel:
            delta = env.max_position_vel * (delta / pos_norm)
            num_max_steps = int(np.floor(pos_norm / env.max_position_vel))
            for _ in range(num_max_steps):
                dx, dy, dz = delta / env.max_position_vel
                action_arrs.append(
                    np.array([dx, dy, dz, 0.0, 0.0, 0.0], dtype=np.float32))
                current_pos = current_pos + delta
            delta = np.subtract((final_x, final_y, final_z), current_pos)
            pos_norm = float(np.linalg.norm(delta))
        if pos_norm > 0:
            delta = delta / env.max_position_vel
            dx, dy, dz = delta
            action_arrs.append(
                np.array([dx, dy, dz, 0.0, 0.0, 0.0], dtype=np.float32))
        return action_arrs

    # Test twisting the jug.
    target_x = state.get(jug, "x")
    target_y = state.get(jug, "y")
    target_z = env.jug_height
    action_arrs = _get_position_action_arrs(state.get(robot, "x"),
                                            state.get(robot, "y"),
                                            state.get(robot, "z"), target_x,
                                            target_y, target_z)
    num_twists = 2
    twist_act_arr = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    action_arrs.extend([twist_act_arr for _ in range(num_twists)])
    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           state,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    twist_amt = num_twists * env.max_angular_vel
    assert abs(traj.states[-1].get(jug, "rot") - twist_amt) < 1e-6
    s = traj.states[-1]

    # The jug is too twisted now, so picking it up should fail.
    target_x, target_y, target_z = env._get_jug_handle_grasp(s, jug)  # pylint: disable=protected-access
    move_action_arrs = _get_position_action_arrs(s.get(robot, "x"),
                                                 s.get(robot, "y"),
                                                 s.get(robot, "z"), target_x,
                                                 target_y, target_z)
    action_arrs.extend(move_action_arrs)
    pick_act_arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    action_arrs.append(pick_act_arr)

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           state,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-1].get(jug, "is_held") < 0.5

    # Test picking up the jug.
    target_x, target_y, target_z = env._get_jug_handle_grasp(state, jug)  # pylint: disable=protected-access
    action_arrs = _get_position_action_arrs(state.get(robot, "x"),
                                            state.get(robot, "y"),
                                            state.get(robot, "z"), target_x,
                                            target_y, target_z)
    pick_act_arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    action_arrs.append(pick_act_arr)

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           state,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-2].get(jug, "is_held") < 0.5
    assert traj.states[-1].get(jug, "is_held") > 0.5
    assert GroundAtom(OnTable, [jug]).holds(traj.states[-2])
    assert not GroundAtom(OnTable, [jug]).holds(traj.states[-1])
    s = traj.states[-1]

    # Test moving and placing the jug at the machine.
    # Offset based on the grasp.
    target_x = env.dispense_area_x - (s.get(jug, "x") - s.get(robot, "x"))
    target_y = env.dispense_area_y - (s.get(jug, "y") - s.get(robot, "y"))
    target_z = s.get(robot, "z")
    move_jug_act_arrs = _get_position_action_arrs(s.get(robot, "x"),
                                                  s.get(robot, "y"),
                                                  s.get(robot, "z"), target_x,
                                                  target_y, target_z)
    action_arrs.extend(move_jug_act_arrs)
    # Drop the jug.
    place_act_arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    action_arrs.append(place_act_arr)

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           state,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-2].get(jug, "is_held") > 0.5
    assert traj.states[-1].get(jug, "is_held") < 0.5
    s = traj.states[-1]

    # Test pressing the machine button.
    # First move to above the button.
    move_to_above_button_act_arrs = _get_position_action_arrs(
        s.get(robot, "x"), s.get(robot, "y"), s.get(robot, "z"), env.button_x,
        env.button_y + 1.0, env.button_z)
    action_arrs.extend(move_to_above_button_act_arrs)
    # Move forward to press the button.
    move_to_press_button_act_arrs = _get_position_action_arrs(
        env.button_x,
        env.button_y + 1.0,
        env.button_z,
        env.button_x,
        env.button_y,
        env.button_z,
    )
    action_arrs.extend(move_to_press_button_act_arrs)

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           state,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-2].get(machine, "is_on") < 0.5
    assert traj.states[-1].get(machine, "is_on") > 0.5
    # The jug should also now be filled.
    assert traj.states[-2].get(jug, "is_filled") < 0.5
    assert traj.states[-1].get(jug, "is_filled") > 0.5
    s = traj.states[-1]

    # Test picking up the filled jug.
    target_x, target_y, target_z = env._get_jug_handle_grasp(s, jug)  # pylint: disable=protected-access
    move_to_pick_act_arrs = _get_position_action_arrs(s.get(robot, "x"),
                                                      s.get(robot, "y"),
                                                      s.get(robot,
                                                            "z"), target_x,
                                                      target_y, target_z)
    action_arrs.extend(move_to_pick_act_arrs)
    pick_act_arr = np.array([0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)
    action_arrs.append(pick_act_arr)

    policy = utils.action_arrs_to_policy(action_arrs)
    traj = utils.run_policy_with_simulator(policy,
                                           env.simulate,
                                           state,
                                           lambda _: False,
                                           max_num_steps=len(action_arrs))
    assert traj.states[-2].get(jug, "is_held") < 0.5
    assert traj.states[-1].get(jug, "is_held") > 0.5
    assert GroundAtom(NotAboveCup, [robot, jug]).holds(traj.states[-1])
    s = traj.states[-1]

    # Check for noop when pouring into nothing.
    pour_act_lst = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    pour_act_arr = np.array(pour_act_lst, dtype=np.float32)
    next_s = env.simulate(s, Action(pour_act_arr))
    assert s.allclose(next_s)

    # Test pouring in each of the cups.
    for cup in cups:
        jug_target_x, jug_target_y, jug_target_z = env._get_pour_position(  # pylint: disable=protected-access
            s, cup)
        target_x = jug_target_x - (s.get(jug, "x") - s.get(robot, "x"))
        target_y = jug_target_y - (s.get(jug, "y") - s.get(robot, "y"))
        target_z = jug_target_z + env.jug_handle_height
        move_to_pour_act_arrs = _get_position_action_arrs(
            s.get(robot, "x"), s.get(robot, "y"), s.get(robot, "z"), target_x,
            target_y, target_z)
        action_arrs.extend(move_to_pour_act_arrs)
        target_liquid = state.get(cup, "target_liquid")
        num_pour_steps = int(np.ceil(target_liquid / env.pour_velocity))
        # Start pouring.
        action_arrs.append(pour_act_arr)
        # Keep pouring.
        action_arrs.extend([pour_act_arr for _ in range(num_pour_steps - 1)])
        # Stop pouring.
        action_arrs.append(-1 * pour_act_arr)

        policy = utils.action_arrs_to_policy(action_arrs)
        traj = utils.run_policy_with_simulator(policy,
                                               env.simulate,
                                               state,
                                               lambda _: False,
                                               max_num_steps=len(action_arrs))
        assert not GroundAtom(CupFilled, [cup]).holds(traj.states[-3])
        assert GroundAtom(CupFilled, [cup]).holds(traj.states[-1])
        assert not GroundAtom(NotAboveCup, [robot, jug]).holds(traj.states[-1])
        s = traj.states[-1]
        # Render a state where we are in the process of pouring.
        env.render_state(traj.states[-2], task)

    # Check for noop when overfilling a cup.
    next_s = env.simulate(s, Action(pour_act_arr))
    assert s.allclose(next_s)

    # Uncomment for debugging.
    # policy = utils.action_arrs_to_policy(action_arrs)
    # monitor = utils.SimulateVideoMonitor(task, env.render_state)
    # traj = utils.run_policy_with_simulator(policy,
    #                                        env.simulate,
    #                                        state,
    #                                        lambda _: False,
    #                                        max_num_steps=len(action_arrs),
    #                                        monitor=monitor)
    # video = monitor.get_video()
    # outfile = "hardcoded_actions_coffee.mp4"
    # utils.save_video(outfile, video)

    ## Test options ##

    # Test MoveToTwistJug and TwistJug.
    MoveToTwistJug = option_name_to_option["MoveToTwistJug"]
    TwistJug = option_name_to_option["TwistJug"]
    option1 = MoveToTwistJug.ground([robot, jug], [])
    option2 = TwistJug.ground([robot, jug], np.array([1.0], dtype=np.float32))
    option_plan = [option1, option2]
    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        state,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    expected_rot = env.jug_init_rot_ub
    assert abs(traj.states[-1].get(jug, "rot") - expected_rot) < 1e-6

    # Test PickJug.
    PickJug = option_name_to_option["PickJug"]
    option = PickJug.ground([robot, jug], [])
    option_plan = [option]

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        state,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(jug, "is_held") < 0.5
    assert traj.states[-1].get(jug, "is_held") > 0.5

    # Test PlaceJugInMachine.
    PlaceJugInMachine = option_name_to_option["PlaceJugInMachine"]
    option = PlaceJugInMachine.ground([robot, jug, machine], [])
    option_plan.append(option)

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        state,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(jug, "is_held") > 0.5
    assert traj.states[-1].get(jug, "is_held") < 0.5
    assert GroundAtom(JugInMachine, [jug, machine]).holds(traj.states[-1])

    # Test TurnMachineOn.
    TurnMachineOn = option_name_to_option["TurnMachineOn"]
    option = TurnMachineOn.ground([robot, machine], [])
    option_plan.append(option)

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        state,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(machine, "is_on") < 0.5
    assert traj.states[-1].get(machine, "is_on") > 0.5

    # Test PickJug from the dispense area.
    option = PickJug.ground([robot, jug], [])
    option_plan.append(option)

    policy = utils.option_plan_to_policy(option_plan)
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        state,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    assert traj.states[-2].get(jug, "is_held") < 0.5
    assert traj.states[-1].get(jug, "is_held") > 0.5

    # Test Pour into each of the cups.
    Pour = option_name_to_option["Pour"]
    for cup in cups:
        option = Pour.ground([robot, jug, cup], [])
        option_plan.append(option)
        policy = utils.option_plan_to_policy(option_plan)
        traj = utils.run_policy_with_simulator(
            policy,
            env.simulate,
            state,
            lambda _: False,
            max_num_steps=1000,
            exceptions_to_break_on={utils.OptionExecutionFailure})
        assert not GroundAtom(CupFilled, [cup]).holds(traj.states[-3])
        assert GroundAtom(CupFilled, [cup]).holds(traj.states[-1])
        s = traj.states[-1]

    # Uncomment for debugging.
    # policy = utils.option_plan_to_policy(option_plan)
    # monitor = utils.SimulateVideoMonitor(task, env.render_state)
    # traj = utils.run_policy_with_simulator(
    #     policy,
    #     env.simulate,
    #     state,
    #     lambda _: False,
    #     max_num_steps=1000,
    #     exceptions_to_break_on={utils.OptionExecutionFailure},
    #     monitor=monitor)
    # video = monitor.get_video()
    # outfile = "hardcoded_options_coffee.mp4"
    # utils.save_video(outfile, video)
