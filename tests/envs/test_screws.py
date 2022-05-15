"""Test cases for the screws environment."""

import numpy as np

from predicators.src import utils
from predicators.src.envs.screws import ScrewsEnv
from predicators.src.structs import Action


def test_screws():
    """Tests for ScrewsEnv class."""
    utils.reset_config({"seed": 0})
    env = ScrewsEnv()
    train_tasks = env.get_train_tasks()
    task = train_tasks[0]
    env.render_state(task.init, task)
    # Goal predicates should be {ScrewInReceptacle}.
    goal_preds = env.goal_predicates
    assert len(goal_preds) == 1
    assert {pred.name for pred in env.goal_predicates} == {"ScrewInReceptacle"}
    # Check that moving the robot beyond the bounds in the
    # x-direction doesn't allow it to move out of bounds.
    gripper_type = sorted(env.types)[0]
    gripper, = task.init.get_objects(gripper_type)
    extreme_left_state = env.simulate(
        task.init,
        Action(
            np.array([-(env.rz_x_ub - env.rz_x_lb), 0.0, 0.0],
                     dtype=np.float32)))
    assert extreme_left_state.get(
        gripper, "pose_x") == env.rz_x_lb + (env.gripper_width / 2.0)
    extreme_right_state = env.simulate(
        task.init,
        Action(
            np.array([(env.rz_x_ub - env.rz_x_lb), 0.0, 0.0],
                     dtype=np.float32)))
    assert extreme_right_state.get(
        gripper, "pose_x") == env.rz_x_ub - (env.gripper_width / 2.0)
    # Check that moving the robot beyond the bounds in the
    # y-direction doesn't allow it to move out of bounds.
    extreme_down_state = env.simulate(
        task.init,
        Action(
            np.array([0.0, -(env.rz_y_ub - env.rz_y_lb), 0.0],
                     dtype=np.float32)))
    assert extreme_down_state.get(gripper, "pose_y") == env.rz_y_lb
    extreme_up_state = env.simulate(
        task.init,
        Action(
            np.array([0.0, (env.rz_y_ub - env.rz_y_lb), 0.0],
                     dtype=np.float32)))
    assert extreme_up_state.get(gripper, "pose_y") == env.rz_y_ub

    # Check that picking up something and then dropping it immediately
    # works (this is never done during normal optimal planning.)
    goal_screw = list(task.goal)[0].objects[0]
    DemagnetizeGripper, MagnetizeGripper, _, MoveToScrew = sorted(env.options)
    demagnetize_gripper_option = DemagnetizeGripper.ground([gripper], [])
    magnetize_gripper_option = MagnetizeGripper.ground([gripper], [])
    move_to_screw_option = MoveToScrew.ground([gripper, goal_screw], [])
    policy = utils.option_plan_to_policy([
        move_to_screw_option, magnetize_gripper_option,
        demagnetize_gripper_option
    ])
    traj = utils.run_policy_with_simulator(
        policy,
        env.simulate,
        task.init,
        lambda _: False,
        max_num_steps=1000,
        exceptions_to_break_on={utils.OptionExecutionFailure})
    # traj.states[-1] is the last state after executing the demagnetization,
    # and traj.states[-3] is the state before executing magnetization.
    # This check thus asserts that magnetizing and then immediately
    # demagnetizing does nothing to the state.
    assert traj.states[-1].allclose(traj.states[-3])
