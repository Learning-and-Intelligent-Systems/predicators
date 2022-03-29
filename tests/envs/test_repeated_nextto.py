"""Test cases for the repeated NextTo environment."""

import numpy as np

from predicators.src import utils
from predicators.src.envs.repeated_nextto import RepeatedNextToEnv
from predicators.src.structs import Action


def test_repeated_nextto():
    """Tests for RepeatedNextTo class."""
    utils.reset_config({"env": "repeated_nextto"})
    env = RepeatedNextToEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert {pred.name for pred in env.predicates} == \
        {"NextTo", "NextToNothing", "Grasped"}
    assert {pred.name for pred in env.goal_predicates} == {"Grasped"}
    assert len(env.options) == 2
    assert len(env.types) == 2
    dot_type = [t for t in env.types if t.name == "dot"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    assert env.action_space.shape == (3, )
    for i, task in enumerate(env.get_test_tasks()):
        state = task.init
        for item in state:
            x = state.get(item, "x")
            assert env.env_lb <= x <= env.env_ub
            if item.type == robot_type:
                continue
            assert 0 <= state.get(item, "grasped") <= 1
        if i == 0:
            # Test rendering by setting up a state that has
            # a grasped dot and a dot next to the robot
            dot0 = dot_type("dot0")
            dot1 = dot_type("dot1")
            robot = robot_type("robby")
            state.set(robot, "x", state.get(dot0, "x"))
            state.set(dot1, "grasped", 1.0)
            env.render_state(state, task, caption="caption")


def test_repeated_nextto_simulate():
    """Tests for the simulate() function."""
    utils.reset_config({
        "env": "repeated_nextto",
        "approach": "nsrt_learning",
    })
    env = RepeatedNextToEnv()
    Move = [o for o in env.options if o.name == "Move"][0]
    Grasp = [o for o in env.options if o.name == "Grasp"][0]
    Grasped = [o for o in env.predicates if o.name == "Grasped"][0]
    dot_type = [t for t in env.types if t.name == "dot"][0]
    robot_type = [t for t in env.types if t.name == "robot"][0]
    dot0 = dot_type("dot0")
    dot1 = dot_type("dot1")
    robby = robot_type("robby")
    task = env.get_train_tasks()[0]
    state = task.init
    atoms = utils.abstract(state, env.predicates)
    for item in state:
        if item != robby:
            assert Grasped([robby, item]) not in atoms
    # Move always succeeds, and clips back into bounds
    midpt = (env.env_lb + env.env_ub) / 2
    state.set(dot0, "x", midpt)
    act = Move.ground([robby, dot0], np.array([1],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    assert abs(next_state.get(robby, "x") - (midpt + 1)) < 1e-4
    state.set(dot0, "x", env.env_lb)
    act = Move.ground([robby, dot0], np.array([-1],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    assert abs(next_state.get(robby, "x") - env.env_lb) < 1e-4
    state.set(dot0, "x", env.env_ub)
    act = Move.ground([robby, dot0], np.array([1],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    assert abs(next_state.get(robby, "x") - env.env_ub) < 1e-4
    # Move to dot1 and change the state
    act = Move.ground([robby, dot1], np.array([0],
                                              dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    state = next_state
    # Grasp success
    act = Grasp.ground([robby, dot1], np.array([],
                                               dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert not state.allclose(next_state)
    # Grasp fails if not at the argument dot
    act = Grasp.ground([robby, dot0], np.array([],
                                               dtype=np.float32)).policy(state)
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
    # Use Action directly for final failure mode: no dot at desired x
    act = Action(np.array([1, 0, 0], dtype=np.float32))
    next_state = env.simulate(state, act)
    assert state.allclose(next_state)
