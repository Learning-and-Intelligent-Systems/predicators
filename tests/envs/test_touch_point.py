"""Test cases for the touch point environment."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from predicators.src import utils
from predicators.src.envs.touch_point import TouchPointEnv
from predicators.src.structs import Action


def test_touch_point():
    """Tests for TouchPointEnv class."""
    utils.reset_config({"env": "touch_point"})
    env = TouchPointEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 1
    assert len(env.goal_predicates) == 1
    assert {pred.name for pred in env.goal_predicates} == {"Touched"}
    assert len(env.options) == 1
    assert len(env.types) == 2
    assert env.action_space.shape == (1, )
    task = env.get_train_tasks()[0]
    state = task.init
    env.render_state(state, task, caption="caption")
    robot, target = sorted(state, key=lambda o: o.name)
    assert robot.name == "robby"
    assert target.name == "target"
    state = utils.create_state_from_dict({
        robot: {
            "x": 0.5,
            "y": 0.1,
        },
        target: {
            "x": 0.5,
            "y": 0.9,
        }
    })
    assert len(task.goal) == 1
    goal_atom = list(task.goal)[0]
    assert goal_atom.predicate.name == "Touched"
    assert goal_atom.objects == [robot, target]
    assert not goal_atom.holds(state)
    action = Action(np.array([np.pi / 2], dtype=np.float32))  # move up
    state = env.simulate(state, action)
    assert abs(state.get(robot, "x") - 0.5) < 1e-7
    act_mag = TouchPointEnv.action_magnitude
    assert abs(state.get(robot, "y") - (0.1 + act_mag)) < 1e-7
    state.set(robot, "y", 0.9 - TouchPointEnv.action_magnitude)
    assert goal_atom.holds(state)
    # Test interface for collecting human demonstrations.
    event_to_action = env.get_event_to_action_fn()
    fig = plt.figure()
    event = matplotlib.backend_bases.KeyEvent("test", fig.canvas, "down")
    with pytest.raises(AssertionError) as e:
        event_to_action(state, event)
    assert "Keyboard controls not allowed" in str(e)
    event = matplotlib.backend_bases.MouseEvent("test",
                                                fig.canvas,
                                                x=1.0,
                                                y=2.0)
    with pytest.raises(AssertionError) as e:
        event_to_action(state, event)
    assert "Out-of-bounds click" in str(e)
    event.xdata = event.x
    event.ydata = event.y
    assert isinstance(event_to_action(state, event), Action)
    plt.close()
