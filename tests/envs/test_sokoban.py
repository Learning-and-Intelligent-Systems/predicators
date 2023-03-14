"""Test cases for the sokoban environment."""

import pytest

from predicators import utils
from predicators.envs.sokoban import SokobanEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options


def test_sokoban():
    """Tests for sokoban env.

    Since the gym environment can be slow to initialize, we group all
    tests together.
    """
    utils.reset_config({
        "env": "sokoban",
        "sokoban_gym_name": "Sokoban-small-v0",
        "num_train_tasks": 1,
        "num_test_tasks": 2,
    })
    env = SokobanEnv()
    assert env.get_name() == "sokoban"
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 12
    Above, At, Below, GoalCovered, IsBox, IsGoal, IsLoc, IsNonGoalLoc, \
        IsPlayer, LeftOf, NoBoxAtLoc, RightOf = sorted(env.predicates)
    assert Above.name == "Above"
    assert At.name == "At"
    assert Below.name == "Below"
    assert GoalCovered.name == "GoalCovered"
    assert IsBox.name == "IsBox"
    assert IsGoal.name == "IsGoal"
    assert IsLoc.name == "IsLoc"
    assert IsNonGoalLoc.name == "IsNonGoalLoc"
    assert IsPlayer.name == "IsPlayer"
    assert LeftOf.name == "LeftOf"
    assert NoBoxAtLoc.name == "NoBoxAtLoc"
    assert RightOf.name == "RightOf"
    assert env.goal_predicates == {GoalCovered}
    options = get_gt_options(env.get_name())
    assert len(options) == 9
    MoveDown, MoveLeft, MoveRight, MoveUp, Noop, PushDown, PushLeft, \
        PushRight, PushUp = sorted(options)
    assert MoveDown.name == "MoveDown"
    assert MoveLeft.name == "MoveLeft"
    assert MoveRight.name == "MoveRight"
    assert MoveUp.name == "MoveUp"
    assert Noop.name == "NoOperation"
    assert PushDown.name == "PushDown"
    assert PushLeft.name == "PushLeft"
    assert PushRight.name == "PushRight"
    assert PushUp.name == "PushUp"
    assert len(env.types) == 1
    object_type, = env.types
    assert object_type.name == "obj"
    assert env.action_space.shape == (9, )
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 12
    train_tasks = [t.task for t in env.get_train_tasks()]
    assert len(train_tasks) == 1
    test_tasks = [t.task for t in env.get_test_tasks()]
    assert len(test_tasks) == 2
    task = test_tasks[1]
    state = env.reset("test", 1)
    assert state.allclose(task.init)
    imgs = env.render()
    assert len(imgs) == 1
    atoms = utils.abstract(state, env.predicates)
    num_boxes = len({a for a in atoms if a.predicate == IsBox})
    num_goals = len({a for a in atoms if a.predicate == IsGoal})
    assert num_boxes == num_goals
    # Hardcode a sequence of actions to achieve one example of GoalCovered.
    # This is brittle, but we don't want to plan because it could be slow
    # without Fast Downward, which is not installed on the test server.
    plan = [
        "MoveDown", "MoveDown", "MoveLeft", "MoveLeft", "MoveUp", "PushRight"
    ]
    option_names = {o.name: o for o in options}
    for name in plan:
        param_option = option_names[name]
        option = param_option.ground([], [])
        assert option.initiable(state)
        action = option.policy(state)
        state = env.step(action)
    atoms = utils.abstract(state, env.predicates)
    # Now one of the goals should be covered.
    assert len({a for a in atoms if a.predicate == GoalCovered}) == 1
    # Cover not implemented methods.
    with pytest.raises(NotImplementedError) as e:
        env.render_state_plt(state, task)
    assert "This env does not use Matplotlib" in str(e)
    with pytest.raises(NotImplementedError) as e:
        env.render_state(state, task)
    assert "A gym environment cannot render arbitrary states." in str(e)
    with pytest.raises(NotImplementedError) as e:
        env.simulate(state, action)
    assert "Simulate not implemented for gym envs." in str(e)
