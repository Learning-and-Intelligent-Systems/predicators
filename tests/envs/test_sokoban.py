"""Test cases for the sokoban environment."""

import numpy as np
import pytest

from predicators import utils
from predicators.envs.sokoban import SokobanEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.perception.sokoban_perceiver import SokobanPerceiver


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
    perceiver = SokobanPerceiver()
    assert env.get_name() == "sokoban"
    assert perceiver.get_name() == "sokoban"
    for env_task in env.get_train_tasks():
        task = perceiver.reset(env_task)
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for env_task in env.get_test_tasks():
        task = perceiver.reset(env_task)
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
    env_train_tasks = env.get_train_tasks()
    assert len(env_train_tasks) == 1
    env_test_tasks = env.get_test_tasks()
    assert len(env_test_tasks) == 2
    env_task = env_test_tasks[1]
    obs = env.reset("test", 1)
    assert all(np.allclose(m1, m2) for m1, m2 in zip(obs, env_task.init_obs))
    imgs = env.render()
    assert len(imgs) == 1
    task = perceiver.reset(env_task)
    with pytest.raises(NotImplementedError):
        perceiver.render_mental_images(env_task.init_obs, env_task)
    state = task.init
    atoms = utils.abstract(state, env.predicates)
    num_boxes = len({a for a in atoms if a.predicate == IsBox})
    num_goals = len({a for a in atoms if a.predicate == IsGoal})
    assert num_boxes == num_goals
    # Hardcode a sequence of actions to achieve one example of GoalCovered.
    # This is brittle, but we don't want to plan because it could be slow
    # without Fast Downward, which is not installed on the test server.
    plan = ["PushUp", "PushUp"]
    option_names = {o.name: o for o in options}
    for name in plan:
        param_option = option_names[name]
        option = param_option.ground([], [])
        assert option.initiable(state)
        action = option.policy(state)
        obs = env.step(action)
        recovered_obs = env.get_observation()
        assert len(obs) == len(recovered_obs) == 4
        assert (np.allclose(m1, m2) for m1, m2 in zip(obs, recovered_obs))
        state = perceiver.step(obs)
        assert not env.goal_reached()
    atoms = utils.abstract(state, env.predicates)
    # Now one of the goals should be covered.
    assert len({a for a in atoms if a.predicate == GoalCovered}) == 1
    # Cover not implemented methods.
    with pytest.raises(NotImplementedError) as e:
        env.render_state_plt(obs, task)
    assert "This env does not use Matplotlib" in str(e)
    with pytest.raises(NotImplementedError) as e:
        env.render_state(obs, task)
    assert "A gym environment cannot render arbitrary states." in str(e)
    with pytest.raises(NotImplementedError) as e:
        env.simulate(obs, action)
    assert "Simulate not implemented for gym envs." in str(e)
