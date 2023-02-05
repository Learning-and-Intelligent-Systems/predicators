"""Test cases for the sandwich env."""

import numpy as np

from predicators import utils
from predicators.envs.sandwich import SandwichEnv
from predicators.structs import Action, GroundAtom, Task


def test_sandwich_properties():
    """Test env object initialization and properties."""
    utils.reset_config({"env": "sandwich"})
    env = SandwichEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 15
    BoardClear, Clear, GripperOpen, Holding, InHolder, IsBread, IsBurger, \
        IsCheese, IsEgg, IsGreenPepper, IsHam, IsLettuce, IsTomato, On, \
        OnBoard = sorted(env.predicates)
    assert BoardClear.name == "BoardClear"
    assert Clear.name == "Clear"
    assert GripperOpen.name == "GripperOpen"
    assert Holding.name == "Holding"
    assert InHolder.name == "InHolder"
    assert IsBread.name == "IsBread"
    assert IsBurger.name == "IsBurger"
    assert IsCheese.name == "IsCheese"
    assert IsEgg.name == "IsEgg"
    assert IsGreenPepper.name == "IsGreenPepper"
    assert IsHam.name == "IsHam"
    assert IsLettuce.name == "IsLettuce"
    assert IsTomato.name == "IsTomato"
    assert On.name == "On"
    assert OnBoard.name == "OnBoard"
    assert env.goal_predicates == {
        IsBread, IsBurger, IsCheese, IsEgg, IsGreenPepper, IsHam, IsLettuce,
        IsTomato, On, OnBoard
    }
    assert len(env.options) == 3
    Pick, PutOnBoard, Stack = sorted(env.options)
    assert Pick.name == "Pick"
    assert PutOnBoard.name == "PutOnBoard"
    assert Stack.name == "Stack"
    assert len(env.types) == 4
    board_type, holder_type, ingredient_type, robot_type = sorted(env.types)
    assert board_type.name == "board"
    assert holder_type.name == "holder"
    assert ingredient_type.name == "ingredient"
    assert robot_type.name == "robot"
    assert env.action_space.shape == (4, )
