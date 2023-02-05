"""Test cases for the sandwich env."""

from predicators import utils
from predicators.envs.sandwich import SandwichEnv
from predicators.structs import GroundAtom


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


def test_sandwich_options():
    """Tests for sandwich parameterized options, predicates, and rendering."""
    # Set up environment
    utils.reset_config({
        "env": "sandwich",
        # "render_state_dpi": 150,  # uncomment for higher-res test videos
    })
    env = SandwichEnv()
    BoardClear, Clear, GripperOpen, _, InHolder, _, _, _, _, _, _, _, _, On, \
        OnBoard = sorted(env.predicates)
    Pick, PutOnBoard, Stack = sorted(env.options)
    board_type, holder_type, _, robot_type = sorted(env.types)

    task = env.get_train_tasks()[0]
    state = task.init
    obj_name_to_obj = {o.name: o for o in state}
    # Select one cuboid and one cylinder to cover the different rendering cases
    ing0 = obj_name_to_obj["bread0"]
    ing1 = obj_name_to_obj["tomato0"]
    robot, = state.get_objects(robot_type)
    board, = state.get_objects(board_type)
    holder, = state.get_objects(holder_type)

    # Test a successful trajectory involving all the options
    option_plan = [
        Pick.ground([robot, ing0], []),
        PutOnBoard.ground([robot, board], []),
        Pick.ground([robot, ing1], []),
        Stack.ground([robot, ing0], [])
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
    assert len(video) == 5  # each option just takes 1 step
    # outfile = "hardcoded_options_sandwich.mp4"
    # utils.save_video(outfile, video)
    init_state = traj.states[0]
    final_state = traj.states[-1]

    assert GroundAtom(BoardClear, [board]).holds(init_state)
    assert not GroundAtom(BoardClear, [board]).holds(final_state)
    assert GroundAtom(GripperOpen, [robot]).holds(init_state)
    assert GroundAtom(GripperOpen, [robot]).holds(final_state)
    assert GroundAtom(InHolder, [ing0, holder]).holds(init_state)
    assert not GroundAtom(InHolder, [ing0, holder]).holds(final_state)
    assert GroundAtom(InHolder, [ing1, holder]).holds(init_state)
    assert not GroundAtom(InHolder, [ing1, holder]).holds(final_state)
    assert not GroundAtom(On, [ing1, ing0]).holds(init_state)
    assert GroundAtom(On, [ing1, ing0]).holds(final_state)
    assert not GroundAtom(OnBoard, [ing0, board]).holds(init_state)
    assert GroundAtom(OnBoard, [ing0, board]).holds(final_state)
    assert not GroundAtom(Clear, [ing1]).holds(init_state)
    assert GroundAtom(Clear, [ing1]).holds(final_state)
    assert not GroundAtom(Clear, [ing0]).holds(final_state)
