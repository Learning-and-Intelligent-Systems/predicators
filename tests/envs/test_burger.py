"""Test cases for the burger environment."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

from predicators import utils
from predicators.envs.burger import BurgerEnv, BurgerNoMoveEnv
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.settings import CFG
from predicators.structs import Action, GroundAtom


def test_burger():
    """Tests for BurgerEnv."""

    utils.reset_config({
        "env": "burger",
        "option_model_terminate_on_repeat": False,
        "sesame_max_skeletons_optimized": 1000,
        "sesame_max_samples_per_step": 1,
        "sesame_task_planner": "fdopt",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "seed": 0
    })
    env = BurgerEnv()
    # This should really test this for however many train/test tasks we will use
    # in our experiments (e.g 50 for each instead of 1), but because the image
    # rendering is so slow, I set this to 1 here.
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 12
    assert len(env.goal_predicates) == 3
    assert len(env.agent_goal_predicates) == 2
    assert env.get_name() == "burger"
    assert len(env.types) == 11
    options = get_gt_options(env.get_name())
    assert len(options) == 5
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 15
    task = env.get_train_tasks()[0]
    # The nsrts that are commented out are used when the "full" goal is used
    # in the tasks. Because we are still getting predicate invention to work
    # in BurgerEnv, we are not yet using the "full" goal. Because the full goal
    # won't be in the environment's task's goals, we won't test a plan that
    # achieves that goal in the tests for now.
    MoveWhenFacingOneStack = [
        n for n in nsrts if n.name == "MoveWhenFacingOneStack"
    ][0]
    MoveWhenFacingTwoStack = [
        n for n in nsrts if n.name == "MoveWhenFacingTwoStack"
    ][0]
    # MoveWhenFacingThreeStack = [
    #     n for n in nsrts if n.name == "MoveWhenFacingThreeStack"
    # ][0]
    # MoveWhenFacingFourStack = [
    #     n for n in nsrts if n.name == "MoveWhenFacingFourStack"
    # ][0]
    MoveFromNothingToOneStack = [
        n for n in nsrts if n.name == "MoveFromNothingToOneStack"
    ][0]
    MoveFromNothingToTwoStack = [
        n for n in nsrts if n.name == "MoveFromNothingToTwoStack"
    ][0]
    # MoveFromNothingToFourStack = [
    #     n for n in nsrts if n.name == "MoveFromNothingToFourStack"
    # ][0]
    # MoveFromOneStackToThreeStack = [
    #     n for n in nsrts if n.name == "MoveFromOneStackToThreeStack"
    # ][0]
    PickSingleAdjacent = [n for n in nsrts
                          if n.name == "PickSingleAdjacent"][0]
    PickFromStack = [n for n in nsrts if n.name == "PickFromStack"][0]
    Place = [n for n in nsrts if n.name == "Place"][0]
    Cook = [n for n in nsrts if n.name == "Cook"][0]
    Slice = [n for n in nsrts if n.name == "Slice"][0]

    grill = [obj for obj in task.init if obj.name == "grill"][0]
    patty = [obj for obj in task.init if obj.name == "patty"][0]
    robot = [obj for obj in task.init if obj.name == "robot"][0]
    tomato = [obj for obj in task.init if obj.name == "lettuce"][0]
    cutting_board = [obj for obj in task.init
                     if obj.name == "cutting_board"][0]
    cheese = [obj for obj in task.init if obj.name == "cheese"][0]
    top_bun = [obj for obj in task.init if obj.name == "top_bun"][0]
    bottom_bun = [obj for obj in task.init if obj.name == "bottom_bun"][0]

    plan = [
        MoveFromNothingToOneStack.ground([robot, tomato]),
        PickSingleAdjacent.ground([robot, tomato]),
        MoveFromNothingToOneStack.ground([robot, cutting_board]),
        Place.ground([robot, tomato, cutting_board]),
        Slice.ground([robot, tomato, cutting_board]),
        MoveWhenFacingTwoStack.ground([robot, patty, tomato, cutting_board]),
        PickSingleAdjacent.ground([robot, patty]),
        MoveFromNothingToOneStack.ground([robot, grill]),
        Place.ground([robot, patty, grill]),
        Cook.ground([robot, patty, grill]),
        PickFromStack.ground([robot, patty, grill]),
        MoveWhenFacingOneStack.ground([robot, bottom_bun, grill]),
        Place.ground([robot, patty, bottom_bun]),
        MoveWhenFacingTwoStack.ground([robot, cheese, patty, bottom_bun]),
        PickSingleAdjacent.ground([robot, cheese]),
        MoveFromNothingToTwoStack.ground([robot, patty, bottom_bun]),
        Place.ground([robot, cheese, patty])
    ]

    # plan = [
    #     MoveWhenFacingOneStack.ground([robot, patty, grill]),
    #     PickSingleAdjacent.ground([robot, patty]),
    #     MoveFromNothingToOneStack.ground([robot, grill]),
    #     Place.ground([robot, patty, grill]),
    #     Cook.ground([robot, patty, grill]),
    #     PickFromStack.ground([robot, patty, grill]),
    #     MoveWhenFacingOneStack.ground([robot, bottom_bun, grill]),
    #     Place.ground([robot, patty, bottom_bun]),
    #     MoveWhenFacingTwoStack.ground([robot, cheese, patty, bottom_bun]),
    #     PickSingleAdjacent.ground([robot, cheese]),
    #     MoveFromNothingToTwoStack.ground([robot, patty, bottom_bun]),
    #     Place.ground([robot, cheese, patty]),
    #     MoveWhenFacingThreeStack.ground(
    #         [robot, tomato, cheese, patty, bottom_bun]),
    #     PickSingleAdjacent.ground([robot, tomato]),
    #     MoveFromNothingToOneStack.ground([robot, cutting_board]),
    #     Place.ground([robot, tomato, cutting_board]),
    #     Slice.ground([robot, tomato, cutting_board]),
    #     PickFromStack.ground([robot, tomato, cutting_board]),
    #     MoveFromOneStackToThreeStack.ground(
    #         [robot, cheese, patty, bottom_bun, cutting_board]),
    #     Place.ground([robot, tomato, cheese]),
    #     MoveWhenFacingFourStack.ground(
    #         [robot, top_bun, tomato, cheese, patty, bottom_bun]),
    #     PickSingleAdjacent.ground([robot, top_bun]),
    #     MoveFromNothingToFourStack.ground(
    #         [robot, tomato, cheese, patty, bottom_bun]),
    #     Place.ground([robot, top_bun, tomato])
    # ]

    option_plan = [n.option.ground(n.option_objs, []) for n in plan]
    policy = utils.option_plan_to_policy(option_plan)
    traj, _ = utils.run_policy(policy,
                               env,
                               "train",
                               0,
                               termination_function=lambda s: False,
                               max_num_steps=CFG.horizon,
                               exceptions_to_break_on={
                                   utils.OptionExecutionFailure,
                                   utils.HumanDemonstrationFailure,
                               },
                               monitor=None)
    assert task.task.goal_holds(traj.states[-1])

    # Test _AdjacentToNothing_holds
    state = task.init.copy()
    abstract_state = utils.abstract(state, env.predicates)
    AdjacentToNothing = [
        p for p in env.predicates if p.name == "AdjacentToNothing"
    ][0]
    assert GroundAtom(AdjacentToNothing, [robot]) in abstract_state

    # Test _OnNothing_holds
    OnNothing = [p for p in env.predicates if p.name == "OnNothing"][0]
    assert GroundAtom(OnNothing,
                      [cheese]) not in utils.abstract(traj.states[-1],
                                                      env.predicates)

    # Test _GoalHack_holds
    GoalHack = [p for p in env.predicates if p.name == "GoalHack"][0]
    assert GroundAtom(GoalHack, [bottom_bun, patty, cheese, tomato, top_bun
                                 ]) in utils.abstract(traj.states[-1],
                                                      env.predicates)

    # Test _Clear_holds -- that a held object is not clear.
    Clear = [p for p in env.predicates if p.name == "Clear"][0]
    assert GroundAtom(Clear,
                      [patty]) not in utils.abstract(traj.states[9],
                                                     env.predicates)

    # Test get_cell_in_direction
    x, y = env.get_cell_in_direction(1, 1, "left")
    assert x == 0 and y == 1
    x, y = env.get_cell_in_direction(1, 1, "right")
    assert x == 2 and y == 1
    x, y = env.get_cell_in_direction(1, 1, "up")
    assert x == 1 and y == 2
    x, y = env.get_cell_in_direction(1, 1, "down")
    assert x == 1 and y == 0
    x, y = env.get_cell_in_direction(1, 1, "no_change")
    assert x == 1 and y == 1

    # Test collision
    state.set(robot, "col", 3)  # robot is at (2, 3))
    action = Action(np.array([1, 0, -1, 0, 0], dtype=np.float32))
    next_state = env.simulate(state, action)
    assert env.get_position(robot,
                            next_state) == env.get_position(robot, state)

    # Test placing on the ground
    state = traj.states[2]
    action = Action(np.array([0, 0, -1, 0, 1], dtype=np.float32))
    next_state = env.simulate(state, action)
    assert env.get_position(tomato, next_state) == env.get_position(
        tomato, traj.states[0])
    assert next_state.get(tomato, "z") == 0

    # Test rendering
    env.render_state_plt(traj.states[0], task)
    env.render_state_plt(traj.states[5], task)
    env.render_state_plt(traj.states[-1], task)
    # Test labeling when a cutting board is not in the state.
    no_cutting_board_state = traj.states[12].copy()
    del no_cutting_board_state.data[env._cutting_board]  # pylint: disable=protected-access
    env.render_state_plt(no_cutting_board_state, task)

    # Test interface for collecting demonstrations
    event_to_action = env.get_event_to_action_fn()
    fig = plt.figure()
    event = matplotlib.backend_bases.KeyEvent("test", fig.canvas, "asdf")
    assert isinstance(event_to_action(state, event), Action)
    event = matplotlib.backend_bases.KeyEvent("test", fig.canvas, "q")
    with pytest.raises(utils.HumanDemonstrationFailure):
        event_to_action(state, event)
    for key in ["w", "a", "s", "d", "left", "right", "down", "up", "e", "f"]:
        event = matplotlib.backend_bases.KeyEvent("test", fig.canvas, key)
        event_to_action(state, event)
    plt.close()

    # Test move option when already adjacent but not facing
    state = task.init.copy()
    state.set(grill, "col", 2)
    state.set(grill, "row", 3)
    Move = [o for o in options if o.name == "Move"][0]
    option = Move.ground([robot, grill], [])
    assert option.initiable(state)
    action = option.policy(state)
    next_state = env.step(action)
    assert next_state.get(robot, "dir") == 0

    state = task.init.copy()
    state.set(grill, "col", 2)
    state.set(grill, "row", 1)
    Move = [o for o in options if o.name == "Move"][0]
    option = Move.ground([robot, grill], [])
    assert option.initiable(state)
    action = option.policy(state)
    next_state = env.step(action)
    assert next_state.get(robot, "dir") == 2

    state = task.init.copy()
    state.set(grill, "col", 1)
    state.set(grill, "row", 2)
    Move = [o for o in options if o.name == "Move"][0]
    option = Move.ground([robot, grill], [])
    assert option.initiable(state)
    action = option.policy(state)
    next_state = env.step(action)
    assert next_state.get(robot, "dir") == 1

    state = task.init.copy()
    state.set(grill, "col", 3)
    state.set(grill, "row", 2)
    state.set(robot, "dir", 1)
    Move = [o for o in options if o.name == "Move"][0]
    option = Move.ground([robot, grill], [])
    assert option.initiable(state)
    action = option.policy(state)
    next_state = env.step(action)
    assert next_state.get(robot, "dir") == 3

    # Test _get_edge_cells_for_object_placement()
    # This isn't a real test because we aren't going to verify that the
    # edge cells that this function outputs satisfy all the constraints we want
    # them to satisy (edge cells such that the robot will never be adjacent to
    # more than one of these edge cells at any time).
    rng = np.random.default_rng(0)
    # 50 is an arbitrary number here. We just want to call this function many
    # times to get all possible outcomes to happen at least once for coverage
    # purposes.
    for _ in range(50):
        _ = env.get_edge_cells_for_object_placement(rng)


def test_burger_no_move():
    """Tests for BurgerNoMoveEnv."""

    utils.reset_config({
        "env": "burger_no_move",
        "option_model_terminate_on_repeat": False,
        "sesame_max_skeletons_optimized": 1000,
        "sesame_max_samples_per_step": 1,
        "sesame_task_planner": "fdopt",
        "burger_no_move_task_type": "combo_burger",
        "burger_dummy_render": True,
        "seed": 0,
        "num_train_tasks": 4,
    })
    env = BurgerNoMoveEnv()
    task = env.get_test_tasks()[0]
    assert len(env.predicates) == 13
    assert len(env.goal_predicates) == 7
    assert len(env.agent_goal_predicates) == 7
    assert len(env.get_vlm_debug_atom_strs([])) == 3
    rng = np.random.default_rng(0)
    assert len(env.get_edge_cells_for_object_placement(
        rng)) == (env.num_cols - 2) * 2 + (env.num_rows - 2) * 2

    # Test _OnGround_holds
    s = task.init
    patty = [o for o in s if o.name == "patty1"][0]
    OnGround = [p for p in env.predicates if p.name == "OnGround"][0]
    assert GroundAtom(OnGround, [patty]) in utils.abstract(s, env.predicates)

    # Test the GoalHacks, but not that rigorously, because we will likely
    # change their definitions.
    abstract_state = utils.abstract(s, env.predicates)
    goalhacks_to_check = ["GoalHack2", "GoalHack3", "GoalHack4", "GoalHack5"]
    for atom in abstract_state:
        assert atom.predicate.name not in goalhacks_to_check

    assert env.get_name() == "burger_no_move"
    options = get_gt_options(env.get_name())
    assert len(options) == 4
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    assert len(nsrts) == 5

    Cook = [n for n in nsrts if n.name == "Cook"][0]
    Slice = [n for n in nsrts if n.name == "Slice"][0]
    PickFromGround = [n for n in nsrts if n.name == "PickFromGround"][0]
    Unstack = [n for n in nsrts if n.name == "Unstack"][0]
    Stack = [n for n in nsrts if n.name == "Stack"][0]

    grill = [obj for obj in s if obj.name == "grill"][0]
    patty = [obj for obj in s if obj.name == "patty1"][0]
    robot = [obj for obj in s if obj.name == "robot"][0]
    lettuce = [obj for obj in s if obj.name == "lettuce1"][0]
    cutting_board = [obj for obj in s if obj.name == "cutting_board"][0]
    top_bun = [obj for obj in s if obj.name == "top_bun1"][0]
    bottom_bun = [obj for obj in s if obj.name == "bottom_bun1"][0]

    plan = [
        PickFromGround.ground((robot, patty)),
        Stack.ground((robot, patty, grill)),
        Cook.ground((robot, patty, grill)),
        Unstack.ground((robot, patty, grill)),
        Stack.ground((robot, patty, bottom_bun)),
        PickFromGround.ground((robot, lettuce)),
        Stack.ground((robot, lettuce, cutting_board)),
        Slice.ground((robot, lettuce, cutting_board)),
        Unstack.ground((robot, lettuce, cutting_board)),
        Stack.ground((robot, lettuce, patty)),
        PickFromGround.ground((robot, top_bun)),
        Stack.ground((robot, top_bun, lettuce))
    ]

    option_plan = [n.option.ground(n.option_objs, []) for n in plan]
    policy = utils.option_plan_to_policy(option_plan)
    traj, _ = utils.run_policy(policy,
                               env,
                               "test",
                               0,
                               termination_function=lambda s: False,
                               max_num_steps=CFG.horizon,
                               exceptions_to_break_on={
                                   utils.OptionExecutionFailure,
                                   utils.HumanDemonstrationFailure,
                               },
                               monitor=None)
    assert task.task.goal_holds(traj.states[-1])

    # Coverage for move part of slice policy.
    # Put the lettuce on the cutting board and have the robot move to it when
    # slicing.
    s.set(lettuce, "row", 0)
    s.set(lettuce, "col", 3)
    s.set(lettuce, "z", 1)
    plan = [Slice.ground((robot, lettuce, cutting_board))]
    option_plan = [n.option.ground(n.option_objs, []) for n in plan]
    policy = utils.option_plan_to_policy(option_plan)
    traj, _ = utils.run_policy(policy,
                               env,
                               "test",
                               0,
                               termination_function=lambda s: False,
                               max_num_steps=CFG.horizon,
                               exceptions_to_break_on={
                                   utils.OptionExecutionFailure,
                                   utils.HumanDemonstrationFailure,
                               },
                               monitor=None)
    assert traj.states[-1].get(robot, "row") == 1
    assert traj.states[-1].get(robot, "col") == 3

    # Coverage for move part of cook policy.
    # Put the patty on the grill and have the robot move to it when cooking.
    s.set(patty, "row", 0)
    s.set(patty, "col", 2)
    s.set(patty, "z", 1)
    plan = [Cook.ground((robot, patty, grill))]
    option_plan = [n.option.ground(n.option_objs, []) for n in plan]
    policy = utils.option_plan_to_policy(option_plan)
    traj, _ = utils.run_policy(policy,
                               env,
                               "test",
                               0,
                               termination_function=lambda s: False,
                               max_num_steps=CFG.horizon,
                               exceptions_to_break_on={
                                   utils.OptionExecutionFailure,
                                   utils.HumanDemonstrationFailure,
                               },
                               monitor=None)
    assert traj.states[-1].get(robot, "row") == 1
    assert traj.states[-1].get(robot, "col") == 2

    utils.reset_config({
        "env": "burger_no_move",
        "option_model_terminate_on_repeat": False,
        "sesame_max_skeletons_optimized": 1000,
        "sesame_max_samples_per_step": 1,
        "sesame_task_planner": "fdopt",
        "burger_no_move_task_type": "more_stacks",
        "burger_dummy_render": True,
        "seed": 0
    })
    env = BurgerNoMoveEnv()
    task = env.get_test_tasks()[0]

    utils.reset_config({
        "env": "burger_no_move",
        "option_model_terminate_on_repeat": False,
        "sesame_max_skeletons_optimized": 1000,
        "sesame_max_samples_per_step": 1,
        "sesame_task_planner": "fdopt",
        "burger_no_move_task_type": "fatter_burger",
        "burger_dummy_render": True,
        "seed": 0,
        "num_train_tasks": 3,
    })
    env = BurgerNoMoveEnv()
    task = env.get_train_tasks()[0]

    utils.reset_config({
        "env": "burger_no_move",
        "option_model_terminate_on_repeat": False,
        "sesame_max_skeletons_optimized": 1000,
        "sesame_max_samples_per_step": 1,
        "sesame_task_planner": "fdopt",
        "burger_no_move_task_type": "fake",
        "burger_dummy_render": True,
        "seed": 0
    })
    env = BurgerNoMoveEnv()
    with pytest.raises(NotImplementedError) as e:
        task = env.get_test_tasks()[0]
    assert "Unrecognized task type: fake." in str(e)
