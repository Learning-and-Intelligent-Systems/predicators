"""Test cases for teacher."""

import pytest

from predicators.src import utils
from predicators.src.envs import create_new_env
from predicators.src.ground_truth_nsrts import _get_predicates_by_names
from predicators.src.structs import DemonstrationQuery, \
    DemonstrationResponse, GroundAtom, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, InteractionRequest, LowLevelTrajectory, \
    PathToStateQuery, PathToStateResponse, Query, Task
from predicators.src.teacher import Teacher, TeacherInteractionMonitor, \
    TeacherInteractionMonitorWithVideo


def test_GroundAtomsHold():
    """Tests for answering queries of type GroundAtomsHoldQuery."""
    utils.reset_config({"env": "cover", "approach": "unittest"})
    env = create_new_env("cover")
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    state = env.get_train_tasks()[0].init
    block_type = [t for t in env.types if t.name == "block"][0]
    target_type = [t for t in env.types if t.name == "target"][0]
    block = block_type("block0")
    target = target_type("target0")
    Covers, IsBlock = _get_predicates_by_names("cover", ["Covers", "IsBlock"])
    Covers = utils.strip_predicate(Covers)
    IsBlock = utils.strip_predicate(IsBlock)
    is_block_block = GroundAtom(IsBlock, [block])
    query = GroundAtomsHoldQuery({is_block_block})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert response.holds[is_block_block]
    covers_block_target = GroundAtom(Covers, [block, target])
    query = GroundAtomsHoldQuery({covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert not response.holds[covers_block_target]
    query = GroundAtomsHoldQuery({covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 1
    assert not response.holds[covers_block_target]
    query = GroundAtomsHoldQuery({is_block_block, covers_block_target})
    response = teacher.answer_query(state, query)
    assert isinstance(response, GroundAtomsHoldResponse)
    assert response.query is query
    assert len(response.holds) == 2
    assert response.holds[is_block_block]
    assert not response.holds[covers_block_target]


def test_DemonstrationQuery():
    """Tests for answering queries of type DemonstrationQuery."""
    utils.reset_config({"env": "cover", "approach": "unittest"})
    env = create_new_env("cover")
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    train_task_idx = 0
    task = train_tasks[train_task_idx]
    state = task.init
    goal = task.goal
    # Test normal usage
    query = DemonstrationQuery(train_task_idx)
    response = teacher.answer_query(state, query)
    assert isinstance(response, DemonstrationResponse)
    assert response.query is query
    assert isinstance(response.teacher_traj, LowLevelTrajectory)
    assert len(response.teacher_traj.actions) == 2
    assert all(atom.holds(response.teacher_traj.states[-1]) for atom in goal)
    assert response.teacher_traj.is_demo
    assert response.teacher_traj.train_task_idx == train_task_idx
    # Test usage when goal is already achieved
    response = teacher.answer_query(response.teacher_traj.states[-1], query)
    assert isinstance(response, DemonstrationResponse)
    assert response.query is query
    assert isinstance(response.teacher_traj, LowLevelTrajectory)
    assert len(response.teacher_traj.actions) == 0
    # Test usage when achieving goal is impossible
    state = train_tasks[0].init
    block = [b for b in state if b.name == "block0"][0]
    IsBlock = _get_predicates_by_names("cover", ["IsBlock"])[0]
    IsBlock = utils.strip_predicate(IsBlock)
    NotIsBlock = IsBlock.get_negation()
    not_is_block_block = GroundAtom(NotIsBlock, [block])
    impossible_task = Task(state, {not_is_block_block})
    train_tasks = [impossible_task]
    teacher = Teacher(train_tasks)
    query = DemonstrationQuery(train_task_idx=0)
    response = teacher.answer_query(state, query)
    assert isinstance(response, DemonstrationResponse)
    assert response.query is query
    assert response.teacher_traj is None


def test_PathToStateQuery():
    """Tests for answering queries of type PathToStateQuery."""
    utils.reset_config({
        "env": "cover_multistep_options",
        "approach": "unittest",
        "cover_multistep_thr_percent": 0.99,
        "cover_multistep_bhr_percent": 0.99
    })
    # Test normal usage.
    env = create_new_env("cover_multistep_options")
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    train_task_idx = 0
    task = train_tasks[train_task_idx]
    state = task.init
    block0, _, _, _, robby, _, _, _, _ = state
    # Create a goal state where robby has just picked block0.
    goal_state = state.copy()
    goal_state.set(block0, "grasp", 1)
    goal_state.set(robby, "x", 0.9095)
    goal_state.set(robby, "y", 0.1)
    goal_state.set(robby, "grip", 1)
    goal_state.set(robby, "holding", 1)
    query = PathToStateQuery(goal_state)
    assert query.cost == 1
    response = teacher.answer_query(state, query)
    assert isinstance(response, PathToStateResponse)
    assert response.query is query
    assert isinstance(response.teacher_traj, LowLevelTrajectory)
    assert len(response.teacher_traj.states) == 3
    assert len(response.teacher_traj.actions) == 2
    final_state = response.teacher_traj.states[-1]
    assert final_state.allclose(goal_state)
    # Although this is a good trajectory, demonstrations are expected to reach
    # task goals, which this is not guaranteed to do.
    assert not response.teacher_traj.is_demo
    # Trajectory should not contain options.
    for action in response.teacher_traj.actions:
        assert not action.has_option()
    # Reset the state to be the previous goal state.
    state = goal_state.copy()
    # Create a new goal state where robby has just placed block0 on target0.
    goal_state = state.copy()
    goal_state.set(block0, "grasp", -1)
    goal_state.set(block0, "x", 0.7339)
    goal_state.set(block0, "y", 0.1)
    goal_state.set(robby, "x", 0.7587)
    goal_state.set(robby, "y", 0.101)
    goal_state.set(robby, "grip", -1)
    goal_state.set(robby, "holding", -1)
    query = PathToStateQuery(goal_state)
    response = teacher.answer_query(state, query)
    assert isinstance(response, PathToStateResponse)
    assert response.query is query
    assert isinstance(response.teacher_traj, LowLevelTrajectory)
    assert len(response.teacher_traj.states) == 4
    assert len(response.teacher_traj.actions) == 3
    final_state = response.teacher_traj.states[-1]
    assert final_state.allclose(goal_state)
    # Test that no trajectory is returned when the query goal state cannot
    # be reached at all.
    utils.update_config({"max_num_steps_option_rollout": 2})
    query = PathToStateQuery(goal_state)
    response = teacher.answer_query(state, query)
    assert response.teacher_traj is None
    # Test that no trajectory is returned when the query would require two
    # options to complete.
    state = task.init
    query = PathToStateQuery(goal_state)
    response = teacher.answer_query(state, query)
    assert response.teacher_traj is None
    # Test that an error is raised when an unsupported environment is used.
    utils.reset_config({"env": "painting", "approach": "unittest"})
    env = create_new_env("painting")
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    task = train_tasks[0]
    state = task.init
    query = PathToStateQuery(state)
    with pytest.raises(NotImplementedError):
        teacher.answer_query(state, query)


def test_TeacherInteractionMonitor():
    """Tests for TeacherInteractionMonitor()."""
    utils.reset_config({
        "env": "cover",
        "cover_initial_holding_prob": 0.0,
        "approach": "unittest",
        "timeout": 1,
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "num_online_learning_cycles": 1
    })
    env = create_new_env("cover")
    HandEmpty = [p for p in env.predicates if p.name == "HandEmpty"][0]
    hand_empty_atom = GroundAtom(HandEmpty, [])
    query_policy = lambda s: GroundAtomsHoldQuery({hand_empty_atom})
    act_policy = lambda _: env.action_space.sample()
    termination_function = lambda s: True  # terminate immediately
    request = InteractionRequest(0, act_policy, query_policy,
                                 termination_function)
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    monitor = TeacherInteractionMonitor(request, teacher)
    assert monitor.get_query_cost() == 0.0
    assert monitor.get_responses() == []
    state = train_tasks[0].init
    action = env.action_space.sample()
    monitor.observe(state, action)
    assert monitor.get_query_cost() == 1.0
    assert len(monitor.get_responses()) == 1
    state = train_tasks[0].init
    action = env.action_space.sample()
    monitor.observe(state, action)
    assert monitor.get_query_cost() == 2.0
    assert len(monitor.get_responses()) == 2
    # Cover not making queries
    env = create_new_env("cover")
    query_policy = lambda s: None
    act_policy = lambda _: env.action_space.sample()
    termination_function = lambda s: True  # terminate immediately
    request = InteractionRequest(0, act_policy, query_policy,
                                 termination_function)
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    monitor = TeacherInteractionMonitor(request, teacher)
    state = env.reset("train", 0)
    action = env.action_space.sample()
    monitor.observe(state, action)


def test_TeacherInteractionMonitorWithVideo():
    """Tests for TeacherInteractionMonitorWithVideo()."""
    utils.reset_config({
        "env": "cover",
        "cover_initial_holding_prob": 0.0,
        "approach": "unittest",
        "timeout": 1,
        "num_train_tasks": 2,
        "num_test_tasks": 1,
        "num_online_learning_cycles": 1
    })
    env = create_new_env("cover")
    HandEmpty = [p for p in env.predicates if p.name == "HandEmpty"][0]
    hand_empty_atom = GroundAtom(HandEmpty, [])
    query_policy = lambda s: GroundAtomsHoldQuery({hand_empty_atom})
    act_policy = lambda _: env.action_space.sample()
    termination_function = lambda s: True  # terminate immediately
    request = InteractionRequest(0, act_policy, query_policy,
                                 termination_function)
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    monitor = TeacherInteractionMonitorWithVideo(env.render, request, teacher)
    assert monitor.get_query_cost() == 0.0
    assert monitor.get_responses() == []
    state = train_tasks[0].init
    action = env.action_space.sample()
    monitor.observe(state, action)
    assert monitor.get_query_cost() == 1.0
    assert len(monitor.get_responses()) == 1
    # Cover not making queries and generating a video
    utils.update_config({
        "make_interaction_videos": True,
    })
    env = create_new_env("cover")
    query_policy = lambda s: None
    act_policy = lambda _: env.action_space.sample()
    termination_function = lambda s: True  # terminate immediately
    request = InteractionRequest(0, act_policy, query_policy,
                                 termination_function)
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    monitor = TeacherInteractionMonitorWithVideo(env.render, request, teacher)
    state = env.reset("train", 0)
    action = env.action_space.sample()
    monitor.observe(state, action)


def test_answer_query():
    """Tests for Teacher.answer_query()."""
    utils.reset_config({"env": "cover", "approach": "unittest"})
    env = create_new_env("cover")
    train_tasks = env.get_train_tasks()
    teacher = Teacher(train_tasks)
    state = env.get_train_tasks()[0].init

    class _MockQuery(Query):
        """Not a real Query type."""

    query = _MockQuery()
    with pytest.raises(NotImplementedError):
        teacher.answer_query(state, query)
