"""Test cases for teacher."""

from predicators.src import utils
from predicators.src.envs import create_env
from predicators.src.ground_truth_nsrts import _get_predicates_by_names
from predicators.src.structs import (DemonstrationQuery, DemonstrationResponse,
                                     GroundAtom, GroundAtomsHoldQuery,
                                     GroundAtomsHoldResponse,
                                     LowLevelTrajectory, Task)
from predicators.src.teacher import Teacher


def test_GroundAtomsHold():
    """Tests for answering queries of type GroundAtomsHoldQuery."""
    utils.reset_config({"env": "cover", "approach": "unittest"})
    env = create_env("cover")
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
    env = create_env("cover")
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
