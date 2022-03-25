"""Tests for PDDL blocks environments."""

from predicators.src import utils
from predicators.src.envs.pddl_env import FixedTasksBlocksPDDLEnv, \
    ProceduralTasksBlocksPDDLEnv


def test_fixed_tasks_blocks_pddl_env():
    """Tests for FixedTasksBlocksPDDLEnv class."""
    utils.reset_config({
        "env": "pddl_blocks_fixed_tasks",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = FixedTasksBlocksPDDLEnv()
    assert {t.name for t in env.types} == {"object", "block"}
    assert {p.name
            for p in env.predicates
            } == {"on", "ontable", "clear", "handempty", "holding"}
    assert {p.name for p in env.goal_predicates} == {"on"}
    assert {o.name
            for o in env.options
            } == {"pick-up", "put-down", "stack", "unstack"}
    assert {o.name
            for o in env.strips_operators
            } == {"pick-up", "put-down", "stack", "unstack"}
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 2
    test_tasks = env.get_test_tasks()
    assert len(test_tasks) == 2
    task = train_tasks[0]
    assert {a.predicate.name for a in task.goal} == {"on"}


def test_procedural_tasks_blocks_pddl_env():
    """Tests for ProceduralTasksBlocksPDDLEnv class."""
    # Note that the procedural generation itself is tested in
    # test_pddl_procedural_generation.
    utils.reset_config({
        "env": "pddl_blocks_procedural_tasks",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })
    env = ProceduralTasksBlocksPDDLEnv()
    assert {t.name for t in env.types} == {"object", "block"}
    assert {p.name
            for p in env.predicates
            } == {"on", "ontable", "clear", "handempty", "holding"}
    assert {p.name for p in env.goal_predicates} == {"on", "ontable"}
    assert {o.name
            for o in env.options
            } == {"pick-up", "put-down", "stack", "unstack"}
    assert {o.name
            for o in env.strips_operators
            } == {"pick-up", "put-down", "stack", "unstack"}
    train_tasks = env.get_train_tasks()
    assert len(train_tasks) == 2
    test_tasks = env.get_test_tasks()
    assert len(test_tasks) == 2
    task = train_tasks[0]
    assert {a.predicate.name for a in task.goal}.issubset({"on", "ontable"})
