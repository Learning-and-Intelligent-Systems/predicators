"""Test cases for the PG4 approach."""

import pytest

from predicators.src import utils
from predicators.src.approaches import ApproachFailure, ApproachTimeout
from predicators.src.approaches.pg4_approach import PG4Approach
from predicators.src.envs import create_new_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.structs import Task


def test_pg4_approach():
    """Tests for PG4Approach().

    Additional tests are in test_pg3_approach().
    """
    env_name = "pddl_easy_delivery_procedural_tasks"
    utils.reset_config({
        "env": env_name,
        "approach": "pg4",
        "num_train_tasks": 1,
        "num_test_tasks": 1,
        "strips_learner": "oracle",
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    approach = PG4Approach(env.predicates, env.options, env.types,
                           env.action_space, train_tasks)
    nsrts = get_gt_nsrts(env.predicates, env.options)
    approach._nsrts = nsrts  # pylint: disable=protected-access
    task = train_tasks[0]

    # Test option execution failure.
    trivial_task = Task(task.init, set())
    policy = approach.solve(trivial_task, 500)
    with pytest.raises(ApproachFailure) as e:
        policy(task.init)
    assert "Option plan exhausted!" in str(e)

    # Test planning timeout.
    with pytest.raises(ApproachTimeout):
        approach.solve(task, -1)

    # Test planning failure.
    approach._nsrts = set()  # pylint: disable=protected-access
    with pytest.raises(ApproachFailure):
        approach.solve(task, 1)
