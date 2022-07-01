"""Test cases for the GLIB explorer class."""

import pytest

from predicators.src import utils
from predicators.src.envs.cover import CoverEnv
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.interaction import create_explorer
from predicators.src.option_model import _OracleOptionModel


@pytest.mark.parametrize("target_predicate", ["Covers", "Holding"])
def test_glib_explorer(target_predicate):
    """Tests for GLIBExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "glib",
        "cover_initial_holding_prob": 0.0,
    })
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    option_model = _OracleOptionModel(env)
    train_tasks = env.get_train_tasks()
    # For testing purposes, score everything except target predicate low.
    score_fn = lambda atoms: target_predicate in str(atoms)
    explorer = create_explorer("glib",
                               env.predicates,
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               nsrts,
                               option_model,
                               babble_predicates=env.predicates,
                               atom_score_fn=score_fn)
    task_idx = 0
    task = env.get_test_tasks()[task_idx]
    policy, termination_function = explorer.get_exploration_strategy(task, 500)
    traj, _ = utils.run_policy(
        policy,
        env,
        "test",
        task_idx,
        termination_function,
        max_num_steps=1000,
    )
    assert len(traj.actions) < 3  # should be able to quickly achieve targets
    final_state = traj.states[-1]
    assert termination_function(final_state)
    init_atoms = utils.abstract(traj.states[0], env.predicates)
    assert target_predicate not in str(init_atoms)
    final_atoms = utils.abstract(final_state, env.predicates)
    assert target_predicate in str(final_atoms)


def test_glib_explorer_failure_cases():
    """Tests failure cases for the GLIBExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "glib",
    })
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    option_model = _OracleOptionModel(env)
    train_tasks = env.get_train_tasks()
    score_fn = lambda _: 0.0
    task_idx = 0
    task = env.get_test_tasks()[task_idx]
    # Test case where there are no possible goals.
    explorer = create_explorer("glib",
                               set(),
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               nsrts,
                               option_model,
                               babble_predicates=env.predicates,
                               atom_score_fn=score_fn)
    policy, _ = explorer.get_exploration_strategy(task, 500)
    act = policy(task.init)  # a random action
    assert env.action_space.contains(act.arr)
    # Test case where no plan can be found (due to timeout).
    explorer = create_explorer("glib",
                               env.predicates,
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               nsrts,
                               option_model,
                               babble_predicates=env.predicates,
                               atom_score_fn=score_fn)
    explorer.get_exploration_strategy(task, -1)
    act = policy(task.init)  # a random action
    assert env.action_space.contains(act.arr)
