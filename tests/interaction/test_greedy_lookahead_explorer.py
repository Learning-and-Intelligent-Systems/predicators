"""Test cases for the greedy lookahead explorer class."""

import pytest

from predicators.src import utils
from predicators.src.envs.cover import CoverEnv
from predicators.src.explorers import create_explorer
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.option_model import _OracleOptionModel
from predicators.src.structs import NSRT


@pytest.mark.parametrize("target_predicate", ["Covers", "Holding"])
def test_greedy_lookahead_explorer(target_predicate):
    """Tests for GreedyLookaheadExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "greedy_lookahead",
        "cover_initial_holding_prob": 0.0,
    })
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    option_model = _OracleOptionModel(env)
    train_tasks = env.get_train_tasks()
    # For testing purposes, score everything except target predicate low.
    score_fn = lambda atoms, _: target_predicate in str(atoms)
    explorer = create_explorer("greedy_lookahead",
                               env.predicates,
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               nsrts,
                               option_model,
                               state_score_fn=score_fn)
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
        exceptions_to_break_on={utils.OptionExecutionFailure},
    )
    some_state_has_target_pred = False
    init_atoms = utils.abstract(traj.states[0], env.predicates)
    assert target_predicate not in str(init_atoms)
    for state in traj.states:
        atoms = utils.abstract(state, env.predicates)
        if target_predicate in str(atoms):
            some_state_has_target_pred = True
            break
    assert some_state_has_target_pred


def test_greedy_lookahead_explorer_failure_cases():
    """Tests failure cases for the GreedyLookaheadExplorer class."""
    utils.reset_config({
        "env": "cover",
        "explorer": "greedy_lookahead",
    })
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    option_model = _OracleOptionModel(env)
    train_tasks = env.get_train_tasks()
    state_score_fn = lambda _1, _2: 0.0
    task_idx = 0
    task = env.get_test_tasks()[task_idx]
    # Test case where we can't sample a ground NSRT.
    explorer = create_explorer("greedy_lookahead",
                               env.predicates,
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               set(),
                               option_model,
                               state_score_fn=state_score_fn)
    policy, _ = explorer.get_exploration_strategy(task, 500)
    with pytest.raises(utils.OptionExecutionFailure):
        policy(task.init)
    # Test case where the option model returns num actions 0.

    def _policy(s, memory, objects, params):
        del s, memory, objects, params  # unused
        raise utils.OptionExecutionFailure("Mock error")

    new_nsrts = set()
    for nsrt in nsrts:
        new_option = utils.SingletonParameterizedOption(
            "LearnedMockOption",
            _policy,
            types=nsrt.option.types,
            params_space=nsrt.option.params_space,
            initiable=nsrt.option.initiable)
        new_nsrt = NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                        nsrt.add_effects, nsrt.delete_effects, set(),
                        new_option, nsrt.option_vars, nsrt._sampler)  # pylint: disable=protected-access
        new_nsrts.add(new_nsrt)
    explorer = create_explorer("greedy_lookahead",
                               env.predicates,
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               new_nsrts,
                               option_model,
                               state_score_fn=state_score_fn)
    policy, _ = explorer.get_exploration_strategy(task, 500)
    with pytest.raises(utils.OptionExecutionFailure):
        policy(task.init)
