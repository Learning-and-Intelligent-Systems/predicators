"""Test cases for the base explorer class."""

import pytest

from predicators.src import utils
from predicators.src.envs.cover import CoverEnv
from predicators.src.explorers import BaseExplorer, create_explorer
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.option_model import _OracleOptionModel


def test_create_explorer():
    """Tests for create_explorer."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    nsrts = get_gt_nsrts(env.predicates, env.options)
    option_model = _OracleOptionModel(env)
    train_tasks = env.get_train_tasks()
    # Greedy lookahead explorer.
    state_score_fn = lambda _1, _2: 0.0
    name = "greedy_lookahead"
    explorer = create_explorer(name,
                               env.predicates,
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               nsrts=nsrts,
                               option_model=option_model,
                               state_score_fn=state_score_fn)
    assert isinstance(explorer, BaseExplorer)
    # GLIB explorer.
    atom_score_fn = lambda _: 0.0
    name = "glib"
    explorer = create_explorer(name,
                               env.predicates,
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               nsrts=nsrts,
                               option_model=option_model,
                               babble_predicates=env.predicates,
                               atom_score_fn=atom_score_fn)
    assert isinstance(explorer, BaseExplorer)
    # Bilevel planning explorer.
    name = "exploit_planning"
    explorer = create_explorer(name,
                               env.predicates,
                               env.options,
                               env.types,
                               env.action_space,
                               train_tasks,
                               nsrts=nsrts,
                               option_model=option_model)
    assert isinstance(explorer, BaseExplorer)
    # Basic explorers.
    for name in [
            "random_actions",
            "random_options",
    ]:
        explorer = create_explorer(name, env.predicates, env.options,
                                   env.types, env.action_space, train_tasks)
        assert isinstance(explorer, BaseExplorer)
    # Failure case.
    with pytest.raises(NotImplementedError):
        create_explorer("Not a real explorer", env.predicates, env.options,
                        env.types, env.action_space, train_tasks)
