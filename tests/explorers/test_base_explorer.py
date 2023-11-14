"""Test cases for the base explorer class."""
import pytest

from predicators import utils
from predicators.envs.cover import CoverEnv
from predicators.explorers import BaseExplorer, create_explorer
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.option_model import _OracleOptionModel


def test_create_explorer():
    """Tests for create_explorer."""
    utils.reset_config({"env": "cover"})
    env = CoverEnv()
    options = get_gt_options(env.get_name())
    nsrts = get_gt_nsrts(env.get_name(), env.predicates, options)
    option_model = _OracleOptionModel(options, env.simulate)
    train_tasks = [t.task for t in env.get_train_tasks()]
    # Greedy lookahead explorer.
    state_score_fn = lambda _1, _2: 0.0
    name = "greedy_lookahead"
    explorer = create_explorer(name,
                               env.predicates,
                               get_gt_options(env.get_name()),
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
                               get_gt_options(env.get_name()),
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
                               get_gt_options(env.get_name()),
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
        explorer = create_explorer(name, env.predicates,
                                   get_gt_options(env.get_name()), env.types,
                                   env.action_space, train_tasks)
        assert isinstance(explorer, BaseExplorer)
    # Failure case.
    with pytest.raises(NotImplementedError):
        create_explorer("Not a real explorer", env.predicates,
                        get_gt_options(env.get_name()), env.types,
                        env.action_space, train_tasks)
