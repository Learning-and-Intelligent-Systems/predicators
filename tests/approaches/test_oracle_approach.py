"""Test cases for the oracle approach class.
"""
from absl import flags
import ml_collections
import numpy as np
import pytest
from predicators.configs.envs import cover_config
from predicators.src.approaches import OracleApproach
from predicators.src.approaches.oracle_approach import _get_gt_ops
from predicators.src.envs import CoverEnv
from predicators.src import utils


def test_cover_get_gt_ops():
    """Tests for _get_gt_ops in CoverEnv.
    """
    # All predicates and options
    flags.env = cover_config.get_config()
    env = CoverEnv()
    operators = _get_gt_ops(env.predicates, env.options)
    assert len(operators) == 2
    pick_operator, place_operator = sorted(operators, key=lambda o: o.name)
    assert pick_operator.name == "Pick"
    assert place_operator.name == "Place"
    env.seed(123)
    train_task = env.get_train_tasks()[0]
    state = train_task.init
    block0, _, _, target0, _ = sorted(state.data.keys())
    assert block0.name == "block0"
    assert target0.name == "target0"
    pick0_operator = pick_operator.ground([block0])
    rng = np.random.RandomState(123)
    pick_param = pick0_operator.sampler(state, rng)
    pick_option = pick0_operator.option.ground(pick_param)
    pick_action = pick_option.policy(state)
    assert env.action_space.contains(pick_action)
    state = env.simulate(state, pick_action)
    place0_operator = place_operator.ground([block0, target0])
    place_param = place0_operator.sampler(state, rng)
    place_option = place0_operator.option.ground(place_param)
    place_action = place_option.policy(state)
    assert env.action_space.contains(place_action)
    # Excluded option
    assert _get_gt_ops(env.predicates, set()) == set()
    # Excluded predicate
    predicates = {p for p in env.predicates if p.name != "Holding"}
    operators = _get_gt_ops(predicates, env.options)
    assert len(operators) == 2
    pick_operator, place_operator = sorted(operators, key=lambda o: o.name)
    for atom in pick_operator.preconditions:
        assert atom.predicate.name != "Holding"
    assert len(pick_operator.add_effects) == 0
    for atom in pick_operator.delete_effects:
        assert atom.predicate.name != "Holding"


def test_get_gt_ops():
    """Test get gt ops alone.
    """
    flags.env = ml_collections.ConfigDict()
    flags.env.name = "Not a real environment"
    with pytest.raises(NotImplementedError):
        _get_gt_ops(set(), set())


def test_oracle_approach():
    """Tests for OracleApproach class.
    """
    flags.env = cover_config.get_config()
    env = CoverEnv()
    env.seed(123)
    approach = OracleApproach(env.simulate, env.predicates, env.options,
                              env.action_space)
    approach.seed(123)
    for task in env.get_train_tasks():
        policy = approach.solve(task, timeout=500)
        _check_policy(task, env.simulate, env.predicates, policy)
    for task in env.get_test_tasks():
        policy = approach.solve(task, timeout=500)
        _check_policy(task, env.simulate, env.predicates, policy)


def _check_policy(task, simulator, predicates, policy):
    state = task.init
    atoms = utils.abstract(state, predicates)
    assert not task.goal.issubset(atoms)  # goal shouldn't be already satisfied
    for _ in range(100):
        act = policy(state)
        state = simulator(state, act)
        atoms = utils.abstract(state, predicates)
        if task.goal.issubset(atoms):
            break
    assert task.goal.issubset(atoms)
