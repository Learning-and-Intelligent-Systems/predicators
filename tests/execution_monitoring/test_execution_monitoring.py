"""Tests for execution monitors."""

import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.cogman import CogMan
from predicators.envs import get_or_create_env
from predicators.execution_monitoring import create_execution_monitor
from predicators.execution_monitoring.expected_atoms_monitor import \
    ExpectedAtomsExecutionMonitor
from predicators.execution_monitoring.mpc_execution_monitor import \
    MpcExecutionMonitor
from predicators.execution_monitoring.trivial_execution_monitor import \
    TrivialExecutionMonitor
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver


def test_create_execution_monitor():
    """Tests for create_execution_monitor()."""
    exec_monitor = create_execution_monitor("trivial")
    assert isinstance(exec_monitor, TrivialExecutionMonitor)

    exec_monitor = create_execution_monitor("mpc")
    assert isinstance(exec_monitor, MpcExecutionMonitor)

    exec_monitor = create_execution_monitor("expected_atoms")
    assert isinstance(exec_monitor, ExpectedAtomsExecutionMonitor)

    with pytest.raises(NotImplementedError) as e:
        create_execution_monitor("not a real monitor")
    assert "Unrecognized execution monitor" in str(e)


def test_expected_atoms_execution_monitor():
    """Tests for ExpectedAtomsExecutionMonitor."""
    # Test that the monitor works in an environment where options take
    # multiple steps.
    env_name = "cover_multistep_options"
    utils.reset_config({
        "env": env_name,
        "approach": "oracle",
        "bilevel_plan_without_sim": True,
    })
    env = get_or_create_env(env_name)
    options = get_gt_options(env.get_name())
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = create_approach("oracle", env.predicates, options, env.types,
                               env.action_space, train_tasks)
    perceiver = create_perceiver("trivial")
    exec_monitor = create_execution_monitor("expected_atoms")
    cogman = CogMan(approach, perceiver, exec_monitor)
    env_task = env.get_test_tasks()[0]
    cogman.reset(env_task)
    obs = env.reset("test", 0)
    # Check that the actions are not ever repeated, since re-planning should
    # cause re-sampling.
    prev_act = None
    for _ in range(10):
        act = cogman.step(obs)
        obs = env.step(act)
        if prev_act is not None:
            assert prev_act != act
        prev_act = act
