"""Tests for execution monitors."""

import pytest

from predicators.execution_monitoring import create_execution_monitor
from predicators.execution_monitoring.mpc_execution_monitor import \
    MpcExecutionMonitor
from predicators.execution_monitoring.trivial_execution_monitor import \
    TrivialExecutionMonitor


def test_create_execution_monitor():
    """Tests for create_execution_monitor()."""
    exec_monitor = create_execution_monitor("trivial")
    assert isinstance(exec_monitor, TrivialExecutionMonitor)

    exec_monitor = create_execution_monitor("mpc")
    assert isinstance(exec_monitor, MpcExecutionMonitor)

    with pytest.raises(NotImplementedError) as e:
        create_execution_monitor("not a real monitor")
    assert "Unrecognized execution monitor" in str(e)
