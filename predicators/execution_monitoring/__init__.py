"""Handle creation of execution monitors."""

from predicators import utils
from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor

__all__ = ["BaseExecutionMonitor"]

# Find the subclasses.
utils.import_submodules(__path__, __name__)


def create_execution_monitor(name: str, ) -> BaseExecutionMonitor:
    """Create an execution monitor given its name."""
    for cls in utils.get_all_subclasses(BaseExecutionMonitor):
        if not cls.__abstractmethods__ and cls.get_name() == name:
            execution_monitor = cls()
            break
    else:
        raise NotImplementedError(f"Unrecognized execution monitor: {name}")
    return execution_monitor
