"""A trivial execution monitor that never suggests replanning."""

from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.structs import State, Task


class TrivialExecutionMonitor(BaseExecutionMonitor):
    """A trivial execution monitor that never suggests replanning."""

    @classmethod
    def get_name(cls) -> str:
        return "trivial"

    def reset(self, task: Task) -> None:
        pass

    def step(self, state: State) -> bool:
        return False
