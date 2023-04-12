"""A model-predictive control monitor that always suggests replanning."""

from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.structs import State, Task


class MpcExecutionMonitor(BaseExecutionMonitor):
    """A model-predictive control monitor that always suggests replanning."""

    @classmethod
    def get_name(cls) -> str:
        return "mpc"

    def reset(self, task: Task) -> None:
        pass

    def step(self, state: State) -> bool:
        return True
