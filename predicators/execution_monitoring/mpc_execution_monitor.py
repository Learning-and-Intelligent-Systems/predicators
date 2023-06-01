"""A model-predictive control monitor that always suggests replanning."""
import logging

from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.structs import State, Task


class MpcExecutionMonitor(BaseExecutionMonitor):
    """A model-predictive control monitor that always suggests replanning."""

    def __init__(self) -> None:
        super().__init__()
        self._curr_plan_timestep = 0

    @classmethod
    def get_name(cls) -> str:
        return "mpc"

    def reset(self, task: Task) -> None:
        logging.info("EXECUTION MONITOR RESET")
        self._curr_plan_timestep = 0

    def step(self, state: State) -> bool:
        # Don't trigger replanning on the 0th
        # timestep.
        if self._curr_plan_timestep == 0:
            self._curr_plan_timestep += 1
            return False
        # Otherwise, trigger replanning.
        return True
