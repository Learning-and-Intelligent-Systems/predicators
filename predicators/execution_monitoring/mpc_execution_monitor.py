"""A model-predictive control monitor that always suggests replanning."""
from predicators.execution_monitoring.base_execution_monitor import \
    BaseExecutionMonitor
from predicators.structs import State


class MpcExecutionMonitor(BaseExecutionMonitor):
    """A model-predictive control monitor that always suggests replanning."""

    @classmethod
    def get_name(cls) -> str:
        return "mpc"

    def step(self, state: State) -> bool:
        # Don't trigger replanning on the 0th
        # timestep.
        if self._curr_plan_timestep < 50:
            self._curr_plan_timestep += 1
            return False
        # Otherwise, trigger replanning.
        return True
