"""A trivial perceiver that assumes observations are already states."""

from predicators.perception.base_perceiver import BasePerceiver
from predicators.structs import EnvironmentTask, Observation, State, Task


class TrivialPerceiver(BasePerceiver):
    """A trivial perceiver that assumes observations are already states."""

    @classmethod
    def get_name(cls) -> str:
        return "trivial"

    def reset(self, env_task: EnvironmentTask) -> Task:
        return env_task.task

    def step(self, observation: Observation) -> State:
        assert isinstance(observation, State)
        return observation
