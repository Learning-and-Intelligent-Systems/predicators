"""A trivial perceiver that assumes observations are already states."""

from predicators.envs import get_or_create_env
from predicators.perception.base_perceiver import BasePerceiver
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Observation, State, Task, \
    Video


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

    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        # Use the environment's render function by default.
        assert isinstance(observation, State)
        env = get_or_create_env(CFG.env)
        return env.render_state(observation, env_task)
