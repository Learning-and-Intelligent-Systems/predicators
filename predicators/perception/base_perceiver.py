"""Base class for perceivers."""

import abc

from predicators.structs import Action, EnvironmentTask, Observation, State, \
    Task, Video


class BasePerceiver(abc.ABC):
    """A perceiver consumes observations and produces states."""

    @classmethod
    @abc.abstractmethod
    def get_name(cls) -> str:
        """Get the unique name of this perceiver."""

    @abc.abstractmethod
    def reset(self, env_task: EnvironmentTask) -> Task:
        """Start a new episode of environment interaction."""

    @abc.abstractmethod
    def step(self, observation: Observation) -> State:
        """Produce a State given the current and past observations."""

    def update_perceiver_with_action(self, action: Action) -> None:
        """In some cases, the perceiver might need to know the action that was
        taken (e.g. if the agent is trying to grasp an object, the perceiver
        needs to know which object this is)."""

    @abc.abstractmethod
    def render_mental_images(self, observation: Observation,
                             env_task: EnvironmentTask) -> Video:
        """Create mental images for the given observation."""
