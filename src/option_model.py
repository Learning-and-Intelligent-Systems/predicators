"""Definitions of option models."""

from __future__ import annotations
import abc
from typing import Callable
from predicators.src import utils
from predicators.src.structs import State, Action, _Option
from predicators.src.settings import CFG
from predicators.src.envs import get_cached_env_instance


def create_option_model(
        name: str, simulator: Callable[[State, Action],
                                       State]) -> _OptionModel:
    """Create an option model given its name."""
    if name == "default":
        return _DefaultOptionModel(simulator)
    if name == "behavior_option_model":
        return _BehaviorOptionModel(simulator)  # pragma: no cover
    raise NotImplementedError(f"Unknown option model: {name}")


class _OptionModel:
    """Struct defining an option model, which computes the next state of the
    world after an option is executed from a given start state."""

    def __init__(self, simulator: Callable[[State, Action], State]):
        self._simulator = simulator

    @abc.abstractmethod
    def get_next_state(self, state: State, option: _Option) -> State:
        """The key method that an option model must implement.

        Returns the next state given a current state and an option.
        Subclasses can choose whether this method should use the
        simulator.
        """
        raise NotImplementedError("Override me!")


class _DefaultOptionModel(_OptionModel):
    """A default option model that just runs options through the simulator to
    figure out the next state."""

    def get_next_state(self, state: State, option: _Option) -> State:
        traj = utils.option_to_trajectory(
            state,
            self._simulator,
            option,
            max_num_steps=CFG.max_num_steps_option_rollout)
        return traj.states[-1]


class _BehaviorOptionModel(_OptionModel):
    """An oracle option model for the BEHAVIOR env."""

    def get_next_state(self, state: State,
                       option: _Option) -> State:  # pragma: no cover
        env = get_cached_env_instance("behavior")
        assert option.memory["model_controller"] is not None
        final_state = option.memory["model_controller"](state, env)
        return final_state
