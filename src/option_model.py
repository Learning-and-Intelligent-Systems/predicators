"""Definitions of option models."""

from __future__ import annotations

import abc
from typing import cast

from predicators.src import utils
from predicators.src.envs import create_env, get_cached_env_instance
from predicators.src.envs.behavior import BehaviorEnv
from predicators.src.settings import CFG
from predicators.src.structs import State, _Option


def create_option_model(name: str) -> _OptionModelBase:
    """Create an option model given its name."""
    if name == "oracle":
        return _OracleOptionModel()
    if name == "behavior_oracle":
        return _BehaviorOptionModel()  # pragma: no cover
    raise NotImplementedError(f"Unknown option model: {name}")


class _OptionModelBase(abc.ABC):
    """Struct defining an option model, which computes the next state of the
    world after an option is executed from a given start state."""

    @abc.abstractmethod
    def get_next_state(self, state: State, option: _Option) -> State:
        """The key method that an option model must implement.

        Returns the next state given a current state and an option.
        """
        raise NotImplementedError("Override me!")


class _OracleOptionModel(_OptionModelBase):
    """An oracle option model that uses the ground truth simulator.

    Runs options through this simulator to figure out the next state.
    """

    def __init__(self) -> None:
        super().__init__()
        env = create_env(CFG.env)
        self._simulator = env.simulate

    def get_next_state(self, state: State, option: _Option) -> State:
        traj = utils.option_to_trajectory(
            state,
            self._simulator,
            option,
            max_num_steps=CFG.max_num_steps_option_rollout)
        return traj.states[-1]


class _BehaviorOptionModel(_OptionModelBase):
    """An oracle option model that is specific to BEHAVIOR, since simulation is
    expensive in this environment."""

    def get_next_state(self, state: State,
                       option: _Option) -> State:  # pragma: no cover
        env_base = get_cached_env_instance("behavior")
        env = cast(BehaviorEnv, env_base)
        assert option.memory.get("model_controller") is not None
        option.memory["model_controller"](state, env.igibson_behavior_env)
        next_state = env.current_ig_state_to_state()
        return next_state
