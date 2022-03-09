"""Definitions of option models."""

from __future__ import annotations
import abc
from typing import cast, Tuple
import numpy as np
from predicators.src import utils
from predicators.src.structs import State, _Option
from predicators.src.settings import CFG
from predicators.src.envs import BaseEnv, get_or_create_env
from predicators.src.envs.behavior import BehaviorEnv


def create_option_model(name: str) -> _OptionModelBase:
    """Create an option model given its name."""
    if name == "oracle":
        env = get_or_create_env(CFG.env)
        return _OracleOptionModel(env)
    if name == "oracle_behavior":
        return _BehaviorOptionModel()  # pragma: no cover
    if name.startswith("oracle"):
        env_name = name[name.index("_") + 1:]
        env = get_or_create_env(env_name)
        return _OracleOptionModel(env)
    raise NotImplementedError(f"Unknown option model: {name}")


class _OptionModelBase(abc.ABC):
    """Struct defining an option model, which computes the next state of the
    world after an option is executed from a given start state."""

    @abc.abstractmethod
    def get_next_state_and_num_actions(
            self, state: State, option: _Option) -> Tuple[State, int]:
        """The key method that an option model must implement.

        Given a current state and an option, returns a tuple of
        (the next state, the number of actions needed to reach it).
        """
        raise NotImplementedError("Override me!")


class _OracleOptionModel(_OptionModelBase):
    """An oracle option model that uses the ground truth simulator.

    Runs options through this simulator to figure out the next state.
    """

    def __init__(self, env: BaseEnv) -> None:
        super().__init__()
        self._name_to_parameterized_option = {o.name: o for o in env.options}
        self._simulator = env.simulate

    def get_next_state_and_num_actions(
            self, state: State, option: _Option) -> Tuple[State, int]:
        # We do not want to actually execute the option; we want to know what
        # *would* happen if we were to execute the option. So, we will make a
        # copy of the option and run that instead. This is important if the
        # option has memory. It is also important when using the option model
        # for one environment with options from another environment. E.g.,
        # using a non-PyBullet environment in the option model while using a
        # PyBullet environment otherwise.
        param_opt = option.parent
        env_param_opt = self._name_to_parameterized_option[param_opt.name]
        assert env_param_opt.types == param_opt.types
        assert np.allclose(env_param_opt.params_space.low,
                           param_opt.params_space.low)
        assert np.allclose(env_param_opt.params_space.high,
                           param_opt.params_space.high)
        env_option = env_param_opt.ground(option.objects, option.params)
        del option  # unused after this
        assert env_option.initiable(state)
        traj = utils.run_policy_with_simulator(
            env_option.policy,
            self._simulator,
            state,
            env_option.terminal,
            max_num_steps=CFG.max_num_steps_option_rollout)
        return traj.states[-1], len(traj.actions)


class _BehaviorOptionModel(_OptionModelBase):
    """An oracle option model that is specific to BEHAVIOR, since simulation is
    expensive in this environment."""

    def get_next_state_and_num_actions(
            self, state: State,
            option: _Option) -> Tuple[State, int]:  # pragma: no cover
        env_base = get_or_create_env("behavior")
        env = cast(BehaviorEnv, env_base)
        assert option.memory.get("model_controller") is not None
        assert option.memory.get("planner_result") is not None
        option.memory["model_controller"](state, env.igibson_behavior_env)
        next_state = env.current_ig_state_to_state()
        plan, _ = option.memory["planner_result"]
        return next_state, len(plan)
