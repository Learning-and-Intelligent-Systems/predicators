"""Definitions of option models.

An option model makes predictions about the result of executing an
option in the environment.
"""

from __future__ import annotations

import abc
from typing import Tuple

import numpy as np

from predicators import utils
from predicators.behavior_utils.behavior_utils import load_checkpoint_state
from predicators.envs import BaseEnv, get_or_create_env
from predicators.envs.behavior import BehaviorEnv
from predicators.settings import CFG
from predicators.structs import DefaultState, State, _Option


def create_option_model(name: str) -> _OptionModelBase:
    """Create an option model given its name."""
    if name == "oracle":
        env = get_or_create_env(CFG.env)
        return _OracleOptionModel(env)
    if name == "oracle_behavior":
        return _BehaviorOptionModel()
    if name.startswith("oracle"):
        env_name = name[name.index("_") + 1:]
        env = get_or_create_env(env_name)
        return _OracleOptionModel(env)
    raise NotImplementedError(f"Unknown option model: {name}")


class _OptionModelBase(abc.ABC):
    """Struct defining an option model, which predicts the next state of the
    world after an option is executed from a given start state."""

    @abc.abstractmethod
    def get_next_state_and_num_actions(self, state: State,
                                       option: _Option) -> Tuple[State, int]:
        """The key method that an option model must implement.

        Given a current state and an option, returns a tuple of (the
        next state, the number of actions needed to reach it).
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

    def get_next_state_and_num_actions(self, state: State,
                                       option: _Option) -> Tuple[State, int]:
        # We do not want to actually execute the option; we want to know what
        # *would* happen if we were to execute the option. So, we will make a
        # copy of the option and run that instead. This is important if the
        # option has memory. It is also important when using the option model
        # for one environment with options from another environment. E.g.,
        # using a non-PyBullet environment in the option model while using a
        # PyBullet environment otherwise. In the case where we are
        # learning options, the learned options will not appear in the
        # env.options set. However, we still want to use the environment
        # options during data collection when we are learning options. In this
        # case, we make a copy of the option itself, rather than reconstructing
        # it from env.options.
        param_opt = option.parent
        if param_opt.name not in self._name_to_parameterized_option:
            assert "Learned" in param_opt.name
            option_copy = param_opt.ground(option.objects,
                                           option.params.copy())
        else:
            env_param_opt = self._name_to_parameterized_option[param_opt.name]
            assert env_param_opt.types == param_opt.types
            assert np.allclose(env_param_opt.params_space.low,
                               param_opt.params_space.low)
            assert np.allclose(env_param_opt.params_space.high,
                               param_opt.params_space.high)
            option_copy = env_param_opt.ground(option.objects, option.params)
        del option  # unused after this
        assert option_copy.initiable(state)

        # Detect if the option gets stuck in a state and terminate immediately
        # if it does. This is a helpful optimization for planning with
        # fine-grained options over long horizons.
        # Note: mypy complains if this is None instead of DefaultState.
        last_state = DefaultState

        def _terminal(s: State) -> bool:
            nonlocal last_state
            if option_copy.terminal(s):
                return True
            if last_state is not DefaultState and last_state.allclose(s):
                raise utils.OptionExecutionFailure("Option got stuck.")
            last_state = s
            return False

        try:
            traj = utils.run_policy_with_simulator(
                option_copy.policy,
                self._simulator,
                state,
                _terminal,
                max_num_steps=CFG.max_num_steps_option_rollout)
        except utils.OptionExecutionFailure:
            # If there is a failure during the execution of the option, treat
            # this as a noop.
            return state, 0
        # Note that in the case of using a PyBullet environment, the
        # second return value (num_actions) will be an underestimate
        # since we are not actually rolling out the option in the full
        # simulator, but that's okay; it leads to optimistic planning.
        return traj.states[-1], len(traj.actions)


class _BehaviorOptionModel(_OptionModelBase):
    """An oracle option model that is specific to BEHAVIOR, since simulation is
    expensive in this environment."""

    def get_next_state_and_num_actions(
            self, state: State,
            option: _Option) -> Tuple[State, int]:  # pragma: no cover
        env = get_or_create_env("behavior")
        assert isinstance(env, BehaviorEnv)
        assert option.memory.get("model_controller") is not None
        assert option.memory.get("planner_result") is not None
        if not CFG.plan_only_eval:
            load_checkpoint_state(state, env, reset=True)
        option.memory["model_controller"](state, env.igibson_behavior_env)
        next_state = env.current_ig_state_to_state()
        plan, _ = option.memory["planner_result"]
        return next_state, len(plan)

    @staticmethod
    def load_state(state: State) -> State:  # pragma: no cover
        """Loads BEHAVIOR state by getting or creating our current BEHAVIOR
        env."""
        env = get_or_create_env("behavior")
        assert isinstance(env, BehaviorEnv)
        load_checkpoint_state(state, env)
        return env.current_ig_state_to_state()
