"""Definitions of option models.

An option model makes predictions about the result of executing an
option in the environment.
"""

from __future__ import annotations

import abc
import logging
from typing import Callable, Set, Tuple

import numpy as np

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import Action, DefaultState, ParameterizedOption, \
    State, _Option


def create_option_model(name: str) -> _OptionModelBase:
    """Create an option model given its name."""
    if name == "oracle":
        env = create_new_env(CFG.env,
                             do_cache=False,
                             use_gui=CFG.option_model_use_gui)
        options = get_gt_options(env.get_name())
        return _OracleOptionModel(options, env.simulate)
    if name.startswith("oracle"):
        env_name = name[name.index("_") + 1:]
        env = create_new_env(env_name,
                             do_cache=False,
                             use_gui=CFG.option_model_use_gui)
        options = get_gt_options(env.get_name())
        return _OracleOptionModel(options, env.simulate)
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

    def __init__(self, options: Set[ParameterizedOption],
                 simulator: Callable[[State, Action], State]) -> None:
        super().__init__()
        self._name_to_parameterized_option = {o.name: o for o in options}
        self._simulator = simulator

    def get_next_state_and_num_actions(self, state: State,
                                       option: _Option) -> Tuple[State, int]:
        # We do not want to actually execute the option; we want to know what
        # *would* happen if we were to execute the option. So, we will make a
        # copy of the option and run that instead. This is important if the
        # option has memory. It is also important when using the option model
        # for one environment with options from another environment. E.g.,
        # using a non-PyBullet environment in the option model while using a
        # PyBullet environment otherwise. In the case where we are
        # learning options, the learned options will not appear in the ground
        # truth options set. However, we still want to use the environment
        # options during data collection when we are learning options. In this
        # case, we make a copy of the option itself, rather than reconstructing
        # it from the ground truth options.
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
        if CFG.option_model_terminate_on_repeat:
            last_state = DefaultState

            def _terminal(s: State) -> bool:
                nonlocal last_state
                if option_copy.terminal(s):
                    logging.debug("Option reached terminal state.")
                    return True
                if last_state is not DefaultState and last_state.allclose(s):
                    logging.debug("Option got stuck.")
                    raise utils.OptionExecutionFailure("Option got stuck.")
                last_state = s
                return False
        else:
            # mypy complains without the lambda, pylint complains with it!
            _terminal = lambda s: option_copy.terminal(s)  # pylint: disable=unnecessary-lambda

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
