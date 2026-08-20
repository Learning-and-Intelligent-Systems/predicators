"""Definitions of option models.

An option model makes predictions about the result of executing an
option in the environment.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, Optional, Set, Tuple

import numpy as np
import pybullet

from predicators import utils
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import Action, DefaultState, LowLevelTrajectory, \
    ParameterizedOption, State, _Option


def _check_wait_termination(option: _Option, state: State, last_state: State,
                            abstract_fn: Callable[[State], Set]) -> bool:
    """Check if a Wait option should terminate based on target atoms or atom
    change.

    Returns True if it should terminate.
    """
    result = utils.check_wait_target_atoms(option, state, abstract_fn)
    if result is True:
        logging.info("Wait terminating: target atoms satisfied")
        return True
    if result is None:
        cur_atoms = abstract_fn(state)
        prev_atoms = abstract_fn(last_state)
        if cur_atoms != prev_atoms:
            logging.info(f"Wait terminating due to atom change: "
                         f"Add: {sorted(cur_atoms - prev_atoms)} "
                         f"Del: {sorted(prev_atoms - cur_atoms)}")
            return True
    return False


def create_option_model(
        name: str,
        use_gui: Optional[bool] = None,
        skip_residual_dynamics: bool = False) -> _OptionModelBase:
    """Create an option model given its name.

    Args:
        name: The name of the option model.
        use_gui: If provided, overrides CFG.option_model_use_gui for the
            environment created by this option model.
        skip_residual_dynamics: If True, the wrapped env runs with its
            delayed ``_domain_specific_step`` dynamics disabled (the
            "base" simulator). Forwarded to the env only when True, so
            non-PyBullet analog envs whose ``__init__`` does not accept
            the kwarg are unaffected by the default.
    """
    gui = CFG.option_model_use_gui if use_gui is None else use_gui
    env_kwargs: Dict[str, Any] = {}
    if skip_residual_dynamics:
        env_kwargs["skip_residual_dynamics"] = True
    if name == "oracle":
        env = create_new_env(CFG.env,
                             do_cache=False,
                             use_gui=gui,
                             **env_kwargs)
        options = get_gt_options(env.get_name())
        model = _OracleOptionModel(options, env.simulate)
        model.sim_env = env
        return model
    if name.startswith("oracle"):
        env_name = name[name.index("_") + 1:]
        env = create_new_env(env_name,
                             do_cache=False,
                             use_gui=gui,
                             **env_kwargs)
        options = get_gt_options(env.get_name())
        model = _OracleOptionModel(options, env.simulate)
        model.sim_env = env
        return model
    raise NotImplementedError(f"Unknown option model: {name}")


class _OptionModelBase(abc.ABC):
    """Struct defining an option model, which predicts the next state of the
    world after an option is executed from a given start state."""

    # The env instance backing this model's simulator, when it wraps
    # one (set by ``create_option_model``). Task-evaluator verdict
    # paths pass it as the transient ``sim_env`` so physics-needing
    # certificates (the domino counterfactual push probe) run against
    # the same belief physics the rollout used.
    sim_env: Optional[Any] = None

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
        self._abstract_function: Optional[Callable] = None
        # Diagnostic: stores the reason when the last call returned 0 actions.
        self.last_execution_failure: str | None = None
        # Stores the full trajectory from the last successful execution.
        self.last_trajectory: LowLevelTrajectory | None = None

    def get_next_state_and_num_actions(self, state: State,
                                       option: _Option) -> Tuple[State, int]:
        self.last_execution_failure = None
        self.last_trajectory = None

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
        # Propagate Wait target atoms through re-grounding
        for key in ("wait_target_atoms", "wait_target_neg_atoms"):
            if key in option.memory:
                option_copy.memory[key] = option.memory[key]
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
                logging.debug("Option reached terminal state.")
                return True
            if (CFG.option_model_terminate_on_repeat
                    and last_state is not DefaultState
                    and last_state.allclose(s)):
                logging.debug("Option got stuck.")
                raise utils.OptionExecutionFailure(
                    f"Option '{option_copy.name}' got stuck: the "
                    f"policy's action did not change the state. "
                    f"This usually means the first motion phase "
                    f"produced a no-op (e.g. IK returned current "
                    f"joints, or finger command matched current "
                    f"finger state).")
            if (CFG.wait_option_terminate_on_atom_change
                    and option_copy.name == "Wait"
                    and last_state is not DefaultState
                    and self._abstract_function is not None
                    and _check_wait_termination(option_copy, s, last_state,
                                                self._abstract_function)):
                logging.debug("Wait option terminating early.")
                return True
            last_state = s
            return False

        try:
            traj = utils.run_policy_with_simulator(
                option_copy.policy,
                self._simulator,
                state,
                _terminal,
                max_num_steps=CFG.max_num_steps_option_rollout)
        except (utils.OptionExecutionFailure, pybullet.error) as e:
            # Treat PyBullet physics engine errors the same as planned
            # execution failures (e.g. GUI/Metal crash on macOS).
            self.last_execution_failure = str(e)
            return state, 0
        # Note that in the case of using a PyBullet environment, the
        # second return value (num_actions) will be an underestimate
        # since we are not actually rolling out the option in the full
        # simulator, but that's okay; it leads to optimistic planning.
        self.last_trajectory = traj
        return traj.states[-1], len(traj.actions)
