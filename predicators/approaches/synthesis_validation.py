"""Synthesis-time validation hooks for the agent sim-learning approach.

These helpers run inside an active synthesis-agent session: they need
approach state (base env, train tasks, predicates, options) but never
re-enter the agent — no sketch-prompt query, no new session — so they
can be invoked from a synthesis tool without disturbing the live
session's prompt or tool set. They live in the approaches layer (not
``code_sim_learning``) because they orchestrate approach state and the
planner; the ``SynthesisBackend`` protocol declares exactly the approach
surface they touch.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.fitting import fit_rule_parameters
from predicators.code_sim_learning.utils import LearnedSimulator, \
    apply_rules, has_latent_rules, has_physics_rules
from predicators.structs import Action, State

if TYPE_CHECKING:
    from predicators.agent_sdk.synthesis_backend import SynthesisBackend

logger = logging.getLogger(__name__)


def build_candidate_option_model(
    approach: "SynthesisBackend",
    rules: List,
    specs: List[ParamSpec],
    residual_features: Dict[str, List[str]],
    base_pred_triples: List[Tuple[State, Action, State]],
    latent_init: Any = None,
) -> Tuple[Any, Dict[str, float], float]:
    """MCMC-fit ``specs`` and build the candidate's option model.

    The front half of the synthesis-session probe: every rollout must
    exercise the candidate simulator at its *deployed* (fitted)
    parameters, never at init_value. Returns ``(option_model,
    fitted_params, fit_sse)``; raises ``RuntimeError`` when fitting
    fails.

    Publishes side effects onto ``approach`` exactly once, here, so the
    two surfaces can never disagree: the candidate ``rules`` /
    ``latent_init`` (the recurrent combined simulator is built from
    instance state) and the fitted params into ``_fitted_params`` *in
    place* (invented predicates hold a ``_ParamsView`` over it - the
    gating rule and the gating predicate must anchor to the same
    values).

    Recurrent (latent-declaring, 5-arg) rules are fit with the latent
    threaded per trajectory; fully-observable rules take the legacy
    per-transition path. Dispatch keys off the candidate rule
    signatures (:func:`has_latent_rules`), as everywhere else in the
    fitting stack.
    """
    # pylint: disable=protected-access
    latent = has_latent_rules(rules)

    # Publish the candidate rules / latent_init *before* building the
    # combined simulator: the recurrent combined sim reads
    # self._residual_rules / self._latent_init / self._fitted_params, so
    # without this it would validate a stale cycle's rules - or, with
    # _residual_rules still None, mis-dispatch a latent candidate onto
    # the 3-arg path. Per-cycle state; overwritten when synthesis
    # finalises.
    approach._residual_rules = rules
    if latent:
        approach._latent_init = latent_init

    try:
        if has_physics_rules(rules):
            # Physics-command rules act through engine stepping, so the
            # teacher-forced objectives below cannot see them; fit
            # against free-running rollouts instead (the same routing
            # sim.fit uses). The joint fit also covers any declared
            # PHYSICAL_PARAMS, which _load_simulator_from_module_file
            # published onto the approach before this runs.
            fit_result, fit_sse = approach._fit_parameters_joint_rollout(
                rules, specs, residual_features)
        elif latent:
            fit_result, fit_sse = approach._fit_parameters_recurrent(
                rules, specs, base_pred_triples, residual_features)
        else:
            fit_result, fit_sse = fit_rule_parameters(rules, specs,
                                                      base_pred_triples,
                                                      residual_features)
        params = fit_result.point_estimate
    except Exception as e:
        raise RuntimeError(f"param fitting failed:\n{e}") from e

    # In place (clear + update, never replace): see docstring.
    approach._fitted_params.clear()
    approach._fitted_params.update(params)

    # Fully-observable rules run through this `learned` object; for
    # recurrent rules _build_combined_simulator bypasses it and threads
    # state.latent through the candidate rules published above.
    learned = LearnedSimulator(
        step_fn=lambda s, c, _r=rules, _p=params:  # type: ignore[misc]
        apply_rules(s, _r, _p, cmds=c),
        name="agent_in_session")
    combined_sim = approach._build_combined_simulator(learned)
    return approach._build_option_model(combined_sim), params, fit_sse
