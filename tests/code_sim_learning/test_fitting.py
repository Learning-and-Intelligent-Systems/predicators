"""Tests for code sim-learning training utilities."""

import numpy as np
import pytest

from predicators import utils
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.fitting import compute_residuals, \
    compute_residuals_recurrent, compute_sse, compute_sse_recurrent, \
    fit_map_lm_recurrent, fit_params, fit_params_recurrent
from predicators.code_sim_learning.utils import has_latent_rules, \
    rollout_predictions
from predicators.structs import Action, Object, State, Type


def _mk_jug_trajectory():
    """A 2-step single-object trajectory: (base, action, next_obs) triples.

    ``bubbling`` rises 0 -> 0.5 -> 1.0 across the trajectory, which a
    recurrent rule can only predict by accumulating a hidden quantity.
    """
    jug = Type("jug", ["bubbling"])
    j = jug("jug0")
    act = Action(np.zeros(1, dtype=np.float32))

    def s(v):
        return State({j: np.array([v], dtype=np.float32)})

    group = [(s(0.0), act, s(0.5)), (s(0.5), act, s(1.0))]
    return j, group


def test_rollout_predictions_threads_latent_for_recurrent_rules():
    """A correct 5-arg rule is rolled out with the latent threaded.

    This is the path the synthesis tools now take for recurrent rules.
    The old per-transition path called such a rule with 3 args and
    raised ``TypeError``, which misled the agent into writing a broken
    3-arg rule.
    """
    j, group = _mk_jug_trajectory()

    def bubbling_rule(state, latent, history, updates, params):
        del state, history
        latent["heat"] = latent.get("heat", 0.0) + params["rate"]
        updates.setdefault(j, {})["bubbling"] = min(1.0, latent["heat"])
        return updates

    rules = [bubbling_rule]
    assert has_latent_rules(rules)

    preds = rollout_predictions(rules, {"rate": 0.5}, [group],
                                latent_init={"heat": 0.0})
    # Latent accumulates across steps: 0.5 then 1.0 (not reset each step).
    assert [round(float(sp.get(j, "bubbling")), 3) for sp, _ in preds] == \
        [0.5, 1.0]
    # And the recurrent SSE agrees with the observations exactly.
    sse = compute_sse_recurrent(rules, [group], {"rate": 0.5}, {"heat": 0.0},
                                {"jug": ["bubbling"]})
    assert sse == 0.0


def test_rollout_predictions_legacy_rules_are_independent():
    """3-arg rules apply per-transition; latent_init is ignored."""
    j, group = _mk_jug_trajectory()

    def legacy_rule(state, updates, params):
        del state
        updates.setdefault(j, {})["bubbling"] = params["const"]
        return updates

    rules = [legacy_rule]
    assert not has_latent_rules(rules)
    preds = rollout_predictions(rules, {"const": 0.3}, [group],
                                latent_init={"heat": 0.0})
    # Each step predicts the constant independently — no accumulation.
    assert [round(float(sp.get(j, "bubbling")), 3) for sp, _ in preds] == \
        [0.3, 0.3]


def test_fit_params_can_skip_training_with_cfg():
    """Test that CFG can disable parameter fitting."""
    utils.reset_config({"code_sim_learning_num_mcmc_steps": 0})
    param_specs = [ParamSpec("rate", 2.5), ParamSpec("threshold", 0.7)]

    result = fit_params(
        simulator_fn=lambda _s, _a, _p: {},
        transitions=[],
        param_specs=param_specs,
        residual_features={},
    )

    assert result.point_estimate == {"rate": 2.5, "threshold": 0.7}
    np.testing.assert_allclose(result.samples, np.array([[2.5, 0.7]]))
    np.testing.assert_allclose(result.log_probs, np.array([0.0]))
    # No info-seeking and no Hessian/warm-start flags -> no Laplace bundle.
    assert result.jacobian is None


def _linear_transitions():
    """k_true * x observations for a 1-param linear simulator."""
    p_type = Type("p", ["x", "v"])
    obj = Object("o", p_type)
    act = Action(np.zeros(1, dtype=np.float32))
    k_true = 3.0
    transitions = []
    for x in (0.2, 0.5, 1.0, 1.5, 2.0):
        s_t = State({obj: np.array([x, 0.0], dtype=np.float32)})
        s_next = State({obj: np.array([x, k_true * x], dtype=np.float32)})
        transitions.append((s_t, act, s_next))

    def simulator_fn(s, _a, params):
        return {obj: {"v": params["k"] * s.get(obj, "x")}}

    return simulator_fn, transitions, {"p": ["v"]}


def test_fit_params_threads_laplace_bundle_when_info_seeking():
    """At 0 MCMC steps, info-seeking attaches the LM Jacobian + MAP."""
    utils.reset_config({
        "code_sim_learning_num_mcmc_steps": 0,
        "agent_explorer_info_seeking": True,
    })
    simulator_fn, transitions, residual_features = _linear_transitions()
    result = fit_params(
        simulator_fn=simulator_fn,
        transitions=transitions,
        param_specs=[ParamSpec("k", 1.0, lo=0.0, hi=10.0)],
        residual_features=residual_features,
        noise_sigma=0.05,
    )
    # LM recovers the true slope (3.0) as the point estimate, not init (1.0).
    assert result.point_estimate["k"] == pytest.approx(3.0, abs=1e-3)
    # The Laplace bundle is populated: one residual per transition, one param.
    assert result.jacobian is not None
    assert result.jacobian.shape == (len(transitions), 1)
    assert result.noise_sigma == pytest.approx(0.05)
    assert result.prior_sigma is not None and result.prior_sigma.shape == (1, )


def test_fit_params_no_bundle_when_lm_fully_disabled():
    """With LM off (no warm-start, no Hessian, no info-seeking), no bundle."""
    utils.reset_config({
        "code_sim_learning_num_mcmc_steps": 0,
        "code_sim_learning_warm_start_with_lm": False,
        "code_sim_learning_log_hessian_identifiability": False,
        "agent_explorer_info_seeking": False,
    })
    simulator_fn, transitions, residual_features = _linear_transitions()
    result = fit_params(
        simulator_fn=simulator_fn,
        transitions=transitions,
        param_specs=[ParamSpec("k", 1.0, lo=0.0, hi=10.0)],
        residual_features=residual_features,
    )
    # No LM ran: point estimate stays at init and no bundle is attached.
    assert result.point_estimate["k"] == pytest.approx(1.0)
    assert result.jacobian is None


def test_fit_params_bundle_from_warm_start_lm():
    """The Laplace bundle is also populated by the default warm-start LM."""
    utils.reset_config({
        "code_sim_learning_num_mcmc_steps": 0,
        "code_sim_learning_warm_start_with_lm": True,
        "agent_explorer_info_seeking": False,
    })
    simulator_fn, transitions, residual_features = _linear_transitions()
    result = fit_params(
        simulator_fn=simulator_fn,
        transitions=transitions,
        param_specs=[ParamSpec("k", 1.0, lo=0.0, hi=10.0)],
        residual_features=residual_features,
        noise_sigma=0.05,
    )
    assert result.point_estimate["k"] == pytest.approx(3.0, abs=1e-3)
    assert result.jacobian is not None
    assert result.jacobian.shape == (len(transitions), 1)


# ── recurrent LM / Laplace path ──────────────────────────────────


def _mk_recurrent_problem():
    """A smooth, non-clamping latent-rate fit problem.

    ``bubbling`` accumulates ``rate`` per step. With ``rate_true = 0.2``
    the observed ramp is 0.2, 0.4 — both below the 1.0 cap, so the
    residual is linear in ``rate`` everywhere in the search box (no flat
    clamp region) and LM recovers it cleanly.
    """
    jug = Type("jug", ["bubbling"])
    j = jug("jug0")
    act = Action(np.zeros(1, dtype=np.float32))

    def s(v):
        return State({j: np.array([v], dtype=np.float32)})

    group = [(s(0.0), act, s(0.2)), (s(0.2), act, s(0.4))]

    def bubbling_rule(state, latent, history, updates, params):
        del state, history
        latent["heat"] = latent.get("heat", 0.0) + params["rate"]
        updates.setdefault(j, {})["bubbling"] = min(1.0, latent["heat"])
        return updates

    rules = [bubbling_rule]
    return j, rules, [group], {"heat": 0.0}, {"jug": ["bubbling"]}, 0.2


def test_compute_residuals_recurrent_matches_sse():
    """sum(r**2) must equal compute_sse_recurrent for the same params."""
    _, rules, trajs, latent_init, feats, _ = _mk_recurrent_problem()
    for rate in (0.2, 0.35, 0.5):
        params = {"rate": rate}
        res = compute_residuals_recurrent(rules, trajs, params, latent_init,
                                          feats)
        sse = compute_sse_recurrent(rules, trajs, params, latent_init, feats)
        # One residual per (step, feature): 2 steps x 1 feature.
        assert res.shape == (2, )
        assert float(np.sum(res**2)) == pytest.approx(sse, abs=1e-9)
    # At the true rate the rollout matches the observations exactly.
    res_true = compute_residuals_recurrent(rules, trajs, {"rate": 0.2},
                                           latent_init, feats)
    assert np.allclose(res_true, 0.0)


def test_fit_map_lm_recurrent_recovers_rate_and_jacobian():
    """LM over latent-threaded residuals recovers the rate, returns J."""
    _, rules, trajs, latent_init, feats, true_rate = _mk_recurrent_problem()
    theta_map, jac = fit_map_lm_recurrent(
        rules,
        trajs,
        [ParamSpec("rate", 0.35, lo=0.01, hi=2.0)],  # perturbed init
        latent_init,
        feats,
    )
    assert theta_map[0] == pytest.approx(true_rate, abs=1e-4)
    assert jac is not None
    # 2 residuals (steps) x 1 param; columns are d(residual)/d(rate) = 1, 2.
    assert jac.shape == (2, 1)
    assert np.allclose(jac[:, 0], [1.0, 2.0], atol=1e-3)


def test_fit_params_recurrent_threads_laplace_bundle_at_mcmc0():
    """Recurrent fit at 0 MCMC steps with info-seeking attaches the bundle."""
    utils.reset_config({
        "code_sim_learning_num_mcmc_steps": 0,
        "agent_explorer_info_seeking": True,
    })
    _, rules, trajs, latent_init, feats, true_rate = _mk_recurrent_problem()
    result = fit_params_recurrent(
        rules=rules,
        trajectories=trajs,
        param_specs=[ParamSpec("rate", 0.35, lo=0.01, hi=2.0)],
        latent_init=latent_init,
        residual_features=feats,
        noise_sigma=0.05,
    )
    # LM MAP (not init) is the point estimate, and the Laplace bundle is set.
    assert result.point_estimate["rate"] == pytest.approx(true_rate, abs=1e-4)
    assert result.jacobian is not None
    assert result.jacobian.shape == (2, 1)
    assert result.noise_sigma == pytest.approx(0.05)
    assert result.prior_sigma is not None and result.prior_sigma.shape == (1, )


def test_fit_params_recurrent_no_bundle_when_lm_fully_disabled():
    """With LM fully off, the recurrent fit stays at init, no bundle."""
    utils.reset_config({
        "code_sim_learning_num_mcmc_steps": 0,
        "code_sim_learning_warm_start_with_lm": False,
        "code_sim_learning_log_hessian_identifiability": False,
        "agent_explorer_info_seeking": False,
    })
    _, rules, trajs, latent_init, feats, _ = _mk_recurrent_problem()
    result = fit_params_recurrent(
        rules=rules,
        trajectories=trajs,
        param_specs=[ParamSpec("rate", 0.35, lo=0.01, hi=2.0)],
        latent_init=latent_init,
        residual_features=feats,
    )
    assert result.point_estimate["rate"] == pytest.approx(0.35)
    assert result.jacobian is None


def test_compute_residuals_matches_sse():
    """sum(r**2) must equal compute_sse for the per-transition pair.

    Locks in the equivalence before any unification of the residual
    loops: the two functions iterate in different orders (predicted
    then unpredicted vs obj x feature) but must agree on the total.
    """
    jug = Type("jug", ["bubbling", "temp"])
    j = jug("jug0")
    act = Action(np.zeros(1, dtype=np.float32))

    def s(bub, temp):
        return State({j: np.array([bub, temp], dtype=np.float32)})

    # Two transitions; the simulator predicts only "bubbling", so
    # "temp" exercises the penalize-unpredicted branch.
    transitions = [
        (s(0.0, 1.0), act, s(0.4, 1.2)),
        (s(0.4, 1.2), act, s(0.9, 1.1)),
    ]
    feats = {"jug": ["bubbling", "temp"]}

    def simulator_fn(state, action, params):
        del action
        return {
            j: {
                "bubbling": float(state.get(j, "bubbling")) + params["rate"]
            }
        }

    for rate in (0.1, 0.4, 0.7):
        params = {"rate": rate}
        res = compute_residuals(simulator_fn, transitions, params, feats)
        sse = compute_sse(simulator_fn, transitions, params, feats)
        # One residual per (transition, object, feature): 2 x 1 x 2.
        assert res.shape == (4, )
        assert float(np.sum(res**2)) == pytest.approx(sse, abs=1e-12)
