"""Tests for AgentSimLearningApproach.score_atom_disagreement.

Validates the param-swap mechanism that turns a parameter ensemble into a
boundary-straddling-detector: a learned predicate whose classifier reads
the approach's live ``_fitted_params`` is evaluated under each ensemble
member, and the across-member disagreement is the info score.

Also covers the exploration-fit MCMC budget: the solver fit follows the
global MCMC budget, while info-seeking exploration can run a separate
once-per-cycle posterior fit used only for the active-experiment
ensemble.
"""

# pylint: disable=protected-access,import-outside-toplevel,unused-import

import numpy as np

from predicators import utils  # noqa: F401  (settles import order)
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.code_sim_learning.fitting import fit_rule_parameters, \
    fit_rule_parameters_latent
from predicators.structs import Action, GroundAtom, Object, Predicate, State, \
    Type

_t = Type("block", ["x"])
_block = Object("b", _t)


def _bare_approach(ensemble, fitted):
    """An approach instance with only the fields the scorer touches."""
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = dict(fitted)
    approach._param_ensemble = [dict(m) for m in ensemble]
    return approach


def _at_target_atom(approach):
    """AtTarget(block) holds iff x < the live fitted threshold."""

    def _classifier(s, o):
        return s.get(o[0], "x") < approach._fitted_params["thresh"]

    return GroundAtom(Predicate("AtTarget", [_t], _classifier), [_block])


def _state(x):
    return State({_block: np.array([x], dtype=np.float32)})


def test_disagreement_high_at_boundary():
    """Disagreement high at boundary."""
    ens = [{"thresh": t} for t in (0.5, 0.3, 0.4, 0.6, 0.7)]
    approach = _bare_approach(ens, {"thresh": 0.5})
    atom = _at_target_atom(approach)
    # x=0.5 splits the ensemble (3 say False, 2 say True) -> nonzero entropy.
    assert approach.score_atom_disagreement(_state(0.5), {atom}) > 0.0


def test_disagreement_zero_far_from_boundary():
    """Disagreement zero far from boundary."""
    ens = [{"thresh": t} for t in (0.5, 0.3, 0.4, 0.6, 0.7)]
    approach = _bare_approach(ens, {"thresh": 0.5})
    atom = _at_target_atom(approach)
    # x=0.05 < every threshold -> all members agree True -> no disagreement.
    assert approach.score_atom_disagreement(_state(0.05), {atom}) == 0.0
    # x=0.95 > every threshold -> all agree False -> no disagreement.
    assert approach.score_atom_disagreement(_state(0.95), {atom}) == 0.0


def test_fitted_params_restored_after_scoring():
    """Fitted params restored after scoring."""
    ens = [{"thresh": t} for t in (0.3, 0.7)]
    approach = _bare_approach(ens, {"thresh": 0.5})
    atom = _at_target_atom(approach)
    approach.score_atom_disagreement(_state(0.5), {atom})
    # The scorer must leave the MAP params exactly as it found them.
    assert approach._fitted_params == {"thresh": 0.5}


def test_singleton_ensemble_scores_zero():
    """Singleton ensemble scores zero."""
    approach = _bare_approach([{"thresh": 0.5}], {"thresh": 0.5})
    atom = _at_target_atom(approach)
    assert approach.score_atom_disagreement(_state(0.5), {atom}) == 0.0


def test_empty_atoms_scores_zero():
    """Empty atoms scores zero."""
    ens = [{"thresh": t} for t in (0.3, 0.7)]
    approach = _bare_approach(ens, {"thresh": 0.5})
    assert approach.score_atom_disagreement(_state(0.5), set()) == 0.0


def test_rebuild_param_ensemble_respects_flag():
    """Rebuild param ensemble respects flag."""
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {"a": 1.0}
    approach._param_specs = []
    approach._param_ensemble = [{"a": 1.0}, {"a": 2.0}]
    approach._last_fit_result = None  # no calibrated fit -> uniform fallback
    approach._rng = np.random.default_rng(0)
    utils.reset_config({"agent_explorer_info_seeking": False})
    approach._rebuild_param_ensemble()
    assert approach._param_ensemble == []  # cleared when off

    from predicators.code_sim_learning.fit_space import ParamSpec
    approach._param_specs = [ParamSpec("a", 1.0, lo=0.0, hi=2.0)]
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_ensemble_size": 5,
        "agent_explorer_info_perturb_frac": 0.2,
    })
    approach._rebuild_param_ensemble()
    assert len(approach._param_ensemble) == 5
    assert approach._param_ensemble[0] == {"a": 1.0}  # member 0 is anchor


def _selector_approach(fit_result):
    from predicators.code_sim_learning.fit_space import ParamSpec
    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {"a": 1.0, "b": 2.0}
    approach._param_specs = [
        ParamSpec("a", 1.0, lo=-10.0, hi=10.0),
        ParamSpec("b", 2.0, lo=-10.0, hi=10.0),
    ]
    approach._last_fit_result = fit_result
    approach._rng = np.random.default_rng(0)
    return approach


def test_select_ensemble_prefers_posterior_when_samples_present():
    """Select ensemble prefers posterior when samples present."""
    from predicators.code_sim_learning.fit_space import FitResult

    # MCMC ran: multi-row samples -> posterior subsample wins.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.1, 2.1], [0.9, 1.9], [1.2, 2.2]]),
                    log_probs=np.array([0.0, 2.0, 1.0]))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 3,
    })
    members, method = approach._select_param_ensemble(3)
    assert method == "posterior-subsample"
    assert members[0] == {"a": 0.9, "b": 1.9}
    rows = {(1.1, 2.1), (0.9, 1.9), (1.2, 2.2)}
    assert all((m["a"], m["b"]) in rows for m in members[1:])


def test_select_ensemble_uses_laplace_when_only_jacobian():
    """Select ensemble uses laplace when only jacobian."""
    from predicators.code_sim_learning.fit_space import FitResult

    # No MCMC (single-row samples) but the Laplace bundle is present.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.0, 2.0]]),
                    log_probs=np.zeros(1),
                    jacobian=np.eye(2),
                    noise_sigma=0.1,
                    prior_sigma=np.array([1.0, 1.0]))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 4,
    })
    members, method = approach._select_param_ensemble(4)
    assert method == "laplace"
    assert len(members) == 4
    assert members[0] == {"a": 1.0, "b": 2.0}


def test_select_ensemble_falls_back_to_uniform_without_calibration():
    """Select ensemble falls back to uniform without calibration."""
    from predicators.code_sim_learning.fit_space import FitResult

    # Single-row samples and no Jacobian (LM skipped/failed) -> uniform.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.0, 2.0]]),
                    log_probs=np.zeros(1))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 4,
        "agent_explorer_info_perturb_frac": 0.2,
    })
    _, method = approach._select_param_ensemble(4)
    assert method == "uniform-perturb"


def test_select_ensemble_uniform_when_calibration_disabled():
    """Select ensemble uniform when calibration disabled."""
    from predicators.code_sim_learning.fit_space import FitResult

    # Posterior samples exist, but the calibration flag is off -> uniform.
    fit = FitResult(names=["a", "b"],
                    samples=np.array([[1.1, 2.1], [0.9, 1.9]]),
                    log_probs=np.zeros(2))
    approach = _selector_approach(fit)
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_calibrated_ensemble": False,
        "agent_explorer_info_ensemble_size": 4,
        "agent_explorer_info_perturb_frac": 0.2,
    })
    _, method = approach._select_param_ensemble(4)
    assert method == "uniform-perturb"


def test_exploration_fit_num_steps_budget():
    """The exploration posterior can request extra MCMC.

    The override never reduces an explicit global solver run; a separate
    exploration-only fit is needed only when this budget exceeds the
    global solver budget.
    """
    # Info-seeking off -> no override (None falls back to the global).
    utils.reset_config({
        "agent_explorer_info_seeking": False,
        "agent_explorer_info_mcmc_steps": 300,
        "code_sim_learning_num_mcmc_steps": 0,
    })
    assert AgentSimLearningApproach._exploration_fit_num_steps() is None
    separate_steps = (
        AgentSimLearningApproach._separate_exploration_fit_num_steps())
    assert separate_steps is None
    # On: the exploration budget applies even with the global at 0.
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_mcmc_steps": 300,
        "code_sim_learning_num_mcmc_steps": 0,
    })
    assert AgentSimLearningApproach._exploration_fit_num_steps() == 300
    separate_steps = (
        AgentSimLearningApproach._separate_exploration_fit_num_steps())
    assert separate_steps == 300
    # A larger global budget wins over a smaller exploration one.
    utils.reset_config({
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_mcmc_steps": 0,
        "code_sim_learning_num_mcmc_steps": 500,
    })
    assert AgentSimLearningApproach._exploration_fit_num_steps() == 500
    separate_steps = (
        AgentSimLearningApproach._separate_exploration_fit_num_steps())
    assert separate_steps is None


def test_exploration_mcmc_does_not_replace_solver_params(monkeypatch):
    """Extra exploration MCMC should not publish into solver params."""
    from predicators.code_sim_learning.fit_space import FitResult, ParamSpec

    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {}
    approach._param_specs = []
    approach._physical_param_specs = []
    approach._param_ensemble = []
    approach._last_fit_result = None
    approach._fit_sse = float("inf")
    approach._rng = np.random.default_rng(0)

    solver_result = FitResult(names=["a"],
                              samples=np.array([[1.0]]),
                              log_probs=np.zeros(1))
    exploration_result = FitResult(
        names=["a"],
        samples=np.array([[2.0], [3.0], [4.0]]),
        log_probs=np.array([0.0, 1.0, 2.0]),
    )
    calls = []

    def _fake_fit(rules,
                  specs,
                  base_pred_triples,
                  residual_features,
                  num_steps=None):
        del rules, specs, base_pred_triples, residual_features
        calls.append(num_steps)
        if num_steps is None:
            return solver_result, 10.0
        return exploration_result, 5.0

    monkeypatch.setattr(
        "predicators.approaches.agent_sim_learning_approach"
        ".fit_rule_parameters", _fake_fit)
    utils.reset_config({
        "agent_sim_learn_oracle_sim_params": False,
        "agent_explorer_info_seeking": True,
        "agent_explorer_info_mcmc_steps": 300,
        "agent_explorer_info_calibrated_ensemble": True,
        "agent_explorer_info_ensemble_size": 3,
        "code_sim_learning_num_mcmc_steps": 0,
    })
    specs = [ParamSpec("a", 1.0, lo=0.0, hi=5.0)]
    # Non-empty triples: with no data the method seeds from the declared
    # inits instead of fitting (the oracle-sim-program no-demos path).
    s = State({_block: np.array([0.0])})
    triples = [(s, Action(np.zeros(1, dtype=np.float32)), s)]
    approach._fit_params_after_synthesis([], specs, triples, {})
    assert calls == [None, 300]
    assert approach._fitted_params == {"a": 1.0}
    assert approach._fit_sse == 10.0
    assert approach._last_fit_result is exploration_result
    assert approach._param_ensemble[0] == {"a": 4.0}
    assert {m["a"]
            for m in approach._param_ensemble[1:]}.issubset({2.0, 3.0, 4.0})


def test_fit_params_no_data_seeds_declared_inits(monkeypatch):
    """With no transitions, params seed from inits and no fit runs.

    This is the oracle-sim-program no-demos path: every demo failed, so
    ``_learn_simulator`` reaches the fit with empty
    ``base_pred_triples`` and must fall back to the declared init values
    instead of fitting.
    """
    from predicators.code_sim_learning.fit_space import ParamSpec

    approach = object.__new__(AgentSimLearningApproach)
    approach._fitted_params = {}
    approach._param_specs = []
    approach._physical_param_specs = []
    approach._param_ensemble = []
    approach._last_fit_result = None
    approach._fit_sse = 0.0
    approach._rng = np.random.default_rng(0)

    def _fail_fit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("fit must not run with no data")

    monkeypatch.setattr(
        "predicators.approaches.agent_sim_learning_approach"
        ".fit_rule_parameters", _fail_fit)
    utils.reset_config({
        "agent_sim_learn_oracle_sim_params": False,
        "agent_explorer_info_seeking": False,
    })
    specs = [ParamSpec("a", 1.5, lo=0.0, hi=5.0)]
    approach._fit_params_after_synthesis([], specs, [], {})
    assert approach._fitted_params == {"a": 1.5}
    assert approach._last_fit_result is None
    assert approach._fit_sse == float("inf")


def test_fit_parameters_num_steps_override_runs_mcmc():
    """``num_steps>0`` runs emcee even when the global budget is 0.

    This is the decoupling the exploration-only fit relies on: tools and
    solver fitting call ``_fit_parameters`` without ``num_steps`` (fast
    path at the global 0), while the separate active-experiment fit
    passes its own budget and gets multi-row posterior samples — exactly
    what upgrades ``_select_param_ensemble`` to posterior-subsample.
    """
    from predicators.code_sim_learning.fit_space import ParamSpec
    utils.reset_config({
        "code_sim_learning_num_mcmc_steps": 0,
        "code_sim_learning_warm_start_with_lm": False,
        "code_sim_learning_log_hessian_identifiability": False,
        "agent_explorer_info_seeking": False,
    })
    specs = [ParamSpec("a", 1.0, lo=0.5, hi=1.5)]
    triple = (_state(0.1), Action(np.zeros(1, dtype=np.float32)), _state(0.1))
    # No override: short-circuits at the global 0 -> single-row samples.
    result, _ = fit_rule_parameters([], specs, [triple], {})
    assert result.samples.shape[0] == 1
    # Override: emcee runs despite the global 0 -> multi-row samples.
    result, _ = fit_rule_parameters([], specs, [triple], {}, num_steps=8)
    assert result.samples.shape[0] > 1


def test_fit_parameters_latent_threads_num_steps(monkeypatch):
    """The recurrent fit forwards the override into fit_params_recurrent."""
    import predicators.code_sim_learning.fitting as fitting_mod
    from predicators.code_sim_learning.fit_space import FitResult, ParamSpec

    captured = {}

    def _fake_fit(**kwargs):
        captured.update(kwargs)
        return FitResult(names=["a"],
                         samples=np.array([[1.0]]),
                         log_probs=np.zeros(1))

    monkeypatch.setattr(fitting_mod, "fit_params_recurrent", _fake_fit)
    monkeypatch.setattr(fitting_mod, "compute_sse_recurrent",
                        lambda *a, **k: 0.0)
    specs = [ParamSpec("a", 1.0, lo=0.0, hi=2.0)]
    result, sse = fit_rule_parameters_latent([],
                                             specs, [[]],
                                             None, {},
                                             num_steps=7)
    assert captured["num_steps"] == 7
    assert sse == 0.0
    assert result.point_estimate == {"a": 1.0}
