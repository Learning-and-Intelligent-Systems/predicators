"""Synthesis-session tools for sim learning (create_synthesis_tools)."""
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from predicators.agent_sdk.synthesis_backend import SynthesisBackend
from predicators.agent_sdk.tools.python_exec import _make_python_exec_tool
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths
from predicators.agent_sdk.tools.snapshots import _ArtifactSnapshotter


@dataclasses.dataclass(frozen=True)
class SynthesisToolkit:
    """What ``create_synthesis_tools`` builds for one synthesis session.

    ``tools`` are the MCP tools to attach; ``fit_runner`` and
    ``residuals_runner`` are the ``sim.fit`` / ``sim.residuals``
    backends (installed as ``ToolContext.probe_fit_provider`` /
    ``probe_residuals_provider``). All three share this session's
    snapshotter, so every report carries consistent
    ``[cycle_XXX_vers_YYY]`` tags.
    """
    tools: list
    fit_runner: Callable[..., str]
    residuals_runner: Callable[..., str]


def moving_feature_scope(
        rollouts: List[Tuple[Any, Any]]) -> Dict[str, List[str]]:
    """Features whose OBSERVED value moves anywhere in the fit data.

    The open-loop report scores global fidelity, so its scope is
    "everything that moves" - independent of the artifact's declared
    RESIDUAL_FEATURES, which describe rule scope and may legitimately be
    empty (a physics-only artifact, or one that concluded no rule is
    needed). A feature is in scope when its observed span across all
    recorded states exceeds the settle tolerance (the same "still
    moving" cutoff the settled-tail truncation uses).

    ``code_sim_learning_rollout_scope_types`` narrows that, and is empty
    by default so this report is unchanged. It exists because
    identifying ONE physical parameter is a different question from
    global fidelity: the arm is commanded, so it reproduces at every
    candidate value and can only dilute the signal -- and with it in
    scope nothing in the episode ever rests, so the rest-point
    segmentation this scope also drives can never cut.

    Module-level rather than a closure inside ``create_synthesis_tools``
    because it captures nothing from it, and the narrowing above is
    worth testing on its own.
    """
    # Deferred, as everywhere else in this module: importing settings at
    # module level here reintroduces an import cycle.
    # pylint: disable-next=import-outside-toplevel
    from predicators.settings import CFG
    tol = CFG.code_sim_learning_rollout_settle_tol
    keep_types = set(CFG.code_sim_learning_rollout_scope_types)
    drop_feats = set(CFG.code_sim_learning_rollout_nonkinematic_features)
    span_lo: Dict[Tuple[str, str], float] = {}
    span_hi: Dict[Tuple[str, str], float] = {}
    for states, _actions in rollouts:
        for state in states:
            for obj in state:
                if keep_types and obj.type.name not in keep_types:
                    continue
                for feat in obj.type.feature_names:
                    if keep_types and feat in drop_feats:
                        continue
                    val = float(state.get(obj, feat))
                    key = (obj.type.name, feat)
                    if key not in span_lo or val < span_lo[key]:
                        span_lo[key] = val
                    if key not in span_hi or val > span_hi[key]:
                        span_hi[key] = val
    out: Dict[str, List[str]] = {}
    for (tn, feat), lo in span_lo.items():
        if span_hi[(tn, feat)] - lo > tol:
            out.setdefault(tn, []).append(feat)
    return {t: sorted(fs) for t, fs in out.items()}


def create_synthesis_tools(
    exec_ns: Dict[str, Any],
    base_pred_triples: list,
    inferred_residual_features: Dict[str, List[str]],
    simulator_file: str,
    versions_dir: str,
    approach: Optional[SynthesisBackend] = None,
    sandbox_dir: Optional[str] = None,
    sandbox_dir_for_agent: Optional[str] = None,
    cycle_index_provider: Optional[Callable[[], int]] = None,
    budget_check: Optional[Callable[[], None]] = None,
) -> SynthesisToolkit:
    """Create the sim-learning synthesis agent's tool surface.

    Returns a :class:`SynthesisToolkit` with ``tools = [run_python]``
    plus the ``fit_runner`` / ``residuals_runner`` behind ``sim.fit``
    and ``sim.residuals``. Plan validation is hand-composed by the
    agent in ``run_python`` (``sim.fit()`` then ``sim.refine`` then a
    continuous ``sim.run`` pass), so there is no separate refinement
    tool.

    The agent's source-of-truth for the simulator is the file at
    ``simulator_file`` (which it edits with ``Write`` / ``Edit``). The
    fit and residuals backends each ``exec`` that file fresh into an
    isolated namespace per call and read ``RESIDUAL_RULES``,
    ``PARAM_SPECS``, ``RESIDUAL_FEATURES`` from it — no namespace state
    leaks across iterations. Before loading, every call also snapshots
    the current contents into ``versions_dir`` as
    ``cycle_XXX_vers_YYY_simulator.py`` (``XXX`` from
    ``cycle_index_provider()``, ``YYY`` resetting per
    ``create_synthesis_tools`` call) so the full history of evaluated
    versions is preserved across cycles; identical-content calls reuse
    the prior snapshot. Each report is prefixed with the version tag
    (``[cycle_XXX_vers_YYY]``).

    * ``run_python`` — executes arbitrary Python in a persistent
      namespace pre-loaded with trajectory data (and, in synthesis
      sessions, the candidate probe ``sim``). It does **not** define
      rules — write ``simulator.py`` for that.
    * ``fit_runner`` (not a tool; bound as ``sim.fit``) — SSE of the
      current ``RESIDUAL_RULES`` at init_value params, plus post-fit
      SSE and fitted values; the joint rollout system-ID path when
      ``PHYSICAL_PARAMS`` is declared; exploratory ``traj_idxs`` /
      ``fixed`` variants that publish nothing.
    * ``residuals_runner`` (not a tool; bound as ``sim.residuals``) —
      per-feature breakdown of where the current rules disagree with
      observations: mismatch counts, mean/max abs error, comparison
      to the no-rule baseline, and worst-N example transitions per
      feature.

    Args:
        exec_ns: Persistent namespace for ``run_python``. Should
            contain ``trajectories``, ``np``, ``ParamSpec``.
        base_pred_triples: ``(s_base, action, s_next_obs)`` triples
            with the base step already advanced — eval/test consume
            ``s_base`` directly so no live env is needed.
        inferred_residual_features: Data-driven default scope used
            when the agent hasn't declared ``RESIDUAL_FEATURES`` in
            ``simulator.py`` yet.
        simulator_file: Host path to the canonical simulator file
            the agent edits. Synthesis tools ``exec`` this file
            fresh on every call.
        versions_dir: Directory to write per-call snapshots into
            (created on first use).
        approach: ``AgentSimLearningApproach`` instance, used by the
            rollout system-ID fit path (raw trajectories, fit env,
            applying identified params). If ``None``, that path
            returns an error.
        sandbox_dir: Host path to the agent's sandbox root.  When set,
            ``run_python`` spills oversize output to
            ``<sandbox_dir>/tool_outputs/run_python/`` instead of
            letting the agent SDK truncate and dump it to
            ``~/.claude/projects/.../tool-results/``.  When ``None``,
            output is always returned inline.
        sandbox_dir_for_agent: Path prefix the agent sees for
            ``sandbox_dir`` (e.g. ``"."`` for local sandbox or
            ``"/sandbox"`` for docker).  Used only when building the
            human-readable path included in the spilled-output message.
        cycle_index_provider: Callable returning the current online
            learning cycle (1-indexed). Read at snapshot time so the
            same tools instance reflects later cycle bumps. If ``None``,
            cycle defaults to 0 (still valid; produces
            ``cycle_000_vers_YYY``).
        budget_check: Callable raising ``ProbeBudgetExceeded`` when the
            session's wall-clock budget is spent. Long-running backends
            (the rollout-residuals sweep) call it between rollouts so a
            budget stop returns the partial report instead of burning
            the rest of the attempt. ``None`` disables the checks.
    """
    # pylint: disable=import-outside-toplevel
    import traceback  # pylint: disable=redefined-outer-name,reimported
    from collections import defaultdict

    from claude_agent_sdk import tool as _sdk_tool
    tool = _make_coercing_tool(_sdk_tool)

    from predicators.code_sim_learning.fit_space import ParamSpec
    from predicators.code_sim_learning.fitting import compute_sse, \
        compute_sse_recurrent, fit_rule_parameters, \
        fit_rule_parameters_latent
    from predicators.code_sim_learning.grid_seed import grid_candidates
    from predicators.code_sim_learning.identifiability import \
        format_identifiability
    from predicators.code_sim_learning.orchestrator import run_rollout_sysid
    from predicators.code_sim_learning.rollout_env import \
        physical_param_anchors
    from predicators.code_sim_learning.rollout_objective import \
        compute_rollout_sse, per_trajectory_rms
    from predicators.code_sim_learning.trajectory_prep import \
        compute_residual_scaling
    from predicators.code_sim_learning.utils import apply_rules, \
        has_latent_rules, has_physics_rules, iter_feature_residuals, \
        read_latent_init, read_physical_param_specs, \
        read_simulator_components, rollout_predictions, \
        stamp_physical_spec_scales
    from predicators.settings import CFG

    # pylint: enable=import-outside-toplevel

    _snapshotter = _ArtifactSnapshotter(
        live_file=simulator_file,
        versions_dir=versions_dir,
        artifact_name="simulator",
        cycle_index_provider=cycle_index_provider,
        missing_file_hint=("Use Write to create it with RESIDUAL_RULES, "
                           "PARAM_SPECS, RESIDUAL_FEATURES."),
    )
    # Spill oversize output from the synthesis tools into the sandbox,
    # so nothing is dumped to ``~/.claude/projects/.../tool-results/``.
    # (``run_python``'s own spill lives in ``_make_python_exec_tool``.)
    _text = _make_spilling_text_result(sandbox_dir,
                                       agent_prefix=sandbox_dir_for_agent)

    def _snapshot_and_load(
            path: str) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
        """Snapshot ``path`` then exec it into a fresh namespace.

        Returns ``(rules, specs, features, latent_init, physical_specs,
        version_tag, error_msg)``; ``error_msg`` is ``None`` on success.
        ``latent_init`` is the optional ``LATENT_INIT`` export (``None``
        for fully- observable simulators) — the synthesis tools need it
        to score recurrent (5-arg) rules through the latent-threaded
        path. ``physical_specs`` is the optional ``PHYSICAL_PARAMS``
        export (system identification); when present, a physics-only
        artifact is valid and missing rules/specs default to empty
        lists. Snapshots are deduped by SHA256, so repeated calls on
        unchanged content reuse the prior ``cycle_XXX_vers_YYY`` tag.
        """
        raw, version_tag, err = _snapshotter.snapshot(path)
        if err is not None:
            return None, None, None, None, None, None, err
        assert raw is not None and version_tag is not None
        ns: Dict[str, Any] = {"np": np, "ParamSpec": ParamSpec}
        try:
            exec(raw.decode("utf-8"), ns)  # pylint: disable=exec-used
        except Exception:  # pylint: disable=broad-except
            return None, None, None, None, None, version_tag, (
                f"[{version_tag}] Error executing {path}:\n"
                f"{_scrub_host_paths(traceback.format_exc())}")
        rules, specs, features = read_simulator_components(ns)
        latent_init = read_latent_init(ns)
        physical_specs = read_physical_param_specs(ns)
        if rules is None:
            if not physical_specs:
                return None, None, None, None, None, version_tag, (
                    f"[{version_tag}] RESIDUAL_RULES missing or empty in "
                    f"{path}.")
            rules = []
        if specs is None:
            if not physical_specs:
                return None, None, None, None, None, version_tag, (
                    f"[{version_tag}] PARAM_SPECS missing or empty in "
                    f"{path}.")
            specs = []
        return (rules, specs, features, latent_init, physical_specs,
                version_tag, None)

    def _groups_for(triples: list) -> List[List[Tuple[Any, Any, Any]]]:
        """Slice flat base-pred triples into per-trajectory groups.

        Recurrent rules thread their latent block within a trajectory,
        so scoring/residuals must regroup the flat triples the same way
        the engine does. Reuses the bound approach's grouping (keyed off
        the same ``_fit_trajectories`` cache the engine uses); falls
        back to a single group when no approach is bound or the lengths
        don't line up — correct for the common single-demo case.
        """
        if approach is not None and hasattr(approach,
                                            "_group_triples_by_trajectory"):
            grouped = approach._group_triples_by_trajectory(  # pylint: disable=protected-access
                triples)
            if grouped:
                return grouped
        return [triples]

    def _evaluate_rollout_fit(rules: list,
                              rule_specs: list,
                              physical_specs: list,
                              latent_init: Any,
                              residual_features: Dict[str, List[str]],
                              scope_note: str,
                              version_tag: str,
                              traj_idxs: Optional[List[int]] = None) -> str:
        """Joint physical+rule system-ID fit on free-running rollouts.

        Reached from ``run_fit`` (``sim.fit``) when the artifact
        declares ``PHYSICAL_PARAMS``. Needs the bound approach for the
        raw (states, actions) trajectories and the dedicated headless
        fit env. With ``traj_idxs=None`` (canonical) the identified
        physical values are applied in place to the approach's planning
        base env on success, so subsequent probe rollouts run against
        the calibrated sim. With ``traj_idxs`` the fit is EXPLORATORY:
        it runs on only those trajectories' data and applies nothing - a
        consistency diagnostic (per-trajectory fits that disagree mean
        heterogeneous data, not a parameter).
        """
        if approach is None:
            return (f"[{version_tag}] Error: the rollout fit "
                    "(PHYSICAL_PARAMS / command-emitting rules) requires a "
                    "bound approach (raw trajectories + base env) — "
                    "unavailable in this session.")
        exploratory = traj_idxs is not None
        if exploratory and not traj_idxs:
            return (f"[{version_tag}] Error: traj_idxs is empty - pass the "
                    "trajectory indices to fit on, or omit it for the "
                    "canonical all-data fit.")
        # Stamp the fit scale (log vs linear) from the env registry —
        # the agent's declaration carries name/init/bounds only.
        physical_specs = stamp_physical_spec_scales(physical_specs,
                                                    approach._base_env)  # pylint: disable=protected-access
        try:
            rollouts = approach._rollout_fit_trajectories(  # pylint: disable=protected-access
                residual_features,
                traj_idxs=traj_idxs)
        except ValueError as e:
            return f"[{version_tag}] Error: {e}"
        if not rollouts:
            return (f"[{version_tag}] Error: no complete (states, actions) "
                    "trajectories are available, so the rollout system-ID fit "
                    "cannot run. PHYSICAL_PARAMS needs full trajectories, not "
                    "isolated transitions.")
        # Factory, not an instance: every rollout runs in a fresh env.
        fit_env = approach._get_rollout_fit_env()  # pylint: disable=protected-access
        physical_names = [s.name for s in physical_specs]
        init_params = {
            s.name: s.init_value
            for s in list(physical_specs) + list(rule_specs)
        }
        anchors = physical_param_anchors(
            approach._base_env,  # pylint: disable=protected-access
            physical_specs)
        try:
            outcome = run_rollout_sysid(
                fit_env,
                rollouts,
                physical_specs,
                residual_features,
                rules=rules,
                rule_specs=rule_specs,
                latent_init=latent_init,
                anchors=anchors,
                # The phase-shared caches key trajectory identity by
                # segment lengths only - exact when every fit sees the
                # full recording set, but two different traj_idxs
                # subsets with equal-length segments would collide and
                # cross-assign verdicts. Exploratory subset fits
                # therefore never share them.
                rms_cache=(None if exploratory else getattr(
                    approach, "_explainability_cache", None)),
                fit_cache=(None if exploratory else getattr(
                    approach, "_sysid_fit_cache", None)),
                fit_cache_key=version_tag)
        except Exception as e:  # pylint: disable=broad-except
            return (
                f"[{version_tag}] Error: rollout system-ID fit failed:\n{e}")
        if outcome.num_survivors == 0:
            # Honest empty-data output: no fit ran, nothing was
            # applied - do NOT print fitted values or identifiability
            # verdicts computed on zero surviving data (chaos makes
            # the probe report "identified" for everything).
            rms_str = ", ".join(f"{r:.4g}" for r in outcome.traj_rms)
            return "\n".join([
                f"[{version_tag}] NO FIT RAN: all {len(rollouts)} "
                "recorded motion segments were unexplainable at ANY "
                "candidate physical parameters (per-segment "
                f"best-achievable RMS [{rms_str}] all above the "
                "trimming threshold).",
                "",
                "Parameters were left at their baselines; nothing was "
                "applied to the planning base env.",
                "",
                "This usually means the recorded interactions are not "
                "repeatable under replay (prolonged scraping/jamming "
                "robot-object contact is chaotic). Collect experiments "
                "whose outcome is dominated by object dynamics: actuate "
                "one or two objects cleanly, then let the scene evolve "
                "and settle on its own.",
            ])
        fitted = outcome.fitted
        applied = outcome.applied
        ident_report = outcome.report
        pre_sse, post_sse = outcome.pre_sse, outcome.post_sse
        if not exploratory:
            approach._apply_identified_physical_params(applied)  # pylint: disable=protected-access
            if hasattr(approach, "_record_sysid_diagnostics"):
                approach._record_sysid_diagnostics(  # pylint: disable=protected-access
                    ident_report, physical_names, outcome.num_survivors,
                    len(rollouts), outcome.traj_rms)
        kept_at_init = sorted(n for n in physical_names
                              if applied[n] != fitted[n])
        if pre_sse > 0:
            pct_str = (f"({(pre_sse - post_sse) / pre_sse * 100:.1f}% "
                       "SSE reduction vs init)")
        else:
            pct_str = "(init SSE was 0)"
        mode_note = (
            f"EXPLORATORY, trajectories {sorted(traj_idxs or [])} only"
            if exploratory else "canonical")
        fit_reason = ("PHYSICAL_PARAMS declared"
                      if physical_specs else "command-emitting rules")
        lines = [
            f"[{version_tag}] JOINT ROLLOUT SYSTEM-ID FIT ({fit_reason}; "
            f"{mode_note}) on {len(rollouts)} motion segments "
            f"(scope: {scope_note}; {len(physical_names)} physical + "
            f"{len(list(rule_specs))} rule params). Residuals are "
            "per-feature normalized (angles wrapped), so SSE/RMS are "
            "dimensionless fractions of typical motion.",
            "",
            f"At init params:   rollout SSE = {pre_sse:.6f}",
            f"After joint fit:  rollout SSE = {post_sse:.6f}  {pct_str}",
            "",
            "Fitted parameters:",
        ]
        if outcome.num_survivors < len(rollouts):
            rms_str = ", ".join(f"{r:.4g}" for r in outcome.traj_rms)
            dropped = len(rollouts) - outcome.num_survivors
            lines.insert(
                1, f"Goodness-of-fit trimming: {dropped}"
                f" of {len(rollouts)} motion segments were unexplainable at "
                "ANY candidate params (per-segment best-achievable RMS: "
                f"[{rms_str}]) and were dropped before fitting; the fit "
                "below used only the explainable ones. Unexplainable "
                "segments are not repeatable under replay - prefer "
                "experiments whose outcome is dominated by object dynamics "
                "rather than prolonged robot-object contact.")
        for name in sorted(fitted):
            init_val = init_params[name]
            fit_val = fitted[name]
            delta = fit_val - init_val
            ppct = (delta / init_val * 100) if init_val != 0 else float("nan")
            kind = "physical" if name in physical_names else "rule"
            lines.append(f"  {name:<28} [{kind:<8}] {init_val:.4f} -> "
                         f"{fit_val:.4f}  (delta={delta:+.4f}, {ppct:+.1f}%)")

        lines.extend([
            "",
            "Identifiability (posterior_std / prior_std; ~1 means the data "
            "did NOT constrain the parameter — its fitted value is "
            "arbitrary, so remove it from PHYSICAL_PARAMS or collect data "
            "that exercises it):",
            format_identifiability(ident_report),
            "",
        ])
        if exploratory:
            lines.append(
                "EXPLORATORY subset fit: nothing was applied or recorded - "
                "the deployed physical params are unchanged. Compare the "
                "identified values across subsets (and against the "
                "canonical no-arg fit): parameters that disagree between "
                "individually-explainable trajectories indicate "
                "heterogeneous data (e.g. an arm-touched episode), not a "
                "parameter value.")
        elif kept_at_init:
            lines.append(
                "Applied to the planning base env: fitted values for the "
                "identified params only; "
                f"{', '.join(kept_at_init)} did not contract (or failed "
                "the sensitivity screen), so their baseline values were "
                "kept (the fitted values above for them are arbitrary). "
                "Probe rollouts (sim.run / sim.refine) now run against the "
                "partially calibrated sim.")
        else:
            lines.append(
                "The identified physical params were applied to the "
                "planning base env; probe rollouts (sim.run / sim.refine) "
                "now run against the calibrated sim.")
        return "\n".join(lines)

    # ── run_python ──────────────────────────────────────────

    # The approach merges the BeliefProbe facade (`sim` over the CANDIDATE
    # simulator.py) into this same namespace - one exec namespace per
    # synthesis session, so helpers defined next to the data are
    # visible to probe sweeps. Since the fit/refine/forward-validate
    # surfaces all live on `sim` now, the probe is unconditional in
    # synthesis (there is no other validation surface). Shared blurb,
    # so the wording cannot drift from the solve-phase explore_python
    # surface.
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.tools.exploration import belief_probe_blurb
    probe_blurb = (
        " This namespace ALSO binds the candidate-simulator probe: " +
        belief_probe_blurb(synthesis_probe=True) +
        " Probe rollouts are CANDIDATE-simulator predictions - do not "
        "mix them up with the recorded real `trajectories`. Nothing the "
        "probe runs is captured; the validation protocol before declaring "
        "the simulator done is: `sim.fit()` (canonical fit report), "
        "`sim.refine(plan, require_goal=True)` (params exist that reach "
        "each subgoal), then a continuous `sim.run` of the refined plan "
        "(the forward pass; a refine-pass/run-fail means a rule is more "
        "permissive than the data).")
    run_python = _make_python_exec_tool(
        tool,
        name="run_python",
        description=(
            "Execute Python code for ad-hoc data exploration. Available "
            "variables: trajectories (List[LowLevelTrajectory]; each has "
            "`is_demo`, `train_task_idx`, `states`, `actions`), train_tasks "
            "(List[Task]; each has `init`, `goal`, `goal_holds(state)`), "
            "is_goal_state (callable: state, task_idx -> bool - do the goal "
            "atoms hold in this one STATE; reaching the goal atoms does not "
            "by itself mean solved), describe_trajectory(traj_idx, "
            "include_states=True, include_atoms=False, max_timesteps=10) "
            "- a per-timestep digest of one trajectory, np, ParamSpec, "
            "and (when the "
            "env defines task evaluators) evaluate_trajectory(states, "
            "actions=None, task_idx=0) -> {reward, solved} - the env's "
            "ground-truth episode scoring over a full TRAJECTORY. "
            "print() output "
            "is returned. The namespace persists across calls. If output "
            "exceeds ~30k chars it is saved to "
            "`tool_outputs/run_python/call_NNNN.txt` in the sandbox and only "
            "a head/tail preview plus that path is returned - use Read/Grep "
            "to inspect the full file. This does NOT define rules - write "
            "`simulator.py` for that; `sim.fit` and `sim.residuals` "
            "load RESIDUAL_RULES, PARAM_SPECS, RESIDUAL_FEATURES from that "
            "file fresh on every call." + probe_blurb),
        exec_ns=exec_ns,
        sandbox_dir=sandbox_dir,
        sandbox_dir_for_agent=sandbox_dir_for_agent,
        text_result=_text,
    )

    # ── run_fit (the ``sim.fit`` backend) ───────────────────────

    def run_fit(path: Optional[str] = None,
                traj_idxs: Optional[List[int]] = None,
                fixed: Optional[Dict[str, float]] = None) -> str:
        """Fit PARAM_SPECS (loaded fresh from ``simulator.py``) and report.

        The backend behind ``sim.fit``. With no arguments this is the
        CANONICAL fit - the same data/fit the probe deploys - and, when
        the file declares PHYSICAL_PARAMS, the identified physical
        values are applied to the planning base env. Any argument makes
        the fit EXPLORATORY: a diagnostic report only, publishing and
        applying nothing.

        ``traj_idxs`` restricts the fit to those trajectories' data
        (step transitions on the rule paths; whole recorded rollouts on
        the system-ID path - a consistency diagnostic across
        trajectories). ``fixed`` pins parameters at given values while
        the rest are fit (rule paths only - the system-ID path rejects
        it; narrow the param's bounds in PHYSICAL_PARAMS instead). Each
        call snapshots the simulator file into simulator_versions/ and
        tags output ``[cycle_XXX_vers_YYY]``.
        """
        p = path or simulator_file
        rules, specs, declared, latent_init, physical_specs, version_tag, \
            err = _snapshot_and_load(p)
        if err:
            return str(err)

        residual_features = (declared if isinstance(declared, dict) else
                             inferred_residual_features)
        scope_note = ("declared" if isinstance(declared, dict) else
                      "inferred (RESIDUAL_FEATURES not declared)")
        canonical = traj_idxs is None and not fixed

        # PHYSICAL_PARAMS declared, or rules on the physics-command
        # channel (a ``cmds`` parameter) -> joint system-identification
        # fit on free-running rollouts (the per-transition /
        # teacher-forced paths below cannot see physical params - State
        # carries no velocities - and cannot see command effects, which
        # only exist through engine stepping).
        # traj_idxs is allowed (exploratory subset fit; applies nothing);
        # fixed is not - pinning a physical param has a versioned channel
        # already (its lo/hi bounds in the PHYSICAL_PARAMS declaration).
        if physical_specs or has_physics_rules(rules):
            if fixed:
                return (f"[{version_tag}] Error: fixed is not supported "
                        "with the rollout fit (PHYSICAL_PARAMS or "
                        "command-emitting rules). Pin a param by narrowing "
                        "its lo/hi bounds in the declaration instead "
                        "(versioned in simulator.py, respected by the "
                        "whole fit stack).")
            return _evaluate_rollout_fit(rules,
                                         specs,
                                         physical_specs or [],
                                         latent_init,
                                         residual_features,
                                         scope_note,
                                         version_tag,
                                         traj_idxs=traj_idxs)

        if fixed:
            known = {s.name for s in specs}
            unknown = sorted(set(fixed) - known)
            if unknown:
                return (f"[{version_tag}] Error: fixed names {unknown} are "
                        f"not in PARAM_SPECS (available: {sorted(known)}).")

        triples = base_pred_triples
        groups = _groups_for(base_pred_triples)
        if traj_idxs is not None:
            bad = sorted(i for i in traj_idxs if not 0 <= i < len(groups))
            if bad:
                return (f"[{version_tag}] Error: traj_idxs {bad} out of "
                        f"range (0-{len(groups) - 1}).")
            groups = [groups[i] for i in traj_idxs]
            triples = [t for g in groups for t in g]
            if not triples:
                return (f"[{version_tag}] Error: the selected trajectories "
                        "contain no step transitions.")

        # Dispatch on the rule signature exactly as the fitting engine
        # does: recurrent (5-arg, latent-declaring) rules are scored with
        # the latent block threaded per trajectory, never through the
        # legacy per-transition path (which would call them with 3 args).
        latent_mode = has_latent_rules(rules)
        # Pinning: wrap the rules so pinned values override whatever the
        # fit engine proposes, and fit only the free specs. Wrapper arg
        # names must be preserved - the engine dispatches recurrent
        # rules by inspecting for a 2nd parameter named ``latent``.
        fit_rules = rules
        fit_specs = list(specs)
        if fixed:
            fixed_vals = dict(fixed)
            if latent_mode:
                fit_rules = [
                    lambda state, latent, history, updates, params, _r=r: _r(
                        state, latent, history, updates, {
                            **params,
                            **fixed_vals
                        }) for r in rules
                ]
            else:
                fit_rules = [
                    lambda state, updates, params, _r=r: _r(
                        state, updates, {
                            **params,
                            **fixed_vals
                        }) for r in rules
                ]
            fit_specs = [s for s in specs if s.name not in fixed_vals]
        init_params = {s.name: s.init_value for s in fit_specs}
        try:
            if latent_mode:
                pre_sse = compute_sse_recurrent(fit_rules, groups, init_params,
                                                latent_init, residual_features)
            else:
                sim_fn = lambda s, _a, prm: apply_rules(  # noqa: E731
                    s, fit_rules, prm)
                pre_sse = compute_sse(sim_fn, triples, init_params,
                                      residual_features)
        except Exception as e:  # pylint: disable=broad-except
            return f"[{version_tag}] Error: SSE computation failed:\n{e}"

        sig_note = ("recurrent (latent threaded per trajectory)"
                    if latent_mode else "per-transition")
        mode_note = ("canonical - the same fit the probe deploys" if canonical
                     else "EXPLORATORY - diagnostic only, nothing published")
        lines = [
            f"[{version_tag}] Fit evaluation on {len(triples)} "
            f"step transitions (scope: {scope_note}; rules: {sig_note}; "
            f"{mode_note}).",
        ]
        if traj_idxs is not None:
            lines.append(f"Restricted to trajectories {sorted(traj_idxs)}.")
        if fixed:
            pinned = ", ".join(f"{n}={v:.4g}"
                               for n, v in sorted(fixed.items()))
            lines.append(f"Pinned parameters: {pinned}.")
        lines += ["", f"At init_value params:  SSE = {pre_sse:.6f}"]

        if not fit_specs:
            lines.append(
                "All parameters are pinned - no fit ran; the SSE above is "
                "the score at the pinned values.")
            return "\n".join(lines)
        try:
            if latent_mode:
                fit_result, post_sse = fit_rule_parameters_latent(
                    fit_rules, fit_specs, groups, latent_init,
                    residual_features)
            else:
                fit_result, post_sse = fit_rule_parameters(
                    fit_rules, fit_specs, triples, residual_features)
            fitted_params = fit_result.point_estimate
        except Exception as e:  # pylint: disable=broad-except
            return f"[{version_tag}] Error: fit_params failed:\n{e}"
        if pre_sse > 0:
            pct = (pre_sse - post_sse) / pre_sse * 100
            pct_str = f"({pct:+.1f}% vs init)"
        else:
            pct_str = "(init SSE was 0)"
        lines.append(f"After fit:             SSE = {post_sse:.6f}  "
                     f"{pct_str}")
        lines.append("")
        lines.append("Fitted parameters:")
        for name in sorted(fitted_params):
            init_val = init_params[name]
            fit_val = fitted_params[name]
            delta = fit_val - init_val
            ppct = ((delta / init_val *
                     100) if init_val != 0 else float("nan"))
            lines.append(f"  {name:<30} {init_val:.4f} -> "
                         f"{fit_val:.4f}  (delta={delta:+.4f}, "
                         f"{ppct:+.1f}%)")

        return "\n".join(lines)

    # ── rollout-mode residuals (open-loop fidelity) ─────────────

    _moving_feature_scope = moving_feature_scope

    def _run_rollout_residuals(rules: list, specs: list, latent_init: Any,
                               version_tag: str, sweep_num_points: int,
                               sweep_params: Optional[Union[str, List[str]]],
                               phys_params: Optional[Dict[str, float]]) -> str:
        """Open-loop rollout fidelity report, with opt-in param probing.

        The ``rollout=True`` branch of ``sim.residuals``. Replays each
        recorded trajectory's actions free-running from its initial
        state (fresh env per rollout, same objective the system-ID fit
        minimizes) and reports the divergence at the current baselines.
        Two mutually exclusive opt-ins ride on top:

        * ``sweep_params`` (list of registry names, or ``"all"``) sweeps
          each named env-registry physical parameter alone across its
          plausible range. The sweep is the interpretive anchor: an
          absolute rollout SSE is meaningless under chaotic replay
          divergence, but "the same data is explained N times better at
          a different friction" is exactly the evidence the
          PHYSICAL_PARAMS declaration decision needs - evidence the
          teacher-forced report structurally cannot surface
          (run_20260728_111805 declined to declare on near-zero
          per-step residuals while the open-loop SSE ratio on the same
          data was ~340x). Costs one fresh-env rollout per candidate
          per motion segment, hence opt-in.
        * ``phys_params`` ({name: value}) scores the same data at ONE
          hypothesized physical-parameter point and reports the SSE
          ratio against the baseline - the composable primitive for
          agent-written targeted sweeps.
        """
        # pylint: disable-next=import-outside-toplevel
        from predicators.agent_sdk.belief_probe import ProbeBudgetExceeded
        if sweep_params is not None and phys_params:
            return (f"[{version_tag}] Error: sweep_params and phys_params "
                    "are mutually exclusive - sweep_params tests each named "
                    "parameter across its box, phys_params scores one "
                    "explicit point. Make two calls if you want both.")
        if approach is None:
            return (f"[{version_tag}] Error: rollout residuals require a "
                    "bound approach (raw trajectories + fit env) - "
                    "unavailable in this session.")
        # Whole trajectories first (no scope -> no truncation) to derive
        # the motion scope, then re-prep with it so the scored rollouts
        # get the same settled-tail truncation and rest-point
        # segmentation the system-ID fit uses.
        whole = approach._rollout_fit_trajectories(None)  # pylint: disable=protected-access
        if not whole:
            return (f"[{version_tag}] Error: no complete (states, actions) "
                    "trajectories are available - the open-loop report "
                    "needs full trajectories, not isolated transitions.")
        scope = _moving_feature_scope(whole)
        if not scope:
            return (f"[{version_tag}] No feature moves beyond the settle "
                    "tolerance anywhere in the recorded data; there is no "
                    "motion to score open-loop.")
        rollouts = approach._rollout_fit_trajectories(scope)  # pylint: disable=protected-access
        fit_env = approach._get_rollout_fit_env()  # pylint: disable=protected-access
        scaling = compute_residual_scaling(rollouts, scope)
        rule_params = {s.name: s.init_value for s in specs}
        info: Dict[str, Dict[str, Any]] = getattr(
            approach._base_env,  # pylint: disable=protected-access
            "get_physical_param_info",
            lambda: {})()
        sweepable = {
            n: e
            for n, e in info.items()
            if e.get("lo") is not None and e.get("hi") is not None
        }
        if sweep_params is None:
            selected: Dict[str, Dict[str, Any]] = {}
        elif isinstance(sweep_params, str):
            if sweep_params != "all":
                return (f"[{version_tag}] Error: sweep_params must be a "
                        "list of registry names or the string 'all', got "
                        f"{sweep_params!r} (available: "
                        f"{sorted(sweepable)}).")
            selected = dict(sweepable)
        else:
            unknown = sorted(set(sweep_params) - set(sweepable))
            if unknown:
                return (f"[{version_tag}] Error: sweep_params {unknown} not "
                        "in the env's physical-param registry (available: "
                        f"{sorted(sweepable)}).")
            selected = {n: sweepable[n] for n in sweep_params}
        if phys_params:
            unknown = sorted(set(phys_params) - set(info))
            if unknown:
                return (f"[{version_tag}] Error: phys_params {unknown} not "
                        "in the env's physical-param registry (available: "
                        f"{sorted(info)}).")
            bad_vals = sorted(
                n for n, v in phys_params.items()
                if not isinstance(v, (int, float)) or not np.isfinite(v))
            if bad_vals:
                return (f"[{version_tag}] Error: phys_params values must be "
                        f"finite numbers; got {bad_vals} = "
                        f"{[phys_params[n] for n in bad_vals]!r}.")
        num_points = max(2, int(sweep_num_points))
        # Rules run inside every rollout evaluation, so a buggy rule
        # (e.g. reading a param this version no longer declares) must
        # come back as a report, not a raw traceback through the tool.
        try:
            baseline_sse = compute_rollout_sse(fit_env, rollouts, rule_params,
                                               scope, [], rules, latent_init,
                                               scaling)
            seg_rms = per_trajectory_rms(fit_env, rollouts, rule_params, scope,
                                         [], rules, latent_init, scaling)
            point_sse: Optional[float] = None
            point_rms: Optional[List[float]] = None
            if phys_params:
                overrides = sorted(phys_params)
                point_sse = compute_rollout_sse(fit_env, rollouts, {
                    **rule_params,
                    **phys_params
                }, scope, overrides, rules, latent_init, scaling)
                point_rms = per_trajectory_rms(fit_env, rollouts, {
                    **rule_params,
                    **phys_params
                }, scope, overrides, rules, latent_init, scaling)
        except Exception as e:  # pylint: disable=broad-except
            return (f"[{version_tag}] Error: open-loop rollout scoring "
                    f"failed (often a RESIDUAL_RULES bug - rules run on "
                    f"every rolled-out step):\n{e}")
        ratio_bar = CFG.code_sim_learning_rollout_consistency_sse_ratio
        scope_str = "; ".join(f"{t}: {', '.join(fs)}"
                              for t, fs in sorted(scope.items()))
        rms_str = ", ".join(f"{r:.4g}" for r in seg_rms)
        lines = [
            f"[{version_tag}] OPEN-LOOP ROLLOUT residual report - each "
            "recorded trajectory's actions replayed free-running from its "
            "initial state on the base sim with the current RESIDUAL_RULES "
            "riding (params at init_value; physical params at the env "
            "registry baselines, NOT any already-applied fit). Errors "
            "COMPOUND across steps here; the per-step (teacher-forced) "
            "report resets to the recorded state every step and therefore "
            "CANNOT see integrated divergence - a wrong physical parameter "
            "can look near-perfect per step and still diverge wildly "
            "open-loop.",
            f"Scope (every feature with observed motion, independent of "
            f"RESIDUAL_FEATURES): {scope_str}.",
            "Residuals are normalized (angles wrapped), Huber-capped, with "
            "endpoint/onset summary terms - the same objective the "
            "system-ID fit minimizes.",
            "",
            f"Rollout SSE at current baselines: {baseline_sse:.6f}",
            f"Per-segment RMS at current baselines: [{rms_str}]  "
            f"({len(rollouts)} motion segments after settled-tail "
            "truncation / rest-point segmentation).",
        ]
        if phys_params:
            assert point_sse is not None and point_rms is not None
            desc = ", ".join(f"{k}={float(v):.4g}"
                             for k, v in sorted(phys_params.items()))
            prms_str = ", ".join(f"{r:.4g}" for r in point_rms)
            lines.append("")
            lines.append(f"SSE at phys_params ({desc}): {point_sse:.6f}")
            lines.append(f"Per-segment RMS at this point: [{prms_str}]")
            if baseline_sse <= 0 and point_sse <= 0:
                lines.append("Both SSEs are exactly 0 (static or perfectly "
                             "reproduced data) - no evidence either way.")
            else:
                better = (baseline_sse /
                          point_sse if point_sse > 0 else float("inf"))
                worse = (point_sse /
                         baseline_sse if baseline_sse > 0 else float("inf"))
                if better >= ratio_bar:
                    lines.append(
                        f"The data is {better:.1f}x better explained at "
                        "this point than at the baseline - strong evidence "
                        "FOR declaring the overridden parameter(s) in "
                        "PHYSICAL_PARAMS.")
                elif worse >= ratio_bar:
                    lines.append(
                        f"The data is {worse:.1f}x WORSE explained at "
                        "this point than at the baseline - evidence "
                        "against this hypothesis.")
                else:
                    lines.append(
                        f"SSE ratio baseline/point = {better:.2f}, within "
                        f"the {ratio_bar:g}x consistency bar - this data "
                        "cannot distinguish the two points.")
        if selected:
            lines.append("")
            lines.append(
                "Physical-parameter sweep (each requested registry param "
                "swept ALONE across its plausible range, all others held "
                "at baseline; SSEs are comparable within this report "
                "only):")
            swept: List[str] = []
            try:
                for name in sorted(selected):
                    entry = selected[name]
                    spec = ParamSpec(name,
                                     float(entry["default"]),
                                     lo=float(entry["lo"]),
                                     hi=float(entry["hi"]),
                                     scale=entry.get("scale", "linear"))
                    cands = [
                        float(v) for v in grid_candidates(spec, num_points)
                    ]
                    try:
                        sses = []
                        for c in cands:
                            # Between-rollout checkpoint: a budget stop
                            # must salvage the completed params below,
                            # not discard minutes of sim time.
                            if budget_check is not None:
                                budget_check()
                            sses.append(
                                compute_rollout_sse(fit_env, rollouts, {
                                    **rule_params, name: c
                                }, scope, [name], rules, latent_init, scaling))
                    except ProbeBudgetExceeded:
                        raise
                    except Exception as e:  # pylint: disable=broad-except
                        lines.append(f"  {name}: sweep failed:\n    {e}")
                        swept.append(name)
                        continue
                    best_i = int(np.argmin(sses))
                    best_sse = sses[best_i]
                    worst_sse = max(sses)
                    base_ratio = (baseline_sse /
                                  best_sse if best_sse > 0 else float("inf"))
                    spread = (worst_sse /
                              best_sse if best_sse > 0 else float("inf"))
                    if worst_sse <= 0:
                        # Every candidate scores exactly 0 (static or
                        # perfectly reproduced data): a flat landscape,
                        # not evidence.
                        verdict = ("flat across the range - this data "
                                   "cannot constrain it (declaring it "
                                   "would fit noise)")
                    elif base_ratio >= ratio_bar:
                        verdict = (f"the data is {base_ratio:.1f}x better "
                                   f"explained at {cands[best_i]:.4g} than "
                                   "at the baseline - strong evidence FOR "
                                   "declaring this parameter in "
                                   "PHYSICAL_PARAMS")
                    elif spread < ratio_bar:
                        verdict = ("flat across the range - this data "
                                   "cannot constrain it (declaring it "
                                   "would fit noise)")
                    else:
                        verdict = (f"best at {cands[best_i]:.4g} "
                                   f"({base_ratio:.1f}x vs baseline) - weak "
                                   "evidence; consider data that exercises "
                                   "it")
                    cand_str = ", ".join(f"{c:.4g} -> {s:.4g}"
                                         for c, s in zip(cands, sses))
                    lines.append(f"  {name} ({spec.scale} scale, baseline "
                                 f"{float(entry['default']):.4g}):")
                    lines.append(f"    {cand_str}")
                    lines.append(f"    {verdict}.")
                    swept.append(name)
            except ProbeBudgetExceeded as e:
                remaining = sorted(set(selected) - set(swept))
                lines.append(
                    f"  SWEEP STOPPED EARLY after {len(swept)}/"
                    f"{len(selected)} parameters ({e}). Parameters "
                    "reported above are complete; still unswept: "
                    f"{remaining} - re-run with sweep_params={remaining} "
                    "if their evidence is still needed.")
        if selected or phys_params:
            lines.extend([
                "",
                "How to read this: replay divergence is chaotic, so the "
                "SSE never reaches 0 even at perfect parameters - compare "
                "ratios, not absolutes (the run's consistency bar is "
                f"{ratio_bar:g}x). A parameter materially better at "
                "another value belongs in PHYSICAL_PARAMS so the "
                "system-ID fit can calibrate it; a flat sweep means this "
                "data cannot distinguish values.",
            ])
        else:
            lines.append("")
            lines.append(
                "No parameter probing requested. Env-registry physical "
                "parameters this data could be tested against:")
            for name in sorted(sweepable):
                entry = sweepable[name]
                lines.append(
                    f"  {name}: baseline {float(entry['default']):.4g}, "
                    f"box [{float(entry['lo']):.4g}, "
                    f"{float(entry['hi']):.4g}] "
                    f"({entry.get('scale', 'linear')} scale)")
            lines.append(
                "An absolute rollout SSE is meaningless under chaotic "
                "replay divergence - only ratios between parameter values "
                "carry evidence. Pass sweep_params=[...] (or 'all') to "
                "sweep each named parameter alone across its box, or "
                "phys_params={name: value} to score one hypothesized "
                "point. Consult a sweep BEFORE deciding the "
                "PHYSICAL_PARAMS declaration, in either direction: 'this "
                "data is explained Nx better at a different value' is the "
                "open-loop evidence the declaration needs, and a flat "
                "sweep is honest evidence the data cannot constrain a "
                "parameter.")
        return "\n".join(lines)

    # ── run_residuals (the ``sim.residuals`` backend) ───────────

    def run_residuals(max_transitions: int = 100,
                      abs_tol: float = 1e-4,
                      rel_tol: float = 1e-3,
                      num_worst_examples: int = 3,
                      fit_params: bool = False,
                      path: Optional[str] = None,
                      rollout: bool = False,
                      sweep_num_points: int = 6,
                      sweep_params: Optional[Union[str, List[str]]] = None,
                      phys_params: Optional[Dict[str, float]] = None) -> str:
        """Per-feature residual report for the current RESIDUAL_RULES.

        The backend behind ``sim.residuals``. Loads the rules fresh
        from ``simulator.py`` and reports, for each feature in
        RESIDUAL_FEATURES (or the inferred fallback), the mismatch
        count, mean/max abs error, and the relative improvement over
        the no-rule baseline (negative means the rules are worse than
        not running them at all), plus the worst-N example transitions
        per feature. Uses init_value params by default;
        ``fit_params=True`` MCMC-fits first (diagnostic only - nothing
        is published). Tolerance: ``|pred - obs| > rel_tol * |obs| +
        abs_tol``. Each call snapshots the simulator file into
        simulator_versions/ and tags output ``[cycle_XXX_vers_YYY]``.

        ``rollout=True`` switches to the OPEN-LOOP report (see
        ``_run_rollout_residuals``): free-running replay divergence at
        the current baselines. On top of that, ``sweep_params`` (a list
        of registry names, or ``"all"``) opts into a per-parameter
        sweep of the env's physical-param registry
        (``sweep_num_points`` candidates each - each candidate costs
        one fresh-env rollout per motion segment, so a full-registry
        sweep takes minutes), and ``phys_params`` ({name: value},
        mutually exclusive with ``sweep_params``) scores the data at
        one hypothesized point and reports the SSE ratio vs the
        baseline. The two modes answer different questions: per-step
        localizes WHICH feature has an unmodeled process; rollout
        answers whether the base physics is globally faithful, which
        per-step residuals cannot see.
        """
        p = path or simulator_file
        rules, specs, declared, latent_init, _physical_specs, version_tag, \
            err = _snapshot_and_load(p)
        if err:
            return str(err)
        if not rollout and (sweep_params is not None or phys_params):
            return (f"[{version_tag}] Error: sweep_params/phys_params apply "
                    "to the open-loop report only - pass rollout=True.")
        if not rollout and has_physics_rules(rules):
            # Command-emitting rules act through engine stepping, which
            # the per-transition report cannot replay; auto-route to the
            # open-loop report rather than scoring a commands-free
            # prediction and reporting phantom residuals.
            note = (f"[{version_tag}] Note: RESIDUAL_RULES emit physics "
                    "commands (a `cmds` parameter), so the per-transition "
                    "report cannot score them; showing the OPEN-LOOP "
                    "rollout report instead (equivalent to rollout=True).")
            report = _run_rollout_residuals(rules, specs, latent_init,
                                            version_tag, sweep_num_points,
                                            sweep_params, phys_params)
            return note + "\n\n" + report
        if rollout:
            return _run_rollout_residuals(rules, specs, latent_init,
                                          version_tag, sweep_num_points,
                                          sweep_params, phys_params)

        residual_features = (declared if isinstance(declared, dict) else
                             inferred_residual_features)
        scope_label = ("declared"
                       if isinstance(declared, dict) else "inferred")

        max_n = int(max_transitions)
        abs_tol = float(abs_tol)
        rel_tol = float(rel_tol)
        n_examples = int(num_worst_examples)
        do_fit = bool(fit_params)

        # Same engine-matching dispatch as run_fit: recurrent
        # rules are fit and rolled out with the latent threaded per
        # trajectory, never called per-transition with 3 args.
        latent_mode = has_latent_rules(rules)
        groups = _groups_for(base_pred_triples)
        if do_fit:
            try:
                if latent_mode:
                    fit_result, _ = fit_rule_parameters_latent(
                        rules, specs, groups, latent_init, residual_features)
                else:
                    fit_result, _ = fit_rule_parameters(
                        rules, specs, base_pred_triples, residual_features)
                t_params = fit_result.point_estimate
                param_label = "fitted"
            except Exception as e:  # pylint: disable=broad-except
                return f"[{version_tag}] Error: param fitting failed:\n{e}"
        else:
            t_params = {s.name: s.init_value for s in specs}
            param_label = "init_value"

        # Predicted next states, latent threaded per trajectory for
        # recurrent rules (legacy rules roll each transition independently).
        # Roll out all groups in flat order, then truncate to max_n so the
        # reported step indices line up with the flat triples slice below.
        try:
            all_preds = rollout_predictions(rules, t_params, groups,
                                            latent_init)
        except Exception as e:  # pylint: disable=broad-except
            return f"[{version_tag}] Error: rule rollout failed:\n{e}"
        triples_rules: List = all_preds[:max_n]
        triples_base: List = [(bs, sn)
                              for bs, _a, sn in base_pred_triples[:max_n]]

        # Per-feature accumulators keyed by (type_name, feat_name).
        rule_n_total: Dict = defaultdict(int)
        rule_n_mismatch: Dict = defaultdict(int)
        rule_sum_err: Dict = defaultdict(float)
        rule_max_err: Dict = defaultdict(float)
        base_n_total: Dict = defaultdict(int)
        base_sum_err: Dict = defaultdict(float)
        worst: Dict = defaultdict(list)
        mismatched_steps: set = set()

        for i, obj, tn, feat, pred, obs in iter_feature_residuals(
                triples_rules, residual_features):
            key = (tn, feat)
            err = abs(pred - obs)
            thr = rel_tol * abs(obs) + abs_tol
            rule_n_total[key] += 1
            rule_sum_err[key] += err
            if err > rule_max_err[key]:
                rule_max_err[key] = err
            if err > thr:
                rule_n_mismatch[key] += 1
                mismatched_steps.add(i)
                worst[key].append((i, obj.name, pred, obs, err))

        for _, _, tn, feat, pred, obs in iter_feature_residuals(
                triples_base, residual_features):
            key = (tn, feat)
            base_n_total[key] += 1
            base_sum_err[key] += abs(pred - obs)

        if not rule_n_total:
            return (f"[{version_tag}] RESIDUAL_FEATURES is empty; "
                    "nothing to report.")

        n_steps = len(triples_rules)
        perfect_steps = n_steps - len(mismatched_steps)
        lines = [
            f"[{version_tag}] Residual report — {n_steps} step transitions, "
            f"scope: {scope_label} RESIDUAL_FEATURES, "
            f"params: {param_label}, "
            f"tol: {rel_tol:g}*|obs| + {abs_tol:g}.",
            f"Steps with all in-scope features within tol: "
            f"{perfect_steps}/{n_steps}.",
            "",
            f"{'feature':<35} {'misses/total':<14} {'mean_err':<10} "
            f"{'max_err':<10} {'vs base':<14}",
        ]
        for key in sorted(rule_n_total):
            tn, feat = key
            n_tot = rule_n_total[key]
            n_mm = rule_n_mismatch[key]
            mean = rule_sum_err[key] / max(1, n_tot)
            mx = rule_max_err[key]
            bn = max(1, base_n_total[key])
            base_mean = base_sum_err[key] / bn
            if base_mean > 0:
                improvement = (base_mean - mean) / base_mean * 100
                vs_base = f"{improvement:+.0f}%"
                if improvement < 0:
                    vs_base += " (worse)"
            elif mean == 0:
                vs_base = "exact"
            else:
                vs_base = "rules add err"
            lines.append(f"{tn + '.' + feat:<35} {f'{n_mm}/{n_tot}':<14} "
                         f"{mean:<10.4f} {mx:<10.4f} {vs_base:<14}")

        if n_examples > 0 and worst:
            lines.append("")
            lines.append(f"Worst {n_examples} mismatches per feature "
                         f"(step N = trajectory transition state[N] -> "
                         f"state[N+1]):")
            for key in sorted(worst):
                tn, feat = key
                entries = sorted(worst[key], key=lambda x: x[4], reverse=True)
                for step, oname, pred, obs, err in entries[:n_examples]:
                    lines.append(f"  step {step:>4}  {oname}.{feat}: "
                                 f"pred={pred:.6f} obs={obs:.6f} "
                                 f"err={err:.6f}")

        return "\n".join(lines)

    return SynthesisToolkit(tools=[run_python],
                            fit_runner=run_fit,
                            residuals_runner=run_residuals)
