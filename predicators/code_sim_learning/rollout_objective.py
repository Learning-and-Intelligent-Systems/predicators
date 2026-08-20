"""The free-running rollout objective for physical system identification.

SSE / residual-vector / per-trajectory-RMS evaluations of the joint
physical+rule forward model, and the rollout Levenberg-Marquardt MAP
fit built on them. See the
:mod:`predicators.code_sim_learning.physical_sysid` module docstring
for why the objective free-runs the base sim.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.fit_space import ParamSpec, to_fit_space
from predicators.code_sim_learning.lm import solve_lm
from predicators.code_sim_learning.rollout_env import RolloutTrajectory, \
    rollout_states
from predicators.code_sim_learning.trajectory_prep import ResidualScaling
from predicators.settings import CFG
from predicators.structs import Action, State

# Relative finite-difference step for the rollout LM Jacobian. scipy's
# default (~sqrt(machine eps) ~ 1e-8) is below the contact solver's
# sensitivity: the rollout comes back bitwise identical and the Jacobian
# is identically zero, stalling LM at init. Swept empirically on the
# domino friction-recovery smoke test (true 0.35, init 0.8): 0.01
# stalls at 0.446 (noise-dominated Jacobian), 0.05 is flaky (0.43),
# while 0.02 recovers 0.353-0.356 in ~21-30 evals. Contact-rich
# landscapes are rough at multiple scales — re-sweep this if a new
# domain's LM fit stalls well above the MCMC-quality SSE.
_ROLLOUT_LM_DIFF_STEP = 2e-2

# Tracks are read once per path per process, not once per candidate theta: a
# sweep evaluates the objective dozens of times and an episode's track is a
# multi-megabyte JSON. Also what stops the post-processing wait below being
# re-entered on every evaluation. Cleared by ``reset_track_cache``, which the
# tests use; a run only ever reads one manifest.
_TRACK_CACHE: Dict[str, List[Any]] = {}


def reset_track_cache() -> None:
    """Forget any loaded tracks, so a new manifest is read fresh."""
    _TRACK_CACHE.clear()


def compute_rollout_sse(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    residual_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any] = (),
    latent_init: Any = None,
    scaling: Optional[ResidualScaling] = None,
    config: Optional[SysIdConfig] = None,
) -> float:
    """Total per-step SSE between free-running rollouts and observations.

    ``params`` is the *joint* parameter dict (physical and rule params in one
    namespace); the ``physical_names`` subset is pushed into the env via
    ``apply_physical_param_overrides`` while the full dict is handed to the
    rules, mirroring how ``compute_sse``/``compute_sse_recurrent`` pass params.
    When ``rules`` are present they run in-the-loop on each *rolled-out*
    base state (latents threaded per trajectory when declared): physics
    commands they emit are queued on the env and shape the remainder of
    the rollout, while a rule's predicted feature overrides the base
    sim's in the scoring — the same precedence ``merge_updates`` uses at
    plan time.

    Observed and simulated states may carry different ``Object`` instances
    (e.g. from separately-constructed envs / real-env trajectories), so
    features are matched by object name.

    With ``scaling``, residuals are wrapped (angular features) and
    normalized per feature (see :class:`ResidualScaling`); the SSE is
    then dimensionless. All evaluations belonging to one fit must share
    the same ``scaling`` object or their SSEs are incomparable.
    """
    return sum(r * r for r in _iter_rollout_residual_terms(
        base_env, trajectories, params, residual_features, physical_names,
        rules, latent_init, scaling, config))


def compute_rollout_residuals(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    residual_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any] = (),
    latent_init: Any = None,
    scaling: Optional[ResidualScaling] = None,
    config: Optional[SysIdConfig] = None,
) -> np.ndarray:
    """Rollout residuals (predicted - observed, scaled) as a flat vector.

    Levenberg-Marquardt counterpart of :func:`compute_rollout_sse` (same
    prediction pipeline, same iteration order — the sim is deterministic,
    so the same theta yields the same vector, as finite-difference
    Jacobians require).
    """
    return np.asarray(list(
        _iter_rollout_residual_terms(base_env, trajectories, params,
                                     residual_features, physical_names, rules,
                                     latent_init, scaling, config)),
                      dtype=float)


def _huberize(residual: float, delta: float) -> float:
    """Soft-cap a residual so its SQUARE follows the Huber loss.

    Returns ``r'`` with ``r'**2 == huber_delta(r)``: identity inside
    ``delta``, ``sign(r) * sqrt(2*delta*|r| - delta**2)`` outside, so
    every squared-residual consumer (SSE, LM least-squares, RMS) uses
    the robust objective through one transform. ``delta <= 0``
    disables. Motivation: a qualitatively-diverged replay's per-step
    residuals are chaos in theta - quadratic growth lets one such
    replay outvote every clean observation and steer the grid fit to a
    wrong basin (measured: SSE 248.7 at theta 0.4746 vs 0.0025 at
    0.4743 on the same recording); linear growth keeps it penalized
    without dominance.
    """
    if delta <= 0:
        return residual
    a = abs(residual)
    if a <= delta:
        return residual
    return float(np.sign(residual) * np.sqrt(2.0 * delta * a - delta * delta))


def _iter_rollout_residual_terms(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    residual_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any],
    latent_init: Any,
    scaling: Optional[ResidualScaling] = None,
    config: Optional[SysIdConfig] = None,
) -> Iterator[float]:
    """Yield per-feature residuals for the joint forward model.

    Shared prediction pipeline behind :func:`compute_rollout_sse` and
    :func:`compute_rollout_residuals`: free-run the base sim at the
    physical slice of ``params``, apply rules (latents threaded) on each
    rolled-out state, and score every in-scope feature against its
    observation. Deterministic iteration order. Without ``scaling`` the
    residual is the raw ``pred - obs`` difference (legacy objective).

    Each per-step residual is Huber-capped (``huber_delta``), and two
    kinds of per-trajectory SUMMARY residuals are appended with weight
    ``summary_weight`` (see the flags in ``settings.py``): the settled
    ENDPOINT residuals (the trajectory's final scored features - the
    rest poses a plan actually depends on, re-emphasized because they
    are 1 of N steps in the per-step sum) and per-object motion ONSET
    residuals (first step any scored feature moves beyond
    ``settle_tol``, normalized by the horizon - event timing is smooth
    in the physical params where mid-flight paths are chaos).
    """
    # pylint: disable=import-outside-toplevel
    from predicators.code_sim_learning.commands import CommandBuffer
    from predicators.code_sim_learning.utils import apply_rules, \
        apply_rules_with_latent, has_latent_rules, init_latent

    # pylint: enable=import-outside-toplevel
    config = config or SysIdConfig.from_cfg()
    delta = config.huber_delta
    summary_w = np.sqrt(max(config.summary_weight, 0.0))
    # Loaded once per objective evaluation, not per trajectory: the fit calls
    # this for every candidate theta, and re-reading the JSON each time would
    # put a file read inside the inner loop of a sweep.
    loaded = _load_scored_track(config) if config.score_observed_only else None
    tracks: List[Any] = loaded or []
    score_intervals = bool(tracks)
    # Once per objective evaluation, and once per EPISODE rather than per
    # scored segment: the positions only line up at the episode's start.
    id_maps = (_episode_id_maps(tracks, trajectories, config)
               if score_intervals else [])
    physical = {n: params[n] for n in physical_names if n in params}
    rules_list = list(rules)
    latent_mode = bool(rules_list) and has_latent_rules(rules_list)

    for traj_index, (states, actions) in enumerate(trajectories):
        latent: Dict[str, Any] = (init_latent(latent_init, params)
                                  if latent_mode else {})
        history: List[Tuple[State, Optional[Action]]] = []
        # Rules run IN the rollout loop, not on its output: physics
        # commands must shape the remainder of this very rollout, or
        # the objective would score a commands-free trajectory.
        # Feature updates stay scoring-side overrides of the rolled-out
        # state (never written back into the physics world).
        updates_per_step: List[Dict[Any, Dict[str, Any]]] = []

        # pylint: disable=cell-var-from-loop
        # (Consumed within this same loop iteration, before rebinding.)
        def _run_rules_post_step(env: Any, sim_state: State, i: int) -> None:
            cmds = CommandBuffer()
            if latent_mode:
                history.append((sim_state, actions[i]))
                step_updates = apply_rules_with_latent(sim_state,
                                                       latent,
                                                       history,
                                                       rules_list,
                                                       params,
                                                       cmds=cmds)
            else:
                step_updates = apply_rules(sim_state,
                                           rules_list,
                                           params,
                                           cmds=cmds)
            updates_per_step.append(step_updates)
            if cmds:
                env.queue_residual_commands(cmds.commands)

        # pylint: enable=cell-var-from-loop
        sim_states = rollout_states(
            base_env,
            states[0],
            actions,
            physical,
            post_step=_run_rules_post_step if rules_list else None)
        if score_intervals:
            # The per-step loop below is skipped entirely rather than added
            # to. Under open-loop nothing corrects the twin, so those steps
            # are the twin's own simulation and including them would let the
            # defect this flag exists to fix outvote the real evidence by
            # thousands of terms to a handful.
            yield from _interval_residual_terms(
                sim_states, _track_for(tracks, traj_index, len(trajectories)),
                id_maps[traj_index], config, summary_w)
            continue
        endpoint_residuals: List[float] = []
        for i, sim_state in enumerate(sim_states):
            obs_state = states[i + 1]
            updates: Dict[Any, Dict[str, Any]] = (updates_per_step[i]
                                                  if rules_list else {})
            obs_by_name = {o.name: o for o in obs_state}
            is_last = i == len(sim_states) - 1
            for obj in sim_state:
                feats = residual_features.get(obj.type.name, [])
                if not feats:
                    continue
                obs_obj = obs_by_name.get(obj.name)
                if obs_obj is None:
                    continue
                feat_updates = updates.get(obj, {})
                for feat in feats:
                    pred = feat_updates.get(feat, sim_state.get(obj, feat))
                    pred_val = float(
                        pred.item() if hasattr(pred, "item") else pred)
                    obs_val = float(obs_state.get(obs_obj, feat))
                    if scaling is not None:
                        res = scaling.residual(obj.type.name, feat, pred_val,
                                               obs_val)
                    else:
                        res = pred_val - obs_val
                    if is_last and summary_w > 0:
                        endpoint_residuals.append(res)
                    yield _huberize(res, delta)
        if summary_w > 0 and sim_states:
            for res in endpoint_residuals:
                yield summary_w * _huberize(res, delta)
            yield from _onset_residuals([states[0]] + sim_states, states,
                                        residual_features, config.settle_tol,
                                        summary_w)


def _load_scored_track(config: SysIdConfig) -> Optional[Any]:
    """The observation track to score against, or None to fall back.

    Missing or unreadable is a WARNING and a fallback to per-step
    scoring, never an empty residual set: scoring nothing would make
    every theta equally good and return the prior centre with a
    confident-looking identifiability report. Failing loud is the whole
    point of the flag.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import load_tracks
    if not config.track_path:
        logging.warning(
            "code_sim_learning_rollout_score_observed_only is on but no "
            "track path is set; falling back to per-step scoring, which "
            "under open-loop execution scores the twin against itself.")
        return None
    cached = _TRACK_CACHE.get(config.track_path)
    if cached is not None:
        return cached
    try:
        tracks = load_tracks(config.track_path, config.track_fallback_fps,
                             config.track_wait_s)
    except (OSError, ValueError, KeyError) as e:
        logging.warning(
            "could not read the observation track %s (%s); falling back to "
            "per-step scoring.", config.track_path, e)
        return None
    tracks = [t for t in tracks if t.angles_deg]
    if not tracks:
        logging.warning(
            "no usable observation track at %s (none carried domino angles, "
            "or none has been post-processed yet); falling back to per-step "
            "scoring.", config.track_path)
        return None
    tracks = [_track_in_world_frame(t, config) for t in tracks]
    _TRACK_CACHE[config.track_path] = tracks
    return tracks


def _track_in_world_frame(track: Any, config: SysIdConfig) -> Any:
    """The same track with its positions in the env's world frame.

    Only the positions move. The angles the onsets are detected from are
    per-domino rotations about the table normal, so a yaw of the whole
    frame leaves every one of them unchanged -- which is why this is a
    matching concern and not a scoring one.

    Identity when the flags are unset, and the track is returned
    untouched rather than rebuilt, so an env whose track already shares
    the twin's frame pays nothing and cannot be perturbed.
    """
    yaw = float(config.track_frame_yaw)
    off_x, off_y = (float(v) for v in config.track_frame_xy)
    if yaw == 0.0 and off_x == 0.0 and off_y == 0.0:
        return track
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)

    def _moved(xy: Dict[int, Any]) -> Dict[int, Any]:
        """Both position sets, or the anchor and the fallback disagree."""
        return {
            obj_id:
            (off_x + cos_y * x - sin_y * y, off_y + sin_y * x + cos_y * y)
            for obj_id, (x, y) in xy.items()
        }

    return dataclasses.replace(track,
                               first_xy=_moved(track.first_xy),
                               pre_cascade_xy=_moved(track.pre_cascade_xy))


def _track_for(tracks: List[Any], index: int, n_trajectories: int) -> Any:
    """Which track scores trajectory ``index``.

    One track per episode, paired in order, which is how the manifest is
    written. When the counts disagree every trajectory is scored against
    the most recent track and the mismatch is logged loudly: the usual
    cause is rest-point segmentation splitting one episode into several
    scored segments, which is legitimate -- every segment of an episode
    does belong to that episode's track -- but the positional pairing
    can no longer say which.
    """
    if len(tracks) == n_trajectories:
        return tracks[index]
    if index == 0:
        logging.warning(
            "%d trajectory/ies but %d track(s): scoring all of them against "
            "the most recent track. Rest-point segmentation splitting an "
            "episode is the usual cause; a per-segment track would need the "
            "episode identity that segmentation drops.", n_trajectories,
            len(tracks))
    return tracks[-1]


def _episode_id_maps(tracks: List[Any], trajectories: List[RolloutTrajectory],
                     config: SysIdConfig) -> List[Dict[str, int]]:
    """Match track ids to object names once per EPISODE, not per segment.

    The match is positional, and the one moment the two position sets
    are known to agree is the track's first frame against the episode's
    INITIAL state. Rest-point segmentation then splits that episode into
    several scored trajectories, and every segment after the first
    starts from a state the episode has already rearranged: a
    pick-and-place moves a domino by ~200 mm, five times the matching
    tolerance, so a per-segment match silently drops those objects and
    the intervals they carry.

    When the counts pair one-to-one, each trajectory is its own episode
    and anchors on its own initial state. Otherwise segmentation has
    split something -- ``_track_for`` scores every trajectory against
    the most recent track -- and the anchor is the FIRST trajectory's
    initial state, which is where the episode began, before the plan
    moved anything.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        match_ids_by_xy, settled_xy_before_cascade, track_name_to_id
    if not trajectories or not tracks:
        return []
    prefix = config.track_object_prefix

    def _map_for(states: List[State], track: Any) -> Dict[str, int]:
        """Match the two settled arrangements, not two arbitrary moments."""
        twin_xy = settled_xy_before_cascade(
            states,
            prefix,
            confirm_deg=config.onset_confirm_deg,
            onset_deg=config.onset_deg,
            min_persist=config.onset_min_persist)
        track_xy = track.pre_cascade_xy or track.first_xy
        name_to_id = match_ids_by_xy(twin_xy, track_xy)
        if not name_to_id:
            logging.warning(
                "falling back to matching track ids by object name, which "
                "assumes the initialization boxes were drawn in the env's "
                "own domino order")
            name_to_id = track_name_to_id(states[0], prefix)
        return name_to_id

    if len(tracks) == len(trajectories):
        return [
            _map_for(list(states), tracks[i])
            for i, (states, _) in enumerate(trajectories)
        ]
    # Segmentation split an episode, so the cascade may be in any segment:
    # the whole episode's states in order, which is what the split came from.
    whole = [s for states, _ in trajectories for s in states]
    shared = _map_for(whole, tracks[-1])
    return [shared] * len(trajectories)


def _interval_residual_terms(sim_states: List[State], track: Any,
                             name_to_id: Dict[str, int], config: SysIdConfig,
                             summary_w: float) -> Iterator[float]:
    """Yield (sim - observed) propagation intervals, in seconds.

    The residual set for a whole trajectory is one term per domino
    after the first to fall. That is a handful of numbers where the
    per-step objective has thousands, and deliberately so: those
    thousands are the twin's own simulation under open-loop, and only
    these carry the real cascade.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.code_sim_learning.observation_track import \
        interval_residuals, propagation_intervals, sim_topple_series, \
        topple_onsets
    if not sim_states:
        return
    # Matched once for the whole episode by _episode_id_maps, because only
    # the episode's initial state is contemporaneous with the track's first
    # frame; see that function.
    if not name_to_id:
        logging.warning(
            "no object name starts with %r, so nothing in the rollout maps "
            "onto the track's domino ids; this trajectory contributes no "
            "interval residuals.", config.track_object_prefix)
        return
    step_s = CFG.pybullet_sim_steps_per_action / 240.0
    sim_series = sim_topple_series(sim_states, step_s, name_to_id)

    def _onsets(series: Any) -> Dict[int, float]:
        """Both sides detected identically, which is the point."""
        return topple_onsets(series,
                             confirm_deg=config.onset_confirm_deg,
                             onset_deg=config.onset_deg,
                             min_persist=config.onset_min_persist)

    sim_intervals = propagation_intervals(_onsets(sim_series))
    obs_intervals = propagation_intervals(_onsets(track.angles_deg))
    # A cascade that fails to propagate in one of the two is the strongest
    # evidence there is, so the stand-in is the track's own span rather than
    # a small number: it must cost more than any real disagreement.
    penalty = max(track.duration_s, step_s * len(sim_states))
    for res in interval_residuals(sim_intervals, obs_intervals, penalty):
        yield summary_w * res


def _onset_residuals(sim_states: List[State], obs_states: List[State],
                     residual_features: Dict[str, List[str]],
                     motion_tol: float, weight: float) -> Iterator[float]:
    """Per-object (sim onset - observed onset) / horizon, weighted.

    The onset is the first state index at which ANY of the object's
    scored features deviates from its initial value by more than
    ``motion_tol`` (the same "still moving" tolerance the settled-tail
    truncation uses); objects that never move in either trajectory
    contribute nothing. Both trajectories share the initial state (the
    rollout is reset to it), so the baselines match by construction.
    """
    horizon = max(len(obs_states) - 1, 1)

    def _onset(states: List[State], obj_name: str, feats: List[str],
               baseline: State) -> Optional[int]:
        base_obj = {o.name: o for o in baseline}[obj_name]
        for t, state in enumerate(states):
            objs = {o.name: o for o in state}
            obj = objs.get(obj_name)
            if obj is None:
                continue
            for feat in feats:
                if abs(
                        float(state.get(obj, feat)) -
                        float(baseline.get(base_obj, feat))) > motion_tol:
                    return t
        return None

    baseline = obs_states[0]
    for obj in baseline:
        feats = residual_features.get(obj.type.name, [])
        if not feats:
            continue
        obs_onset = _onset(obs_states, obj.name, feats, baseline)
        sim_onset = _onset(sim_states, obj.name, feats, baseline)
        if obs_onset is None and sim_onset is None:
            continue
        obs_t = obs_onset if obs_onset is not None else horizon
        sim_t = sim_onset if sim_onset is not None else horizon
        yield weight * (sim_t - obs_t) / horizon


def fit_map_lm_rollout(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    physical_specs: Sequence[ParamSpec],
    residual_features: Dict[str, List[str]],
    rules: Sequence[Any] = (),
    rule_specs: Sequence[ParamSpec] = (),
    latent_init: Any = None,
    max_nfev: int = 200,
    scaling: Optional[ResidualScaling] = None,
    prior_centers: Optional[np.ndarray] = None,
    prior_sigmas: Optional[np.ndarray] = None,
    noise_sigma: float = 0.05,
    fixed_physical: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """MAP estimate of the joint physical+rule theta via Levenberg-Marquardt.

    ``fixed_physical`` holds extra physical params at EXPLICIT constant
    values throughout the fit (pushed into the env alongside the fitted
    ones, not left to the env registry's defaults) - used by the anchor
    ablation to pin reverted params at exactly the anchor it records.

    Rollout counterpart of :func:`training.fit_map_lm`, built on
    :func:`compute_rollout_residuals` and the shared bound-aware
    ``solve_lm`` core. Uses a coarse relative finite-difference step
    (``_ROLLOUT_LM_DIFF_STEP``) because simulation residuals are flat
    under scipy's default ~1e-8 perturbations. Returns ``(theta_map,
    jacobian_at_optimum)``; the Jacobian feeds both the identifiability
    report and the Laplace exploration ensemble.

    When ``prior_centers``/``prior_sigmas`` (FIT-space, one per joint
    param) are given, the Gaussian prior is folded into the LM
    objective as extra residual rows ``noise_sigma * (z - c) / sigma``,
    so the point estimate is a true MAP: a direction the data leaves
    flat stays exactly at its anchor instead of drifting on rollout
    noise (a flat likelihood's finite-difference gradient is pure
    noise). The prior rows are stripped from the returned Jacobian -
    its consumers (Laplace ensemble, Hessian diagnostic) add the prior
    term themselves and would otherwise double-count it.
    """
    all_specs = list(physical_specs) + list(rule_specs)
    names = [s.name for s in all_specs]
    fixed = dict(fixed_physical or {})
    physical_names = list(fixed) + [s.name for s in physical_specs]
    use_prior = prior_centers is not None and prior_sigmas is not None

    def residuals_fn(theta: np.ndarray) -> np.ndarray:
        params = {n: float(theta[i]) for i, n in enumerate(names)}
        params.update(fixed)
        res = compute_rollout_residuals(base_env, trajectories, params,
                                        residual_features, physical_names,
                                        rules, latent_init, scaling)
        if not use_prior:
            return res
        z = to_fit_space(all_specs, theta)
        prior_res = noise_sigma * (z - prior_centers) / prior_sigmas
        return np.concatenate([res, prior_res])

    theta_map, jac = solve_lm(residuals_fn,
                              all_specs,
                              max_nfev,
                              "rollout",
                              diff_step=_ROLLOUT_LM_DIFF_STEP)
    if use_prior and jac is not None and jac.shape[0] > len(all_specs):
        jac = jac[:-len(all_specs)]
    return theta_map, jac


def per_trajectory_rms(
    base_env: Any,
    trajectories: List[RolloutTrajectory],
    params: Dict[str, float],
    residual_features: Dict[str, List[str]],
    physical_names: Sequence[str],
    rules: Sequence[Any] = (),
    latent_init: Any = None,
    scaling: Optional[ResidualScaling] = None,
    config: Optional[SysIdConfig] = None,
) -> List[float]:
    """RMS rollout residual of each trajectory separately at ``params``.

    The per-trajectory analogue of :func:`compute_rollout_sse`: how well
    can the model explain THIS trajectory at these parameters? Without
    ``scaling`` the RMS is in the scored features' native units (meters
    / radians per residual); with it, a dimensionless fraction of
    typical motion.
    """
    out: List[float] = []
    for traj in trajectories:
        res = compute_rollout_residuals(base_env, [traj], params,
                                        residual_features, physical_names,
                                        rules, latent_init, scaling, config)
        out.append(float(np.sqrt(np.mean(res**2))) if res.size else 0.0)
    return out
