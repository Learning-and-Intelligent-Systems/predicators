"""Trajectory preparation for the rollout system-ID objective.

Settled-tail truncation, rest-point segmentation (multiple shooting),
and the per-(type, feature) residual scaling shared by every SSE/RMS
evaluation of one fit. See the
:mod:`predicators.code_sim_learning.physical_sysid` module docstring
for the identification problem these serve.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import numpy as np

from predicators.code_sim_learning.config import SysIdConfig
from predicators.code_sim_learning.rollout_env import RolloutTrajectory
from predicators.structs import Action, State

logger = logging.getLogger(__name__)


def _active_step_indices(states: List[State], actions: List[Action],
                         residual_features: Dict[str, List[str]],
                         motion_tol: float) -> List[int]:
    """Indices of steps where any scored feature moved more than
    ``motion_tol``.

    Shared observed-delta scan behind :func:`truncate_settled_tail` and
    :func:`split_at_rest_points`: step ``i`` compares ``states[i]`` to
    ``states[i + 1]`` (objects matched by name) and counts as active as
    soon as one in-scope feature's per-step delta exceeds the tolerance.
    """
    active: List[int] = []
    for i in range(len(actions)):
        s_prev, s_next = states[i], states[i + 1]
        prev_by_name = {o.name: o for o in s_prev}
        for obj in s_next:
            feats = residual_features.get(obj.type.name, [])
            prev_obj = prev_by_name.get(obj.name)
            if not feats or prev_obj is None:
                continue
            if any(
                    abs(
                        float(s_next.get(obj, f)) -
                        float(s_prev.get(prev_obj, f))) > motion_tol
                    for f in feats):
                active.append(i)
                break
    return active


def truncate_settled_tail(
    trajectory: RolloutTrajectory,
    residual_features: Dict[str, List[str]],
    motion_tol: Optional[float] = None,
    margin: Optional[int] = None,
    config: Optional[SysIdConfig] = None,
) -> RolloutTrajectory:
    """Cut a recorded trajectory once its scored features have settled.

    Scans the OBSERVED per-step deltas of the ``residual_features`` and
    keeps everything up to the last step where any of them moved by more
    than ``motion_tol``, plus a ``margin`` of settle steps (so the
    rollout is still scored on coming to rest at the right pose). The
    static remainder is dropped: it contains no physics signal — the
    scored bodies no longer move — but re-scores whatever pose
    divergence the free-running rollout has accumulated on every
    remaining step, which is exactly the chaos-amplification term that
    drowned the friction signal in run_20260705_203314. Intermediate
    still phases are safe: the cut is anchored to the LAST motion, so a
    push -> settle -> second push trajectory keeps both pushes.

    A trajectory whose scored features never move carries no signal at
    all; it is truncated to the first ``margin`` steps (kept non-empty
    so callers' trajectory counts stay meaningful) and logged.
    """
    config = config or SysIdConfig.from_cfg()
    if motion_tol is None:
        motion_tol = config.settle_tol
    if margin is None:
        margin = config.settle_margin
    states, actions = trajectory
    active = _active_step_indices(states, actions, residual_features,
                                  motion_tol)
    last_active = active[-1] if active else -1
    if last_active < 0:
        logger.warning(
            "truncate_settled_tail: no scored feature ever moved more than "
            "%g in a %d-step trajectory; keeping only the first %d steps "
            "(the trajectory carries no physical-parameter signal).",
            motion_tol, len(actions), margin)
    keep = min(len(actions), last_active + 1 + margin)
    if keep >= len(actions):
        return trajectory
    return states[:keep + 1], actions[:keep]


@dataclass
class ResidualScaling:
    """Per-(type, feature) residual semantics for the rollout objective.

    ``angular`` features (declared on their :class:`~predicators.structs
    .Type` via ``angular_features``) have their prediction errors
    wrapped to [-pi, pi] before scoring, so equivalent orientations
    (a settled domino at roll -pi vs +pi) do not read as a (2*pi)^2
    error per step. Every residual is then divided by its feature's
    ``scales`` entry, making residuals dimensionless fractions of
    typical motion: without this, radians and meters share one implicit
    unit and rotation errors drown position information (measured on
    run_20260711_141026: roll+yaw carried 83% of the post-fit SSE, with
    max errors of exactly 2*pi and pi - pure representation artifacts).
    """

    angular: FrozenSet[Tuple[str, str]]
    scales: Dict[Tuple[str, str], float] = field(default_factory=dict)

    def residual(self, type_name: str, feat: str, pred: float,
                 obs: float) -> float:
        """Scaled (and, for angular features, wrapped) ``pred - obs``."""
        key = (type_name, feat)
        diff = pred - obs
        if key in self.angular:
            diff = (diff + np.pi) % (2.0 * np.pi) - np.pi
        return diff / self.scales.get(key, 1.0)

    def signature(self) -> Tuple:
        """Hashable identity for caching explainability verdicts."""
        return (tuple(sorted(self.angular)),
                tuple(sorted(
                    (k, round(v, 12)) for k, v in self.scales.items())))


def compute_residual_scaling(
    trajectories: List[RolloutTrajectory],
    residual_features: Dict[str, List[str]],
    config: Optional[SysIdConfig] = None,
) -> Optional[ResidualScaling]:
    """Data-derived :class:`ResidualScaling` for a fit's trajectory set.

    Angular features come from each Type's declared ``angular_features``
    metadata (the states carry their types, so no env handle is
    needed). Scales: angular features get a constant ``pi`` (the
    largest possible wrapped error, so a full topple-direction mistake
    scores ~0.5); linear features get their observed span (max - min)
    across ALL observed states of the fit data, floored at
    ``CFG.code_sim_learning_rollout_feature_scale_floor`` so static
    features do not amplify sensor noise. Computed from observations
    only, so it is deterministic per dataset and MUST be shared across
    every SSE/RMS evaluation of one fit - per-trajectory scales would
    make trimming verdicts incomparable.

    Returns ``None`` when ``code_sim_learning_rollout_scale_residuals``
    is off (raw, unwrapped residuals - the legacy objective).
    """
    config = config or SysIdConfig.from_cfg()
    if not config.scale_residuals:
        return None
    floor = config.feature_scale_floor
    angular: Set[Tuple[str, str]] = set()
    lo: Dict[Tuple[str, str], float] = {}
    hi: Dict[Tuple[str, str], float] = {}
    for states, _actions in trajectories:
        for state in states:
            for obj in state:
                feats = residual_features.get(obj.type.name, [])
                if not feats:
                    continue
                type_angular = set(getattr(obj.type, "angular_features", ()))
                for feat in feats:
                    key = (obj.type.name, feat)
                    if feat in type_angular:
                        angular.add(key)
                        continue
                    val = float(state.get(obj, feat))
                    lo[key] = min(lo.get(key, val), val)
                    hi[key] = max(hi.get(key, val), val)
    scales = {key: max(hi[key] - lo_val, floor) for key, lo_val in lo.items()}
    for key in angular:
        scales[key] = float(np.pi)
    return ResidualScaling(angular=frozenset(angular), scales=scales)


def split_at_rest_points(
    trajectory: RolloutTrajectory,
    residual_features: Dict[str, List[str]],
    motion_tol: Optional[float] = None,
    min_rest_steps: Optional[int] = None,
    margin: Optional[int] = None,
    config: Optional[SysIdConfig] = None,
) -> List[RolloutTrajectory]:
    """Split a recording into independently-scored rest-anchored segments.

    Multiple shooting for chaotic contact dynamics: free-running an
    entire manipulation trajectory lets small early divergence compound
    across phases, which shifts the SSE minimum away from the true
    parameters (replay-divergence bias - observed pulling the fitted
    friction both above and below truth on different runs). Cutting at
    rest points (every scored feature quiescent for at least
    ``min_rest_steps`` consecutive steps) bounds the compounding
    horizon while keeping each segment's zero-velocity re-anchor exact:
    the observed anchor state genuinely is at rest, so no momentum is
    discarded (the failure mode that rules out per-step teacher
    forcing, see the :mod:`.physical_sysid` module docstring).

    Each segment spans from the at-rest state just before its first
    motion to ``margin`` steps after its last motion (so it is still
    scored on settling at the right pose). Fully-static stretches
    between segments are dropped: they carry no parameter signal but
    would re-score accumulated divergence every step. A trajectory with
    no scored motion at all yields ``[]``.

    Trimming/consistency then operate per segment, so one chaotic phase
    (e.g. a scraping robot push) no longer discards the clean cascade
    recorded seconds later in the same episode.
    """
    config = config or SysIdConfig.from_cfg()
    if motion_tol is None:
        motion_tol = config.settle_tol
    if min_rest_steps is None:
        min_rest_steps = config.segment_min_rest_steps
    if margin is None:
        margin = config.settle_margin
    states, actions = trajectory
    num_steps = len(actions)
    active = _active_step_indices(states, actions, residual_features,
                                  motion_tol)
    if not active:
        return []
    runs: List[Tuple[int, int]] = []
    run_start = active[0]
    prev = active[0]
    for i in active[1:]:
        if i - prev > min_rest_steps:
            runs.append((run_start, prev))
            run_start = i
        prev = i
    runs.append((run_start, prev))
    segments: List[RolloutTrajectory] = []
    for a, b in runs:
        end = min(num_steps, b + 1 + margin)
        segments.append((states[a:end + 1], actions[a:end]))
    return segments
