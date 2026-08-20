"""Utilities for the code sim-learning module.

Two groups live here:

Rule-DSL primitives — the API residual-rule simulators (oracle modules
and agent-synthesized code) are written against:

* ``apply_rules`` / ``apply_rules_with_latent`` — run a list of rule
  functions on a state, return feature updates (``ResidualUpdate``);
  the latent variant threads a hidden-state dict and ``History``. Both
  thread an optional :class:`~predicators.code_sim_learning.commands.
  CommandBuffer` into rules that declare a ``cmds`` parameter (the
  physics-command output channel); ``has_physics_rules`` is the
  dispatch signal that such rules exist and must be scored by
  env-in-the-loop rollout matching.
* ``merge_updates`` — overwrite features in a ``State`` with values
  from a ``ResidualUpdate``.
* ``rollout_predictions`` / ``iter_feature_residuals`` — teacher-forced
  rollouts of a rule set over recorded trajectories.
* ``sigmoid`` / ``SOFT_EPS`` — building blocks for differentiable
  soft gates in residual rules.
* ``objs_by_type`` / ``init_latent`` / ``has_latent_rules`` — rule
  introspection and setup helpers.

Simulator-artifact decoding — how a synthesized ``simulator.py``
namespace is unpacked:

* ``read_simulator_components`` — pull the ``RESIDUAL_RULES``,
  ``PARAM_SPECS``, ``RESIDUAL_FEATURES`` triple out of a namespace
  (oracle module globals or agent-synthesized exec namespace).
* ``read_latent_init`` / ``read_physical_param_specs`` /
  ``stamp_physical_spec_scales`` — optional artifact entries.
* ``LearnedSimulator`` — fail-soft wrapper around a step function.
"""

from __future__ import annotations

import inspect
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, \
    Optional, Sequence, Tuple

import numpy as np

from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.structs import Action, Object, State

logger = logging.getLogger(__name__)

# ── Type aliases ──────────────────────────────────────────────────

# {Object: {feature_name: new_value}} — the dict that rule functions
# accumulate into.
ResidualUpdate = Dict[Object, Dict[str, float]]

# {param_name: value} — the params dict passed to rule functions.
Params = Dict[str, float]

# ── Soft-gate building blocks ─────────────────────────────────────

# Default smoothing scale for parameter-dependent soft gates. Small
# enough that gates are ~99% saturated when the operand is one
# threshold-width into the active region, large enough to give MCMC a
# usable gradient near the cliff. 0.02 is in the right ballpark for
# both spatial thresholds (~0.05–0.15 m) and water-level thresholds
# (~0.3–1.3). Override per call site as needed.
SOFT_EPS = 0.02


def sigmoid(z: float) -> float:
    """Numerically-stable scalar sigmoid."""
    if z >= 0:
        return 1.0 / (1.0 + np.exp(-z))
    ez = np.exp(z)
    return ez / (1.0 + ez)


def objs_by_type(state: State) -> Dict[str, List[Object]]:
    """Group state objects by type name."""
    groups: Dict[str, List[Object]] = {}
    for o in state:
        groups.setdefault(o.type.name, []).append(o)
    return groups


# ── Primitives ────────────────────────────────────────────────────


def apply_rules(state: State,
                rules: List,
                params: Dict[str, float],
                cmds: Optional[CommandBuffer] = None) -> ResidualUpdate:
    """Apply residual rules sequentially and return feature updates.

    Each rule has signature ``rule(state, updates, params) -> updates``,
    optionally extended with a trailing ``cmds`` parameter (the
    physics-command channel, see
    :mod:`predicators.code_sim_learning.commands`). ``cmds`` is threaded
    only into rules that declare it; when the caller passes ``None`` a
    throwaway buffer is used so command-emitting rules still run, but
    their commands are discarded - callers that can execute commands
    must pass their own buffer and hand its contents to the env. Values
    are normalised to plain floats (rules may return numpy scalars).
    """
    buf = cmds if cmds is not None else CommandBuffer()
    updates: ResidualUpdate = {}
    for rule in rules:
        if _rule_accepts_cmds(rule):
            updates = rule(state, updates, params, cmds=buf)
        else:
            updates = rule(state, updates, params)
    return {
        obj: {feat: float(val)
              for feat, val in feat_dict.items()}
        for obj, feat_dict in updates.items()
    }


# ── Recurrent rule support (latent + history) ─────────────────────

# Read-only history prefix handed to recurrent rules:
# [(state_0, action_0), (state_1, action_1), ..., (state_t, action_t)]
# Most recent last. The first entry's action is ``None``. Typed as
# ``Sequence`` (covariant) so callers can pass a stricter
# ``List[Tuple[State, Action]]`` without an invariance complaint —
# rules treat history as read-only.
History = Sequence[Tuple[State, Optional[Action]]]


@lru_cache(maxsize=None)
def _rule_accepts_latent(rule: Callable) -> bool:
    """Return True iff ``rule`` declares a `latent` parameter or **kwargs.

    Used by :func:`apply_rules_with_latent` to thread the sample's
    `latent` state-feature block / `history` only into rules that opted
    in. Cached because rule callables are reused across many simulator
    invocations and ``inspect.signature`` isn't free.
    """
    try:
        params = inspect.signature(rule).parameters
    except (TypeError, ValueError):
        return False
    if "latent" in params:
        return True
    return any(p.kind == inspect.Parameter.VAR_KEYWORD
               for p in params.values())


@lru_cache(maxsize=None)
def _rule_accepts_cmds(rule: Callable) -> bool:
    """True iff ``rule`` declares an explicit ``cmds`` parameter.

    Unlike :func:`_rule_accepts_latent`, a bare ``**kwargs`` does NOT
    count: declaring ``cmds`` doubles as the fit-routing signal
    (:func:`has_physics_rules` forces the env-in-the-loop rollout
    objective), so it must be an explicit, auditable opt-in rather
    than something a generic signature acquires by accident.
    """
    try:
        params = inspect.signature(rule).parameters
    except (TypeError, ValueError):
        return False
    return "cmds" in params


def has_physics_rules(rules: Iterable[Callable]) -> bool:
    """True iff any rule declares a ``cmds`` param (physics commands).

    The dispatch signal that a simulator acts through the engine:
    command effects only exist through physics stepping, so fitting and
    residual scoring must run the env-in-the-loop rollout objective
    (teacher-forced pure-function scoring cannot see them). Mirrors
    :func:`has_latent_rules`: keyed off the rule *signatures* so it is
    correct on both the oracle and the agent-synthesis path.
    """
    return any(_rule_accepts_cmds(r) for r in rules)


def has_latent_rules(rules: Iterable[Callable]) -> bool:
    """True iff any rule declares a `latent` param (recurrent 5-arg).

    The dispatch signal that distinguishes a partially-observable
    simulator (carries a latent block) from a fully-observable one: it
    keys off the rule *signatures*, so it is correct on both the oracle
    path (where ``LATENT_INIT`` may not have been loaded) and the agent-
    synthesis path. Empty / all-legacy rule lists return False, so
    fully-observable approaches take their existing non-latent paths
    unchanged.
    """
    return any(_rule_accepts_latent(r) for r in rules)


def apply_rules_with_latent(
    state: State,
    latent: Dict[str, Any],
    history: History,
    rules: List,
    params: Dict[str, float],
    cmds: Optional[CommandBuffer] = None,
) -> ResidualUpdate:
    """Apply rules with a ``latent`` state-feature block and read-only
    ``history``.

    Each rule is either:

    * **Legacy 3-arg**: ``rule(state, updates, params) -> updates``.
      Called without latent/history; latent and history are ignored.
    * **Recurrent 5-arg**: ``rule(state, latent, history, updates,
      params) -> updates``. ``latent`` is mutated in place — the
      same dict object passed in by the caller is threaded across
      steps.

    Either form may additionally declare a trailing ``cmds`` parameter
    (the physics-command channel); it is threaded exactly as in
    :func:`apply_rules`, with a throwaway buffer when the caller passes
    ``None``.

    Signature is inspected once per rule (cached). Values are
    normalised to plain floats. The returned update dict has the
    same shape as ``apply_rules``'s output.
    """
    buf = cmds if cmds is not None else CommandBuffer()
    updates: ResidualUpdate = {}
    for rule in rules:
        extra = {"cmds": buf} if _rule_accepts_cmds(rule) else {}
        if _rule_accepts_latent(rule):
            updates = rule(state, latent, history, updates, params, **extra)
        else:
            updates = rule(state, updates, params, **extra)
    return {
        obj: {feat: float(val)
              for feat, val in feat_dict.items()}
        for obj, feat_dict in updates.items()
    }


def init_latent(
    latent_init: Optional[Dict[str, Any]],
    params: Dict[str, float],
) -> Dict[str, Any]:
    """Build the initial latent state-feature block for a fresh rollout.

    ``latent_init`` follows the same convention as ``PARAM_SPECS``: it
    may be ``None`` (empty block), a plain ``Dict[str, Any]``, or a
    zero-arg callable returning such a dict. Values may be
    :class:`~predicators.code_sim_learning.fit_space.ParamSpec`
    instances, in which case the corresponding entry from
    ``params[name]`` is used (falling back to ``init_value`` if the
    param hasn't been fit yet) — this lets MCMC fit the initial
    latent value alongside rate parameters.
    """
    if latent_init is None:
        return {}
    if callable(latent_init):
        latent_init = latent_init()
    if not isinstance(latent_init, dict):
        return {}
    out: Dict[str, Any] = {}
    for k, v in latent_init.items():
        if isinstance(v, ParamSpec):
            out[k] = params.get(v.name, v.init_value)
        else:
            out[k] = v
    return out


def merge_updates(
    base_state: State,
    updates: ResidualUpdate,
) -> State:
    """Overwrite features in *base_state* with values from *updates*."""
    if not updates:
        return base_state

    new_data = {}
    for obj in base_state:
        arr = base_state[obj].copy()
        if obj in updates:
            for feat_name, new_val in updates[obj].items():
                idx = obj.type.feature_names.index(feat_name)
                arr[idx] = new_val
        new_data[obj] = arr

    merged = base_state.copy()
    merged.data = new_data
    return merged


def rollout_predictions(
    rules: List,
    params: Dict[str, float],
    groups: Sequence[Sequence[Tuple[State, Action, State]]],
    latent_init: Any = None,
) -> List[Tuple[State, State]]:
    """Per-transition ``(predicted_next_state, observed_next_state)`` pairs.

    Mirrors exactly how the fitting engine runs the rules, so a tool that
    builds residuals from this can never disagree with the engine on the
    rule call convention:

    * **Recurrent (5-arg) rules** — when any rule declares a ``latent``
      param (:func:`has_latent_rules`), the ``latent`` block is built once
      per trajectory group via :func:`init_latent` and threaded across that
      group's steps, with a growing read-only ``history`` prefix. This is
      the same threading :func:`compute_sse_recurrent` does.
    * **Legacy (3-arg) rules** — each transition's base state is rolled
      independently through :func:`apply_rules`; ``latent``/``history`` are
      ignored and ``groups`` only controls iteration order.

    ``groups`` is a list of per-trajectory triple lists, each
    ``[(base_state, action, next_obs), ...]`` (the latent threads within a
    trajectory, not across). Output is flattened in group-then-step order,
    so passing ``[base_pred_triples]`` reproduces the flat input order.
    """
    latent_mode = has_latent_rules(rules)
    out: List[Tuple[State, State]] = []
    for group in groups:
        latent: Dict[str, Any] = (init_latent(latent_init, params)
                                  if latent_mode else {})
        history: List[Tuple[State, Optional[Action]]] = []
        for base_state, action, s_next_obs in group:
            if latent_mode:
                history.append((base_state, action))
                updates = apply_rules_with_latent(base_state, latent, history,
                                                  rules, params)
            else:
                updates = apply_rules(base_state, rules, params)
            s_pred = (merge_updates(base_state, updates)
                      if updates else base_state)
            out.append((s_pred, s_next_obs))
    return out


def iter_feature_residuals(
    triples: Iterable[Tuple[State, State]],
    feature_scope: Optional[Dict[str, List[str]]] = None,
) -> Iterator[Tuple[int, Object, str, str, float, float]]:
    """Yield ``(step_idx, obj, type_name, feat, pred_val, obs_val)``.

    Walks each ``(s_pred, s_obs)`` pair and emits one tuple per
    ``(object, feature)``. If ``feature_scope`` is provided, only
    features listed under each type name are emitted; otherwise every
    feature in the type's ``feature_names`` is emitted. Used by both the
    residual-based feature-discovery scan and the per-feature residual
    report so the two stay in sync.
    """
    for i, (s_pred, s_obs) in enumerate(triples):
        for obj in s_pred:
            tn = obj.type.name
            feats: Sequence[str] = (feature_scope.get(tn, []) if feature_scope
                                    is not None else obj.type.feature_names)
            for feat in feats:
                yield (
                    i,
                    obj,
                    tn,
                    feat,
                    float(s_pred.get(obj, feat)),
                    float(s_obs.get(obj, feat)),
                )


# ── Module-namespace loader ───────────────────────────────────────


def read_simulator_components(
    ns: Mapping[str, Any],
) -> Tuple[Optional[List], Optional[List], Optional[Dict[str, List[str]]]]:
    """Pull the simulator triple from a namespace (module or exec dict).

    Looks for three names by convention:

    * ``RESIDUAL_RULES`` — non-empty list of rule functions.
    * ``PARAM_SPECS``   — list of ``ParamSpec``, **or** a zero-arg
      callable returning such a list. The callable form lets oracle
      modules defer CFG-dependent values until consumption time, so the
      module can be imported before CFG is finalized; the agent's
      saved-file form normally just uses a list.
    * ``RESIDUAL_FEATURES`` — ``{type_name: [feature_names]}`` dict.

    Returns ``(rules, specs, features)`` with ``None`` for any
    missing-or-malformed component; callers decide how to react.

    The optional fourth component ``LATENT_INIT`` (used by the
    recurrent partial-observability approach) is read separately via
    :func:`read_latent_init` so existing callers don't have to grow
    a fourth tuple element.
    """
    # Legacy simulator files (pre-rename snapshots under logs/) still
    # define PROCESS_RULES / PROCESS_FEATURES; accept them as aliases.
    rules = ns.get("RESIDUAL_RULES", ns.get("PROCESS_RULES"))
    if not isinstance(rules, list) or not rules:
        rules = None

    specs = ns.get("PARAM_SPECS")
    if callable(specs):
        specs = specs()
    if not isinstance(specs, list) or not specs:
        specs = None

    features = ns.get("RESIDUAL_FEATURES", ns.get("PROCESS_FEATURES"))
    if features is not None and not isinstance(features, dict):
        features = None

    return rules, specs, features


def read_latent_init(ns: Mapping[str, Any]) -> Optional[Any]:
    """Pull ``LATENT_INIT`` (optional) from a simulator namespace.

    ``LATENT_INIT`` declares the initial values for the latent
    state-feature block used by the partial-observability approach.
    Returns ``None`` if not present or malformed; in that case the
    caller should default to an empty block.

    Accepted shapes:

    * ``Dict[str, Any]`` — literal initial values.
    * ``Callable[[], Dict[str, Any]]`` — zero-arg factory, called at
      consumption time. Mirrors the callable-``PARAM_SPECS`` pattern.
    """
    latent_init = ns.get("LATENT_INIT")
    if latent_init is None:
        return None
    if not (callable(latent_init) or isinstance(latent_init, dict)):
        return None
    return latent_init


def read_physical_param_specs(ns: Mapping[str, Any]) -> Optional[List]:
    """Pull ``PHYSICAL_PARAMS`` (optional) from a simulator namespace.

    ``PHYSICAL_PARAMS`` declares base-sim physical parameters to identify —
    a sparse subset of what the env reveals via
    ``get_physical_param_info()`` — as a list of ``ParamSpec`` (init value
    = the agent's hypothesis, bounds from the revealed info), **or** a
    zero-arg callable returning such a list (mirroring the callable
    ``PARAM_SPECS`` pattern). When present, these are fit *jointly* with
    ``PARAM_SPECS`` against free-running base-sim rollouts
    (:mod:`predicators.code_sim_learning.physical_sysid`), and a
    physics-only artifact (no ``RESIDUAL_RULES``) becomes valid. Returns
    ``None`` if absent or malformed.
    """
    specs = ns.get("PHYSICAL_PARAMS")
    if callable(specs):
        specs = specs()
    if not isinstance(specs, list) or not specs:
        return None
    return specs


def stamp_physical_spec_scales(specs: List, base_env: Any) -> List:
    """Stamp each physical ParamSpec's fit ``scale`` from the env registry.

    The env's ``get_physical_param_info()`` is the source of truth for
    which parameters are scale-like (fitted in log-space): agents copy
    name/init/bounds into their ``PHYSICAL_PARAMS`` but may omit
    ``scale``, and a silently-linear friction fit has no grid resolution
    at the low end of a decades-spanning box (measured: fitted 0.0114
    for a true 0.1 on run_20260706_171526). A registry entry that
    declares ``scale`` therefore overrides the agent's declaration;
    parameters the registry does not mark keep whatever the agent
    declared (default linear).
    """
    getter = getattr(base_env, "get_physical_param_info", None)
    info = getter() if callable(getter) else {}
    stamped = []
    for s in specs:
        scale = (info.get(s.name) or {}).get("scale",
                                             getattr(s, "scale", "linear"))
        stamped.append(
            ParamSpec(s.name, s.init_value, lo=s.lo, hi=s.hi, scale=scale))
    return stamped


# ── LearnedSimulator ──────────────────────────────────────────────


class LearnedSimulator:
    """Wraps a step-level simulator function (handwritten or LLM-synthesized).

    The function predicts residual dynamics — features like
    water_volume, heat_level, spilled_level that aren't captured by
    rigid body physics, and (for rules on the physics-command channel)
    the commands to queue on the base env for its next action.
    """

    # (state, command buffer) -> feature updates. The buffer is always
    # provided by predict_step; step functions whose rules don't use
    # the command channel simply leave it empty.
    StepFn = Callable[[State, CommandBuffer], ResidualUpdate]

    def __init__(self,
                 step_fn: StepFn,
                 name: str = "learned_simulator") -> None:
        self._step_fn = step_fn
        self.name = name

    def predict_step(self,
                     state: State,
                     cmds: Optional[CommandBuffer] = None) -> ResidualUpdate:
        """Predict residual feature updates for a single timestep.

        ``cmds`` collects physics commands emitted by command-channel
        rules; callers that can execute them pass their own buffer,
        others may omit it (commands are then discarded).

        Fails soft: agent-written simulators may raise on states they
        never anticipated, so any exception is logged and treated as "no
        update" rather than crashing the rollout.
        """
        buf = cmds if cmds is not None else CommandBuffer()
        try:
            return self._step_fn(state, buf)
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("Simulator '%s' step raised: %s", self.name, e)
            return {}
