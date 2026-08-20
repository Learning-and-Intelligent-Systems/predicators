"""Partially-observable ground-truth simulator for pybullet_boil.

Sibling of ``gt_simulator.py`` for the partially-observable (PO) setting,
where the jug's ``heat_level`` is *not* an observable feature: the agent
sees only the derived ``bubbling_level`` and must infer the hidden
heating process. This module is the answer-key for that inference — it
encodes the latent heat explicitly and maps it to the observable through
the same monotone ramp the environment uses.

Differences from the fully-observable ``gt_simulator.py``:

* ``_heating`` uses the recurrent 5-arg signature
  ``rule(state, latent, history, updates, params)``. It carries the
  hidden per-jug heat in ``latent["heat"]`` (a ``{jug_name: heat}`` dict
  threaded across steps by ``compute_sse_recurrent``) and writes only the
  *observable* ``bubbling_level`` via
  ``clip((heat - BUBBLING_ONSET) * BUBBLING_RAMP, 0, 1)`` — it never reads
  or writes a ``heat_level`` feature, which does not exist in PO mode.
* ``LATENT_INIT`` declares the initial latent block (heat = 0 per jug).
  It is the *callable* form so each rollout gets its own nested dict
  (a module-level literal would be shared across trajectories by
  ``init_latent`` and silently accumulate).
* ``RESIDUAL_FEATURES`` scopes the fit to the jug observables
  (``water_volume``, ``bubbling_level``); the fully-observable module's
  spill/happiness chain is dropped here, keeping the reference focused on
  the partially-observable signal.
* Gates are *hard* (no sigmoid smoothing). The recurrent fit
  (``fit_params_recurrent``) now has an optional LM path
  (``fit_map_lm_recurrent``) feeding the Hessian diagnostic and the
  Laplace ensemble, but a hard gate makes a threshold param's residual
  flat-with-a-cliff: LM's finite-difference Jacobian column for it is
  ~0, so Laplace reports it as wide-open (prior-driven) rather than
  pinning the boundary. The *rates* are still recovered cleanly from the
  smooth bubbling ramp. For calibrated uncertainty on the hard-gated
  thresholds, run with ``num_mcmc_steps > 0`` (the posterior-subsample
  ensemble) or soft-gate them as the FO module does.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import History, Params, \
    ResidualUpdate, objs_by_type
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import State

# ── Constants ────────────────────────────────────────────────────

# Physical defaults matching pybullet_boil.py exactly. water_fill_speed
# is derived from CFG at spec-build time (env uses
# CFG.boil_water_fill_speed * water_height_to_level_ratio).
HEATING_SPEED = 0.03
MAX_JUG_WATER_CAPACITY = 1.3
FAUCET_ALIGN_THRESHOLD = 0.1
BURNER_ALIGN_THRESHOLD = 0.05
FAUCET_X_LEN = 0.15
_WATER_HEIGHT_TO_LEVEL_RATIO = 10

# Bubbling readout (env's PO projection of the hidden heat): bubbling
# stays 0 until heat crosses BUBBLING_ONSET, then ramps linearly to 1.0
# at heat == 1.0. These are env constants, not learned parameters.
BUBBLING_ONSET = 0.85
BUBBLING_RAMP = 1.0 / (1.0 - BUBBLING_ONSET)  # ≈ 6.667

# ── Residual rules ────────────────────────────────────────────────


def _water_filling(state: State, updates: ResidualUpdate,
                   params: Params) -> ResidualUpdate:
    """Faucet on + nearest non-held jug aligned and under capacity → fill.

    Fully observable: ``water_volume`` is a visible feature, so no
    latent is needed (legacy 3-arg rule). Alignment / capacity gates are
    hard (no smoothing) — the recurrent fit is gradient-free, so the
    differentiability the FO module's soft gates provided is
    unnecessary.
    """
    objs = objs_by_type(state)
    for faucet in objs.get("faucet", []):
        if state.get(faucet, "is_on") <= 0.5:
            continue

        fx = float(state.get(faucet, "x"))
        fy = float(state.get(faucet, "y"))
        frot = float(state.get(faucet, "rot"))
        out_x = fx + params["faucet_x_len"] * np.cos(frot)
        out_y = fy - params["faucet_x_len"] * np.sin(frot)

        # Closest non-held jug picks up the catch (matches the original
        # "first aligned wins" semantics for single-jug tasks).
        best_jug, best_dist = None, float("inf")
        for jug in objs.get("jug", []):
            if state.get(jug, "is_held") > 0.5:
                continue
            jx = float(state.get(jug, "x"))
            jy = float(state.get(jug, "y"))
            d = float(np.hypot(out_x - jx, out_y - jy))
            if d < best_dist:
                best_jug, best_dist = jug, d

        if best_jug is None or best_dist >= params["faucet_align_threshold"]:
            continue
        water = float(state.get(best_jug, "water_volume"))
        new_water = min(params["max_jug_water_capacity"],
                        water + params["water_fill_speed"])
        updates.setdefault(best_jug, {})["water_volume"] = new_water

    return updates


def _heating(  # pylint: disable=unused-argument
        state: State, latent: Dict[str, Any], history: History,
        updates: ResidualUpdate, params: Params) -> ResidualUpdate:
    """Burner on + jug with water aligned → accumulate hidden heat, surfaced
    through the observable ``bubbling_level``.

    ``heat_level`` is not observable in PO mode, so this rule carries the
    per-jug heat in ``latent["heat"]`` (a ``{jug_name: heat}`` dict
    threaded across steps) and emits only the derived observable
    ``bubbling_level``. The readout is the env's exact monotone ramp:
    bubbling is 0 until heat crosses ``BUBBLING_ONSET``, then ramps to 1.0
    at heat == 1.0. Heat never decreases (no cooling in the env), so
    neither does bubbling. This is Pattern B (physical latent + monotone
    readout) from the recurrent approach prompt.

    ``history`` is unused: the carried heat is a sufficient statistic of
    the on-burner steps so far, so no look-back is needed. (Like the FO
    module, this reference omits the env's one-step burner warm-up, where
    heat begins only on the *second* consecutive on-step — a ~1-step
    phase offset out of the ~34 steps to boil.)
    """
    heats: Dict[str, float] = latent.setdefault("heat", {})
    objs = objs_by_type(state)
    burners = objs.get("burner", [])

    for jug in objs.get("jug", []):
        heat = float(heats.get(jug.name, 0.0))
        # Heat accumulates only while the jug (with water, not held) sits
        # on a turned-on, aligned burner.
        if (state.get(jug, "is_held") <= 0.5
                and state.get(jug, "water_volume") > 0.0):
            jx = float(state.get(jug, "x"))
            jy = float(state.get(jug, "y"))
            for burner in burners:
                if state.get(burner, "is_on") <= 0.5:
                    continue
                bx = float(state.get(burner, "x"))
                by = float(state.get(burner, "y"))
                if float(np.hypot(bx - jx, by - jy)) < \
                        params["burner_align_threshold"]:
                    heat = min(1.0, heat + params["heating_speed"])
                    break  # one increment per step regardless of count
        heats[jug.name] = heat
        # Monotone readout of the latent onto the observable (Pattern B).
        bubbling = max(0.0, min(1.0, (heat - BUBBLING_ONSET) * BUBBLING_RAMP))
        updates.setdefault(jug, {})["bubbling_level"] = bubbling

    return updates


# ── Latent block ─────────────────────────────────────────────────


def _latent_init() -> Dict[str, Dict[str, float]]:
    """Fresh per-jug heat block for a new rollout.

    Callable (not a literal) so every ``init_latent`` call gets its own
    nested ``{jug_name: heat}`` dict; a shared module-level literal
    would accumulate heat across trajectories.
    """
    return {"heat": {}}


# ── Param specs ──────────────────────────────────────────────────


def _build_param_specs() -> List[ParamSpec]:
    """Build at call time so CFG-driven values match the current run.

    Only the *rates* (fill / heating speed) are exposed as learnable
    here: they are identifiable from the smooth observable ramps even
    under hard gates. The geometric thresholds are passed through as
    fixed specs at their true values for parity with the FO module, but
    are not meaningfully identifiable from a hard gate.
    """
    water_fill_speed = (CFG.boil_water_fill_speed *
                        _WATER_HEIGHT_TO_LEVEL_RATIO)
    return [
        ParamSpec("water_fill_speed", water_fill_speed, lo=0.0),
        ParamSpec("heating_speed", HEATING_SPEED, lo=0.0),
        ParamSpec("max_jug_water_capacity", MAX_JUG_WATER_CAPACITY, lo=0.0),
        ParamSpec("faucet_x_len", FAUCET_X_LEN, lo=0.0),
        ParamSpec("faucet_align_threshold", FAUCET_ALIGN_THRESHOLD, lo=0.0),
        ParamSpec("burner_align_threshold", BURNER_ALIGN_THRESHOLD, lo=0.0),
    ]


# ── Public API: consumed by read_simulator_components ────────────
# Same contract as agent-synthesized simulator files, plus the optional
# LATENT_INIT export read by the recurrent partial-observability
# approach. PARAM_SPECS is bound to the *callable* so CFG-dependent
# defaults resolve when the loader pulls the value, after CFG is final.

RESIDUAL_RULES = [_water_filling, _heating]

PARAM_SPECS = _build_param_specs

LATENT_INIT = _latent_init

RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "jug": ["water_volume", "bubbling_level"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletBoilPOGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """PO GT residual-dynamics simulator for pybullet_boil.

    Claims ``pybullet_boil`` only in partially-observable mode; the
    fully-observable ``gt_simulator.py`` claims it otherwise, so
    ``get_gt_simulator``'s env-name dispatch resolves to exactly one
    module per run.
    """

    @classmethod
    def get_env_names(cls) -> set:
        if CFG.partially_observable:
            return {"pybullet_boil"}
        return set()
