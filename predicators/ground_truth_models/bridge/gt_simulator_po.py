"""Ground-truth simulator program for pybullet_bridge (glue construction)
residual dynamics -- partially-observable variant.

In PO mode the per-joint ``cure_*`` dwell counters are hidden: each
joint's progress is carried recurrently in ``latent["cure"]`` keyed by
``"<owner>|<face>"`` (the wet face uniquely identifies the joint; the
mate is geometric), and never written to an observable feature. Only
the wet-glue flags and the discrete ``attached_*`` flips are emitted --
the Pattern A (counter + threshold) archetype, with the twist that the
consequence is kinematic: once attached, the base sim welds the pair.
The threshold is identifiable from *when* joints latch across
trajectories.

Gates are hard (the recurrent fit is gradient-free).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import History, Params, \
    ResidualUpdate, objs_by_type
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import Object, State

CURE_THRESHOLD = 25.0
APPLY_GLUE_RADIUS = 0.02
STACK_ALIGN_TOL = 0.025
LATERAL_PERP_TOL = 0.03
SEAT_X_WINDOW = 0.045
SEAT_Y_TOL = 0.03
LEG_HALF = (0.025, 0.025, 0.05)
SPAN_HALF = (0.05, 0.025, 0.025)
BOTTLE_HALF_H = 0.03
DAB_MARGIN = 0.005
GLUE_FACES = ("top", "end_a", "end_b")
ATTACH_SLOTS = ("top", "bottom", "end_a", "end_b")
_BLOCK_ORDER = [f"leg{i}" for i in range(4)] + [f"span{i}" for i in range(3)]
_BLOCK_IDX = {name: i for i, name in enumerate(_BLOCK_ORDER)}


def _half(state: State, blk: Object) -> Tuple[float, float, float]:
    return LEG_HALF if state.get(blk, "upright") > 0.5 else SPAN_HALF


def _face_dir(state: State, blk: Object,
              face: str) -> Tuple[float, float, float]:
    if face == "top":
        return (0.0, 0.0, 1.0)
    yaw = float(state.get(blk, "rot"))
    sign = -1.0 if face == "end_a" else 1.0
    return (sign * float(np.cos(yaw)), sign * float(np.sin(yaw)), 0.0)


def _dab_point(state: State, blk: Object,
               face: str) -> Tuple[float, float, float]:
    x, y, z = (float(state.get(blk, f)) for f in ("x", "y", "z"))
    hx, _, hz = _half(state, blk)
    if face == "top":
        return (x, y, z + hz + DAB_MARGIN)
    dx, dy, _ = _face_dir(state, blk, face)
    return (x + dx * hx, y + dy * hx, z + hz + DAB_MARGIN)


def _find_mate(state: State, blocks: List[Object], blk: Object, face: str,
               params: Params) -> Optional[Object]:
    """Hard-gated mate detection mirroring the env's geometry."""
    if state.get(blk, "is_held") > 0.5:
        return None
    for other in blocks:
        if other == blk or state.get(other, "is_held") > 0.5:
            continue
        if face == "top":
            hz_b = _half(state, blk)[2]
            hz_o = _half(state, other)[2]
            dz = float(state.get(
                other, "z")) - (float(state.get(blk, "z")) + hz_b + hz_o)
            if abs(dz) >= 0.02:
                continue
            dx = float(state.get(other, "x")) - float(state.get(blk, "x"))
            dy = float(state.get(other, "y")) - float(state.get(blk, "y"))
            if state.get(other, "upright") > 0.5:
                if float(np.hypot(dx, dy)) < params["stack_align_tol"]:
                    return other
            elif abs(dx) < params["seat_x_window"] and \
                    abs(dy) < SEAT_Y_TOL:
                return other
        else:
            dx_dir, dy_dir, _ = _face_dir(state, blk, face)
            dx = float(state.get(other, "x")) - float(state.get(blk, "x"))
            dy = float(state.get(other, "y")) - float(state.get(blk, "y"))
            dz = float(state.get(other, "z")) - float(state.get(blk, "z"))
            if abs(dz) >= 0.015:
                continue
            proj = dx * dx_dir + dy * dy_dir
            perp = abs(-dx * dy_dir + dy * dx_dir)
            ext = _half(state, blk)[0] + _half(state, other)[0]
            if ext - 0.01 <= proj <= ext + 0.012 and \
                    perp < params["lateral_perp_tol"]:
                return other
    return None


def _gluing(state: State, latent: Dict[str, Any], history: History,
            updates: ResidualUpdate, params: Params) -> ResidualUpdate:
    """Recurrent hidden-counter rule: wet faces near the held bottle's tip;
    tick a hidden per-joint counter while a wet face is in aligned resting
    contact; emit only the wet-glue flags and the discrete ``attached_*``
    latches."""
    del history
    cures: Dict[str, float] = latent.setdefault("cure", {})
    objs = objs_by_type(state)
    blocks = objs.get("block", [])
    bottles = objs.get("bottle", [])

    # Defaults: carry glue and attachment observables.
    glue_next: Dict[Any, Dict[str, float]] = {}
    for blk in blocks:
        glue_next[blk] = {
            face: float(state.get(blk, f"glue_{face}"))
            for face in GLUE_FACES
        }
        for slot in ATTACH_SLOTS:
            updates.setdefault(blk, {})[f"attached_{slot}"] = float(
                state.get(blk, f"attached_{slot}"))

    # 1. Glue application: nearest unattached dry face within radius.
    held = [b for b in bottles if state.get(b, "is_held") > 0.5]
    if held:
        tip = np.array([
            float(state.get(held[0], "x")),
            float(state.get(held[0], "y")),
            float(state.get(held[0], "z")) - BOTTLE_HALF_H
        ])
        best, best_d = None, float(params["apply_glue_radius"])
        for blk in blocks:
            for face in GLUE_FACES:
                if glue_next[blk][face] > 0.5 or \
                        state.get(blk, f"attached_{face}") >= 0:
                    continue
                d = float(
                    np.linalg.norm(tip -
                                   np.array(_dab_point(state, blk, face))))
                if d < best_d:
                    best, best_d = (blk, face), d
        if best is not None:
            glue_next[best[0]][best[1]] = 1.0

    # 2. Curing: hidden counters keyed by the wet face.
    for blk in blocks:
        for face in GLUE_FACES:
            key = f"{blk.name}|{face}"
            if state.get(blk, f"attached_{face}") >= 0:
                cures.pop(key, None)
                continue
            if glue_next[blk][face] <= 0.5:
                cures.pop(key, None)
                continue
            mate = _find_mate(state, blocks, blk, face, params)
            if mate is None:
                cures[key] = 0.0
                continue
            prog = cures.get(key, 0.0) + 1.0
            cures[key] = prog
            if prog >= params["cure_threshold"]:
                if face == "top":
                    mate_slot = "bottom"
                else:
                    dx_dir, dy_dir, _ = _face_dir(state, blk, face)
                    m_yaw = float(state.get(mate, "rot"))
                    mbx, mby = float(np.cos(m_yaw)), float(np.sin(m_yaw))
                    mate_slot = "end_b" if mbx * dx_dir + mby * dy_dir < 0 \
                        else "end_a"
                if state.get(mate, f"attached_{mate_slot}") >= 0:
                    continue
                updates[blk][f"attached_{face}"] = float(_BLOCK_IDX[mate.name])
                updates.setdefault(mate, {})[f"attached_{mate_slot}"] = \
                    float(_BLOCK_IDX[blk.name])
                glue_next[blk][face] = 0.0
                cures.pop(key, None)

    for blk in blocks:
        for face in GLUE_FACES:
            updates.setdefault(blk, {})[f"glue_{face}"] = \
                glue_next[blk][face]
    return updates


def _latent_init() -> Dict[str, Dict[str, float]]:
    return {"cure": {}}


def _build_param_specs() -> List[ParamSpec]:
    return [
        ParamSpec("cure_threshold", CURE_THRESHOLD, lo=1.0, hi=100.0),
        ParamSpec("apply_glue_radius", APPLY_GLUE_RADIUS, lo=0.0),
        ParamSpec("stack_align_tol", STACK_ALIGN_TOL, lo=0.0),
        ParamSpec("lateral_perp_tol", LATERAL_PERP_TOL, lo=0.0),
        ParamSpec("seat_x_window", SEAT_X_WINDOW, lo=0.0),
    ]


RESIDUAL_RULES = [_gluing]
PARAM_SPECS = _build_param_specs
LATENT_INIT = _latent_init
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "block": [f"glue_{f}"
              for f in GLUE_FACES] + [f"attached_{s}" for s in ATTACH_SLOTS]
}


class PyBulletBridgePOGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_bridge (partially
    observable)."""

    @classmethod
    def get_env_names(cls) -> set:
        if CFG.partially_observable:
            return {"pybullet_bridge"}
        return set()
