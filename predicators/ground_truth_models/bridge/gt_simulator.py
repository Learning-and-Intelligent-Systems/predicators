"""Ground-truth simulator program for pybullet_bridge (glue construction)
residual dynamics -- fully-observable variant.

The residual slow processes the base rigid-body sim cannot model:

1. Glue application: while the bottle is held with its tip near a
   face's dab point, that face's wet-glue flag flips on.
2. Curing: while a wet face is in aligned resting contact with another
   block (neither held), that joint's ``cure_*`` counter ticks; at
   ``cure_threshold`` the joint irreversibly latches -- the glue is
   consumed and both blocks record the partner in their ``attached_*``
   slot (whereupon the base sim's weld constraint makes them move as
   one rigid body).

Alignment gates are softened with sigmoids so the residual is
differentiable in the threshold parameters (for the Levenberg-Marquardt
Jacobian); held gates and the discrete attachment latch stay hard.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import SOFT_EPS, Params, \
    ResidualUpdate, objs_by_type, sigmoid
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import Object, State

# Physical defaults matching pybullet_bridge.py.
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


def _top_mate_weight(state: State, other: Object, blk: Object,
                     params: Params) -> float:
    """Soft weight that ``other`` rests on ``blk``'s top face (covers both the
    leg-stack and the span-seat geometry)."""
    if state.get(other, "is_held") > 0.5 or state.get(blk, "is_held") > 0.5:
        return 0.0
    hz_b = _half(state, blk)[2]
    hz_o = _half(state, other)[2]
    dz = float(state.get(other,
                         "z")) - (float(state.get(blk, "z")) + hz_b + hz_o)
    if abs(dz) >= 0.02:
        return 0.0
    dx = float(state.get(other, "x")) - float(state.get(blk, "x"))
    dy = float(state.get(other, "y")) - float(state.get(blk, "y"))
    if state.get(other, "upright") > 0.5:
        # Leg on leg: circular xy alignment.
        return sigmoid(
            (params["stack_align_tol"] - float(np.hypot(dx, dy))) / SOFT_EPS)
    # Span seated on leg: the leg under the span's footprint.
    x_w = sigmoid((params["seat_x_window"] - abs(dx)) / SOFT_EPS)
    y_w = sigmoid((SEAT_Y_TOL - abs(dy)) / SOFT_EPS)
    return x_w * y_w


def _end_mate_weight(state: State, blk: Object, face: str, other: Object,
                     params: Params) -> float:
    """Soft weight that ``other`` butts against ``blk``'s end face."""
    if state.get(other, "is_held") > 0.5 or state.get(blk, "is_held") > 0.5:
        return 0.0
    dx_dir, dy_dir, _ = _face_dir(state, blk, face)
    dx = float(state.get(other, "x")) - float(state.get(blk, "x"))
    dy = float(state.get(other, "y")) - float(state.get(blk, "y"))
    dz = float(state.get(other, "z")) - float(state.get(blk, "z"))
    if abs(dz) >= 0.015:
        return 0.0
    proj = dx * dx_dir + dy * dy_dir
    perp = abs(-dx * dy_dir + dy * dx_dir)
    ext = _half(state, blk)[0] + _half(state, other)[0]
    proj_w = sigmoid((0.012 - (proj - ext)) / SOFT_EPS) * \
        sigmoid(((proj - ext) + 0.01) / SOFT_EPS)
    perp_w = sigmoid((params["lateral_perp_tol"] - perp) / SOFT_EPS)
    return proj_w * perp_w


def _find_mate(state: State, blocks: List[Object], blk: Object, face: str,
               params: Params) -> Tuple[Optional[Object], float]:
    best: Optional[Object] = None
    best_w = 0.0
    for other in blocks:
        if other == blk:
            continue
        if face == "top":
            w = _top_mate_weight(state, other, blk, params)
        else:
            w = _end_mate_weight(state, blk, face, other, params)
        if w > best_w:
            best, best_w = other, w
    return best, best_w


def _block_index(blocks: List[Object]) -> Dict[str, int]:
    del blocks  # the index is fixed by name, not task contents
    # Fixed name order matching the env: leg0..leg3, span0..span2.
    full = [f"leg{i}" for i in range(4)] + [f"span{i}" for i in range(3)]
    return {name: i for i, name in enumerate(full)}


def _glue_application(state: State, updates: ResidualUpdate,
                      params: Params) -> ResidualUpdate:
    """Wet the single nearest face within the bottle tip's radius."""
    objs = objs_by_type(state)
    blocks = objs.get("block", [])
    bottles = objs.get("bottle", [])
    # Carry existing glue by default.
    for blk in blocks:
        for face in GLUE_FACES:
            updates.setdefault(blk, {})[f"glue_{face}"] = float(
                state.get(blk, f"glue_{face}"))
    held = [b for b in bottles if state.get(b, "is_held") > 0.5]
    if not held:
        return updates
    bottle = held[0]
    tip = np.array([
        float(state.get(bottle, "x")),
        float(state.get(bottle, "y")),
        float(state.get(bottle, "z")) - BOTTLE_HALF_H
    ])
    best, best_d = None, float(params["apply_glue_radius"])
    for blk in blocks:
        for face in GLUE_FACES:
            if state.get(blk, f"glue_{face}") > 0.5:
                continue
            if state.get(blk, f"attached_{face}") >= 0:
                continue
            d = float(
                np.linalg.norm(tip - np.array(_dab_point(state, blk, face))))
            if d < best_d:
                best, best_d = (blk, face), d
    if best is not None:
        blk, face = best
        updates.setdefault(blk, {})[f"glue_{face}"] = 1.0
    return updates


def _curing(state: State, updates: ResidualUpdate,
            params: Params) -> ResidualUpdate:
    """Tick each wet aligned joint's counter; latch attachment at the threshold
    (hard latch; the counter accumulation is soft-gated)."""
    objs = objs_by_type(state)
    blocks = objs.get("block", [])
    idx_of = _block_index(blocks)

    # Carry attachment slots by default.
    for blk in blocks:
        for slot in ATTACH_SLOTS:
            updates.setdefault(blk, {})[f"attached_{slot}"] = float(
                state.get(blk, f"attached_{slot}"))
        for face in GLUE_FACES:
            updates.setdefault(blk, {})[f"cure_{face}"] = 0.0

    for blk in blocks:
        for face in GLUE_FACES:
            if state.get(blk, f"attached_{face}") >= 0:
                # Latched joints keep their final counter value.
                updates[blk][f"cure_{face}"] = float(
                    state.get(blk, f"cure_{face}"))
                continue
            wet = updates[blk].get(f"glue_{face}",
                                   float(state.get(blk, f"glue_{face}")))
            if wet <= 0.5:
                continue
            mate, w = _find_mate(state, blocks, blk, face, params)
            prog = w * (float(state.get(blk, f"cure_{face}")) + 1.0)
            updates[blk][f"cure_{face}"] = prog
            if mate is not None and prog >= params["cure_threshold"]:
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
                updates[blk][f"attached_{face}"] = float(idx_of[mate.name])
                updates.setdefault(mate, {})[f"attached_{mate_slot}"] = \
                    float(idx_of[blk.name])
                updates[blk][f"glue_{face}"] = 0.0
    return updates


def _build_param_specs() -> List[ParamSpec]:
    return [
        ParamSpec("cure_threshold", CURE_THRESHOLD, lo=1.0, hi=100.0),
        ParamSpec("apply_glue_radius", APPLY_GLUE_RADIUS, lo=0.0),
        ParamSpec("stack_align_tol", STACK_ALIGN_TOL, lo=0.0),
        ParamSpec("lateral_perp_tol", LATERAL_PERP_TOL, lo=0.0),
        ParamSpec("seat_x_window", SEAT_X_WINDOW, lo=0.0),
    ]


RESIDUAL_RULES = [_glue_application, _curing]
PARAM_SPECS = _build_param_specs
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "block":
    [f"glue_{f}" for f in GLUE_FACES] + [f"cure_{f}" for f in GLUE_FACES] +
    [f"attached_{s}" for s in ATTACH_SLOTS]
}


class PyBulletBridgeGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_bridge (fully
    observable)."""

    @classmethod
    def get_env_names(cls) -> set:
        # In PO mode the cure counters are not State features, so this
        # (FO) simulator does not apply; gt_simulator_po.py claims the
        # env.
        if CFG.partially_observable:
            return set()
        return {"pybullet_bridge"}
