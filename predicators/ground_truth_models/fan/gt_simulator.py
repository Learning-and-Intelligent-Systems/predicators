"""Ground-truth simulator program for pybullet_fan residual dynamics.

While a fan is on, the rule emits a world-frame force on the ball
along the fan's facing direction (``cmds.apply_force``): a continuous
push re-applied on every physics substep, like real wind. The base sim's engine
handles everything downstream - contact stops against the obstacle
walls and boundary slabs, sliding along their faces, corner
deflection. The env applies its wind inside ``_domain_specific_step``
(``_simulate_fans``), which the approaches' base sims skip
(``skip_residual_dynamics=True``); this program is that hidden step's
learned-space counterpart. The env's own wind goes through the same
residual-command executor (same post-step emission, same held
re-application), so a rule emitting the env's force is bit-identical
to the env.

The magnitude works jointly with the ball's high linear damping (see
``PyBulletFanBaseEnv.ball_linear_damping``): the damping sets the
terminal speed of the held push (~0.00224 m/action free-field) while
the force sits ~60% above the ~0.036 N stiction/seam creep threshold,
so the ball rolls reliably from rest and across the two-table seam
instead of stick-slipping.

Because commands act through engine stepping, this artifact is scored
and fit by free-running rollout matching (``has_physics_rules``
routing), never teacher-forced.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from predicators.code_sim_learning.commands import CommandBuffer
from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.code_sim_learning.utils import Params, ResidualUpdate, \
    objs_by_type
from predicators.ground_truth_models import GroundTruthSimulatorFactory
from predicators.settings import CFG
from predicators.structs import State

# ── Constants ────────────────────────────────────────────────────

# Held-mode force (N), equal to the env's wind_force_magnitude
# (steady-state free field: 0.00224 m/action). Fitted init.
WIND_FORCE = 0.06


def _wind_blowing(state: State, updates: ResidualUpdate, params: Params,
                  cmds: CommandBuffer) -> ResidualUpdate:
    """Each on-fan blows the ball along its local +X; the engine owns contacts.

    The force is re-emitted every step any fan is on and expires with
    the next action otherwise, so fans-off means no wind - the measured
    zero coasting comes from the ball's high damping in the base sim.
    Simultaneous fans sum as force vectors.
    """
    objs = objs_by_type(state)
    balls = objs.get("ball", [])
    if not balls:
        return updates

    # Wind direction: each on-fan blows along its local +X (rot is the
    # base Z euler), summed per axis for simultaneous fans.
    sign = -1.0 if CFG.fan_fans_blow_opposite_direction else 1.0
    fx = fy = 0.0
    for fan in objs.get("fan", []):
        if state.get(fan, "is_on") <= 0.5:
            continue
        rot = float(state.get(fan, "rot"))
        fx += sign * float(np.cos(rot)) * params["wind_force"]
        fy += sign * float(np.sin(rot)) * params["wind_force"]
    if fx == 0.0 and fy == 0.0:
        return updates
    for ball in balls:
        cmds.apply_force(ball, (fx, fy, 0.0))
    return updates


# ── Public API: consumed by read_simulator_components ────────────
# Same contract used by agent-synthesized simulator files.

RESIDUAL_RULES = [_wind_blowing]

PARAM_SPECS: List[ParamSpec] = [
    ParamSpec("wind_force", WIND_FORCE, lo=0.0, hi=0.2),
]

# Features the wind dynamics own. Scored by the rollout objective
# against observations; NOT overwritten at plan time - the engine moves
# them under the emitted force.
RESIDUAL_FEATURES: Dict[str, List[str]] = {
    "ball": ["x", "y"],
}

# ── Factory binding ──────────────────────────────────────────────


class PyBulletFanGroundTruthSimulatorFactory(GroundTruthSimulatorFactory):
    """GT residual-dynamics simulator for pybullet_fan.

    The actual simulator components (``RESIDUAL_RULES``,
    ``PARAM_SPECS``, ``RESIDUAL_FEATURES``) live as module globals
    above; this class only pins the env-name binding so
    ``get_gt_simulator`` can locate the right module via the factory
    registry.
    """

    @classmethod
    def get_env_names(cls) -> set:
        return {"pybullet_fan"}
