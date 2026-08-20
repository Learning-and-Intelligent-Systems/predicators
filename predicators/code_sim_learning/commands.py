"""Physics-command vocabulary for residual rules.

Residual rules have two output channels. *Feature overwrites*
(``ResidualUpdate``) suit non-spatial residual processes (heating,
filling, curing). *Physics commands* - this module - suit residuals
that move rigid bodies through space: a rule that declares a ``cmds``
parameter receives a :class:`CommandBuffer` and may queue generic
rigid-body actuation - forces, torques, velocity overrides - expressed
against the symbolic ``State``'s objects. The buffer is pure data:
rules never touch a physics client or body ids. The base sim executes
queued commands during the physics substeps of its *next* action (see
``PyBulletEnv.queue_residual_commands``), the same cadence at which a
hidden ``_domain_specific_step`` acts, so the engine - not the rule -
resolves every contact the commanded motion runs into.

Command semantics:

* Commands act for the duration of ONE env action (all of its physics
  substeps) and then expire. A persistent process (wind while a fan is
  on) is expressed by the rule re-emitting the command every step it
  applies; a rule that stops emitting stops the effect.
* Commands are queued by object NAME. The executing env resolves names
  against its own object registry at application time, so rules built
  from one env's ``State`` drive rollouts in a separately-constructed
  env (the sysID fresh-env-per-rollout pattern) without object-identity
  gymnastics.
* Rules that need this channel must be scored by env-in-the-loop
  rollout matching: a force's effect on the next state only exists
  through engine stepping, so the teacher-forced pure-function
  objectives cannot see it (``has_physics_rules`` is the dispatch
  signal, mirroring ``has_latent_rules``).

The vocabulary is deliberately generic - exactly what any rigid-body
engine offers - and carries no domain nouns, so a fan env's wind, a
conveyor's surface speed, and a magnet's pull are all expressible
without touching this module.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple, Union

Vec3 = Tuple[float, float, float]


def _as_vec3(value: Sequence[float], what: str) -> Vec3:
    """Validate and normalise a 3-vector of finite floats."""
    try:
        vec = tuple(float(v) for v in value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{what} must be a sequence of 3 numbers, "
                         f"got {value!r}") from e
    if len(vec) != 3:
        raise ValueError(f"{what} must have exactly 3 components, "
                         f"got {len(vec)}")
    for v in vec:
        # NaN/inf would silently poison the physics state; reject at
        # queue time where the offending rule is still on the stack.
        if not math.isfinite(v):
            raise ValueError(f"{what} components must be finite, got {vec}")
    return vec  # type: ignore[return-value]


def _obj_name(obj: Any, what: str) -> str:
    """Accept a structs ``Object`` (anything with ``.name``) or a str."""
    if isinstance(obj, str):
        return obj
    name = getattr(obj, "name", None)
    if isinstance(name, str):
        return name
    raise ValueError(f"{what} must be a State object (or its name), "
                     f"got {obj!r}")


@dataclass(frozen=True)
class ApplyForce:
    """World-frame force (Newtons) on a body's center of mass.

    Re-applied before every physics substep of the following env action
    - a continuous push across that action, like wind or a magnet
    (PyBullet clears external forces after each ``stepSimulation``, so
    continuous actuation requires this re-application).
    """
    obj_name: str
    force: Vec3


@dataclass(frozen=True)
class ApplyTorque:
    """World-frame torque (N*m) on a body.

    Held semantics as in :class:`ApplyForce`.
    """
    obj_name: str
    torque: Vec3


@dataclass(frozen=True)
class SetVelocity:
    """Kinematic velocity override, re-asserted before every substep.

    ``None`` leaves that component (linear or angular) to the engine.
    """
    obj_name: str
    linear: Optional[Vec3]
    angular: Optional[Vec3]


PhysicsCommand = Union[ApplyForce, ApplyTorque, SetVelocity]


class CommandBuffer:
    """Accumulator a rule writes physics commands into.

    Callers that can execute commands construct one buffer per step,
    pass it to the rules (the ``cmds`` opt-in parameter), and hand
    ``buffer.commands`` to the env; callers that cannot execute them
    (pure-function scoring paths) let ``apply_rules`` construct a
    throwaway buffer so command-emitting rules still run - the routing
    guards at the fit entry points are responsible for never *scoring*
    such rules teacher-forced.
    """

    def __init__(self) -> None:
        self._commands: List[PhysicsCommand] = []

    # ── Rule-facing API ──────────────────────────────────────────

    def apply_force(self, obj: Any, force: Sequence[float]) -> None:
        """Queue a world-frame force (N) on ``obj`` for the next action.

        The force is held across all of the action's physics substeps
        (see :class:`ApplyForce`).
        """
        self._commands.append(
            ApplyForce(_obj_name(obj, "apply_force obj"),
                       _as_vec3(force, "apply_force force")))

    def apply_torque(self, obj: Any, torque: Sequence[float]) -> None:
        """Queue a world-frame torque (N*m) on ``obj`` for the next action."""
        self._commands.append(
            ApplyTorque(_obj_name(obj, "apply_torque obj"),
                        _as_vec3(torque, "apply_torque torque")))

    def set_velocity(self,
                     obj: Any,
                     linear: Optional[Sequence[float]] = None,
                     angular: Optional[Sequence[float]] = None) -> None:
        """Queue a velocity override on ``obj`` for the next action."""
        if linear is None and angular is None:
            raise ValueError("set_velocity needs linear and/or angular.")
        self._commands.append(
            SetVelocity(
                _obj_name(obj, "set_velocity obj"),
                _as_vec3(linear, "set_velocity linear")
                if linear is not None else None,
                _as_vec3(angular, "set_velocity angular")
                if angular is not None else None))

    # ── Consumer API ─────────────────────────────────────────────

    @property
    def commands(self) -> List[PhysicsCommand]:
        """The queued commands, in emission order."""
        return list(self._commands)

    def __len__(self) -> int:
        return len(self._commands)

    def __bool__(self) -> bool:
        return bool(self._commands)
