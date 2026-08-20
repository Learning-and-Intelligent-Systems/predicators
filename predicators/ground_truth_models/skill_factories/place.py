"""Place skill factory: creates a multi-phase place controller.

This module provides ``create_place_skill``, which builds a
``ParameterizedOption`` that places a held object by:

  1. Moving directly to the release position (collision-free via BiRRT).
  2. Opening the gripper to release.
  3. Retreating back up to ``config.transport_z``.

When ``use_move_above=True``, an extra MoveAbove phase is inserted before
the descent, moving to ``config.transport_z`` first.

The placement target ``(target_x, target_y, target_yaw)`` and
``release_z`` are all provided as continuous parameters -- no callback
is needed.

Continuous parameters: ``(target_x, target_y, release_z, target_yaw)``

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_place_skill,
    )

    Place = create_place_skill(
        name="Place",
        types=[robot_type],
        config=config,
    )
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from predicators.ground_truth_models.skill_factories.base import \
    _RELEASE_CLEAR_SLACK, _RELEASE_OPEN_STEP, Phase, PhaseAction, PhaseSkill, \
    SkillConfig, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Place.
_PLACE_PARAMS = [
    ("target_x (world x position for placement)", 0.4, 1.1),
    ("target_y (world y position for placement)", 1.1, 1.6),
    ("release_z (world z height to open gripper)", 0.5, 0.6),
    ("target_yaw (placement orientation in radians)", -np.pi, np.pi),
]


def create_place_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    use_move_above: bool = False,
    param_defs: Optional[Sequence[Tuple[str, float, float]]] = None,
    compensate_held_offset: bool = False,
) -> ParameterizedOption:
    """Create a multi-phase place skill that releases a held object.

    By default (``use_move_above=False``), the skill moves directly to the
    release position, relying on BiRRT for collision avoidance:

        0. **MoveToDrop** -- Move to ``(target_x, target_y, release_z)``.
        1. **OpenFingers** -- Release the object.
        2. **Retreat** -- Rise to ``config.transport_z``.

    With ``use_move_above=True``, an extra phase is prepended:

        0. **MoveAbove** -- Move to ``(target_x, target_y, transport_z)``.
        1. **Descend** -- Lower to ``release_z``.
        2. **OpenFingers** -- Release the object.
        3. **Retreat** -- Rise to ``config.transport_z``.

    When ``config.release_until_ungrasped`` is set, the release is
    grasp-relative: **OpenFingers** opens gradually just until the
    simulator drops the grasp constraint (observed as ``is_held``
    flipping in the state), **ClearFingers** opens a few millimetres
    more so the pads clear the released object, **Retreat** holds that
    width, and a final **FullyOpenFingers** phase opens fully at
    transport height - so a placement only needs side clearance for
    roughly the held object's thickness, not the full opening span.

    Continuous parameters:
        ``(target_x, target_y, release_z, target_yaw)`` -- placement
        position, orientation, and release height.

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration (``config.transport_z`` is used).
        use_move_above: If True, add a MoveAbove phase before descending.
        param_defs: Optional override for the continuous parameter
            definitions (``(description, low, high)`` triples). Must keep
            the canonical order ``(target_x, target_y, release_z,
            target_yaw)`` -- the phases index params positionally. Use
            this when an env needs wider bounds (e.g. releasing above a
            tall structure) than the ``_PLACE_PARAMS`` defaults.
        compensate_held_offset: If True, shift the EE target xy by the
            live (EE - held object) offset read from the state, so the
            HELD OBJECT (not the gripper) lands at ``(target_x,
            target_y)``. A grasp near the arm's reach limit can leave
            the object hanging ~2 cm off the EE (IK residual at the
            grasp pose); without compensation that error transfers
            verbatim to every placement.

    Returns:
        A ``ParameterizedOption`` implementing the place skill.
    """
    if param_defs is None:
        param_defs = _PLACE_PARAMS
    assert len(param_defs) == len(_PLACE_PARAMS), \
        "param_defs must keep the canonical (x, y, release_z, yaw) order"
    params_space, params_description = build_params_space(param_defs)

    def _open_fingers_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        del params
        robot_obj = objects[0]
        current = cfg.fingers_state_to_joint(cfg.robot,
                                             state.get(robot_obj, "fingers"))
        target = cfg.open_fingers_joint
        return current, target

    def _current_fingers(state: State, robot_obj: Object,
                         cfg: SkillConfig) -> float:
        return cfg.fingers_state_to_joint(cfg.robot,
                                          state.get(robot_obj, "fingers"))

    def _release_open_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        # One anchored opening step from the measured grasp width - big
        # enough to exceed every env's _finger_action_tol so the
        # simulator drops the grasp constraint on the first action; the
        # phase terminates (via _nothing_held) as soon as the state
        # reflects the release, so the width stays bounded by
        # grasp + _RELEASE_OPEN_STEP. Anchored (see
        # Phase.anchor_finger_target) so the target does not ratchet
        # while waiting for is_held to flip.
        del params
        current = _current_fingers(state, objects[0], cfg)
        return current, min(cfg.open_fingers_joint,
                            current + _RELEASE_OPEN_STEP)

    def _clear_fingers_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        # Anchored at phase entry (see Phase.anchor_finger_target): the
        # width at which the release was observed, plus slack so the pads
        # clear the released object before the hold-width retreat.
        del params
        current = _current_fingers(state, objects[0], cfg)
        return current, min(cfg.open_fingers_joint,
                            current + _RELEASE_CLEAR_SLACK)

    def _nothing_held(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> bool:
        del objects, params, cfg
        for obj in state:
            if "is_held" in obj.type.feature_names and \
                    state.get(obj, "is_held") > 0.5:
                return False
        return True

    def _held_xy_offset(state: State,
                        robot_obj: Object) -> Tuple[float, float]:
        """(EE - held object) xy offset, or (0, 0) if nothing is held."""
        if not compensate_held_offset:
            return 0.0, 0.0
        for obj in state:
            if obj == robot_obj or \
                    "is_held" not in obj.type.feature_names:
                continue
            if state.get(obj, "is_held") > 0.5:
                return (state.get(robot_obj, "x") - state.get(obj, "x"),
                        state.get(robot_obj, "y") - state.get(obj, "y"))
        return 0.0, 0.0

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        x, y, yaw = float(params[0]), float(params[1]), float(params[3])
        off_x, off_y = _held_xy_offset(state, objects[0])
        return x + off_x, y + off_y, cfg.transport_z, yaw

    def _drop_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del cfg  # unused
        x, y = float(params[0]), float(params[1])
        drop_z, yaw = float(params[2]), float(params[3])
        off_x, off_y = _held_xy_offset(state, objects[0])
        return x + off_x, y + off_y, drop_z, yaw

    # With release_until_ungrasped, the drop-pose opening is
    # grasp-relative instead of full-span: open gradually until the
    # simulator drops the grasp constraint, open a few millimetres more
    # to clear the pads, retreat HOLDING that width (an "open" nudge
    # would keep widening next to the placed object's neighbors, and a
    # "closed" nudge would re-pinch it), and only open fully once back
    # at transport height, clear of the scene.
    partial_release = config.release_until_ungrasped

    phases = []
    if use_move_above:
        phases.append(make_move_to_phase("MoveAbove", _above_pose, "closed"))
    phases.append(
        make_move_to_phase("Descend" if use_move_above else "MoveToDrop",
                           _drop_pose,
                           "closed",
                           check_release_clearance=True))
    if partial_release:
        phases.extend([
            Phase(
                name="OpenFingers",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_release_open_target,
                finger_direction="open",
                terminal_fn=_nothing_held,
                anchor_finger_target=True,
            ),
            Phase(
                name="ClearFingers",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_clear_fingers_target,
                finger_direction="open",
                anchor_finger_target=True,
                # The default grasp_tol accepts ~2 cm of finger error -
                # wider than the whole clear-slack travel, which would
                # terminate the phase before the fingers move.
                finger_tol=1e-6,
            ),
            make_move_to_phase("Retreat", _above_pose, "hold"),
            Phase(
                name="FullyOpenFingers",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_open_fingers_target,
                finger_direction="open",
            ),
        ])
    else:
        phases.extend([
            Phase(
                name="OpenFingers",
                action_type=PhaseAction.CHANGE_FINGERS,
                target_fn=_open_fingers_target,
                finger_direction="open",
            ),
            make_move_to_phase("Retreat", _above_pose, "open"),
        ])

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description,
                      base_mode="home").build()
