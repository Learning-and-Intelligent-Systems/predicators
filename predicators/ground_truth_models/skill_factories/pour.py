"""Pour skill factory: creates a multi-phase pour controller.

This module provides ``create_pour_skill``, which builds a
``ParameterizedOption`` that pours from a held container (e.g. jug) into a
target (e.g. cup) by:

  1. Moving to the pour position (cup + offsets, adjusted for jug displacement).
  2. Tilting the end-effector to a fixed pour angle (π/4).

The tilt phase uses incremental IK (``use_motion_planning=False``) for fine
orientation control.

Continuous parameters: none — all offsets are fixed constants.

The ``get_target_pose_fn`` callback should return the **cup position**
``(cup_x, cup_y, cup_z, yaw)``.  The skill internally computes the robot
EE target by applying the fixed offsets and the jug-to-robot displacement.

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_pour_skill,
    )

    def _get_cup_pose(state, objects, params, config):
        _, jug, cup = objects
        return (state.get(cup, "x"), state.get(cup, "y"),
                state.get(cup, "z"), config.robot_init_wrist)

    Pour = create_pour_skill(
        name="Pour",
        types=[robot_type, jug_type, cup_type],
        config=config,
        get_target_pose_fn=_get_cup_pose,
    )
"""

from typing import List, Sequence, Tuple

import numpy as np
import pybullet as p

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.pybullet_helpers.geometry import Pose
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Pour (none remaining).
_POUR_PARAMS: List[Tuple[str, float, float]] = []

# Fixed pour tilt angle (radians).
_POUR_TILT = np.pi / 4

# Fixed absolute z height for pour target.
_POUR_Z = 0.65625

# Fixed y offset from cup to pour target.
_POUR_Y_OFF = -0.135


def create_pour_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
) -> ParameterizedOption:
    """Create a multi-phase pour skill that tilts to pour liquid.

    Phases:
        0. **MoveToTarget** -- Move to the pour position at pour height.
        1. **TiltToPour** -- Tilt the EE to a fixed pour angle (π/4). Uses
           incremental IK, not BiRRT, for fine orientation control.

    Continuous parameters:
        None -- all offsets are fixed constants.

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  ``[robot, jug, cup]``.
        config: Shared skill configuration.
        get_target_pose_fn: Callback returning the **cup position** as
            ``(x, y, z, yaw)`` from ``(state, objects, params, config)``.

    Returns:
        A ``ParameterizedOption`` implementing the pour skill.
    """
    params_space, params_description = build_params_space(_POUR_PARAMS)
    _empty = np.array([], dtype=np.float32)

    def _robot_ee_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        """Compute robot EE target from cup position + offsets + jug
        displacement."""
        del params
        # Cup position from callback
        cx, cy, _, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        # Pour target for jug (all offsets are fixed constants)
        pour_x, pour_y = cx, cy + _POUR_Y_OFF
        # Jug base z = robot EE z minus handle-to-base distance
        robot_obj, jug_obj = objects[0], objects[1]
        jug_x = state.get(jug_obj, "x")
        jug_y = state.get(jug_obj, "y")
        handle_h = 0.1
        jug_z = state.get(robot_obj, "z") - handle_h
        # Robot target = current robot + displacement to move jug to pour pos
        robot_x = state.get(robot_obj, "x") + (pour_x - jug_x)
        robot_y = state.get(robot_obj, "y") + (pour_y - jug_y)
        robot_z = state.get(robot_obj, "z") + (_POUR_Z - jug_z)
        return (robot_x, robot_y, robot_z, yaw)

    def _tilt_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[Pose, Pose, str]:
        robot_obj = objects[0]
        current_position = (state.get(robot_obj,
                                      "x"), state.get(robot_obj, "y"),
                            state.get(robot_obj, "z"))
        current_orn = p.getQuaternionFromEuler(
            [0, state.get(robot_obj, "tilt"),
             state.get(robot_obj, "wrist")])
        current_pose = Pose(current_position, current_orn)
        tx, ty, tz, tyaw = _robot_ee_target(state, objects, params, cfg)
        target_orn = p.getQuaternionFromEuler([0, _POUR_TILT, tyaw])
        target_pose = Pose((tx, ty, tz), target_orn)
        return current_pose, target_pose, "closed"

    phases = [
        # Phase 0: Move to pour position
        make_move_to_phase("MoveToTarget", _robot_ee_target, "closed"),
        # Phase 1: Tilt EE to pour liquid into target
        Phase(
            name="TiltToPour",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=_tilt_target,
            terminal_fn=None,
        ),
    ]

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description).build()
