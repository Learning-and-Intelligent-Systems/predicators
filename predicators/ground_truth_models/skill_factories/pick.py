"""Pick skill factory: creates a multi-phase pick-and-lift controller.

This module provides ``create_pick_skill``, which builds a
``ParameterizedOption`` that picks up an object by:

  1. Moving above the object at ``config.transport_z`` (closed gripper).
  2. Descending to the grasp height (open gripper, collision-free via BiRRT).
  3. Closing the gripper.
  4. Lifting slightly above the grasp height.

The caller supplies a single callback ``get_target_pose_fn`` that extracts
the object's ``(x, y, z, yaw)`` from the current state.  All environment-
specific logic lives in this callback; the factory handles motion planning,
IK, and phase sequencing.

Continuous parameters: ``(grasp_z_offset,)``

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_pick_skill,
    )

    config = SkillConfig(
        robot=pybullet_robot,
        open_fingers_joint=pybullet_robot.open_fingers,
        closed_fingers_joint=pybullet_robot.closed_fingers,
        fingers_state_to_joint=MyEnv._fingers_state_to_joint,
        transport_z=0.8,
    )

    def _get_jug_pose(state, objects, params, config):
        _, jug = objects
        return (state.get(jug, "x"), state.get(jug, "y"),
                state.get(jug, "z"), state.get(jug, "rot"))

    PickJug = create_pick_skill(
        name="PickJug",
        types=[robot_type, jug_type],
        config=config,
        get_target_pose_fn=_get_jug_pose,
    )
"""

from typing import Optional, Sequence, Tuple

import numpy as np

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Pick.
_PICK_PARAMS = [
    ("grasp_z_offset (height above object origin to close gripper; low "
     "values can put the gripper in contact with the object or its support "
     "at the grasp pose, making the grasp config infeasible)", 0.0, 0.1),
]


def create_pick_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
    approach_open: bool = False,
    anchor_lift: bool = False,
    grasp_finger_tol: Optional[float] = None,
    lift_dz: float = 0.01,
    param_defs: Optional[Sequence[Tuple[str, float, float]]] = None,
) -> ParameterizedOption:
    """Create a multi-phase pick skill that grasps and lifts an object.

    Phases:
        0. **MoveAbove** -- Move above the object at ``config.transport_z``
           with closed gripper.
        1. **MoveToGrasp** -- Descend to object z + ``grasp_z_offset``
           with open gripper (collision-free via BiRRT).
        2. **Grasp** -- Close fingers.
        3. **LiftSlightly** -- Lift slightly above the grasp height.

    Continuous parameters:
        ``(grasp_z_offset,)`` -- offset added to z returned by
        ``get_target_pose_fn`` for the descend height.

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration (``config.transport_z`` is used).
        get_target_pose_fn: Callback returning ``(x, y, z, yaw)`` from
            ``(state, objects, params, config)``.  ``params`` will be empty.
        approach_open: If True, the MoveAbove phase travels with OPEN
            fingers. The default closed-finger approach reopens the
            fingers only gradually during the descend, so the still-closed
            gripper can ram a light object and drag it a few cm before the
            grasp -- breaking tight downstream placement tolerances.
        anchor_lift: If True, the LiftSlightly phase lifts straight up
            from the xy cached at descend time instead of re-reading the
            (now held) object's xy each step. A held object hangs at a
            small offset from the EE, so a chasing lift target is
            unreachable and the lift can spin or fail IK near the reach
            limit.
        grasp_finger_tol: Optional override for the Grasp phase's finger
            terminal tolerance (squared). Needed when the grasped object
            is wide enough to block the fingers above the default
            terminal (target + sqrt(config.grasp_tol)).
        lift_dz: How far LiftSlightly rises above the grasp height.
            Raise it in cluttered scenes: with the default 1 cm, the
            just-closed gripper can end the pick still grazing (~1 mm)
            a neighboring object, which then invalidates the NEXT
            option's BiRRT start config -- unrecoverable by replanning
            since the arm physically stays put.
        param_defs: Optional override for the continuous parameter
            definitions (``(description, low, high)`` triples). The
            default box spans the whole plausible range for any hand,
            so on a short-fingered arm most of it is dead: below the
            collision edge the grasp pose is infeasible, and above the
            reach edge the fingers close on nothing. Narrow it when the
            env knows its object and its robot -- the dead ends are what
            a sampler spends its budget on.

    Returns:
        A ``ParameterizedOption`` implementing the pick skill.
    """
    if param_defs is None:
        param_defs = _PICK_PARAMS
    assert len(param_defs) == len(_PICK_PARAMS), \
        "param_defs must keep the canonical (grasp_z_offset,) order"
    params_space, params_description = build_params_space(param_defs)
    _empty = np.array([], dtype=np.float32)
    _shared: dict = {}

    def _close_fingers_target(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float]:
        del params
        robot_obj = objects[0]
        current = cfg.fingers_state_to_joint(cfg.robot,
                                             state.get(robot_obj, "fingers"))
        target = cfg.closed_fingers_joint - 0.01
        return current, target

    def _above_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del params
        x, y, _, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        return x, y, cfg.transport_z, yaw

    def _descend_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        grasp_z_offset = float(params[0])
        x, y, z, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        grasp_z = z + grasp_z_offset
        _shared["grasp_z"] = grasp_z
        _shared["grasp_xy_yaw"] = (x, y, yaw)
        return x, y, grasp_z, yaw

    def _slight_lift_pose(
        state: State,
        objects: Sequence[Object],
        params: Array,
        cfg: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        del params
        if anchor_lift:
            x, y, yaw = _shared["grasp_xy_yaw"]
        else:
            x, y, _, yaw = get_target_pose_fn(state, objects, _empty, cfg)
        return x, y, _shared["grasp_z"] + lift_dz, yaw

    phases = []
    phases.extend([
        make_move_to_phase("MoveAbove", _above_pose,
                           "open" if approach_open else "closed"),
        # Validate the grasp goal IK: the gripper descends to envelop the
        # target, and an imprecise (unvalidated) IK config can clip the target
        # object, making BiRRT reject a reachable grasp. See Phase.validate_ik.
        make_move_to_phase("MoveToGrasp",
                           _descend_pose,
                           "open",
                           validate_ik=True),
        Phase(
            name="Grasp",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=_close_fingers_target,
            terminal_fn=None,
            finger_direction="close",
            finger_tol=grasp_finger_tol,
        ),
        make_move_to_phase("LiftSlightly",
                           _slight_lift_pose,
                           "closed",
                           allow_shallow_held_object_contacts=True)
    ])

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description,
                      base_mode="home").build()
