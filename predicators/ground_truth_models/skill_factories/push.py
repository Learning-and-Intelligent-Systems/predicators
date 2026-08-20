"""Push skill factory: creates a multi-phase push controller.

This module provides ``create_push_skill``, which builds a
``ParameterizedOption`` that pushes an object (e.g. domino, switch, button)
using a standard 4-waypoint trajectory:

  1. Closing the gripper.
  2. Moving above & behind the target at ``config.transport_z``.
  3. Descending to contact height (target z + ``contact_z_offset``).
  4. Pushing to the target position along its facing direction.
  5. Retreating to ``config.robot_home_pos``.
  6. Opening the gripper.

The "facing direction" is derived from the yaw returned by
``get_target_pose_fn`` as ``(sin(yaw), cos(yaw))``.  "Behind" means
opposite to the facing direction.

``config.robot_home_pos`` **must** be set.

Continuous parameters: ``(approach_distance, contact_z_offset)``

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_push_skill,
    )

    config = SkillConfig(
        robot=pybullet_robot,
        open_fingers_joint=pybullet_robot.open_fingers,
        closed_fingers_joint=pybullet_robot.closed_fingers,
        fingers_state_to_joint=MyEnv._fingers_state_to_joint,
        robot_home_pos=(MyEnv.robot_init_x, MyEnv.robot_init_y,
                        MyEnv.robot_init_z),
    )

    def _get_domino_pose(state, objects, params, config):
        _, domino = objects
        return (state.get(domino, "x"), state.get(domino, "y"),
                state.get(domino, "z"), state.get(domino, "rot"))

    Push = create_push_skill(
        name="Push",
        types=[robot_type, domino_type],
        config=config,
        get_target_pose_fn=_get_domino_pose,
    )
"""

from typing import Callable, List, Sequence, Tuple

import numpy as np

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space
from predicators.ground_truth_models.skill_factories.move_to import \
    make_move_to_phase
from predicators.settings import CFG
from predicators.structs import Array, Object, ParameterizedOption, State, Type

# Canonical continuous parameters for Push. The approach upper bound
# leaves room for a side-oriented gripper (yaw offset 0),
# whose body extends along the approach axis: a descend waypoint closer
# than ~0.07 m can itself collide with the pushed object. That caveat is
# stated in the description because the tool-facing params text is the
# only place an agent learns the advertised range's usable interior.
_PUSH_PARAMS = [
    ("approach_distance (dist behind target along facing dir to start push; "
     "small values put the descend waypoint inside the gripper's own "
     "footprint along the approach axis, colliding with the target)", 0.00,
     0.10),
    ("contact_z_offset (height above target z for contact; near-zero values "
     "descend into the target/support and can stall, near-max values may "
     "pass over a short target)", 0.0, 0.11),
]


def resolve_ee_yaw_offset(config: SkillConfig) -> float:
    """The EE yaw offset Push should use, in radians.

    Which face of the gripper leads into the object is a property of the
    hand, so it comes from the robot unless the config forces one.
    """
    if CFG.skill_push_ee_yaw_offset is None:
        return config.robot.push_ee_yaw_offset
    return float(CFG.skill_push_ee_yaw_offset)


def create_push_skill(
    name: str,
    types: Sequence[Type],
    config: SkillConfig,
    get_target_pose_fn: TargetPoseFn,
) -> ParameterizedOption:
    """Create a multi-phase push skill with a standard 4-waypoint trajectory.

    Phases:
        0. **CloseFingers** -- Close the gripper before approaching.
        1. **Waypoint_0** -- Move above & behind the target at
           ``config.transport_z``, offset by ``approach_distance``
           opposite the facing direction.
        2. **Waypoint_1** -- Descend to contact height
           (target z + ``contact_z_offset``) at the same behind position.
        3. **Waypoint_2** -- Push forward to the target position.
        4. **Waypoint_3** -- Retreat to ``config.robot_home_pos``.
        5. **OpenFingers** -- Open the gripper.

    Continuous parameters:
        ``(approach_distance, contact_z_offset)``

    Args:
        name: Option name used for logging and matching.
        types: Ordered object types.  First element must be the robot type.
        config: Shared skill configuration.  ``config.robot_home_pos`` and
            ``config.transport_z`` must be set.
        get_target_pose_fn: Callback returning ``(x, y, z, yaw)`` from
            ``(state, objects, params, config)``.  ``params`` will be empty.

    Returns:
        A ``ParameterizedOption`` implementing the push skill.
    """
    if config.robot_home_pos is None:
        raise ValueError(
            "config.robot_home_pos must be set for create_push_skill.")

    params_space, params_description = build_params_space(_PUSH_PARAMS)
    _empty = np.array([], dtype=np.float32)

    # -- Standard 4-waypoint trajectory ----------------------------------

    def _waypoints(
        ox: float,
        oy: float,
        oz: float,
        oyaw: float,
        cfg: SkillConfig,
        s_offset_x: float,
        s_offset_z: float,
    ) -> List[Tuple[float, float, float, float, str]]:
        assert cfg.robot_home_pos is not None
        obj_xy = np.array([ox, oy])
        facing = np.array([np.sin(oyaw), np.cos(oyaw)])
        behind_xy = obj_xy - facing * s_offset_x
        push_xy = obj_xy
        home_xy = np.array(cfg.robot_home_pos[:2])
        home_z = cfg.robot_home_pos[2]
        ee_yaw = oyaw + resolve_ee_yaw_offset(cfg)
        return [
            (*behind_xy, cfg.transport_z, ee_yaw, "closed"),
            (*behind_xy, oz + s_offset_z, ee_yaw, "closed"),
            (*push_xy, oz + s_offset_z, ee_yaw, "closed"),
            (*home_xy, home_z, ee_yaw, "closed"),
        ]

    # -- Phase construction -----------------------------------------------

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

    def _make_waypoint_position_fn(
        waypoint_idx: int,
    ) -> Callable[[State, Sequence[Object], Array, SkillConfig], Tuple[
            float, float, float, float]]:

        def _get_target(
            state: State,
            objects: Sequence[Object],
            params: Array,
            cfg: SkillConfig,
        ) -> Tuple[float, float, float, float]:
            s_ox = float(params[0])
            s_oz = float(params[1])
            x, y, z, yaw = get_target_pose_fn(state, objects, _empty, cfg)
            wps = _waypoints(x, y, z, yaw, cfg, s_ox, s_oz)
            wx, wy, wz, wyaw, _ = wps[waypoint_idx]
            return wx, wy, wz, wyaw

        return _get_target

    phases: List[Phase] = []
    phases.append(
        Phase(name="CloseFingers",
              action_type=PhaseAction.CHANGE_FINGERS,
              target_fn=_close_fingers_target,
              finger_direction="close"))

    for i in range(4):
        # Waypoint_2 (push into target) and Waypoint_3 (retreat from target)
        # expect robot-object contact, so suppress collision diagnostics.
        #
        # They must also NOT be motion-planned: their goal poses sit at (or
        # inside) the pushed object, and BiRRT plans a COLLISION-FREE path.
        # What that does is a knife-edge of scene and hand geometry. If the
        # goal config registers as colliding (every sim scene so far),
        # planning fails and the ``expect_contact`` fallback quietly runs
        # incremental IK. If it squeaks past the ~1 mm
        # ``pybullet_birrt_contact_margin`` (the real captured scenes), BiRRT
        # SUCCEEDS by routing around -- measured: over the block's top, 59 mm
        # out to the side, past the block, then down onto the goal from the
        # far side, so the last hop struck the block AGAINST the push
        # direction and toppled it backwards. The direction of travel is the
        # payload of a stroke, and only stepping IK straight at the target
        # guarantees it -- identically in sim and on the real bench.
        #
        # Deliberately not fixed by dropping the pushed object from the
        # planner's collision set: the planner then remains free to detour
        # around a BYSTANDER near the stroke (the same wrong-direction strike
        # one object over), and the restricted Push variant grounds as
        # ``[robot]`` alone, where an index into ``objects`` silently misses.
        # A stroke that cannot go straight should fail and be resampled, not
        # rerouted.
        phases.append(
            make_move_to_phase(
                name=f"Waypoint_{i}",
                get_target_pose_fn=_make_waypoint_position_fn(i),
                finger_status="closed",
                expect_contact=(i >= 2),
                use_motion_planning=(False if i >= 2 else None)))

    phases.append(
        Phase(name="OpenFingers",
              action_type=PhaseAction.CHANGE_FINGERS,
              target_fn=_open_fingers_target,
              finger_direction="open"))

    return PhaseSkill(name,
                      types,
                      params_space,
                      config,
                      phases,
                      params_description=params_description,
                      base_mode="home").build()
