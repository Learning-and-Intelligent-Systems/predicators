"""Reusable parameterized skill factories for PyBullet environments.

This package provides factory functions that build ``ParameterizedOption``
instances for common robot manipulation primitives.  Each factory encapsulates
the multi-phase motion logic (move, grasp, release, etc.) and delegates
environment-specific target computation to a caller-supplied callback.

Available factories
-------------------
- ``create_pick_skill``   -- Pick up an object.
- ``create_place_skill``  -- Place a held object.
- ``create_push_skill``   -- Push through waypoints.
- ``create_pour_skill``   -- Pour from a held container.
- ``create_move_to_skill``-- Move EE to a target pose.
- ``create_wait_option``  -- Hold current pose (no-op).

Shared signature pattern
------------------------
Most factory functions share the same first three arguments::

    create_<X>_skill(
        name: str,            # Option name for logging/matching
        types: Sequence[Type],# Object types (robot first)
        config: SkillConfig,  # Shared environment configuration
        ...                   # Skill-specific arguments
    )

Each factory builds its ``params_space`` internally from canonical parameter
definitions (e.g. ``_PICK_PARAMS``, ``_PLACE_PARAMS``).  The exception is
``create_move_to_skill``, which takes an explicit ``params_space`` argument.

``create_place_skill`` uses ``(name, types, config, use_move_above=False)``
-- target position comes entirely from continuous params, so no callback is
needed.  ``create_wait_option`` uses ``(name, config, robot_type)`` since it
always operates on a single robot type with no parameters.

Callback convention
-------------------
Every factory (except ``create_place_skill`` and ``create_wait_option``)
takes a ``get_target_pose_fn`` callback (typed as ``TargetPoseFn``) with
the uniform signature::

    def get_target_pose_fn(
        state: State,
        objects: Sequence[Object],
        params: Array,
        config: SkillConfig,
    ) -> Tuple[float, float, float, float]:
        '''Return (x, y, z, yaw) for the skill target.'''

Building blocks for custom skills
----------------------------------
- ``make_move_to_phase`` -- Create a single MOVE_TO_POSE phase for use
  in custom ``PhaseSkill`` compositions.
- ``Phase``, ``PhaseAction``, ``PhaseSkill`` -- Low-level primitives for
  building skills with non-standard phase sequences.

Quick start example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_pick_skill, create_place_skill, create_wait_option,
    )

    config = SkillConfig(
        robot=pybullet_robot,
        open_fingers_joint=pybullet_robot.open_fingers,
        closed_fingers_joint=pybullet_robot.closed_fingers,
        fingers_state_to_joint=MyEnv._fingers_state_to_joint,
    )

    def _get_obj_pose(state, objects, params, config):
        _, obj = objects
        return (state.get(obj, "x"), state.get(obj, "y"),
                state.get(obj, "z"), 0.0)

    Pick = create_pick_skill("Pick", [robot_type, obj_type],
                             config, _get_obj_pose)
"""

from predicators.ground_truth_models.skill_factories.base import Phase, \
    PhaseAction, PhaseSkill, SkillConfig, TargetPoseFn, build_params_space, \
    shared_skill_robot, shared_skill_simulator
from predicators.ground_truth_models.skill_factories.move_to import \
    create_move_to_skill, make_move_to_phase
from predicators.ground_truth_models.skill_factories.pick import \
    create_pick_skill
from predicators.ground_truth_models.skill_factories.place import \
    create_place_skill
from predicators.ground_truth_models.skill_factories.pour import \
    create_pour_skill
from predicators.ground_truth_models.skill_factories.push import \
    create_push_skill
from predicators.ground_truth_models.skill_factories.wait import \
    create_wait_option

__all__ = [
    "Phase",
    "PhaseAction",
    "PhaseSkill",
    "SkillConfig",
    "TargetPoseFn",
    "build_params_space",
    "create_move_to_skill",
    "make_move_to_phase",
    "create_pick_skill",
    "create_place_skill",
    "create_pour_skill",
    "create_push_skill",
    "create_wait_option",
    "shared_skill_robot",
    "shared_skill_simulator",
]
