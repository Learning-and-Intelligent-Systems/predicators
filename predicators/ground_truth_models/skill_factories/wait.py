"""Wait skill factory: holds current pose with finger drift resistance.

This module provides ``create_wait_option``, which builds a
``ParameterizedOption`` that holds the robot's current joint positions
while nudging fingers toward their current open/closed state to resist
drift.  The option is always initiable; it never terminates unless the
config sets ``wait_quiescence_eps``, in which case it terminates once
the non-robot scene has stopped moving (see ``SkillConfig``).

Example::

    from predicators.ground_truth_models.skill_factories import (
        SkillConfig, create_wait_option,
    )

    Wait = create_wait_option("Wait", config, robot_type)
"""

import weakref
from typing import Dict, Optional, Sequence, Tuple, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.skill_factories.base import SkillConfig
from predicators.structs import Action, Array, Object, ParameterizedOption, \
    State, Type, _Option


def note_external_state_change(option: _Option, state: State) -> None:
    """Tell ``option`` that ``state`` was set from outside, not moved into.

    ``Wait`` ends once the scene holds still for several consecutive steps.
    Writing perception into the twin replaces object poses without the
    scene having moved, so counting that jump would zero the tally at
    every look and ``Wait`` would never see the scene settle. This keeps
    the tally and moves the comparison point past the jump, so the jump is
    skipped rather than counted as motion.

    A no-op for options that track no quiescence.
    """
    memory = option.memory
    if "quiescence_prev" not in memory:
        return
    robot_obj = option.objects[0]
    scene_objs = sorted((o for o in state if o != robot_obj), key=str)
    if not scene_objs:
        return
    memory["quiescence_prev"] = state.vec(scene_objs)
    # The cached identity names the pre-resync state, so drop it.
    memory.pop("quiescence_sref", None)


def create_wait_option(
    name: str,
    config: SkillConfig,
    robot_type: Type,
    params_description: Optional[Tuple[str, ...]] = None,
) -> ParameterizedOption:
    """Create a wait (no-op) option that holds the robot's current pose.

    Nudges fingers toward their current open/closed state to resist drift
    and keeps all other joints at their current positions.  With
    ``config.wait_quiescence_eps`` unset the option never terminates
    (the executor's option-rollout cap ends it); when set, it terminates
    once every non-robot object's features have changed by less than the
    eps for ``config.wait_quiescence_steps`` consecutive steps.

    Args:
        name: Option name (e.g. "Wait").
        config: Shared skill configuration.  See ``SkillConfig``.
        robot_type: The robot ``Type`` object.

    Returns:
        A ``ParameterizedOption`` with ``initiable=True`` always.

    Example::

        wait = create_wait_option("Wait", config, robot_type)
    """
    robot = config.robot
    mid_point = (config.open_fingers_joint + config.closed_fingers_joint) / 2

    def _initiable(state: State, memory: Dict, objects: Sequence[Object],
                   params: Array) -> bool:
        del state, objects, params
        # A grounded option can be re-run (validation rollouts reuse the
        # grounded plan); stale quiescence tracking from a previous run
        # would terminate the new run instantly.
        memory.pop("quiescence_prev", None)
        memory.pop("quiescence_count", None)
        memory.pop("quiescence_sref", None)
        return True

    def _terminal(state: State, memory: Dict, objects: Sequence[Object],
                  params: Array) -> bool:
        del params
        if config.wait_quiescence_eps is None:
            return False
        robot_obj = objects[0]
        scene_objs = sorted((o for o in state if o != robot_obj), key=str)
        if not scene_objs:
            return False
        # terminal() can be consulted more than once on the same state
        # (executor loop + monitors); recounting a zero delta would let
        # repeated queries stand in for settled physics steps. Identity
        # via weakref, NOT id(): the allocator reuses a freed state's id,
        # which would silently swallow real steps.
        last_ref = memory.get("quiescence_sref")
        if last_ref is not None and last_ref() is state:
            return (memory.get("quiescence_count", 0) >=
                    config.wait_quiescence_steps)
        memory["quiescence_sref"] = weakref.ref(state)
        vec = state.vec(scene_objs)
        prev = memory.get("quiescence_prev")
        memory["quiescence_prev"] = vec
        if prev is None or prev.shape != vec.shape:
            memory["quiescence_count"] = 0
            return False
        if float(np.max(np.abs(vec - prev))) < config.wait_quiescence_eps:
            count = memory.get("quiescence_count", 0) + 1
        else:
            count = 0
        memory["quiescence_count"] = count
        return count >= config.wait_quiescence_steps

    def _policy(state: State, memory: Dict, objects: Sequence[Object],
                params: Array) -> Action:
        del memory, params
        robot_obj = objects[0]

        current_joint = config.fingers_state_to_joint(
            robot, state.get(robot_obj, "fingers"))
        if current_joint > mid_point:  # currently open -- nudge open
            finger_delta = config.finger_action_nudge_magnitude
        else:  # currently closed -- nudge closed
            finger_delta = -config.finger_action_nudge_magnitude

        pb_state = cast(utils.PyBulletState, state)
        joint_positions = pb_state.joint_positions.copy()
        f_action = joint_positions[robot.left_finger_joint_idx] + finger_delta
        joint_positions[robot.left_finger_joint_idx] = f_action
        joint_positions[robot.right_finger_joint_idx] = f_action

        # Pad base-action dims with zeros for mobile robots so the action
        # matches the (arm + base) action space; a no-op for fixed bases.
        action_arr = np.array(joint_positions, dtype=np.float32)
        n_action = robot.action_space.shape[0]
        if action_arr.shape[0] < n_action:
            action_arr = np.concatenate([
                action_arr,
                np.zeros(n_action - action_arr.shape[0], dtype=np.float32)
            ])
        return Action(
            np.clip(action_arr, robot.action_space.low,
                    robot.action_space.high))

    return ParameterizedOption(
        name,
        types=[robot_type],
        params_space=Box(0, 1, (0, )),
        policy=_policy,
        initiable=_initiable,
        terminal=_terminal,
        params_description=params_description,
    )
