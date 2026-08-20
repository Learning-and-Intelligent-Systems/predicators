"""Counterfactual clean-push probe for the domino cascade certificate.

The certificate's legitimacy question is "does the layout the robot
built actually work as a domino cascade?", and this module answers it
by physics instead of geometric forensics: from the recorded pre-push
state it re-runs the episode's own Push - the REAL Push skill, with the
episode's recorded continuous parameters - in a probe world where every
robot link except the two fingertips has its collision geometry masked
off. The skill's controller, waypoint path, speed profile, and
follow-through are exactly the plan's own; the arm's body is
intangible, so collateral arm contact (the confound the old
swept-corridor attribution rules had to guess at) cannot contribute: if
the goal topples in the probe, the layout genuinely cascades under the
legal fingertip push; if it never does, the real episode's topples are
owed to the robot's body, not the built chain.

This replaced an earlier synthetic finger-box replay (a fingertip-sized
box driven along hand-computed waypoints): instrumented replays of
run_20260716_133656 showed the box under-delivering so badly that the
pushed green never crossed 11 deg while the real skill toppled it every
time - two genuine cascades were rejected, and the agent misread the
rejections as a scoring rule. Executing the actual skill removes the
whole fidelity gap by construction.

The replay stops when the skill advances past its push stroke
(Waypoint_2): the closing home-sweep retreat and OpenFingers phases are
deliberately excluded because a fingertip retreat knock is collateral,
not the built chain - with them included a retreat graze could certify
a layout that never cascades. During the settle window the arm holds
the stroke's final commanded pose, exactly like the skill's
position-controlled motors dwell before retreating.

Corner layouts are contact-history knife-edge (see
``min_block_utils``), and a single replay can diverge from the real
rollout in either direction (residual solver state is the one quantity
``_set_state`` cannot pin down), so the probe retries the SAME replay a
few times and certifies on the first success. Velocities are zeroed
before every attempt.

The probe runs in a dedicated probe env - a fresh instance of the
certifying env's class, physics mirrored via
``DominoComponent.physical_param_override`` - so it never contaminates
the certifying env's world (cross-episode residuals, the collision
masks) and stays valid for both the true env and an agent's belief env:
each side probes with its own physics, which is exactly the trust
contract of belief-side verdicts. States transfer between the worlds by
the same mechanism the option model relies on: same-class envs build
the same body pool in the same order, so the pybullet ids stamped on
the state's objects coincide.

Belief-side substrate: when the certifying env carries a
``probe_process_model_factory`` (stamped by a sim-learning approach on
its belief env, never on the real env), each replay attempt applies the
agent's fitted residual rules after every physics step and writes the
merged state back - the same combined-substrate contract the option
model's ``combined_simulate`` uses at plan time. The belief verdict is
a PREDICTION of the real evaluator's verdict, so it should run the
agent's full current world model; a deliberately rules-free belief
replay at miscalibrated base physics rejected every legitimate relay in
run_20260727_210818 seed2 and taught the agent a phantom task rule.
The real evaluator never sets a factory, so real episodes are still
judged on pure env physics.
"""

import logging
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p

from predicators import utils
from predicators.structs import Action, GroundAtom, Object, \
    ParameterizedOption, State

# One belief-side verification rollout's process model: called after
# every probe physics step with the post-step state and the executed
# action, returns the state with the learned residual rules applied (or
# the input state unchanged). A factory (fresh callable per replay
# attempt) rather than a bare callable so recurrent rules can thread
# their latent per rollout. See ``BaseEnv.probe_process_model_factory``.
ProbeProcessStep = Callable[[State, Action], State]
ProbeProcessModelFactory = Callable[[], ProbeProcessStep]

# Fallback Push parameters (approach_distance, contact_z_offset) for
# episodes whose step labels carry no continuous parameters (legacy
# 2-tuple labels, label-free certification). Matches the task
# generators' canonical probe push (``min_block_utils._PUSH_PARAMS``).
_CANONICAL_PUSH_PARAMS: Tuple[float, ...] = (0.04, 0.05)

# Identical-replay attempts: knife-edge cascades flip on residual
# solver state between the real rollout and a replay (the one quantity
# _set_state cannot reconstruct), so any success across a few replays
# is a success of the plan's own push.
_NUM_ATTEMPTS = 3

# The Push skill's phase list (``skill_factories/push.py``):
# 0 CloseFingers, 1 Waypoint_0 (behind-transport), 2 Waypoint_1
# (descend), 3 Waypoint_2 (push stroke), 4 Waypoint_3 (home-sweep
# retreat), 5 OpenFingers. The replay ends when the skill advances to
# the retreat: its home sweep is the recorded collateral-knock vector
# and never contributes to the intended topple.
_RETREAT_PHASE_IDX = 4

# Cap on replayed skill steps per green (a real Push runs ~50; a stall
# past this is a failed push, not a cascade).
_MAX_REPLAY_STEPS = 200

# No-op env steps after the stroke for the cascade to propagate and
# settle, with the arm holding the stroke's final commanded pose. A hop
# takes at most ~24 env steps (~2 s) and recorded chains run 2-4 hops.
_SETTLE_STEPS = 100


def _zero_all_velocities(physics_client_id: int) -> None:
    """Kill residual velocities on every body (see the velocity-residual
    lesson: pose resets do not clear velocities)."""
    for body in range(p.getNumBodies(physicsClientId=physics_client_id)):
        body_id = p.getBodyUniqueId(body, physicsClientId=physics_client_id)
        p.resetBaseVelocity(body_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=physics_client_id)


def _goal_shortfall(final: State, goal: Set[GroundAtom]) -> List[str]:
    """Names of goal atoms that do not hold in ``final``."""
    return [
        str(atom) for atom in sorted(goal, key=str) if not atom.holds(final)
    ]


def _apply_process_step(probe_env: Any, process_step: ProbeProcessStep,
                        state: State, action: Action) -> State:
    """Apply the learned process model to a post-step state, write back.

    Mirrors the combined simulator's plan-time contract: the rule-merged
    state is written into the probe world (so physics continues from it,
    exactly as ``PyBulletEnv.simulate``'s allclose guard does at plan
    time) and becomes the state the replay threads forward. Fail-soft: a
    crashing process model leaves the base-sim state in charge for this
    step, same as ``LearnedSimulator.predict_step``.
    """
    # pylint: disable=protected-access
    try:
        merged = process_step(state, action)
    except Exception as e:  # pylint: disable=broad-except
        logging.debug(
            "[cascade probe] process model step failed (%s); "
            "using the base-sim state.", e)
        return state
    if merged is not state and not merged.allclose(state):
        probe_env._set_state(merged)
    return merged


def _ensure_fingertips_only_collision(probe_env: Any) -> None:
    """Mask collision on every robot link except the two fingertips.

    Idempotent per probe env (the masks persist across ``_set_state``
    because the body pool is fixed and never rebuilt). Group/mask 0
    makes a link collide with nothing, so the arm's body passes through
    the scene while the fingertips - the only link the legal push may
    deliver force through - keep their default filters.
    """
    # pylint: disable=protected-access
    if getattr(probe_env, "_probe_fingertips_only", False):
        return
    robot = probe_env._pybullet_robot
    cid = probe_env._physics_client_id
    keep = {robot.left_finger_id, robot.right_finger_id}
    num_joints = p.getNumJoints(robot.robot_id, physicsClientId=cid)
    for link in range(-1, num_joints):
        if link not in keep:
            p.setCollisionFilterGroupMask(robot.robot_id,
                                          link,
                                          0,
                                          0,
                                          physicsClientId=cid)
    probe_env._probe_fingertips_only = True


def _replay_push_skill(
        probe_env: Any,
        push_option: ParameterizedOption,
        robot: Object,
        green: Object,
        params: Tuple[float, ...],
        process_step: Optional[ProbeProcessStep] = None) -> Optional[Action]:
    """Run the real Push skill on ``green`` up to the end of its stroke.

    Executes the skill's own policy step by step in the probe world and
    stops as soon as it advances past Waypoint_2 (the push stroke) - the
    retreat's returned action is never executed. With ``process_step``
    the learned process model is applied (and written back) after every
    physics step, so the skill's controller and the cascade both evolve
    on the combined substrate. Returns the last executed action (the
    stroke's final commanded pose, for the settle dwell), or the last
    action anyway if the skill stalled/failed short of the stroke's end
    (the settle then scores whatever the partial push achieved - an
    honest non-cascade).
    """
    # pylint: disable=protected-access
    objects = [robot, green][:len(push_option.types)]
    option = push_option.ground(objects, np.asarray(params, dtype=np.float32))
    state = probe_env._get_state()
    last_action: Optional[Action] = None
    if not option.initiable(state):
        return None
    try:
        for _ in range(_MAX_REPLAY_STEPS):
            if option.terminal(state):
                break
            action = option.policy(state)
            # policy() advances the phase when the previous one has
            # terminated, so a retreat-phase action is detected here and
            # never executed.
            if option.memory.get("phase_idx", 0) >= _RETREAT_PHASE_IDX:
                break
            probe_env.step(action)
            last_action = action
            state = probe_env._get_state()
            if process_step is not None:
                state = _apply_process_step(probe_env, process_step, state,
                                            action)
    except (utils.OptionExecutionFailure, p.error) as e:
        logging.debug("[cascade probe] push replay aborted: %s", e)
    return last_action


def run_counterfactual_push_probe(
        probe_env: Any,
        pre_push_state: State,
        greens: Sequence[Object],
        goal: Set[GroundAtom],
        push_params: Optional[Tuple[float, ...]] = None,
        push_option: Optional[ParameterizedOption] = None,
        process_model_factory: Optional[ProbeProcessModelFactory] = None,
        num_attempts: Optional[int] = None) -> Tuple[bool, str]:
    """Does the plan's own push, delivered by the fingertips alone, cascade to
    the goal?

    Re-runs the real Push skill (the episode's recorded ``push_params``,
    falling back to the generators' canonical push when None) from
    ``pre_push_state`` in ``probe_env`` (a dedicated same-class env with
    every non-fingertip robot link collision-masked; see module
    docstring), retrying the identical replay a few times. With
    ``process_model_factory`` (belief-side verdicts on an env whose
    approach learned residual rules) each attempt replays on the
    COMBINED substrate: a fresh per-attempt process step is applied and
    written back after every physics step, replay and settle alike, so
    the verdict predicts what the agent's full current model says - the
    real evaluator, which never sets a factory, still probes pure env
    physics. Returns ``(ok, detail)``: ``ok`` on the first attempt
    whose settled state satisfies every goal atom - or, after all
    attempts fail, the goal atoms the closest run left unsatisfied.
    """
    # pylint: disable=protected-access
    assert greens, "probe needs at least one pushed green block"
    assert push_option is not None, \
        "probe needs the real Push skill (see run_counterfactual_cascade_probe)"
    params = tuple(push_params) if push_params else _CANONICAL_PUSH_PARAMS
    assert len(params) >= 2, f"malformed push params {params}"
    _ensure_fingertips_only_collision(probe_env)
    cid = probe_env._physics_client_id
    robot = next(o for o in pre_push_state if o.type.name == "robot")
    substrate = ("with the fitted residual rules riding on the base sim"
                 if process_model_factory is not None else
                 "on the base sim alone")
    attempts = num_attempts if num_attempts is not None else _NUM_ATTEMPTS
    best_shortfall: Optional[List[str]] = None
    for attempt in range(attempts):
        # Fresh per attempt: recurrent rules thread a latent per rollout.
        process_step = (process_model_factory()
                        if process_model_factory is not None else None)
        probe_env._set_state(pre_push_state)
        _zero_all_velocities(cid)
        hold_action: Optional[Action] = None
        for green in greens:
            action = _replay_push_skill(probe_env, push_option, robot, green,
                                        params, process_step)
            if action is not None:
                hold_action = action
        # Dwell: the skill holds its final commanded pose until the
        # phase terminates; keep pressing while the cascade propagates.
        if hold_action is None:
            hold_action = Action(
                np.array(probe_env._pybullet_robot.initial_joint_positions))
        for _ in range(_SETTLE_STEPS):
            probe_env.step(hold_action)
            if process_step is not None:
                _apply_process_step(probe_env, process_step,
                                    probe_env._get_state(), hold_action)
        shortfall = _goal_shortfall(probe_env._get_state(), goal)
        if not shortfall:
            source = "the plan's" if push_params else "the canonical"
            detail = (f"{source} push (approach {params[0]:.2f} m, contact "
                      f"height +{params[1]:.2f} m), replayed with the real "
                      f"skill and fingertips-only collision {substrate}, "
                      f"cascades to the goal (attempt {attempt + 1})")
            return True, detail
        if best_shortfall is None or len(shortfall) < len(best_shortfall):
            best_shortfall = shortfall
    assert best_shortfall is not None
    source = "the plan's own" if push_params else "the canonical"
    detail = (f"{source} push (approach {params[0]:.2f} m, contact height "
              f"+{params[1]:.2f} m) on {', '.join(g.name for g in greens)}, "
              f"replayed with the real skill and fingertips-only collision "
              f"{substrate}, reaches the goal at none of {attempts} "
              f"attempts; closest run left {', '.join(best_shortfall)} "
              f"unsatisfied")
    logging.debug("[cascade probe] %s", detail)
    return False, detail
