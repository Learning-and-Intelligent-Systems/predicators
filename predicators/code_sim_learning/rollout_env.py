"""Env-facing rollout plumbing for physical system identification.

Fresh-env construction/disposal, velocity zeroing, sticky-override
pinning, env-registry anchors, and the free-running rollout itself.
See the :mod:`predicators.code_sim_learning.physical_sysid` module
docstring for why identification free-runs the base sim instead of
teacher-forcing per step.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pybullet as p

from predicators.code_sim_learning.fit_space import ParamSpec
from predicators.structs import Action, State

# (states, actions) with len(states) == len(actions) + 1 and states[0] at rest.
RolloutTrajectory = Tuple[List[State], List[Action]]

# Monotonic count of free-running rollouts executed by this process.
# Each SSE evaluation runs one rollout per trajectory, so this is the
# honest unit of sysID compute; the fit orchestrators snapshot it
# around their stages to log where the budget actually went.
_NUM_ROLLOUTS = 0


def num_rollouts_run() -> int:
    """Total :func:`rollout_states` invocations so far (cost telemetry)."""
    return _NUM_ROLLOUTS


def _zero_all_velocities(base_env: Any) -> None:
    """Zero every velocity in the env's client: base velocities of all bodies
    AND joint velocities of articulated bodies (the robot arm).

    ``_set_state`` rewrites poses but leaves velocities untouched, and
    its per-component diff skips joints whose positions already match
    the requested state, so without this a rollout inherits residual
    momentum from the previous rollout and no longer starts at rest. The
    joint pass matters as much as the base pass: measured up to ~1.8
    rad/s residual arm-joint velocity between rollouts, which chaotic
    contact amplifies into a 20-40% same-theta SSE jitter
    (run_20260705_203314). Fixed-base bodies ignore the base reset and
    fixed joints ignore the joint reset, so blanket-zeroing is safe.
    """
    pcid = base_env._physics_client_id  # pylint: disable=protected-access
    for i in range(p.getNumBodies(physicsClientId=pcid)):
        bid = p.getBodyUniqueId(i, physicsClientId=pcid)
        p.resetBaseVelocity(bid, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0],
                            physicsClientId=pcid)
        for j in range(p.getNumJoints(bid, physicsClientId=pcid)):
            pos = p.getJointState(bid, j, physicsClientId=pcid)[0]
            p.resetJointState(bid,
                              j,
                              pos,
                              targetVelocity=0.0,
                              physicsClientId=pcid)


def physical_param_anchors(
        base_env: Any,
        physical_specs: Sequence[ParamSpec]) -> Dict[str, float]:
    """Env-registry baseline values for the declared physical params.

    ``get_physical_param_info()`` defaults report the env's believed
    baseline WITHOUT any fit (the value the sysID revert path restores
    to), which makes them the right anchor for everything that must not
    drift with the agent's per-call declarations: the Gaussian prior
    center, the held-at values of grid sweeps, and the fallback applied
    for parameters the data does not constrain. Anchoring these at the
    agent's declared inits instead lets a re-declared init (a) change
    the explainability candidate grid call-to-call, flipping trimming
    verdicts on identical data, and (b) smuggle an unsupported
    hypothesis into the planner when the fit does not contract (e.g. a
    declared restitution of 0.15 surviving as "kept init" against a
    baseline of 0.02). Params the env does not reveal are absent from
    the result (callers fall back to the declared init).
    """
    getter = getattr(base_env, "get_physical_param_info", None)
    info = getter() if callable(getter) else {}
    return {
        s.name: float(info[s.name]["default"])
        for s in physical_specs if s.name in info
    }


def _pin_all_physical_params(base_env: Any,
                             physical_params: Dict[str, float]) -> None:
    """Apply ``physical_params`` with every OTHER registry param pinned to its
    env default.

    The env-side override is sticky per param, so on any env that
    outlives one evaluation (a caller-owned instance rather than a per-
    rollout factory build) a fit declaring a SUBSET of the params (e.g.
    only rolling_friction after an earlier fit touched lateral_friction)
    would silently inherit stale values from previous evaluations.
    Pinning also anchors undeclared params at the env's believed
    baseline on fresh builds, so every rollout evaluates the same
    nuisance physics regardless of env lifetime.
    """
    info: Dict[str, Dict] = getattr(base_env, "get_physical_param_info",
                                    lambda: {})()
    full = {name: float(spec["default"]) for name, spec in info.items()}
    full.update(physical_params)
    base_env.apply_physical_param_overrides(full)


def dispose_env(env: Any) -> None:
    """Free a fresh rollout env, releasing EVERY client it owns.

    Delegates to ``env.dispose()`` so envs with secondary worlds (the
    domino env's counterfactual-probe client) release them too - a raw
    ``p.disconnect(_physics_client_id)`` leaked the probe world per
    fresh validation env (~150MB each, machine-freezing across
    parallel runs).
    """
    dispose = getattr(env, "dispose", None)
    if callable(dispose):
        dispose()
        return
    p.disconnect(env._physics_client_id)  # pylint: disable=protected-access


def rollout_states(
    base_env: Any,
    init_state: State,
    actions: List[Action],
    physical_params: Dict[str, float],
    post_step: Optional[Callable[[Any, State, int],
                                 None]] = None) -> List[State]:
    """Free-run the base sim from ``init_state`` under ``actions``.

    ``post_step(env, state, i)`` (optional) runs after each action with
    the live env, the post-step state, and the step index. The rollout
    objective uses it to run the residual rules in-the-loop: rules that
    emit physics commands queue them on ``env`` there, so the commands
    shape the remainder of THIS rollout (feature-update rules remain
    scoring-side only; see ``_iter_rollout_residual_terms``).

    ``base_env`` is either an env instance or a zero-arg FACTORY: a
    factory is invoked to build a fresh env for this single rollout and
    the fresh env's PyBullet client is disconnected before returning.
    Fresh per-rollout worlds are what make repeated evaluations of the
    same theta deterministic - state-level resets on a shared env leave
    history-dependent residuals (near-matching bodies skipped by the
    reconstruction diff, auxiliary robot joints no reset touches), and
    even a bit-identical ``p.restoreState`` world diverges after a
    heavy-contact rollout via solver-internal state. Measured on
    run_20260708_213258: same-theta SSE alternated 0.15/78 on a shared
    env, which corrupted the grid seed and floored the identifiability
    probe (noise floor 82.9 -> every param "NOT identified").

    Resets once to ``init_state`` (zeroing velocities so the rollout
    begins at rest, matching how a recorded cascade starts), applies the
    candidate physics in place (undeclared registry params pinned to env
    defaults, see :func:`_pin_all_physical_params`), then steps WITHOUT
    resetting so momentum accrues in-sim. Returns the post-step state
    after each action (length == ``len(actions)``).
    """
    global _NUM_ROLLOUTS  # pylint: disable=global-statement
    _NUM_ROLLOUTS += 1
    env = base_env() if callable(base_env) else base_env
    try:
        _pin_all_physical_params(env, physical_params)
        env._set_state(init_state)  # pylint: disable=protected-access
        _zero_all_velocities(env)
        # Re-apply after _set_state in case a reset path ever touches
        # dynamics.
        _pin_all_physical_params(env, physical_params)
        out: List[State] = []
        for i, action in enumerate(actions):
            state = env.step(action)
            if post_step is not None:
                post_step(env, state, i)
            out.append(state)
        return out
    finally:
        if env is not base_env:
            dispose_env(env)
