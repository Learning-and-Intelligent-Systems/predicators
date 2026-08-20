"""Trajectory-level legitimacy certificate for domino cascade tasks.

The domino evaluator's terminated/success checks (``DominoEvaluator``)
are functions of the final state only (goal atoms + toppled-blue
count), which are blind to HOW the target fell. Observed reward hacks
all bypass the intended causal chain "push the green start block ->
dominoes knock each other over -> target falls": pushing a placed blue
directly, sweeping the target with the gripper or a carried block
during Place, knocking the target while an option flails, relocating
the green start block (or the targets themselves) to skip the chain,
and toppling a block with the robot's body so the arm - not the
cascade - supplies the energy.

The certificate decides legitimacy in two layers:

1. **Staging integrity** (pure state/action rules): only the blue
   movable dominoes are the robot's to rearrange. Until the first Push
   on the green start block, the scene must stay as staged - nothing
   topples, and every non-movable domino (the green, the targets,
   heavy blocks) is never held, stays within ``_stage_tolerance()`` of
   its staged xy, and stands upright (below the tilting band, so a
   pre-tilted target cannot hand the probe a half-fallen scene). Only
   the green may ever be the target of a Push, and once anything
   topples such a Push must exist.
2. **The counterfactual push probe** (physics, via the injected
   ``probe``): on goal-reaching episodes, the episode's own Push skill
   re-runs from the recorded pre-push state - the real controller with
   the plan's recorded continuous parameters - in a dedicated
   same-physics world where only the robot's fingertips can touch
   anything (the arm's body is collision-masked; see
   ``cascade_probe``). The goal atoms must topple under that push.

The probe is the sole authority on HOW the goal fell: it re-derives
the outcome from the staged layout and the push alone, so anything the
real episode does after the push - a stalled hop, arm contact, a late
topple - neither earns nor voids the bonus. This replaced first the
old swept-corridor / shoved-relay / robot-strike attribution rules and
then the interim green-first / onset-chaining timing rules: forensic
reconstruction of per-block causality produced both misses and false
rejections (run_20260715_220941: a genuine green-on-blue knock
measured 7 mm of modeled corridor clearance with the end-effector
nearby and was charged to the robot; same-step onset ties on corner
layouts), while the probe answers the only question that matters -
does the layout the robot built actually cascade to the goal under a
clean push? Arm collateral cannot help (the probe's arm is
intangible), so any hack that needs the robot's body to reach the goal
fails the probe. The deliberate flip side: a working layout certifies
even if the real episode also used the arm after the push - the bonus
rewards the layout, which the probe verifies, not the execution.

Layer 1 is a pure function over ``State`` sequences and runs
identically everywhere; layer 2 needs a physics rollout, so the caller
injects ``probe`` (``DominoEvaluator`` binds it from the certifying
env - the true env env-side, the agent's belief env in sandbox
verdicts, each side probing with its own physics). A goal-reaching
episode with no probe available fails closed: with the forensic rules
gone, an uncertifiable success must not score.
"""

import logging
import math
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.structs import GroundAtom, Object, State, StepOption

# The name of the option through which the robot is allowed to topple
# the green start block.
_PUSH_OPTION_NAME = "Push"

# Consecutive non-held states a domino must spend at or past
# ``fallen_threshold`` before that counts as a topple rather than as
# noise.
_TOPPLE_MIN_STEPS = 3

# Minimum base displacement for a movable blue to count as consumed by
# the cascade (shoved off its stand) in ``count_movable_blocks_used``.
# Well above resting jitter and release-settle skids (millimeters); a
# genuine transmitting slide covers centimeters (the recorded
# slide-relay episode measured 0.12 m).
RELAY_MIN_SLIDE = 0.02

# Signature of the injected counterfactual probe: (pre-push state,
# pushed greens in push order, goal atoms, the episode's Push continuous
# parameters or None) -> (ok, human-readable detail). See
# ``PyBulletDominoComposedEnv.run_counterfactual_cascade_probe``.
CascadeProbe = Callable[
    [State, Sequence[Object], Set[GroundAtom], Optional[Tuple[float, ...]]],
    Tuple[bool, str]]


def count_movable_blocks_used(states: Sequence[State]) -> int:
    """Count movable (blue) dominoes the episode consumed.

    A blue is consumed when it has toppled in the final state, or when
    it was displaced at least ``RELAY_MIN_SLIDE`` within a span where it
    was not held: that is a cascade shove (robot transport is excluded
    by the held gating), so a slide-relay that knocks its successor
    over without itself toppling is charged the same as a toppled relay
    and staying upright earns no cost discount.

    A pure function of the states (roles recovered from color features,
    topple from the roll angle, shoves from not-held displacement) so
    ``DominoEvaluator`` needs no live env handle and stays picklable.
    """
    count = 0
    final = states[-1]
    for obj in final:
        if obj.type.name != "domino":
            continue
        # pylint: disable=protected-access
        if not DominoComponent._MovableBlock_holds(final, [obj]):
            continue
        if abs(final.get(obj, "roll")) >= DominoComponent.fallen_threshold:
            count += 1
            continue
        anchor: Optional[State] = None
        for state in states:
            if state.get(obj, "is_held") > 0.5:
                anchor = None
                continue
            if anchor is None:
                anchor = state
                continue
            if math.hypot(
                    state.get(obj, "x") - anchor.get(obj, "x"),
                    state.get(obj, "y") -
                    anchor.get(obj, "y")) >= RELAY_MIN_SLIDE:
                count += 1
                break
    return count


def _stage_tolerance() -> float:
    """Max xy drift of a non-movable domino from its staged pose that still
    counts as "left where it stands".

    A legitimate push tips the green block about its base edge, so the
    base slides at most a couple of centimeters before the block leaves
    the upright band; relocating any block anywhere useful (next to the
    target, or even one chain gap over) moves it by at least ``pos_gap``
    ~= 0.098 m. One domino width sits safely between the two regimes.
    """
    from predicators.envs.pybullet_domino.env import \
        PyBulletDominoComposedEnv  # pylint: disable=import-outside-toplevel
    return PyBulletDominoComposedEnv.domino_width


def _role_label(state: State, domino: Object) -> str:
    """Human-readable role of a non-movable ``domino``, for error messages."""
    # pylint: disable=protected-access
    if DominoComponent._StartBlock_holds(state, [domino]):
        return "green start block"
    if DominoComponent._TargetDomino_holds(state, [domino]):
        return "target domino"
    if DominoComponent._HeavyBlock_holds(state, [domino]):
        return "gray block"
    return "non-movable domino"


def _topple_onset(states: Sequence[State], domino: Object) -> Optional[int]:
    """State index where ``domino``'s final fall began, or None.

    A domino has a topple event iff its |roll| holds at or past
    ``fallen_threshold`` for ``_TOPPLE_MIN_STEPS`` consecutive states
    while not held, or does so in a run that reaches the end of the episode.
    The run restarts whenever the domino is held or comes back inside the
    threshold, so a domino set down crooked or a domino swinging in the
    gripper both stay quiet.

    The onset is then walked back to the moment it last left the upright
    band (|roll| < ``domino_roll_threshold``) before that first full
    topple, so placement wobbles that recover never register and a
    wobble that precedes the real fall is not mistaken for it.

    That backward search may not cross a carry. A domino cannot have
    been falling since before the robot picked it up and moved it, so
    the search floor is the step after it was last released. Without
    that floor the search walks straight through the pick-and-place: a
    bridge domino that lands a few degrees off plumb never re-enters the
    upright band, so the last "standing" moment found is the one before
    it was ever picked up, and a cascade that arrives long after the
    push gets dated to the grasp. A real run was rejected exactly that
    way -- the search stopped at step 21, the step the gripper closed.

    A domino that was staged and then rested below ``fallen_threshold``
    until something reached it is standing, not falling, so its fall is
    dated to the fall itself. Being set down slightly crooked is not the
    beginning of a topple; the robot dropping it over still is, because
    then the full topple lands at the release and is caught there.
    """
    fall_idx: Optional[int] = None
    run_start: Optional[int] = None
    for t, state in enumerate(states):
        held = state.get(domino, "is_held") > 0.5
        fallen = abs(state.get(domino,
                               "roll")) >= DominoComponent.fallen_threshold
        # A carry breaks the run.
        if held or not fallen:
            run_start = None
            continue
        if run_start is None:
            run_start = t
        if t - run_start + 1 >= _TOPPLE_MIN_STEPS:
            fall_idx = run_start
            break
    if fall_idx is None and run_start is not None:
        # The run was still going when the episode ended. A topple in the
        # last states has no room to persist, and discarding it would
        # empty ``onsets`` and pass the episode unexamined -- turning a
        # false reject into a false accept, which is the worse failure.
        fall_idx = run_start
    if fall_idx is None:
        return None
    # Step after the last release before the fall; 0 if never carried.
    floor = 0
    for t in range(fall_idx - 1, -1, -1):
        if states[t].get(domino, "is_held") > 0.5:
            floor = t + 1
            break
    onset = None
    for t in range(fall_idx - 1, floor - 1, -1):
        state = states[t]
        if state.get(domino, "is_held") > 0.5:
            continue
        if abs(state.get(domino,
                         "roll")) < DominoComponent.domino_roll_threshold:
            onset = t + 1
            break
    if onset is not None:
        return onset
    if floor > 0:
        # Staged by the robot and never seen dead upright since, i.e. it
        # sat where it was placed until something reached it.
        return fall_idx
    # Never observed upright and non-held before falling: use the first
    # non-held index (conservative earliest).
    for t in range(fall_idx + 1):
        if states[t].get(domino, "is_held") <= 0.5:
            return t
    return fall_idx


def _push_on_green_spans(
        step_options: Sequence[StepOption], greens: Sequence[Object],
        domino_names: Set[str]) -> Tuple[List[Tuple[int, int]], bool]:
    """Maximal runs of consecutive action indices whose option is a Push on a
    green start block, plus whether any option label was missing.

    A Push whose objects include no domino at all is the restricted
    variant (``domino_restricted_push``), which always targets the
    inferred start block - counted as a push on green. A Push that
    explicitly names a non-green domino is not.
    """
    green_names = {g.name for g in greens}
    push_idxs = []
    any_unknown = False
    for i, step_option in enumerate(step_options):
        if step_option is None:
            any_unknown = True
            continue
        name, object_names = step_option[0], step_option[1]
        if name == _PUSH_OPTION_NAME and (
                green_names & set(object_names)
                or not domino_names & set(object_names)):
            push_idxs.append(i)
    spans: List[Tuple[int, int]] = []
    for i in push_idxs:
        if spans and i == spans[-1][1] + 1:
            spans[-1] = (spans[-1][0], i)
        else:
            spans.append((i, i))
    return spans, any_unknown


def _pushed_greens_in_order(step_options: Sequence[StepOption],
                            greens: Sequence[Object],
                            spans: Sequence[Tuple[int, int]]) -> List[Object]:
    """The green blocks the episode pushed, ordered by their first push.

    A restricted Push names no domino; it targets the inferred start
    block, which is unambiguous exactly when there is one green.
    """
    by_name = {g.name: g for g in greens}
    ordered: List[Object] = []
    for start, _ in spans:
        step_option = step_options[start]
        assert step_option is not None
        object_names = step_option[1]
        named = [by_name[n] for n in object_names if n in by_name]
        for g in named or list(greens):
            if g not in ordered:
                ordered.append(g)
    return ordered


def _push_params_of_span(step_options: Sequence[StepOption],
                         span_start: int) -> Optional[Tuple[float, ...]]:
    """The continuous Push parameters recorded on a span's first label.

    Returns None for legacy 2-tuple labels (agent-authored label lists,
    old tests) or empty parameter tuples - the probe then falls back to
    the canonical push.
    """
    step_option = step_options[span_start]
    assert step_option is not None
    if len(step_option) > 2 and step_option[2]:
        return tuple(step_option[2])
    return None


def check_cascade_legitimacy(
        states: Sequence[State],
        goal: Set[GroundAtom],
        step_options: Optional[Sequence[StepOption]] = None,
        probe: Optional[CascadeProbe] = None) -> Tuple[bool, str]:
    """Certify that the episode's topples are a genuine push-seeded cascade.

    Rules (any violation fails the whole episode):
      (a)  the Push option is only ever legal on the green start block -
           a Push that names any other domino fails outright - and once
           anything topples, a Push on the green must exist;
      (b)  nothing topples before the robot's first Push on the green:
           the scene must stay standing until the push;
      (c)  every non-movable domino - the green start block, the
           targets, heavy blocks - is at its staged pose when the push
           happens: never held up to that point, within
           ``_stage_tolerance()`` of its staged xy, and upright (below
           the tilting band, so a pre-tilted block cannot hand the
           probe a half-fallen scene). Only the blue movable blocks are
           the robot's to carry and place;
      (d)  when the goal atoms hold at the episode's end, the injected
           counterfactual ``probe`` must reproduce the cascade: the
           episode's own Push skill re-run (real controller, the
           plan's recorded continuous parameters) on the green(s) from
           the recorded pre-push state - same physics, only the
           fingertips collidable - must reach the goal atoms. The probe
           alone decides how the goal fell (see the module docstring);
           it runs only on goal-reaching episodes because only those
           have a success bonus at stake, and a goal-reaching episode
           with no probe available fails closed.

    ``step_options`` labels each transition ``states[t] -> states[t+1]``
    (action index ``t``) with the producing option; when it is None the
    action rules (a)/(b) are skipped, the staging rule anchors to the
    state just before the first topple onset, and the probe falls back
    to that state as its pre-push state. ``goal`` feeds the probe's
    success check and the error messages - all dominoes are held to the
    same rules.

    Returns ``(ok, reason)`` with a human-readable reason on failure.
    """
    if len(states) < 2:
        return True, ""
    dominoes = [obj for obj in states[0] if obj.type.name == "domino"]
    domino_names = {d.name for d in dominoes}
    # pylint: disable=protected-access
    greens = [
        d for d in dominoes
        if DominoComponent._StartBlock_holds(states[0], [d])
    ]
    # Only the blue movable blocks are the robot's to arrange; every
    # other domino (the green start block, the targets, heavy blocks) is
    # scenery that must be toppled where the task staged it.
    non_movables = [
        d for d in dominoes
        if not DominoComponent._MovableBlock_holds(states[0], [d])
    ]
    onsets: Dict[Object, int] = {}
    for d in dominoes:
        onset = _topple_onset(states, d)
        if onset is not None:
            onsets[d] = onset
    if not onsets:
        return True, ""
    if not greens:
        toppled = sorted(d.name for d in onsets)
        return False, (f"{', '.join(toppled)} toppled but there is no green "
                       "start block in the scene to seed a cascade")

    # Action rules (a)/(b).
    pre_push_idx: Optional[int] = None
    pushed_greens: List[Object] = list(greens)
    push_params: Optional[Tuple[float, ...]] = None
    if step_options is not None:
        for i, step_option in enumerate(step_options):
            if step_option is None:
                continue
            name, object_names = step_option[0], step_option[1]
            if name == _PUSH_OPTION_NAME:
                foreign = sorted((set(object_names) & domino_names) -
                                 {g.name
                                  for g in greens})
                if foreign:
                    return False, (
                        f"the robot pushed {', '.join(foreign)} (Push at "
                        f"step {i}) - only the green start block may be "
                        "pushed")
        spans, any_unknown = _push_on_green_spans(step_options, greens,
                                                  domino_names)
        if any_unknown:
            logging.warning(
                "[cascade certificate] some actions lack option labels; "
                "action rules may be incomplete for those steps.")
        if not spans:
            return False, ("dominoes toppled but the green start block was "
                           "never pushed (no Push on it in the episode)")
        first_push = spans[0][0]
        pre_push_idx = first_push
        pushed_greens = _pushed_greens_in_order(step_options, greens, spans)
        push_params = _push_params_of_span(step_options, first_push)
        for d, t in sorted(onsets.items(), key=lambda kv: kv[1]):
            # Action index first_push produces state index first_push+1.
            if t <= first_push:
                return False, (
                    f"{d.name} started falling at step {t}, before the "
                    "green start block was first pushed (step "
                    f"{first_push + 1}) - the scene must stay standing "
                    "until the push")
    if pre_push_idx is None:
        # Label-free fallback: anchor to the state just before the
        # first fall.
        pre_push_idx = max(min(onsets.values()) - 1, 0)
    pre_push_idx = min(pre_push_idx, len(states) - 1)
    pre_push = states[pre_push_idx]

    # Rule (c): staging integrity of every non-movable domino at the
    # pre-push snapshot. Measured here rather than per-onset: robot
    # staging necessarily precedes the push, while everything the
    # cascade itself does to a block (shoves, slides, topples) happens
    # after it, so the pre-push snapshot cleanly separates the two.
    stage_tol = _stage_tolerance()
    for d in non_movables:
        for t in range(pre_push_idx + 1):
            if states[t].get(d, "is_held") > 0.5:
                return False, (
                    f"the {_role_label(states[0], d)} {d.name} was picked up "
                    f"(held at step {t}, before the push) - only the blue "
                    "movable blocks may be carried; it must be toppled where "
                    "it stands, not relocated")
        drift = math.hypot(
            pre_push.get(d, "x") - states[0].get(d, "x"),
            pre_push.get(d, "y") - states[0].get(d, "y"))
        if drift > stage_tol:
            return False, (
                f"the {_role_label(states[0], d)} {d.name} moved "
                f"{drift:.2f} m from its staged pose before the push - "
                "only the blue movable blocks may be rearranged")
        lean = abs(pre_push.get(d, "roll"))
        if lean >= DominoComponent.domino_roll_threshold:
            return False, (
                f"the {_role_label(states[0], d)} {d.name} was leaning "
                f"{math.degrees(lean):.1f} deg when the push happened - "
                "it must still stand upright as staged; a pre-tilted "
                "block is a disturbed scene, not a cascade")

    # Rule (d): the counterfactual push probe, on goal-reaching episodes.
    if not all(atom.holds(states[-1]) for atom in goal):
        return True, ""
    if probe is None:
        return False, (
            "the goal atoms hold, but no counterfactual push probe is "
            "available to verify the cascade - an unverifiable success "
            "cannot be certified (bind sim_env at the evaluator call site)")
    ok, detail = probe(states[pre_push_idx], pushed_greens, goal, push_params)
    if not ok:
        return False, (
            "the goal atoms hold, but a clean counterfactual push on "
            f"{', '.join(g.name for g in pushed_greens)} from the pre-push "
            f"scene does not reproduce the cascade ({detail}) - the goal "
            "topples are owed to the robot's body, not the built layout")
    return True, ""
