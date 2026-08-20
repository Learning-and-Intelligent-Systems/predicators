"""Tests for the domino cascade legitimacy certificate.

Pure-State tests (no PyBullet): trajectories are synthesized as step-
function roll profiles, so each domino's topple onset is exactly the
step where its roll jumps past the fallen threshold. The counterfactual
push probe (rule (c)) is exercised through injected fakes here; its
physics lives in ``cascade_probe`` and is integration-tested in
``test_pybullet_domino_composed.py``.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from predicators.envs.pybullet_domino.cascade_certificate import \
    _TOPPLE_MIN_STEPS, _topple_onset, check_cascade_legitimacy, \
    count_movable_blocks_used
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.structs import GroundAtom, Object, Predicate, State, Type

_DOMINO_TYPE = Type("domino",
                    ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"])
_ROBOT_TYPE = Type("robot", ["x", "y", "z"])
# Never holds: the probe (which only runs on goal-reaching final states)
# stays out of the pure-rule tests below.
_TOPPLED = Predicate("Toppled", [_DOMINO_TYPE], lambda s, o: False)
# Holds on genuinely fallen blocks: for the probe-path tests, whose
# trajectories must reach the goal.
_TOPPLED_REAL = Predicate(
    "Toppled", [_DOMINO_TYPE],
    lambda s, o: abs(float(s.get(o[0], "roll"))) >= abs(-0.6))

_GREEN = DominoComponent.start_domino_color
_BLUE = DominoComponent.domino_color
_PURPLE = DominoComponent.target_domino_color

# Past fallen_threshold (10 deg ~= 0.175). Negative: with the env's yaw
# convention a NEGATIVE roll falls along (-sin yaw, cos yaw), i.e. +y at
# yaw 0, which is the direction the synthetic chains below progress.
_FALLEN_ROLL = -0.6

# A straight chain along y with hops well inside the 0.18 m reach.
_CHAIN_Y = {"green": 1.0, "blue1": 1.098, "blue2": 1.196, "target": 1.294}


def _make_objects(names: Sequence[str]) -> Dict[str, Object]:
    return {name: Object(name, _DOMINO_TYPE) for name in names}


def _color_for(name: str) -> Tuple[float, float, float, float]:
    if name.startswith("green"):
        return _GREEN
    if name.startswith("target"):
        return _PURPLE
    return _BLUE


def _build_states(
    objs: Dict[str, Object],
    num_steps: int,
    onsets: Dict[str, int],
    *,
    positions: Optional[Dict[str, Tuple[float, float]]] = None,
    position_profiles: Optional[Dict[str, Sequence[Tuple[float,
                                                         float]]]] = None,
    roll_profiles: Optional[Dict[str, Sequence[float]]] = None,
    held_spans: Optional[Dict[str, Tuple[int, int]]] = None,
    yaws: Optional[Dict[str, float]] = None,
    robot_xyz: Optional[Tuple[float, float, float]] = None,
    robot_profile: Optional[Sequence[Tuple[float, float, float]]] = None,
) -> List[State]:
    """Build a trajectory of ``num_steps + 1`` states.

    ``onsets[name] = t`` gives a step-function roll: 0 before ``t``,
    fallen after. ``roll_profiles`` overrides the roll sequence for a
    domino entirely; ``position_profiles`` gives a per-step (x, y)
    sequence for a domino (e.g. a relocation); ``held_spans[name] =
    (a, b)`` sets is_held on state indices [a, b]; ``yaws`` overrides
    the default 0 yaw; ``robot_xyz`` adds a robot object parked at that
    end-effector position for the whole episode, or ``robot_profile``
    adds one that moves along a per-step (x, y, z) sequence.
    """
    add_robot = robot_xyz is not None or robot_profile is not None
    robot = Object("robot", _ROBOT_TYPE) if add_robot else None
    states = []
    for t in range(num_steps + 1):
        data = {}
        for name, obj in objs.items():
            if position_profiles is not None and name in position_profiles:
                x, y = position_profiles[name][t]
            elif positions is not None and name in positions:
                x, y = positions[name]
            else:
                x, y = 0.7, _CHAIN_Y[name]
            if roll_profiles is not None and name in roll_profiles:
                roll = roll_profiles[name][t]
            elif name in onsets:
                roll = _FALLEN_ROLL if t >= onsets[name] else 0.0
            else:
                roll = 0.0
            held = 0.0
            if held_spans is not None and name in held_spans:
                lo, hi = held_spans[name]
                held = 1.0 if lo <= t <= hi else 0.0
            yaw = yaws.get(name, 0.0) if yaws is not None else 0.0
            data[obj] = np.array(
                [x, y, 0.475, yaw, roll, *_color_for(name)[:3], held],
                dtype=np.float32)
        if robot is not None:
            xyz = robot_profile[t] if robot_profile is not None else robot_xyz
            data[robot] = np.array(xyz, dtype=np.float32)
        states.append(State(data))
    return states


def _goal(objs: Dict[str, Object]) -> set:
    return {GroundAtom(_TOPPLED, [objs["target"]])}


def _options(
    spans: Sequence[Tuple[str, Tuple[str, ...], int, int]],
    num_actions: int,
) -> List[Optional[Tuple[Any, ...]]]:
    """Build a per-action option labeling from (name, objects, lo, hi) spans
    over action indices; unlabeled actions become Wait.

    Labels are loose tuples so callers may stamp params (3-tuple form)
    onto them.
    """
    labels: List[Optional[Tuple[Any, ...]]] = [("Wait", ("robot", ))
                                               for _ in range(num_actions)]
    for name, objects, lo, hi in spans:
        for i in range(lo, hi + 1):
            labels[i] = (name, objects)
    return labels


def test_legit_chain_passes():
    """Push green -> cascade through two blues -> target: accepted."""
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    states = _build_states(objs, 30, {
        "green": 5,
        "blue1": 10,
        "blue2": 14,
        "target": 18
    })
    # Push on green during actions 0-7, then Wait while the cascade runs.
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_k0_pure_push_passes():
    """Green adjacent to target, no blues needed: accepted."""
    objs = _make_objects(["green", "target"])
    states = _build_states(objs,
                           20, {
                               "green": 5,
                               "target": 11
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098)
                           })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 20)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_green_relocation_fails():
    """Picking up green and placing it next to the target: rejected.

    The zero-blue exploit from run_20260712_122549: relocate the green
    start block adjacent to the target, push it, topple the target with
    no chain at all. Rule (a2) rejects the pick regardless of where the
    block ends up.
    """
    objs = _make_objects(["green", "target"])
    staged = (0.7, 1.0)
    relocated = (0.7, 1.28)  # adjacent to the target at (0.7, 1.378)
    green_xy = [staged] * 3 + [relocated] * 28
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 18
                           },
                           positions={"target": (0.7, 1.378)},
                           position_profiles={"green": green_xy},
                           held_spans={"green": (3, 8)})
    step_options = _options([("Pick", ("robot", "green"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "picked up" in reason


def test_green_slide_relocation_fails():
    """Sliding green next to the target without ever holding it: rejected.

    Rule (a3): at its topple onset the green block must be within the
    stage tolerance of its initial pose, so a gripper-sweep relocation
    earns no bonus even though is_held never fires.
    """
    objs = _make_objects(["green", "target"])
    green_xy = [(0.7, 1.0)] * 5 + [(0.7, 1.28)] * 26
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 18
                           },
                           positions={"target": (0.7, 1.378)},
                           position_profiles={"green": green_xy})
    step_options = _options([("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "staged pose" in reason


def test_green_pickup_after_cascade_passes():
    """Picking the fallen green up after the cascade stays legal.

    Episodes run past the goal (terminate_on_goal_reached=False), and
    exploration routinely fiddles with fallen blocks afterwards - that
    must not void an already-legitimate cascade (real case: the two
    reward-0.75 train episodes of run_20260712_122549, green picked up
    at steps 360/287 of 500).
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "blue2": 14,
                               "target": 18
                           },
                           held_spans={"green": (24, 28)})
    step_options = _options([("Push", ("robot", "green"), 0, 7),
                             ("Pick", ("robot", "green"), 23, 27)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_green_push_jitter_passes():
    """Small base slide during the push itself stays within tolerance."""
    objs = _make_objects(["green", "target"])
    green_xy = [(0.7, 1.0)] * 11 + [(0.7, 1.02)] * 20
    states = _build_states(objs,
                           30, {
                               "green": 12,
                               "target": 16
                           },
                           positions={"target": (0.7, 1.098)},
                           position_profiles={"green": green_xy})
    step_options = _options([("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_target_relocation_fails():
    """Carrying the targets into a line in front of green: rejected.

    The zero-blue exploit from run_20260715_084342 task 1: leave green
    and the blues untouched, but pick the targets up and set them down
    in front of green, so one push cascades straight through them. It
    reached the goal with no chain at all and scored a full reward=1.0.
    Rule (a2) rejects the pick: only blues are the robot's to carry.
    """
    objs = _make_objects(["green", "blue1", "target"])
    staged = (0.9, 1.32)
    relocated = (0.7, 1.098)  # lined up in front of green at (0.7, 1.0)
    target_xy = [staged] * 3 + [relocated] * 28
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 18
                           },
                           positions={"green": (0.7, 1.0)},
                           position_profiles={"target": target_xy},
                           held_spans={"target": (3, 8)})
    step_options = _options([("Pick", ("robot", "target"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target domino" in reason
    assert "picked up" in reason


def test_target_slide_relocation_fails():
    """Sliding a target into place without ever holding it: rejected.

    Rule (a3) backstops (a2): a gripper-sweep relocation of a target
    before the cascade starts earns no bonus even though is_held never
    fires.
    """
    objs = _make_objects(["green", "target"])
    target_xy = [(0.9, 1.32)] * 5 + [(0.7, 1.098)] * 26
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 18
                           },
                           positions={"green": (0.7, 1.0)},
                           position_profiles={"target": target_xy})
    step_options = _options([("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "staged pose" in reason


def test_target_shoved_by_cascade_before_toppling_passes():
    """A target the cascade shoves before it tips is not charged to the robot.

    Rule (a3) measures staged-pose drift at the cascade's START, not at
    each block's own onset, precisely so this stays legal: the chain
    rams the target, slides it several centimeters (further than the
    stage tolerance), and only then tips it over. That displacement is
    the cascade's doing, not a relocation.
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    # Struck at step 15, shoved 0.08 m (> the 0.07 m stage tolerance),
    # topples at 18. Untouched through the cascade's start (step 5).
    target_xy = [(0.7, 1.25)] * 16 + [(0.7, 1.33)] * 15
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "blue2": 14,
                               "target": 18
                           },
                           position_profiles={"target": target_xy})
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_blue_relocation_passes():
    """Carrying the blues into a chain is the task, not an exploit.

    The mirror of ``test_target_relocation_fails``: rules (a2)/(a3)
    cover every domino EXCEPT the blue movable blocks, which the robot
    is expressly given to arrange.
    """
    objs = _make_objects(["green", "blue1", "target"])
    blue_xy = [(0.9, 1.32)] * 3 + [(0.7, 1.098)] * 28
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "blue1": 18,
                               "target": 22
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.196)
                           },
                           position_profiles={"blue1": blue_xy},
                           held_spans={"blue1": (3, 8)})
    step_options = _options([("Pick", ("robot", "blue1"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_blue_placed_slightly_crooked_passes():
    """A bridge domino set down off plumb is standing, not falling.

    A real placement does not land at exactly zero roll, so the blue
    never re-enters the upright band (5 degrees) after release. The
    onset search must not walk back through the carry on that account
    and date the cascade's topple to the grasp: run_20260807_105851 was
    rejected exactly that way, reporting that the blue "started falling
    at step 21" -- the step the gripper closed -- while the cascade
    actually reached it long after the push. Sim never showed it,
    because a placed domino settles there at exactly zero roll.
    """
    objs = _make_objects(["green", "blue1", "target"])
    crooked = np.deg2rad(6.0)
    # Above the upright band but below fallen: resting, not toppled.
    assert DominoComponent.domino_roll_threshold < crooked
    assert crooked < DominoComponent.fallen_threshold
    # Upright, carried (3-8), then resting crooked until the cascade.
    rolls = [0.0] * 3 + [crooked] * 15 + [_FALLEN_ROLL] * 13
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "blue1": 18,
                               "target": 22
                           },
                           roll_profiles={"blue1": rolls},
                           held_spans={"blue1": (3, 8)})
    step_options = _options([("Pick", ("robot", "blue1"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_blue_dropped_over_before_the_push_still_fails():
    """The guard above must not excuse actually knocking one down.

    Same staging, but the blue is on its side the moment it is released
    -- before the push. The fall then lands at the release and is caught
    there, so a crooked resting pose being forgiven does not forgive a
    robot that drops a domino over.
    """
    objs = _make_objects(["green", "blue1", "target"])
    rolls = [0.0] * 8 + [_FALLEN_ROLL] * 23
    states = _build_states(objs,
                           30, {
                               "green": 14,
                               "target": 22
                           },
                           roll_profiles={"blue1": rolls},
                           held_spans={"blue1": (3, 8)})
    step_options = _options([("Pick", ("robot", "blue1"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 10, 13)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "blue1" in reason and "before" in reason


def test_push_placed_blue_fails():
    """Pushing a blue directly: rejected outright by the only-green rule."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 30, {"blue1": 10, "target": 15})
    step_options = _options([("Push", ("robot", "blue1"), 5, 12)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "only the green start block may be pushed" in reason


def test_push_placed_blue_without_options_probe_decides():
    """Same hack, label-free: with no bonus at stake it is accepted as
    worthless; once the goal holds, the probe (which replays a push on the
    GREEN from the pre-onset scene) is the layer that rejects it."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 30, {"blue1": 10, "target": 15})
    # Goal unreached (never-holds Toppled): no bonus at stake, accepted.
    ok, reason = check_cascade_legitimacy(states, _goal(objs), None)
    assert ok, reason
    # Goal reached: the probe decides, and a blue-push layout cannot
    # cascade from a green push.
    probe = _FakeProbe(ok=False, detail="green push reaches nothing")
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          None,
                                          probe=probe)
    assert not ok
    assert "green push reaches nothing" in reason


def test_place_knock_fails():
    """Target knocked during Place before any push: rejected by (a0)."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 40, {"target": 3, "green": 25, "blue1": 29})
    step_options = _options([("Place", ("robot", ), 0, 9),
                             ("Push", ("robot", "green"), 20, 27)], 40)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "target" in reason and "before" in reason


def test_place_knock_without_options_probe_decides():
    """Same hack, label-free: the anchor falls back to the pre-onset state.

    That state precedes the knock, so the probe replays the true staged
    scene and a success that needed the knock cannot reproduce.
    """
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 40, {"target": 3, "green": 25, "blue1": 29})
    ok, reason = check_cascade_legitimacy(states, _goal(objs), None)
    assert ok, reason  # goal unreached: no bonus at stake
    probe = _FakeProbe(ok=False, detail="staged scene does not cascade")
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          None,
                                          probe=probe)
    assert not ok
    assert "staged scene does not cascade" in reason
    # The probe was anchored before the knock (first onset at 3).
    assert probe.calls[0][0] is states[2]


def test_flail_knock_rejected_by_probe():
    """Fist knocks the target during Push(green), green stays up: the goal
    holds but the staged layout cannot cascade, so the probe rejects."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs, 30, {"target": 15})
    step_options = _options([("Push", ("robot", "green"), 5, 29)], 30)
    probe = _FakeProbe(ok=False, detail="green never reaches the target")
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=probe)
    assert not ok
    assert "green never reaches the target" in reason


def test_spontaneous_late_tip_probe_decides():
    """A topple long after the cascade settled: no timing forensics anymore -
    the probe alone decides whether the staged layout deserves the bonus."""
    objs = _make_objects(["green", "blue1", "target"])
    late = 45
    num = late + 5
    states = _build_states(objs, num, {
        "green": 5,
        "blue1": 10,
        "target": late
    })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    # Goal unreached: accepted as worthless (no bonus at stake).
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason
    # Goal reached + probe says the layout cannot cascade: rejected.
    probe = _FakeProbe(ok=False, detail="chain stops at blue1")
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=probe)
    assert not ok
    assert "chain stops at blue1" in reason


def test_stalled_relay_probe_decides():
    """Hand-relaying a stalled cascade: the probe is the sole arbiter.

    If the staged layout genuinely cascades on its own, the episode
    certifies even though the real rollout needed help after the push
    (the bonus rewards the layout, which the probe verifies, not the
    execution); if the layout cannot cascade, it is rejected.
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    relay = 84
    num = relay + 10
    states = _build_states(objs, num, {
        "green": 5,
        "blue1": 10,
        "blue2": relay,
        "target": relay + 4
    })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    bad_layout = _FakeProbe(ok=False, detail="stalls at blue2")
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=bad_layout)
    assert not ok
    assert "stalls at blue2" in reason
    good_layout = _FakeProbe(ok=True)
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=good_layout)
    assert ok, reason


def _slide_profile(start: Tuple[float, float], stops: Dict[int, Tuple[float,
                                                                      float]],
                   num_steps: int) -> List[Tuple[float, float]]:
    """Step-function xy profile: at ``start`` until each ``stops[t]`` takes
    over from step ``t`` on."""
    profile = []
    pos = start
    for t in range(num_steps + 1):
        if t in stops:
            pos = stops[t]
        profile.append(pos)
    return profile


def test_slide_relay_passes():
    """A rammed block that slides into the target without toppling is a genuine
    cascade link and must pass the pure rules.

    The run_20260713_133936 task-3 false rejection under the old
    geometric attribution: the green fell onto a staged blue, which
    never toppled - it slid 0.12 m into the purple and knocked it over
    (then rocked back upright). The timing rule accepts it (the
    target's onset is one window from the green's); whether the slide
    genuinely transmits is the counterfactual probe's call.
    """
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    # blue1 starts inside green's corridor and slides to contact range
    # of the target between the green's onset (10) and the target's
    # onset (18).
    profile = _slide_profile((0.7, 1.10), {
        11: (0.7, 1.13),
        12: (0.7, 1.17),
        13: (0.7, 1.21),
        14: (0.7, 1.254)
    }, num)
    states = _build_states(objs,
                           num, {
                               "green": 10,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.27)
                           },
                           position_profiles={"blue1": profile})
    step_options = _options([("Push", ("robot", "green"), 0, 7)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason
    # The relay was consumed even though it never toppled: it costs the
    # same as a toppled blue, so staying upright earns no discount.
    assert count_movable_blocks_used(states) == 1


def test_count_used_includes_displaced_blues():
    """count_movable_blocks_used charges toppled blues AND blues the cascade
    shoved (not-held displacement), but never robot-transported ones."""
    objs = _make_objects(["green", "blue1", "blue2", "blue3", "blue4"])
    num = 20
    # blue2 is shoved 0.05 m while free; blue4 moves only while held.
    shoved = _slide_profile((0.9, 1.2), {12: (0.9, 1.25)}, num)
    carried = _slide_profile((1.0, 1.2), {8: (1.0, 1.4)}, num)
    states = _build_states(objs,
                           num, {
                               "green": 5,
                               "blue1": 10
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "blue3": (0.8, 1.2)
                           },
                           position_profiles={
                               "blue2": shoved,
                               "blue4": carried
                           },
                           held_spans={"blue4": (6, 15)})
    assert count_movable_blocks_used(states) == 2


def test_tight_chain_with_robot_adjacent_passes():
    """A solid corridor-overlap knock stays legitimate even when the EE is
    centimeters away: pushing the green in a tightly staged chain necessarily
    ends the push with the EE next to the first relay."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 9,
                               "target": 14
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           },
                           robot_xyz=(0.7, 1.05, 0.5))
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_held_block_roll_excursion_ignored():
    """A carried blue tilts freely; only unheld topples count."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    # blue1 is held (and tilted) on steps 3-8, upright after placement,
    # then falls via the cascade from step 16 on.
    profile = [0.0] * (num + 1)
    for t in range(3, 9):
        profile[t] = 1.0
    for t in range(16, num + 1):
        profile[t] = _FALLEN_ROLL
    states = _build_states(objs,
                           num, {
                               "green": 12,
                               "target": 20
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           },
                           roll_profiles={"blue1": profile},
                           held_spans={"blue1": (3, 8)})
    step_options = _options([("Pick", ("robot", "blue1"), 0, 2),
                             ("Place", ("robot", ), 3, 8),
                             ("Push", ("robot", "green"), 9, 13)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_placement_wobble_ignored():
    """A wobble that never reaches the fallen threshold is no event."""
    objs = _make_objects(["green", "blue1", "target"])
    num = 30
    profile = [0.0] * (num + 1)
    for t in range(4, 7):
        profile[t] = 0.12  # ~7 deg: tilting, not fallen
    states = _build_states(objs,
                           num, {
                               "green": 12,
                               "target": 18
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.9, 1.0),
                               "target": (0.7, 1.098)
                           },
                           roll_profiles={"blue1": profile})
    step_options = _options([("Push", ("robot", "green"), 9, 13)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_tie_onsets_allowed():
    """Green and its neighbor starting to fall on the same step pass."""
    objs = _make_objects(["green", "target"])
    states = _build_states(objs,
                           20, {
                               "green": 5,
                               "target": 5
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098)
                           })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 20)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_same_step_tie_attributes_through_knocker():
    """Two non-green blocks sharing an onset step certify.

    Regression for run_20260713_172854 seed0 task0: a tight staircase
    crosses two hops within one recorded step, so knocker (domino_2)
    and victim (domino_1) read the same onset - the old single-pass
    geometric attribution falsely rejected the victim. The timing rule
    has no such ordering problem: both onsets sit one window from the
    green's.
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    states = _build_states(
        objs,
        30,
        {
            "green": 5,
            "blue2": 9,  # knocker: solidly inside green's corridor
            "blue1": 9,  # victim: only inside blue2's corridor
            "target": 13
        },
        positions={
            "green": (0.7, 1.0),
            "blue2": (0.7, 1.098),
            # 0.26 from green (clearance ~0.07, outside tolerance) but
            # 0.162 from blue2 (inside its 0.18 m reach).
            "blue1": (0.7, 1.26),
            "target": (0.7, 1.358)
        })
    step_options = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_green_toppled_outside_push_probe_decides():
    """Green falling long after its Push (e.g. swept by a later Place).

    There are no onset-in-span forensics anymore - once the goal holds,
    the probe replays the recorded push from the pre-push scene and
    decides on the layout.
    """
    objs = _make_objects(["green", "target"])
    green_onset = 40
    num = green_onset + 15
    states = _build_states(objs,
                           num, {
                               "green": green_onset,
                               "target": green_onset + 4
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "target": (0.7, 1.098)
                           })
    step_options = _options([("Push", ("robot", "green"), 0, 5),
                             ("Place", ("robot", ), 6, num - 1)], num)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason  # goal unreached: no bonus at stake
    probe = _FakeProbe(ok=False, detail="the push does not topple the green")
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=probe)
    assert not ok
    assert "the push does not topple the green" in reason
    # The probe replays from the recorded push, not the later sweep.
    assert probe.calls[0][0] is states[0]


def test_no_topples_passes():
    """Nothing fell: nothing to certify."""
    objs = _make_objects(["green", "target"])
    states = _build_states(objs, 10, {})
    ok, reason = check_cascade_legitimacy(states, _goal(objs), None)
    assert ok, reason


def test_onsets_during_wait_after_push_pass():
    """Cascade unfolding during Wait steps after the push is the normal case
    and must pass the action rules."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           40, {
                               "green": 5,
                               "blue1": 20,
                               "target": 24
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    # Push ends at action 6; every later onset happens during Wait.
    step_options = _options([("Push", ("robot", "green"), 0, 6)], 40)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_restricted_push_counts_as_green_push():
    """The restricted Push variant grounds only the robot; it always targets
    the inferred start block, so it must satisfy the action rules."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "target": 14
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    step_options = _options([("Push", ("robot", ), 0, 7)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason


def test_domino_evaluator_certify():
    """DominoEvaluator._certify applies the cascade rules to its own goal (the
    evaluator seam consumed by BaseEnv.check_episode_trajectory /
    BaseEnv.evaluate_episode)."""
    # Local import: pulls in PyBullet, which the rest of this file avoids.
    from predicators.envs.pybullet_domino.env import \
        DominoEvaluator  # pylint: disable=import-outside-toplevel
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "target": 14
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    evaluator = DominoEvaluator(_goal(objs))
    honest = _options([("Push", ("robot", "green"), 0, 7)], 30)
    ok, reason = evaluator._certify(states, honest)  # pylint: disable=protected-access
    assert ok, reason
    hacked = _options([("Push", ("robot", "blue1"), 0, 7)], 30)
    ok, reason = evaluator._certify(states, hacked)  # pylint: disable=protected-access
    assert not ok
    assert "green" in reason


def test_domino_evaluator_reward_decomposition():
    """The DominoEvaluator's reward is the certified-success bonus minus the
    per-block cost; termination is purely physical (an illegitimate topple
    still terminates), and no oracle K* lives on the evaluator."""
    # Local import: pulls in PyBullet, which the rest of this file avoids.
    from predicators.envs.pybullet_domino.env import \
        DominoEvaluator  # pylint: disable=import-outside-toplevel
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    from predicators.utils import \
        reset_config  # pylint: disable=import-outside-toplevel
    reset_config({"domino_block_cost": 0.05, "domino_min_block_num_blues": 4})

    # Toppled with a real classifier so terminated() reflects the states.
    toppled = Predicate(
        "Toppled", [_DOMINO_TYPE],
        lambda s, o: abs(float(s.get(o[0], "roll"))) >= abs(_FALLEN_ROLL))
    objs = _make_objects(["green", "blue1", "target"])
    goal = {GroundAtom(toppled, [objs["target"]])}
    states = _build_states(objs,
                           30, {
                               "green": 5,
                               "blue1": 10,
                               "target": 14
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    # The pure block count sees exactly one consumed BLUE (blue1
    # toppled; green also fell but is filtered by color; target is
    # purple).
    assert count_movable_blocks_used(states) == 1
    evaluator = DominoEvaluator(goal)
    honest = _options([("Push", ("robot", "green"), 0, 7)], 30)
    hacked = _options([("Push", ("robot", "blue1"), 0, 7)], 30)

    class _ProbeEnv:
        """Quacks like a domino env whose probe always certifies."""

        def run_counterfactual_cascade_probe(self,
                                             pre_push_state,
                                             greens,
                                             goal_atoms,
                                             push_params=None):
            """Always certify."""
            del pre_push_state, greens, goal_atoms, push_params
            return True, "fake probe: cascades"

    # Legitimate topple: bonus minus one block's cost.
    assert evaluator.terminated(states[-1])
    assert abs(
        evaluator.reward(states, honest, sim_env=_ProbeEnv()) -
        (1.0 - CFG.domino_block_cost)) < 1e-9
    # Illegitimate topple (Push on a blue): still terminated, bonus
    # gated by the only-green label rule, cost still paid.
    assert evaluator.terminated(states[-1])
    assert abs(
        evaluator.reward(states, hacked, sim_env=_ProbeEnv()) -
        (-CFG.domino_block_cost)) < 1e-9
    # The evaluator ships on the agent-facing Task, so no oracle
    # quantity may live on it: K* travels env-side via
    # EnvironmentTask.offline_task_metrics instead.
    assert evaluator.offline_metrics(states, honest) == {"k_used": 1.0}
    assert not hasattr(evaluator, "k_star")
    # The stated objective is public and K*-free, and states the exact
    # reward arithmetic (agents have induced false reward rules from
    # opaque scores when only the qualitative form was given).
    description = evaluator.objective_description()
    assert str(CFG.domino_block_cost) in description
    assert "reward = (1.0 if certified success else 0.0)" in description
    assert "K*" not in description


def test_domino_evaluator_budget_assert():
    """A scene whose staged movables could out-cost the success bonus is a
    config error, caught at construction against the scene's actual count
    (``num_movables``); omitting it falls back to the min-block budget flag."""
    # Local import: pulls in PyBullet, which the rest of this file avoids.
    from predicators.envs.pybullet_domino.env import \
        DominoEvaluator  # pylint: disable=import-outside-toplevel
    from predicators.utils import \
        reset_config  # pylint: disable=import-outside-toplevel
    reset_config({"domino_block_cost": 0.05, "domino_min_block_num_blues": 4})
    goal = _goal(_make_objects(["green", "target"]))
    DominoEvaluator(goal)  # flag default: 0.05 * 4 < 1
    DominoEvaluator(goal, num_movables=19)  # 0.05 * 19 < 1
    with pytest.raises(AssertionError):
        DominoEvaluator(goal, num_movables=20)  # 0.05 * 20 == 1


def _real_goal(objs: Dict[str, Object]) -> set:
    """A goal that actually holds once the target's roll reads fallen."""
    return {GroundAtom(_TOPPLED_REAL, [objs["target"]])}


class _FakeProbe:
    """Recording fake for the counterfactual push probe."""

    def __init__(self, ok: bool, detail: str = "fake detail") -> None:
        self.ok = ok
        self.detail = detail
        self.calls: List[Tuple[State, Tuple[str, ...], frozenset,
                               Optional[Tuple[float, ...]]]] = []

    def __call__(self, pre_push_state, greens, goal, push_params):
        self.calls.append((pre_push_state, tuple(g.name for g in greens),
                           frozenset(goal), push_params))
        return self.ok, self.detail


def _legit_chain_case(push_params: Optional[Tuple[float, ...]] = None):
    """A rule-abiding, goal-reaching chain: the probe decides its fate.

    ``push_params`` stamps the Push labels with continuous parameters
    (3-tuple labels); None leaves legacy 2-tuple labels.
    """
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 12,
                               "blue1": 16,
                               "target": 20
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.7, 1.098),
                               "target": (0.7, 1.196)
                           })
    step_options = _options([("Push", ("robot", "green"), 10, 13)], 30)
    if push_params is not None:
        step_options = [
            label + (push_params, )
            if label is not None and label[0] == "Push" else label
            for label in step_options
        ]
    return objs, states, step_options


def test_probe_accepts_goal_reaching_episode():
    """Rule (c): a passing counterfactual probe certifies the episode, and the
    probe receives the recorded pre-push state and the pushed green."""
    objs, states, step_options = _legit_chain_case()
    probe = _FakeProbe(ok=True)
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=probe)
    assert ok, reason
    assert len(probe.calls) == 1
    pre_push_state, greens, goal, push_params = probe.calls[0]
    # The pre-push state is the one the first Push action (index 10)
    # acts on: everything still upright.
    assert pre_push_state is states[10]
    assert abs(pre_push_state.get(objs["green"], "roll")) < 1e-6
    assert greens == ("green", )
    assert goal == frozenset(_real_goal(objs))
    # Legacy 2-tuple labels carry no parameters.
    assert push_params is None


def test_probe_receives_plan_push_params():
    """Rule (c): 3-tuple labels hand the episode's own Push continuous
    parameters to the probe, so it replays the plan's push."""
    objs, states, step_options = _legit_chain_case(push_params=(0.08, 0.05))
    probe = _FakeProbe(ok=True)
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=probe)
    assert ok, reason
    assert probe.calls[0][3] == (0.08, 0.05)


def test_probe_rejects_goal_reaching_episode():
    """Rule (c): a failing counterfactual probe voids the success even though.

    every pure rule passes - the layout, not the arm, must do the work.
    """
    objs, states, step_options = _legit_chain_case()
    probe = _FakeProbe(ok=False, detail="no stroke reaches the goal")
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=probe)
    assert not ok
    assert "counterfactual push" in reason
    assert "no stroke reaches the goal" in reason
    assert "robot's body" in reason


def test_probe_skipped_when_goal_not_reached():
    """Rule (c) only gates goal-reaching episodes: with no success bonus at
    stake the probe must not run (it costs a physics rollout)."""
    objs, states, step_options = _legit_chain_case()
    probe = _FakeProbe(ok=False)
    # _TOPPLED never holds, so the goal is unreached at the final state.
    ok, reason = check_cascade_legitimacy(states,
                                          _goal(objs),
                                          step_options,
                                          probe=probe)
    assert ok, reason
    assert not probe.calls


def test_goal_reached_without_probe_fails_closed():
    """A goal-reaching episode with no probe available must not certify:

    with the forensic rules gone, an unverifiable success cannot score.
    """
    objs, states, step_options = _legit_chain_case()
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=None)
    assert not ok
    assert "no counterfactual push probe is available" in reason


def test_pre_tilted_non_movable_fails():
    """A non-movable left leaning (past the tilting band, short of fallen) at
    the push is a disturbed scene: the probe would inherit a half-fallen
    layout, so staging integrity rejects it before the probe runs."""
    objs = _make_objects(["green", "blue1", "target", "target2"])
    num = 30
    positions = {
        "green": (0.7, 1.0),
        "blue1": (0.7, 1.098),
        "target": (0.7, 1.196),
        "target2": (0.9, 1.196),
    }
    states = _build_states(
        objs,
        num,
        {
            "green": 12,
            "blue1": 16,
            "target": 20
        },
        positions=positions,
        # target2 never topples but leans 0.12 rad (~6.9 deg, past the
        # 5 deg tilting threshold) the whole episode - e.g. nudged
        # against a staged blue during staging.
        roll_profiles={"target2": [0.12] * (num + 1)})
    step_options = _options([("Push", ("robot", "green"), 10, 13)], num)
    probe = _FakeProbe(ok=True)
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          step_options,
                                          probe=probe)
    assert not ok
    assert "target2" in reason and "leaning" in reason
    assert not probe.calls


def test_probe_without_labels_uses_pre_onset_state():
    """Without option labels the probe still runs, anchored to the state just
    before the first topple onset, and pushes every green."""
    objs, states, _ = _legit_chain_case()
    probe = _FakeProbe(ok=True)
    ok, reason = check_cascade_legitimacy(states,
                                          _real_goal(objs),
                                          None,
                                          probe=probe)
    assert ok, reason
    assert len(probe.calls) == 1
    pre_push_state, greens, _, push_params = probe.calls[0]
    # First onset is the green's at state 12; the anchor is one earlier.
    assert pre_push_state is states[11]
    assert greens == ("green", )
    assert push_params is None


def test_push_on_target_rejected_by_label():
    """A Push explicitly naming the target is rejected outright, whatever the
    kinematics say."""
    objs = _make_objects(["green", "blue1", "target"])
    states = _build_states(objs,
                           30, {
                               "green": 12,
                               "target": 20
                           },
                           positions={
                               "green": (0.7, 1.0),
                               "blue1": (0.9, 1.3),
                               "target": (0.7, 1.098)
                           })
    step_options = _options([("Push", ("robot", "green"), 10, 13),
                             ("Push", ("robot", "target"), 17, 19)], 30)
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert not ok
    assert "only the green start block may be pushed" in reason


def test_domino_evaluator_binds_probe_from_sim_env():
    """DominoEvaluator._certify binds the counterfactual probe off the passed
    sim_env (never storing it), and memoizes the verdict per trajectory."""
    # Local import: pulls in PyBullet, which the rest of this file avoids.
    from predicators.envs.pybullet_domino.env import \
        DominoEvaluator  # pylint: disable=import-outside-toplevel

    class _FakeEnv:
        """Quacks like a domino env for the probe binding."""

        def __init__(self) -> None:
            self.probe = _FakeProbe(ok=False, detail="fake env says no")

        def run_counterfactual_cascade_probe(self,
                                             pre_push_state,
                                             greens,
                                             goal,
                                             push_params=None):
            """Delegate to the fake probe."""
            return self.probe(pre_push_state, greens, goal, push_params)

    objs, states, step_options = _legit_chain_case()
    evaluator = DominoEvaluator(_real_goal(objs))
    fake_env = _FakeEnv()
    ok, reason = evaluator._certify(  # pylint: disable=protected-access
        states,
        step_options,
        sim_env=fake_env)
    assert not ok
    assert "fake env says no" in reason
    # Same (states, labels, sim_env): memoized, the probe is not re-run.
    ok2, _ = evaluator._certify(  # pylint: disable=protected-access
        states,
        step_options,
        sim_env=fake_env)
    assert not ok2
    assert len(fake_env.probe.calls) == 1
    # No sim_env: a goal-reaching episode fails closed - with the
    # forensic rules gone an unverifiable success must not score.
    ok3, reason3 = evaluator._certify(  # pylint: disable=protected-access
        states, step_options)
    assert not ok3
    assert "no counterfactual push probe is available" in reason3
    # Leak-freedom: the transient env was never stored on the evaluator.
    assert all(v is not fake_env for v in vars(evaluator).values())


# --- Topple debounce ------------------------------------------------
#
# A single state past fallen_threshold is not a topple. The state right
# after is_held clears catches the domino mid-release, still settling,
# several degrees off plumb; reading that as a fall dated a real run's
# cascade to the grasp that preceded it by ~100 steps. See
# _TOPPLE_MIN_STEPS.

# Past fallen_threshold, like _FALLEN_ROLL, but used for transients that
# must NOT register.
_SPIKE_ROLL = -0.35


def test_release_transient_is_not_a_topple():
    """One state past the threshold as a placed domino settles: not a fall."""
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    # Held 3-8; state 9 (the first non-held one) catches it mid-release.
    rolls = [0.0] * 9 + [_SPIKE_ROLL] + [0.0] * 21
    states = _build_states(objs,
                           30, {},
                           roll_profiles={"blue1": rolls},
                           held_spans={"blue1": (3, 8)})
    assert _topple_onset(states, objs["blue1"]) is None


def test_release_transient_of_two_states_is_not_a_topple():
    """A transient just under _TOPPLE_MIN_STEPS still does not register."""
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    n = _TOPPLE_MIN_STEPS - 1
    rolls = [0.0] * 9 + [_SPIKE_ROLL] * n + [0.0] * (22 - n)
    states = _build_states(objs,
                           30, {},
                           roll_profiles={"blue1": rolls},
                           held_spans={"blue1": (3, 8)})
    assert _topple_onset(states, objs["blue1"]) is None


def test_sustained_fall_after_release_registers_at_its_first_state():
    """Dropped over rather than set down: the run persists, so it counts."""
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    rolls = [0.0] * 9 + [_FALLEN_ROLL] * 22
    states = _build_states(objs,
                           30, {},
                           roll_profiles={"blue1": rolls},
                           held_spans={"blue1": (3, 8)})
    # Onset is the first state of the run, not the state that completed it.
    assert _topple_onset(states, objs["blue1"]) == 9


def test_topple_reaching_the_end_of_the_episode_registers():
    """A run too short to persist, but ending the episode, still counts.

    Dropping it would empty ``onsets`` and pass the episode unexamined -
    a false accept, which is worse than the false reject being fixed.
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    for run in range(1, _TOPPLE_MIN_STEPS):
        rolls = [0.0] * (31 - run) + [_FALLEN_ROLL] * run
        states = _build_states(objs, 30, {}, roll_profiles={"blue1": rolls})
        assert _topple_onset(states, objs["blue1"]) == 31 - run, \
            f"run of {run} reaching the end should register"


def test_a_carry_breaks_the_run():
    """Tilt in the gripper never accumulates, however long the carry."""
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    # Swinging past the threshold the whole way, but held throughout.
    rolls = [0.0] * 3 + [_FALLEN_ROLL] * 20 + [0.0] * 8
    states = _build_states(objs,
                           30, {},
                           roll_profiles={"blue1": rolls},
                           held_spans={"blue1": (3, 22)})
    assert _topple_onset(states, objs["blue1"]) is None


def test_place_transient_before_the_push_does_not_reject_the_run():
    """Regression: the 20260813 real run, rejected by a release transient.

    Pick and place a blue, whose first non-held state catches it
    settling; push green much later; the cascade then runs. Before the
    debounce the transient was a topple predating the push, so the
    episode was rejected with the fall dated to the grasp.
    """
    objs = _make_objects(["green", "blue1", "blue2", "target"])
    # blue1: held 3-8, one settling state at 9, upright until the cascade
    # reaches it at 30.
    blue1_rolls = ([0.0] * 9 + [_SPIKE_ROLL] + [0.0] * 20 +
                   [_FALLEN_ROLL] * 11)
    states = _build_states(objs,
                           40, {
                               "green": 25,
                               "blue2": 33,
                               "target": 36
                           },
                           roll_profiles={"blue1": blue1_rolls},
                           held_spans={"blue1": (3, 8)})
    step_options = _options([("Pick", ("robot", "blue1"), 0, 3),
                             ("Place", ("robot", ), 4, 8),
                             ("Push", ("robot", "green"), 20, 24)], 40)
    # The transient is not a topple, so blue1's fall is the cascade.
    assert _topple_onset(states, objs["blue1"]) == 30
    ok, reason = check_cascade_legitimacy(states, _goal(objs), step_options)
    assert ok, reason
