"""Unit tests for predicators.pybullet_helpers.real_robot_bridge.

Two halves, deliberately split by what they need:

* The optionality tests (this module imports with babyrobot absent, and the
  factory raises one clear error naming the submodule only when called) must
  run **everywhere**, including on CI, where the private submodule is not
  checked out. They must never skip -- they exist to catch exactly the
  regression that would break a submodule-less checkout.
* The segment tests construct ``babyrobot.realrobot.messages.Segment``, which is
  the shared contract type, so they ``importorskip("babyrobot")``.
"""
# Deferred imports are the subject of this file, not an oversight: babyrobot is
# absent on CI, and the helpers under test must be reachable without it.
# pylint: disable=import-outside-toplevel,import-error
import ast
import builtins
import sys

import numpy as np
import pytest

from predicators import utils
from predicators.pybullet_helpers.real_robot_bridge import _RELEASE_EPS, \
    GripperJointLayout, MissingBabyRobotError, _make_perception, \
    _split_actions, make_real_robot
from predicators.settings import CFG
from predicators.structs import Action

# The Franka layout: 7 arm joints then the 2 finger joints. The waypoint width
# is not cosmetic -- babyrobot's Segment rejects anything but 7 joints.
_N_ARM = 7
_LAYOUT = GripperJointLayout(left_finger_joint_idx=7,
                             right_finger_joint_idx=8,
                             open_fingers=0.04,
                             closed_fingers=0.0)


def _action(arm_value, fingers):
    """A joint-target action: 7 identical arm joints plus both finger joints.

    One value per action keeps the expected waypoints readable.
    """
    arm = [arm_value] * _N_ARM
    return Action(np.array([*arm, fingers, fingers], dtype=np.float32))


# -- optionality: these must run with or without the submodule ---------------


def test_no_module_level_babyrobot_import():
    """babyrobot is imported only inside function bodies, so this module -- and
    everything that imports it, up to `predicators.envs` -- loads on a checkout
    that cannot clone the private submodule.

    Checked on the parse tree rather than by reloading, so it holds
    whether or not babyrobot happens to be installed here.
    """
    from predicators.pybullet_helpers import real_robot_bridge as mod
    with open(mod.__file__, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:  # top level only; nested imports are the point
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("babyrobot"), \
                f"babyrobot imported at module level: {name}"


def test_make_real_robot_raises_naming_the_submodule(monkeypatch):
    """With babyrobot unimportable, the factory raises ONE clear error naming
    the submodule and the install command -- and only when actually called."""
    real_import = builtins.__import__

    def _no_babyrobot(name, *args, **kwargs):
        if name.startswith("babyrobot"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_babyrobot)
    for mod_name in list(sys.modules):
        if mod_name.startswith("babyrobot"):
            monkeypatch.delitem(sys.modules, mod_name)

    with pytest.raises(MissingBabyRobotError) as excinfo:
        make_real_robot()
    msg = str(excinfo.value)
    assert "submodules/BabyRobotPredicator" in msg
    assert "git submodule update --init" in msg
    assert "pip install -e" in msg


def test_gripper_joint_layout_finger_idxs():
    """The layout exposes the finger entries that arm waypoints drop."""
    assert _LAYOUT.finger_joint_idxs == (7, 8)


def test_no_cameras_survives_the_command_line():
    """A launcher config asking for no cameras has to actually get them off.

    ``utils.string_to_python_object`` maps "none" to Python ``None`` on
    the way in from the command line, so a config's ``"none"`` never
    reaches here as a string. Setting the string directly (as the test
    below does) exercises a path no shipped config can take.
    """
    assert utils.string_to_python_object("none") is None
    utils.reset_config(
        {"real_robot_perception": utils.string_to_python_object("none")})
    assert _make_perception() is None


def test_perception_kinds_and_unknown_names():
    """"none" really means no cameras, and an unrecognised name fails loudly
    rather than silently leaving the robot blind."""
    utils.reset_config({"real_robot_perception": "none"})
    assert _make_perception() is None
    utils.reset_config({"real_robot_perception": "telepathy"})
    with pytest.raises(ValueError, match="unknown real_robot_perception"):
        _make_perception()
    utils.reset_config({"real_robot_perception": "none"})


def test_closed_loop_is_the_default():
    """The defaults look between options and use the live cameras -- i.e.
    running on hardware closes the loop without extra flags.

    Pinned because each is individually a knob someone might flip for a
    one-off and forget to restore.
    """
    utils.reset_config({})
    assert CFG.real_robot_observe_at_option_boundary is True
    assert CFG.real_robot_perception == "zed"


def test_live_perception_is_built_lazily_with_our_table_height():
    """The default source is the live ZED session, and it is handed OUR table
    height.

    babyrobot's own default differs from ``domino_real_table_z``, and
    perception and the base -> world transplant have to agree about
    where the table is; a silent mismatch there is a whole bench
    session. Also checks construction opens no cameras -- ``RealRobot``
    opens the session, so building one must stay free.
    """
    pytest.importorskip("babyrobot")
    from babyrobot.realrobot.perception import DominoPerception

    utils.reset_config({"domino_real_table_z": -0.041})
    perception = _make_perception()

    assert isinstance(perception, DominoPerception)
    assert perception._table_z == pytest.approx(-0.041)  # pylint: disable=protected-access
    assert perception._readers is None  # pylint: disable=protected-access


def test_scene_file_perception_replays_the_capture():
    """The cameraless stand-in reports the configured scene, which is what lets
    the closed loop be exercised at a desk."""
    pytest.importorskip("babyrobot")
    from babyrobot.realrobot.perception import FileDominoPerception

    utils.reset_config({"real_robot_perception": "scene_file"})
    assert isinstance(_make_perception(), FileDominoPerception)
    utils.reset_config({"real_robot_perception": "none"})


# -- _split_actions: needs the shared Segment type ---------------------------


def test_split_actions_emits_gripper_transitions_and_arm_moves():
    """A pick-and-place shape splits into open/move/close/move/open/move, with
    the finger joints dropped from every waypoint."""
    pytest.importorskip("babyrobot")
    actions = [
        _action(0.0, 0.04),  # open: approach
        _action(0.1, 0.04),
        _action(0.1, 0.0),  # close: grasp
        _action(0.2, 0.0),  # still closed: carry
        _action(0.2, 0.04),  # open: release
    ]
    segments = _split_actions(actions, _LAYOUT)

    assert [s.type for s in segments] == \
        ["gripper", "move", "gripper", "move", "gripper", "move"]
    assert [s.command for s in segments if s.type == "gripper"] == \
        ["open", "close", "open"]
    moves = [s for s in segments if s.type == "move"]
    # Consecutive same-gripper steps coalesce; the fingers are dropped, so each
    # waypoint is the 7 arm joints only.
    assert [len(m.waypoints) for m in moves] == [2, 2, 1]
    # approx: the actions are float32, the waypoints plain floats.
    assert moves[0].waypoints[0] == pytest.approx((0.0, ) * _N_ARM)
    assert moves[0].waypoints[1] == pytest.approx((0.1, ) * _N_ARM)
    assert all(len(wp) == _N_ARM for m in moves for wp in m.waypoints)


def _commands(widths, layout):
    """The gripper commands a finger-width sequence splits into, in order."""
    segments = _split_actions([_action(0.0, w) for w in widths], layout)
    return [s.command for s in segments if s.type == "gripper"]


# closed_fingers well above a firm grasp, as on the real domino scene.
_TIGHT_LAYOUT = GripperJointLayout(left_finger_joint_idx=7,
                                   right_finger_joint_idx=8,
                                   open_fingers=0.04,
                                   closed_fingers=0.02)


def test_split_actions_judges_a_release_against_the_grasp_width():
    """A release is a widening from where the grasp SETTLED, not a crossing of
    some absolute mark.

    Both absolute rules tried here were wrong. A nearest-value test kept
    the hand shut through the release AND the retreat, so the object was
    dropped from transport height. Anchoring at ``closed_fingers``
    instead only moved the problem: a firm grasp settles far tighter
    than ``closed_fingers`` -- a real Pick bottoms out at 0.0 against a
    closed of 0.02 -- so the whole release still sat under the mark.
    """
    pytest.importorskip("babyrobot")
    # grasp 0.005, release +0.01, clear, retreat holding, full open at height.
    widths = [0.04, 0.04, 0.005, 0.005, 0.015, 0.019, 0.019, 0.04]

    assert _commands(widths, _TIGHT_LAYOUT) == ["open", "close", "open"]


def test_split_actions_releases_before_the_retreat():
    """The release must be commanded while the arm is still at the drop pose.

    This is the property that matters on hardware: an ``open`` emitted
    after the retreat waypoints drops the object from transport height.
    """
    pytest.importorskip("babyrobot")
    widths = [0.04, 0.005, 0.005, 0.015, 0.019, 0.019, 0.019, 0.04]
    segments = _split_actions([_action(0.0, w) for w in widths], _TIGHT_LAYOUT)
    kinds = [(s.type, getattr(s, "command", None)) for s in segments]
    release = kinds.index(("gripper", "open"), 1)
    # Waypoints remain after the release -- those are the retreat.
    assert any(k[0] == "move" for k in kinds[release + 1:])


def test_split_actions_ignores_wobble_while_closing():
    """The closing motion is not monotonic, so a bare "any widening" test would
    report a release mid-grasp and drop the object on the spot."""
    pytest.importorskip("babyrobot")
    # 0.010 -> 0.01183 is a real +1.8mm wobble seen while closing.
    widths = [0.04, 0.010, 0.01183, 0.005, 0.0, 0.0122, 0.0122]

    assert _commands(widths, _TIGHT_LAYOUT) == ["open", "close", "open"]


def test_split_actions_can_grasp_again_after_releasing():
    """Holding open after a release must not prevent a genuine re-grasp."""
    pytest.importorskip("babyrobot")
    widths = [0.04, 0.005, 0.015, 0.019, 0.04, 0.004, 0.004]

    assert _commands(widths,
                     _TIGHT_LAYOUT) == ["open", "close", "open", "close"]


def test_release_epsilon_stays_below_the_skill_layers_open_step():
    """``_RELEASE_EPS`` is a local copy of a skill-layer quantity.

    ``skill_factories`` imports ``pybullet_helpers``, so the bridge
    cannot import it back without inverting the layering. Pin the two
    together here instead: a release opens by ``_RELEASE_OPEN_STEP``
    from the grasp width, so the epsilon has to sit below that, and
    above the few millimetres of wobble seen while closing.
    """
    from predicators.ground_truth_models.skill_factories.base import \
        _RELEASE_OPEN_STEP
    assert 0.002 < _RELEASE_EPS < _RELEASE_OPEN_STEP


def test_split_actions_keeps_a_tighter_than_closed_grasp_closed():
    """The widening test is one-sided, and has to be.

    A real Pick descends through finger values well BELOW
    closed_fingers, so a symmetric ``abs(v - closed) <= tol`` would read
    the whole carry as an open hand and never close the gripper.
    """
    pytest.importorskip("babyrobot")
    layout = GripperJointLayout(left_finger_joint_idx=7,
                                right_finger_joint_idx=8,
                                open_fingers=0.04,
                                closed_fingers=0.02)
    for tighter in (0.0, 0.005, 0.0125, 0.02):
        segments = _split_actions([_action(0.0, tighter)], layout)
        assert segments[0].command == "close", \
            f"finger value {tighter} should be a grasp, not a release"


def test_split_actions_is_stateless_across_calls():
    """Gripper tracking restarts every call, so a chunk that begins already
    holding an object re-emits its leading `close`.

    RealRobot deduplicates that command session-wide -- this test pins
    the split's half of that contract.
    """
    pytest.importorskip("babyrobot")
    closed = _action(0.0, 0.0)
    for _ in range(2):
        segments = _split_actions([closed], _LAYOUT)
        assert segments[0].type == "gripper"
        assert segments[0].command == "close"


def test_split_actions_empty_input():
    """No actions means nothing to ship."""
    pytest.importorskip("babyrobot")
    assert not _split_actions([], _LAYOUT)


def test_split_actions_matches_layout_read_off_a_robot():
    """`gripper_joint_layout_from_robot` reproduces the four numbers the split
    needs, so the env and the helpers cannot disagree about the layout."""
    pytest.importorskip("babyrobot")
    from predicators.pybullet_helpers.real_robot_bridge import \
        gripper_joint_layout_from_robot

    class _FakeRobot:
        left_finger_joint_idx = 7
        right_finger_joint_idx = 8
        open_fingers = 0.04
        closed_fingers = 0.0

    assert gripper_joint_layout_from_robot(_FakeRobot()) == _LAYOUT


class _RecordingRobot:
    """A stand-in arm that records StepRequests instead of moving."""

    def __init__(self, observations=()):
        self.requests = []
        self._observations = tuple(observations)

    def step(self, req):
        """Record the StepRequest and reply with the canned observations."""
        from babyrobot.realrobot.messages import StepReply
        self.requests.append(req)
        return StepReply(
            observations=self._observations if req.observe else ())


def test_execute_chunks_ships_one_chunk_without_observing():
    """Open-loop shipping: a single chunk goes out and no observation is
    requested, so the caller's state stays the sim's prediction."""
    pytest.importorskip("babyrobot")
    from predicators.pybullet_helpers.real_robot_bridge import execute_chunks

    robot = _RecordingRobot()
    actions = [_action(0.0, 0.04), _action(0.1, 0.0)]
    assert not execute_chunks(robot, [actions], _LAYOUT)

    assert len(robot.requests) == 1
    req = robot.requests[0]
    assert len(req.chunks) == 1
    assert req.observe is False
    assert [s.type for s in req.chunks[0]] == \
        ["gripper", "move", "gripper", "move"]

    # An empty buffer ships nothing at all.
    assert not execute_chunks(robot, [[]], _LAYOUT)
    assert len(robot.requests) == 1


def test_execute_chunks_segments_each_chunk_separately():
    """Per-option shipping: each chunk is segmented on its own and they go out
    in order, in one call, so the robot executes them back to back."""
    pytest.importorskip("babyrobot")
    from predicators.pybullet_helpers.real_robot_bridge import execute_chunks

    robot = _RecordingRobot()
    first = [_action(0.0, 0.04), _action(0.1, 0.0)]  # opens, then closes
    second = [_action(0.2, 0.0)]  # already closed
    execute_chunks(robot, [first, second], _LAYOUT)

    req = robot.requests[0]
    assert len(req.chunks) == 2
    assert [s.type for s in req.chunks[0]] == \
        ["gripper", "move", "gripper", "move"]
    # The second chunk re-emits its leading "close" because the split is
    # stateless per chunk; RealRobot drops the repeat session-wide, which is
    # what makes per-option shipping safe.
    assert [s.type for s in req.chunks[1]] == ["gripper", "move"]
    assert req.chunks[1][0].command == "close"


def test_execute_chunks_returns_one_observation_per_chunk():
    """Observing: the reply's observations come back to the caller in chunk
    order, which is what lets the wrapper sync the twin per option."""
    pytest.importorskip("babyrobot")
    from babyrobot.realrobot.observations.domino import DominoObservation

    from predicators.pybullet_helpers.real_robot_bridge import execute_chunks

    seen = (DominoObservation(stamp=1.0), DominoObservation(stamp=2.0))
    robot = _RecordingRobot(observations=seen)
    got = execute_chunks(robot, [[_action(0.0, 0.04)], [_action(0.1, 0.0)]],
                         _LAYOUT,
                         observe=True,
                         settle_s=0.25)

    assert robot.requests[0].observe is True
    assert robot.requests[0].settle_s == 0.25
    assert got == list(seen)


def test_execute_chunks_drops_empty_chunks():
    """An empty chunk is not shipped, so it cannot consume one of the
    observations the caller is about to line up against its chunks."""
    pytest.importorskip("babyrobot")
    from predicators.pybullet_helpers.real_robot_bridge import execute_chunks

    robot = _RecordingRobot()
    execute_chunks(robot, [[], [_action(0.0, 0.04)], []], _LAYOUT)
    assert len(robot.requests[0].chunks) == 1


def test_reset_arm_passes_joints_through():
    """`reset_arm` hands the requested home joints to the robot and returns
    what the arm reports."""
    pytest.importorskip("babyrobot")
    from predicators.pybullet_helpers.real_robot_bridge import reset_arm

    home = (0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7)

    class _HomingRobot:

        def __init__(self):
            self.requested = None

        def reset_arm(self, req):
            """Echo the requested joints back as the arm's new position."""
            from babyrobot.realrobot.messages import ResetArmReply
            self.requested = req.joints
            return ResetArmReply(joints=req.joints)

    robot = _HomingRobot()
    assert reset_arm(robot, home) == home
    assert robot.requested == home


def test_make_real_robot_dry_constructs_an_armless_robot():
    """A dry RealRobot builds no arm, so the whole real path is exercisable at
    a desk; `dry` / `has_perception` are readable off the instance."""
    pytest.importorskip("babyrobot")
    robot = make_real_robot(dry=True)
    try:
        assert robot.dry is True
        assert robot.has_perception is False
    finally:
        robot.close()


def test_zed_recorder_session_matches_what_the_recorder_calls():
    """Pin ``EpisodeRecorder``'s stub against the real class.

    The recorder's own tests drive a stub session, which is what keeps
    them hardware-free -- and is also how a stub silently drifts from
    the thing it stands for. This one asserts the four calls exist with
    the keywords the recorder passes, so a rename upstream fails here
    rather than on the bench. Skips without the submodule, like the
    other contract tests in this file.
    """
    import inspect

    pytest.importorskip("pose_estimation.record_zed_video")
    from pose_estimation.record_zed_video import ZedRecorderSession

    init = inspect.signature(ZedRecorderSession.__init__).parameters
    for name in ("serials", "resolution", "fps", "out_dir"):
        assert name in init, f"ZedRecorderSession lost the {name!r} argument"

    start = inspect.signature(ZedRecorderSession.start_take).parameters
    for name in ("stamp", "max_frames"):
        assert name in start, f"start_take lost the {name!r} argument"

    stop = inspect.signature(ZedRecorderSession.stop_take).parameters
    for name in ("export_mp4", "export_depth"):
        assert name in stop, f"stop_take lost the {name!r} argument"

    for name in ("open", "close"):
        assert callable(getattr(ZedRecorderSession, name, None)), \
            f"ZedRecorderSession lost {name}()"


def test_markerless_staging_matches_what_the_snapshot_rebuild_calls():
    """Pin the snapshot rebuild's use of the markerless staging.

    ``_default_runner`` calls ``run_stages`` with a
    ``MarkerlessCapture`` and resolves prompt boxes itself, because
    ``run_stages`` reads ``boxes`` and not ``boxes_json`` -- a
    distinction worth a test, since getting it wrong would silently fall
    back to the drag window on every episode instead of failing.
    """
    import inspect

    pytest.importorskip("babyrobot.scene.capture_markerless")
    from babyrobot.scene.capture_markerless import MarkerlessCapture, \
        load_boxes, resolve_python, run_stages

    stages = inspect.signature(run_stages).parameters
    assert list(stages) == ["python", "svo", "bundle", "config"]
    assert callable(resolve_python)
    assert callable(load_boxes)

    fields = MarkerlessCapture.__dataclass_fields__
    for name in ("camera", "boxes", "frames", "resolution", "table_z",
                 "z_mode", "viz"):
        assert name in fields, f"MarkerlessCapture lost {name!r}"
    # The reason the runner resolves the file itself.
    source = inspect.getsource(run_stages)
    assert "config.boxes" in source
    assert "config.boxes_json" not in source


def test_markerless_driver_takes_the_arguments_post_processing_passes():
    """Pin the batch driver's interface.

    Automatic post-processing launches ``run_markerless.sh <svo> <out>
    given`` with BOXES and SERIAL in the environment. That is a
    positional shell contract with no type checker behind it, so a
    rename upstream would otherwise surface as a background job that
    fails silently and a track that never appears.
    """
    import os

    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import _default_script
    script = _default_script()
    if not os.path.exists(script):
        pytest.skip("submodule not checked out")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    assert "<svo> <outdir> [box-source]" in text, \
        "run_markerless.sh's positional interface changed"
    for var in ("BOXES", "SERIAL", "MAX_FRAMES"):
        assert var in text, f"run_markerless.sh no longer reads {var}"


def test_the_driver_honours_the_trim_request_once_it_can():
    """TRIM=1 is set unconditionally, because a driver that predates.

    --trim-motion ignores it rather than failing -- which is what lets it be
    configured before the submodule has it.

    The cost of that tolerance is that a silent no-op looks exactly like
    a working one. This says which of the two is in front of us, so
    "trimming is on" is never assumed.
    """
    import os

    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import _default_script
    script = _default_script()
    if not os.path.exists(script):
        pytest.skip("submodule not checked out")
    with open(script, encoding="utf-8") as f:
        text = f.read()
    if "trim-motion" not in text:
        pytest.skip(
            "this submodule predates --trim-motion; TRIM is being set and "
            "ignored, so takes are NOT being trimmed")
    assert "TRIM" in text, "the driver has --trim-motion but reads no TRIM"


# The domino bench: fingers rest at the faces rather than closing through.
_STALLING_LAYOUT = GripperJointLayout(left_finger_joint_idx=7,
                                      right_finger_joint_idx=8,
                                      open_fingers=0.04,
                                      closed_fingers=0.01)


def test_split_actions_does_not_read_a_carried_object_as_a_release():
    """Regression, measured on a Pick in the twin.

    Grasp commands ``closed - 0.01`` = 0.00000, deliberately past the
    domino. The fingers STALL on it at 0.00658 -- they are force-capped
    so they rest at the faces instead of closing through -- and the
    carry phases that follow command ``achieved - 1mm`` = 0.00558. That
    is wider than the grasp COMMAND, which is all this splitter sees, so
    a 5mm release epsilon called it a release: every Pick shipped close,
    open, close and the hand visibly opened around the domino it had
    just taken.
    """
    pytest.importorskip("babyrobot")
    widths = [0.04, 0.04, 0.0, 0.00558, 0.00558, 0.00611, 0.00612]

    assert _commands(widths, _STALLING_LAYOUT) == ["open", "close"]


def test_split_actions_still_sees_a_real_release_on_that_bench():
    """The other half of the same margin: a genuine release measures
    0.0122 on this scale and must still register, or the object is
    carried away instead of being put down."""
    pytest.importorskip("babyrobot")
    widths = [0.04, 0.0, 0.00558, 0.0122, 0.0122]

    assert _commands(widths, _STALLING_LAYOUT) == ["open", "close", "open"]


def test_release_eps_sits_between_the_two_measured_widths():
    """The constant is fitted, not derived, so pin what it was fitted to.

    A carry command of 0.00558 must not clear it and a release of
    0.0122 must. If someone retunes the grasp depth or the finger
    force, these are the two numbers to re-measure.
    """
    assert 0.00558 <= _RELEASE_EPS < 0.0122
