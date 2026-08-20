"""Tests for the real-robot executor: chunking, twin re-sync, what the agent
sees.

These run against a **stub** env and a **stub** robot, and must never skip: a
suite that silently skipped without the private submodule would hide exactly
the regressions it exists to catch. That is asserted below.

The executor is deliberately reachable without babyrobot because it only ever
touches the robot through ``real_robot_bridge.execute_chunks`` / ``reset_arm``,
which these tests replace with recorders. Building the ``Segment`` objects
those helpers ship is babyrobot's contract, covered in
``test_real_robot_bridge.py``, which does skip.
"""
import ast
import inspect
import json
import os
import pathlib
from typing import Any, List, Optional, cast

import numpy as np
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.real_robot_bridge import GripperJointLayout
from predicators.pybullet_helpers.real_robot_executor import \
    OptionBoundaryBuffer, RealRobotExecutor, _dump_look, \
    _per_object_divergence
from predicators.pybullet_helpers.real_robot_executor import \
    _snapshot_perception as build_snapshot_perception
from predicators.pybullet_helpers.real_robot_executor import attach_real_robot
from predicators.pybullet_helpers.real_robot_recorder import EpisodeRecorder, \
    episode_stamp
from predicators.pybullet_helpers.real_robot_snapshot import \
    MarkerlessSnapshotPerception
from predicators.structs import Action, Object, ParameterizedOption, State, \
    Type

_LAYOUT = GripperJointLayout(left_finger_joint_idx=7,
                             right_finger_joint_idx=8,
                             open_fingers=0.04,
                             closed_fingers=0.0)

# A minimal object type with positions, so divergence has something to measure.
_BLOCK_TYPE = Type("block", ["x", "y", "z"])
_BLOCK = Object("block0", _BLOCK_TYPE)


def _state(x: float, joints: Optional[List[float]] = None) -> State:
    """A one-object state carrying joint positions, as the twin's does.

    A ``PyBulletState`` specifically: ``_set_state`` trusts
    ``simulator_state["joint_positions"]`` only when it is there, and
    falls back to IK -- which drops wrist roll -- when it is not.
    """
    return utils.PyBulletState(
        {_BLOCK: np.array([x, 0.0, 0.0])},
        simulator_state={
            "joint_positions":
            joints if joints is not None else [float(i) for i in range(9)]
        })


class _StubObservation:
    """Stands in for a babyrobot observation.

    Opaque to the executor: handed straight to the env's
    ``state_from_observation``.
    """

    def __init__(self, x: float) -> None:
        self.x = x


class _StubEnv:
    """A twin with no PyBullet: it records what was asked of it.

    Implements only what the executor touches -- note that is *not*
    ``reset``/``step``, since the env now calls the executor rather than
    the other way round. Passed where a ``PyBulletEnv`` is declared via
    one cast, which is what keeps these tests free of a physics client.
    """

    def __init__(self) -> None:
        self.synced: List[State] = []
        self._observation = _state(0.0)

    def get_observation(self) -> State:
        """The twin's current state."""
        return self._observation

    def set_observation(self, state: State) -> None:
        """Put the twin in a given state, as a reset would."""
        self._observation = state

    def sync_to_state(self, state: State) -> None:
        """Adopt ``state`` as the twin's world, as PyBullet's would."""
        self.synced.append(state)
        self._observation = state

    def gripper_joint_layout(self) -> GripperJointLayout:
        """The finger layout the splitter needs."""
        return _LAYOUT

    def state_from_observation(self, obs: Any, prev_state: State) -> State:
        """Move the block to wherever the observation says it is."""
        del prev_state
        return _state(obs.x)

    def task_from_observation(self, obs: Any, train_or_test: str) -> Any:
        """Name the observation and the split it was rebuilt for."""
        return (obs, train_or_test)


class _StubRobot:
    """A robot that records nothing and moves nothing."""

    def __init__(self, has_perception: bool = True) -> None:
        self.has_perception = has_perception
        self.dry = True


def _as_env(env: _StubEnv) -> PyBulletEnv:
    """The one cast these tests need.

    The executor declares a ``PyBulletEnv`` because that is where
    ``sync_to_state`` / ``gripper_joint_layout`` live. The stub provides
    them; what it does not provide is a physics client, which the
    executor never touches.
    """
    return cast(PyBulletEnv, env)


@pytest.fixture(autouse=True)
def _never_write_into_the_repo(tmp_path, monkeypatch):
    """Run every test in this module from a scratch directory.

    ``EpisodeRecorder``'s default track directory is the relative
    ``logs/zed_tracks``, and it rewrites ``tracks.json`` there whenever a
    take closes. A test that forgets ``track_dir`` therefore writes into
    the repository -- and if a real run is in flight, over ITS manifest,
    with stub paths that point at takes which never existed. That
    happened: the suite clobbered run_20260817_171402's manifest while
    it was waiting for post-processing, and the fit fell back to
    per-step scoring having been pointed at a take named by a stub.

    Isolating the working directory fixes every present and future test
    at once, which passing ``track_dir`` one call site at a time does
    not.
    """
    monkeypatch.chdir(tmp_path)


@pytest.fixture(name="recorder")
def recorder_fixture(monkeypatch):
    """Replace the two bridge helpers with recorders.

    Patched at the ``real_robot_executor`` module -- where the names
    were imported to -- so nothing reaches babyrobot.
    """

    class _Recorder:
        """Captures what was shipped and what came back."""

        def __init__(self):
            self.shipped = []
            self.observe_flags = []
            self.settle = []
            self.homed = []
            self.to_return = []

        def execute_chunks(self,
                           robot,
                           chunks,
                           layout,
                           observe=False,
                           settle_s=0.0):
            """Record one shipment; reply with the queued observations."""
            del robot, layout
            self.shipped.append([list(c) for c in chunks])
            self.observe_flags.append(observe)
            self.settle.append(settle_s)
            if not observe:
                return []
            return [self.to_return.pop(0) for _ in chunks if self.to_return]

        def reset_arm(self, robot, joints):
            """Record the homing request."""
            del robot
            self.homed.append(list(joints))
            return tuple(joints)

    rec = _Recorder()
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.execute_chunks",
        rec.execute_chunks)
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        rec.reset_arm)
    return rec


def _option(name="Pick"):
    """A grounded option, so actions can carry an option boundary."""
    param_opt = ParameterizedOption(name, [], Box(0, 1, (1, )),
                                    lambda s, m, o, p: Action(p),
                                    lambda s, m, o, p: True,
                                    lambda s, m, o, p: False)
    return param_opt.ground([], [0.5])


def _act(option: Any, terminal: bool = False) -> Action:
    """A joint-target action carrying ``option``.

    ``terminal`` decides whether this action ends its option, which is
    the boundary the executor ships on.
    """
    action = Action(np.zeros(9, dtype=np.float32))
    option.terminal = lambda _obs, _t=terminal: _t
    action.set_option(option)
    return action


def _executor(env, robot=None, **kwargs):
    """An executor over the stubs, with settings passed in, not configured."""
    return RealRobotExecutor(_as_env(env), robot or _StubRobot(), **{
        "settle_s": 0.25,
        **kwargs
    })


# -- optionality: this file must run without the private submodule -----------
def test_executor_has_no_module_level_babyrobot_import():
    """The executor reaches the robot only through the bridge helpers, so it
    imports with babyrobot absent -- and these tests must never skip."""
    source = inspect.getsourcefile(RealRobotExecutor)
    assert source is not None
    with open(source, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("babyrobot"), \
                f"babyrobot imported at module level: {name}"


def test_recorder_has_no_module_level_submodule_import():
    """Same contract for the recorder, and it needs its own test: the executor
    imports this module at module level, so a top-level import here would break
    a checkout without the submodule just as surely -- and the ZED recorder
    lives under ``pose_estimation``, not ``babyrobot``, so the name to look for
    is different."""
    source = inspect.getsourcefile(EpisodeRecorder)
    assert source is not None
    with open(source, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    private = ("babyrobot", "pose_estimation")
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith(private), \
                f"private submodule imported at module level: {name}"


# -- the buffer, on its own --------------------------------------------------
def test_buffer_returns_a_chunk_only_at_a_boundary():
    """Actions accumulate until the option ends; a partial option is not handed
    out, because the arm would execute half a skill."""
    buffer = OptionBoundaryBuffer()
    option = _option()

    assert buffer.add(_act(option), None) is None
    assert buffer.add(_act(option), None) is None
    chunk = buffer.add(_act(option, terminal=True), None)

    assert chunk is not None and len(chunk) == 3
    assert not buffer  # emptied by the handover


def test_buffer_does_not_disturb_a_stateful_terminal():
    """``Wait`` counts consecutive settled steps in its own memory, and its
    policy already consults it once per step.

    A boundary check that counted as well would insert a second sample
    per step, so ``Wait`` would call the scene settled in half the steps
    it really takes -- and on the arm, a look would be spent every time.
    """
    settle_steps = 3

    def _terminal(state: Any, memory: Any, objects: Any, params: Any) -> bool:
        del state, objects, params
        memory["count"] = memory.get("count", 0) + 1
        return cast(bool, memory["count"] >= settle_steps)

    param_opt = ParameterizedOption("Wait", [],
                                    Box(0, 1, (1, )),
                                    policy=lambda s, m, o, p: Action(p),
                                    initiable=lambda s, m, o, p: True,
                                    terminal=_terminal)
    option = param_opt.ground([], [0.5])
    buffer = OptionBoundaryBuffer()

    def _carry(opt: Any) -> Action:
        """An action carrying ``opt`` with its real ``terminal`` intact."""
        action = Action(np.zeros(9, dtype=np.float32))
        action.set_option(opt)
        return action

    for _ in range(2):
        option.terminal(None)  # the option policy's own call, which counts
        buffer.add(_carry(option), None)  # the executor's, which must not
    assert option.memory["count"] == 2

    # So the boundary is still the third settled step, not the second.
    assert option.terminal(None)


def test_buffer_ignores_actions_with_no_option():
    """An action carrying no option has no boundary to attribute it to, so it
    is never buffered and never shipped alone."""
    buffer = OptionBoundaryBuffer()
    assert buffer.add(Action(np.zeros(9, dtype=np.float32)), None) is None
    assert not buffer


def test_buffer_discard_reports_what_was_lost():
    """The count is what the caller warns with, so it has to be real."""
    buffer = OptionBoundaryBuffer()
    option = _option()
    buffer.add(_act(option), None)
    buffer.add(_act(option), None)

    assert buffer.discard() == 2
    assert buffer.discard() == 0


# -- construction ------------------------------------------------------------
def test_construction_names_a_missing_hook():
    """An env without the domain conversions is rejected by name, not by an
    AttributeError several hundred robot-moving actions later."""

    class _NoHooks(_StubEnv):  # pylint: disable=abstract-method
        """An env missing one required conversion."""
        state_from_observation = None  # type: ignore[assignment]

    with pytest.raises(TypeError) as exc:
        _executor(_NoHooks())
    assert "state_from_observation" in str(exc.value)
    # The hook that IS present is not named.
    assert "task_from_observation" not in str(exc.value)


def test_construction_rejects_observing_without_perception():
    """Asking to look at the scene with no cameras fails at construction,
    rather than raising inside the first option boundary."""
    with pytest.raises(ValueError) as exc:
        _executor(_StubEnv(), _StubRobot(has_perception=False))
    assert "perception" in str(exc.value)


def test_blind_run_needs_no_perception():
    """A blind open-loop run is supported, so a robot with no cameras is fine.

    -- provided nothing else asks to look, human resets included.
    """
    assert _executor(_StubEnv(),
                     _StubRobot(has_perception=False),
                     observe_at_boundaries=False,
                     human_reset=False) is not None


def test_human_reset_without_cameras_is_refused():
    """A human reset rebuilds the task from a look, so it cannot be honoured
    without perception.

    Better to say so at construction than to raise on the first task
    request, after the human has already been asked to stand by.
    """
    with pytest.raises(ValueError) as exc:
        _executor(_StubEnv(),
                  _StubRobot(has_perception=False),
                  observe_at_boundaries=False,
                  human_reset=True)
    assert "human reset" in str(exc.value)


# -- reset -------------------------------------------------------------------
def test_reset_homes_the_arm_to_the_twins_joints(recorder):
    """The arm is homed to the twin's home configuration with the finger joints
    dropped, because that is where the first option's waypoints start."""
    env = _StubEnv()
    env.set_observation(_state(0.0, joints=[float(i) for i in range(9)]))
    executor = _executor(env)

    executor.after_reset("test", 0, env.get_observation())

    # 9 joints in, 7 out: entries 7 and 8 are the fingers.
    assert recorder.homed == [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]


def test_both_splits_execute(recorder):
    """Real mode is a property of an executor being attached, not of the split,
    so an exploration episode drives the arm like an evaluation one."""
    env = _StubEnv()
    executor = _executor(env)

    executor.after_reset("train", 0, env.get_observation())
    executor.after_reset("test", 0, env.get_observation())

    assert len(recorder.homed) == 2


def test_reset_drops_a_partial_option_from_the_previous_episode(
        recorder, caplog):
    """An episode cut short leaves half an option buffered.

    Half a skill is not worth executing on the arm, so it is dropped --
    loudly, because motion the caller asked for is silently not
    happening.
    """
    env = _StubEnv()
    executor = _executor(env)
    option = _option()
    executor.after_step(_act(option), env.get_observation())
    executor.after_step(_act(option), env.get_observation())
    assert recorder.shipped == []  # nothing shipped mid-option

    with caplog.at_level("WARNING"):
        executor.after_reset("test", 0, env.get_observation())

    assert "dropping 2 buffered action" in caplog.text
    assert recorder.shipped == []  # and still nothing shipped


# -- chunking ----------------------------------------------------------------
def test_ships_once_per_option_boundary(recorder):
    """Each option's actions go out when that option ends -- one shipment per
    boundary, carrying only that option's actions."""
    env = _StubEnv()
    executor = _executor(env)
    recorder.to_return = [_StubObservation(1.0), _StubObservation(2.0)]

    first, second = _option("Pick"), _option("Place")
    executor.after_step(_act(first), env.get_observation())
    executor.after_step(_act(first, terminal=True), env.get_observation())
    executor.after_step(_act(second), env.get_observation())
    executor.after_step(_act(second, terminal=True), env.get_observation())

    assert len(recorder.shipped) == 2
    assert [len(c) for chunks in recorder.shipped for c in chunks] == [2, 2]
    assert recorder.observe_flags == [True, True]
    assert recorder.settle == [0.25, 0.25]


def test_actions_without_an_option_are_not_shipped(recorder):
    """An optionless action rolls the twin forward without reaching the arm,
    and the observation passes through untouched."""
    env = _StubEnv()
    executor = _executor(env)

    obs = env.get_observation()
    returned = executor.after_step(Action(np.zeros(9, dtype=np.float32)), obs)

    assert recorder.shipped == []
    assert returned is obs


# -- twin re-sync ------------------------------------------------------------
def test_boundary_observation_is_written_into_the_twin(recorder):
    """The perceived state reaches the twin.

    Without this the correction would survive exactly one step: the env
    advances its own physics from its own bodies and returns a state
    derived from them.
    """
    env = _StubEnv()
    executor = _executor(env)
    recorder.to_return = [_StubObservation(0.5)]

    executor.after_step(_act(_option(), terminal=True), env.get_observation())

    assert len(env.synced) == 1
    assert env.synced[0].get(_BLOCK, "x") == pytest.approx(0.5)


def test_the_caller_gets_the_twin_not_the_observation(recorder):
    """The library's observation type never escapes: what comes back is the
    twin's State, so CogMan never learns a robot exists."""
    env = _StubEnv()
    executor = _executor(env)
    recorder.to_return = [_StubObservation(0.5)]

    returned = executor.after_step(_act(_option(), terminal=True),
                                   env.get_observation())

    assert isinstance(returned, State)
    assert not isinstance(returned, _StubObservation)
    # And it is the CORRECTED twin, not the pre-sync prediction.
    assert returned.get(_BLOCK, "x") == pytest.approx(0.5)


def test_no_sync_when_not_observing(recorder):
    """A blind run executes motion and never looks, so the twin is never
    corrected."""
    env = _StubEnv()
    executor = _executor(env,
                         _StubRobot(has_perception=False),
                         observe_at_boundaries=False,
                         human_reset=False)

    executor.after_step(_act(_option(), terminal=True), env.get_observation())

    assert recorder.observe_flags == [False]
    assert not env.synced


def test_large_divergence_is_surfaced(recorder, caplog):
    """A scene that disagrees with the twin is reported rather than swallowed.

    ``_set_state``'s own reconstruction check cannot catch this: it
    measures whether PyBullet could realize the state it was asked to
    write, and a toppled domino is perfectly realizable.
    """
    env = _StubEnv()
    executor = _executor(env, divergence_atol=0.02)
    recorder.to_return = [_StubObservation(0.5)]  # twin says 0.0

    with caplog.at_level("WARNING"):
        executor.after_step(_act(_option(), terminal=True),
                            env.get_observation())

    assert executor.last_divergence == pytest.approx(0.5)
    assert "0.500" in caplog.text


def test_small_divergence_is_quiet(recorder, caplog):
    """Placement jitter and perception noise are below the tolerance and must
    not cry wolf on every option."""
    env = _StubEnv()
    executor = _executor(env, divergence_atol=0.02)
    recorder.to_return = [_StubObservation(0.001)]

    with caplog.at_level("WARNING"):
        executor.after_step(_act(_option(), terminal=True),
                            env.get_observation())

    assert executor.last_divergence == pytest.approx(0.001)
    assert "real robot:" not in caplog.text
    # Quiet, but still corrected.
    assert len(env.synced) == 1


# -- attachment --------------------------------------------------------------
def test_attach_is_a_noop_when_not_executing():
    """The call site reads as one unconditional line, so a dry run must leave
    the env with no executor at all."""
    utils.reset_config({"env": "cover", "real_robot_execute": False})
    env = _StubEnv()
    assert attach_real_robot(cast(Any, env)) is None


def test_attach_rejects_a_non_pybullet_env():
    """The twin is what turns an option into a joint trajectory, so a non-
    PyBullet env cannot be driven and says so."""
    utils.reset_config({"env": "cover", "real_robot_execute": True})
    with pytest.raises(TypeError) as exc:
        attach_real_robot(cast(Any, object()), _StubRobot())
    assert "PyBullet" in str(exc.value)


# -- per-look observability --------------------------------------------------

_OTHER = Object("block1", _BLOCK_TYPE)


def _two_object_state(x0: float, x1: float) -> State:
    """A two-object state, so a per-object breakdown has something to break
    down."""
    return State({
        _BLOCK: np.array([x0, 0.0, 0.0]),
        _OTHER: np.array([x1, 0.0, 0.0]),
    })


def test_per_object_divergence_names_the_object_and_orders_by_distance():
    """The max alone cannot tell a knocked object from a shared offset.

    ``_max_position_divergence`` answers "how bad", which is what the
    tolerance tests; this answers "which", which is what a human reads.
    """
    predicted = _two_object_state(0.0, 0.0)
    perceived = _two_object_state(0.01, 0.05)

    result = _per_object_divergence(predicted, perceived)

    assert [obj.name for obj, _ in result] == ["block1", "block0"]
    assert result[0][1] == pytest.approx(0.05)
    assert result[1][1] == pytest.approx(0.01)


def test_dump_look_writes_both_sides_of_the_comparison(tmp_path):
    """A dumped look records the prediction beside the perception.

    Recording only the divergence would leave a session unable to say
    WHERE things were, which is what makes a capture re-examinable
    offline.
    """
    utils.reset_config({
        "seed": 0,
        "real_robot_observation_dump_dir": str(tmp_path),
    })
    predicted = _two_object_state(0.0, 0.0)
    perceived = _two_object_state(0.01, 0.05)

    _dump_look(3, predicted, perceived,
               _per_object_divergence(predicted, perceived), 0.05)

    written = sorted(tmp_path.glob("*.json"))
    assert [f.name for f in written] == ["look_0003.json"]
    record = json.loads(written[0].read_text(encoding="utf-8"))
    assert record["look"] == 3
    assert record["worst_divergence"] == pytest.approx(0.05)
    assert record["predicted"]["block1"] == [0.0, 0.0, 0.0]
    assert record["perceived"]["block1"] == [0.05, 0.0, 0.0]
    assert record["per_object"]["block1"] == pytest.approx(0.05)


def test_dump_look_is_off_by_default(tmp_path):
    """Dumping is opt-in: an unset directory writes nothing at all."""
    utils.reset_config({"seed": 0, "real_robot_observation_dump_dir": ""})
    predicted = _two_object_state(0.0, 0.0)
    perceived = _two_object_state(0.01, 0.0)

    _dump_look(1, predicted, perceived,
               _per_object_divergence(predicted, perceived), 0.01)

    assert not list(tmp_path.iterdir())


# -- one arrangement, both splits --------------------------------------------
def test_one_reset_serves_both_splits(monkeypatch):
    """A physical reset arranges one scene, so both splits rebuild from it.

    Consuming the look on whichever split asked first left the other one
    holding the captured-scene task: with the online loop off, main.py
    requests the train tasks during setup, so the TEST task -- the one
    that gets solved -- silently stayed on the scene JSON while the arm
    executed in the real scene.
    """
    looks = []

    def _fake_reset_env(robot, joints):
        del robot, joints
        looks.append(len(looks))
        return _StubObservation(float(len(looks)))

    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_env",
        _fake_reset_env)
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        lambda robot, joints: tuple(joints))
    ex = _executor(_StubEnv(), human_reset=True)

    train = ex.tasks_for("train")
    test = ex.tasks_for("test")

    # One look, not two: the person arranged the scene once.
    assert len(looks) == 1
    assert ex.resets_done == 1
    # Both splits rebuilt from THAT observation, each for its own split.
    assert train is not None and test is not None
    assert train[0][0] is test[0][0]
    assert train[0][1] == "train" and test[0][1] == "test"


def test_a_second_arrangement_replaces_the_first(monkeypatch):
    """The next episode's reset supersedes the cached look.

    Otherwise every later episode would keep rebuilding from the first
    scene the person ever arranged.
    """
    seen = []

    def _fake_reset_env(robot, joints):
        del robot, joints
        seen.append(len(seen))
        return _StubObservation(float(len(seen)))

    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_env",
        _fake_reset_env)
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        lambda robot, joints: tuple(joints))
    ex = _executor(_StubEnv(), human_reset=True)

    first = ex.tasks_for("train")
    ex.after_reset("train", 0, _state(0.0))  # an episode began; one is owed
    second = ex.tasks_for("test")

    assert len(seen) == 2
    assert first[0][0] is not second[0][0]


def test_captured_scene_task_is_refused_while_the_cameras_are_live(
        monkeypatch):
    """Live cameras plus no look is a plan written for a world that is not
    there.

    Silent in every other way: the JSON's poses look like a scene and
    planning against them succeeds, so the run only reveals itself on
    the arm.
    """
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        lambda robot, joints: tuple(joints))
    utils.reset_config({
        "real_robot_perception": "zed",
        "real_robot_allow_captured_scene_task": False,
    })
    ex = _executor(_StubEnv(), human_reset=False)

    with pytest.raises(ValueError, match="no one has looked at the scene"):
        ex.tasks_for("test")


def test_replaying_a_recorded_plan_may_keep_the_captured_scene(monkeypatch):
    """The one case that wants those exact poses says so explicitly."""
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        lambda robot, joints: tuple(joints))
    utils.reset_config({
        "real_robot_perception": "zed",
        "real_robot_allow_captured_scene_task": True,
    })
    ex = _executor(_StubEnv(), human_reset=False)

    assert ex.tasks_for("test") is None


def test_no_cameras_means_nothing_to_be_stale_about(monkeypatch):
    """A cameraless run has no truer scene to compare against, so the captured
    one stands without complaint."""
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        lambda robot, joints: tuple(joints))
    utils.reset_config({"real_robot_perception": "scene_file"})
    ex = _executor(_StubEnv(), human_reset=False)

    assert ex.tasks_for("test") is None


# -- episode recording -------------------------------------------------------
class _StubSession:
    """Stands in for babyrobot's ZedRecorderSession.

    Records the lifecycle calls, so a test can assert the cameras are
    opened once for the run and a take is started and stopped per
    episode, with no SDK anywhere near it.
    """

    def __init__(self, errors=None, stop_raises=False):
        self.opens = 0
        self.closes = 0
        self.started = []
        self.stopped = []
        self.recording = False
        # The two the bench actually has, in the order the recorder is given
        # them -- so a test that does not pick a camera exercises the default.
        self.serials = ["32294776", "30264679"]
        self._errors = errors or []
        self._stop_raises = stop_raises

    def open(self):
        """Open the cameras."""
        self.opens += 1

    def start_take(self, stamp=None, max_frames=None):
        """Begin a take, returning its directory.

        Refuses a second concurrent take, as the real session does --
        that refusal is the collision a snapshot has to be sequenced
        around, so the stub has to be able to express it.
        """
        if self.recording:
            raise RuntimeError("already recording; call stop_take() first")
        self.started.append((stamp, max_frames))
        self.recording = True
        return "take_" + str(stamp)

    def stop_take(self, export_mp4=False, export_depth=False):
        """End a take, returning its meta.json contents."""
        self.stopped.append({"mp4": export_mp4, "depth": export_depth})
        self.recording = False
        if self._stop_raises:
            raise RuntimeError("grab thread for ZED 32294776 failed")
        return {
            "take_dir": "take_" + str(self.started[-1][0]),
            "errors": list(self._errors),
            "timestamp_clock": "SDK_DEFAULT",
            "sdk_version": "3.8.2",
            "svo_ext": ".svo2",
        }

    def close(self):
        """Release the cameras."""
        self.closes += 1


def _recording_executor(session, **kwargs):
    """An executor wired to a recorder over ``session``."""
    return _executor(_StubEnv(),
                     observe_at_boundaries=False,
                     human_reset=False,
                     recorder=EpisodeRecorder(session),
                     **kwargs)


@pytest.mark.usefixtures("recorder")
def test_cameras_open_once_for_the_run_not_once_per_episode():
    """A learning cycle is many episodes, and per-episode camera init and
    warmup would otherwise be paid every time."""
    session = _StubSession()
    ex = _recording_executor(session)

    for _ in range(3):
        ex.after_reset("train", 0, _state(0.0))
        ex.after_episode(True)

    assert session.opens == 1
    assert len(session.started) == 3
    assert len(session.stopped) == 3


@pytest.mark.usefixtures("recorder")
def test_open_loop_records_the_motion_not_the_simulating(tmp_path):
    """The take brackets the batch, not the episode.

    Under open-loop the arm does nothing between the reset and the ship,
    so recording from the reset captures the twin simulating -- a static
    scene, and on run_20260817_165815 the larger half of the take (258 s
    recorded against 153 s of motion). Trimming cannot recover it: the
    scan keeps everything between the first and last movement, and the
    arm homing at the reset opens that window.
    """
    session = _StubSession()
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   open_loop_episode=True,
                   recorder=EpisodeRecorder(session, track_dir=str(tmp_path)))

    ex.after_reset("train", 0, _state(0.0))
    assert not session.started, "recording started while the twin simulates"

    obs = _state(0.0)
    option = _option("Push")
    ex.after_step(_act(option, terminal=False), obs)
    ex.after_step(_act(option, terminal=True), obs)
    ex.after_episode(True)

    assert len(session.started) == 1, "the batch was not recorded"
    assert len(session.stopped) == 1
    assert session.started[0][0].endswith("_train0_ep001")


@pytest.mark.usefixtures("recorder")
def test_per_boundary_still_records_the_whole_episode(tmp_path):
    """Shipping option by option puts motion throughout the episode, so there
    the take must still span it."""
    session = _StubSession()
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   recorder=EpisodeRecorder(session, track_dir=str(tmp_path)))

    ex.after_reset("train", 0, _state(0.0))

    assert len(session.started) == 1, "the episode was not being recorded"


def test_nothing_is_left_recording_when_nothing_ships(recorder):
    """Shipping must NOT happen on an abnormal end, and no take may be left
    running afterwards -- one would record until the disk fills.

    Under open-loop the take is opened at the ship, so an episode that
    never ships never opens one; the invariant holds by never starting
    rather than by stopping. What must not happen is a take still
    recording at the end.
    """
    session = _StubSession()
    ex = _recording_executor(session, open_loop_episode=True)
    ex.after_reset("train", 0, _state(0.0))
    _run_episode(ex, ["Pick", "Place"], recorder, completed=False)

    assert recorder.shipped == [], "an incomplete episode ships nothing"
    assert session.recording is False, "a take was left running"
    assert len(session.started) == len(session.stopped)


@pytest.mark.usefixtures("recorder")
def test_a_take_opened_before_the_ship_is_stopped_when_nothing_ships():
    """The per-boundary path DOES open its take at the reset, so there the stop
    is what keeps the invariant."""
    session = _StubSession()
    ex = _recording_executor(session)
    ex.after_reset("train", 0, _state(0.0))
    ex.after_episode(False)

    assert len(session.started) == 1
    assert len(session.stopped) == 1
    assert session.recording is False


@pytest.mark.usefixtures("recorder")
def test_a_take_is_stopped_even_when_shipping_raises(monkeypatch):
    """Recording teardown belongs in a finally: the arm failing is not a reason
    to leave a camera recording."""
    session = _StubSession()

    def _boom(*args, **kwargs):
        raise RuntimeError("arm refused the batch")

    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.execute_chunks",
        _boom)
    ex = _recording_executor(session, open_loop_episode=True)
    ex.after_reset("train", 0, _state(0.0))
    obs = _state(0.0)
    option = _option("Pick")
    ex.after_step(_act(option, terminal=False), obs)
    ex.after_step(_act(option, terminal=True), obs)

    with pytest.raises(RuntimeError, match="arm refused"):
        ex.after_episode(True)

    assert len(session.stopped) == 1
    assert session.recording is False


@pytest.mark.usefixtures("recorder")
def test_a_take_reporting_camera_errors_is_marked_unusable():
    """A camera that dropped out mid-episode yields a short track that looks
    perfectly well formed, so the episode is marked rather than trusted."""
    session = _StubSession(errors=["ZED 30264679 stopped grabbing"])
    rec = EpisodeRecorder(session)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   recorder=rec)

    ex.after_reset("train", 0, _state(0.0))
    ex.after_episode(True)

    assert rec.takes == [("take_" + session.started[0][0], False)]


@pytest.mark.usefixtures("recorder")
def test_a_clean_take_is_marked_usable():
    """The other half of the same contract."""
    session = _StubSession()
    rec = EpisodeRecorder(session)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   recorder=rec)

    ex.after_reset("train", 0, _state(0.0))
    ex.after_episode(True)

    take_dir, usable = rec.takes[0]
    assert usable is True
    assert take_dir == rec.last_take_dir


@pytest.mark.usefixtures("recorder")
def test_a_failed_stop_does_not_take_the_run_down_with_it():
    """By the time a take is stopped the arm has already moved, so a recording
    problem must not destroy the run around it."""
    session = _StubSession(stop_raises=True)
    rec = EpisodeRecorder(session)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   recorder=rec)
    ex.after_reset("train", 0, _state(0.0))

    ex.after_episode(True)  # must not raise

    assert rec.takes == [("<failed>", False)]


@pytest.mark.usefixtures("recorder")
def test_exports_are_off_during_a_run():
    """stop_take can export depth inline, but that is the expensive offline
    work: doing it here would serialise post-processing into the episode loop
    and undo open-loop execution."""
    session = _StubSession()
    ex = _recording_executor(session)
    ex.after_reset("train", 0, _state(0.0))
    ex.after_episode(True)

    assert session.stopped == [{"mp4": False, "depth": False}]


@pytest.mark.usefixtures("recorder")
def test_a_take_left_open_is_closed_before_the_next_episode():
    """An episode that never finished must not have the next one appended to
    its recording."""
    session = _StubSession()
    ex = _recording_executor(session)

    ex.after_reset("train", 0, _state(0.0))
    # No after_episode: the episode ended some other way.
    ex.after_reset("train", 0, _state(0.0))

    assert len(session.stopped) == 1, "the orphaned take was closed"
    assert len(session.started) == 2


def test_recording_is_refused_alongside_a_live_zed_perception():
    """Both want the same cameras, and a ZED admits one owner."""
    utils.reset_config({
        "real_robot_execute": True,
        "real_robot_record_episodes": True,
        "real_robot_perception": "zed",
    })
    with pytest.raises(ValueError, match="one owner"):
        attach_real_robot(cast(Any, _StubEnv()))


def test_take_names_carry_the_episode_they_came_from():
    """A learning cycle revisits the same task index many times, so the stamp
    has to distinguish episodes -- and sort chronologically."""
    first = episode_stamp("train", 0, 1)
    second = episode_stamp("train", 0, 2)

    assert first.endswith("_train0_ep001")
    assert second.endswith("_train0_ep002")
    assert first < second


class _StubProcessor:
    """Stands in for the markerless pipeline, recording what was launched."""

    def __init__(self, started=True):
        self.launched = []
        self.waited = 0
        self.boxes = None
        self._started = started

    def set_boxes(self, boxes_json):
        """Record the boxes this run will reuse."""
        self.boxes = boxes_json

    def launch(self, svo, bundle, serial):
        """Record a launch; return a handle, or None if it could not start."""
        self.launched.append((svo, bundle, serial))
        return object() if self._started else None

    def pending(self):
        """Nothing is ever really running here."""
        return 0

    def wait_all(self, timeout=None):
        """Record the join."""
        del timeout
        self.waited += 1


def test_each_usable_take_is_post_processed(tmp_path):
    """The bridge between recording and scoring: predicators runs the pipeline
    itself rather than leaving takes for a human."""
    session = _StubSession()
    processor = _StubProcessor()
    rec = EpisodeRecorder(session,
                          processor=processor,
                          track_dir=str(tmp_path))
    rec.open()
    rec.start_episode("ep1")
    rec.stop_episode()

    assert len(processor.launched) == 1
    svo, bundle, _serial = processor.launched[0]
    assert svo.endswith(".svo2")
    assert str(tmp_path) in bundle


def test_an_unusable_take_is_not_post_processed(tmp_path):
    """A track fitted to a recording that lost a camera mid-episode is a well-
    formed track of the wrong thing."""
    session = _StubSession(errors=["ZED 30264679 stopped grabbing"])
    processor = _StubProcessor()
    rec = EpisodeRecorder(session,
                          processor=processor,
                          track_dir=str(tmp_path))
    rec.open()
    rec.start_episode("ep1")
    rec.stop_episode()

    assert not processor.launched


def test_the_manifest_records_every_episode_in_order(tmp_path):
    """Written as each take closes, so a run killed mid-way still leaves a
    valid document -- and unusable episodes are recorded, not omitted."""
    session = _StubSession()
    rec = EpisodeRecorder(session,
                          processor=_StubProcessor(),
                          track_dir=str(tmp_path))
    rec.open()
    for i in range(2):
        rec.start_episode(f"ep{i}")
        rec.stop_episode()

    manifest = json.loads(
        (tmp_path / "tracks.json").read_text(encoding="utf-8"))

    assert [e["episode"] for e in manifest["episodes"]] == [1, 2]
    assert all(e["usable"] for e in manifest["episodes"])
    assert all(e["track"].endswith("dominoes_traj.json")
               for e in manifest["episodes"])


def test_closing_waits_for_outstanding_post_processing(tmp_path):
    """The jobs are detached so they overlap the next episode, but the run must
    not exit while one is still writing a track."""
    processor = _StubProcessor()
    rec = EpisodeRecorder(_StubSession(),
                          processor=processor,
                          track_dir=str(tmp_path))
    rec.open()

    rec.close()

    assert processor.waited == 1


def test_takes_are_left_alone_when_processing_is_off(tmp_path):
    """Off means recorded and left for a human, which is the default."""
    rec = EpisodeRecorder(_StubSession(), track_dir=str(tmp_path))
    rec.open()
    rec.start_episode("ep1")
    rec.stop_episode()

    manifest = json.loads(
        (tmp_path / "tracks.json").read_text(encoding="utf-8"))
    assert "processing" not in manifest["episodes"][0]


def test_the_configured_camera_is_the_one_fitted(tmp_path):
    """Markerless is single-camera and the two are not interchangeable: one is
    6x better on orientation, the other tracks 18% more frames.

    The session's first serial is an arbitrary default, so it must be
    overridable.
    """
    session = _StubSession()
    session.serials = ["32294776", "30264679"]
    processor = _StubProcessor()
    rec = EpisodeRecorder(session,
                          processor=processor,
                          track_dir=str(tmp_path),
                          camera="30264679")
    rec.open()
    rec.start_episode("ep")
    rec.stop_episode()

    svo, _bundle, serial = processor.launched[0]
    assert serial == "30264679"
    assert svo.endswith("zed_30264679.svo2")


def test_a_camera_the_session_does_not_record_is_refused(tmp_path):
    """Every episode would otherwise fail at post-processing with a missing
    file, long after the run has cost something."""
    session = _StubSession()
    session.serials = ["32294776"]
    rec = EpisodeRecorder(session, track_dir=str(tmp_path), camera="99999999")

    with pytest.raises(ValueError, match="not one of the cameras"):
        _ = rec.fit_camera


def test_still_frames_are_trimmed_at_stage_1(tmp_path):
    """An episode take makes its own dead air: recording starts at the reset
    and the twin then simulates every option with the arm parked.

    Measured at 152 s of a 420 s take, which SAM-2 would otherwise
    process at ~0.5 s a frame.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import \
        MarkerlessTrackProcessor
    boxes = tmp_path / "boxes.json"
    boxes.write_text('{"boxes": [{"id": 0, "box": [1, 2, 3, 4]}]}',
                     encoding="utf-8")
    seen = {}

    def _launcher(argv, env, log_path):
        """Capture the environment the driver would be started with."""
        del argv, log_path
        seen.update(env)
        return _DoneJob()

    processor = MarkerlessTrackProcessor(script=str(boxes),
                                         boxes_json=str(boxes),
                                         trim=True,
                                         trim_args="--trim-pad 5",
                                         launcher=_launcher)
    processor.launch("take.svo2", str(tmp_path / "bundle"), "30264679")

    assert seen["TRIM"] == "1"
    assert seen["TRIM_ARGS"] == "--trim-pad 5"
    assert seen["SERIAL"] == "30264679"


def test_trimming_can_be_turned_off(tmp_path):
    """Off must leave the driver's environment clean rather than passing an
    empty TRIM, which the shell would read as set."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import \
        MarkerlessTrackProcessor
    boxes = tmp_path / "boxes.json"
    boxes.write_text('[[1, 2, 3, 4]]', encoding="utf-8")
    seen = {}

    def _launcher(argv, env, log_path):
        """Capture the environment the driver would be started with."""
        del argv, log_path
        seen.update(env)
        return _DoneJob()

    processor = MarkerlessTrackProcessor(script=str(boxes),
                                         boxes_json=str(boxes),
                                         trim=False,
                                         launcher=_launcher)
    processor.launch("take.svo2", str(tmp_path / "bundle"), "30264679")

    assert "TRIM" not in seen


def test_speed_settings_reach_the_driver(tmp_path):
    """Worker count and the overlay switch are passed as the driver's own env
    vars, since that is the only channel a detached script has."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import \
        MarkerlessTrackProcessor
    boxes = tmp_path / "boxes.json"
    boxes.write_text('[[1, 2, 3, 4]]', encoding="utf-8")
    seen = {}

    def _launcher(argv, env, log_path):
        """Capture the environment the driver would be started with."""
        del argv, log_path
        seen.update(env)
        return _DoneJob()

    processor = MarkerlessTrackProcessor(script=str(boxes),
                                         boxes_json=str(boxes),
                                         jobs=30,
                                         viz=False,
                                         launcher=_launcher)
    processor.launch("take.svo2", str(tmp_path / "bundle"), "30264679")

    assert seen["JOBS"] == "30"
    # "0" rather than absent: the driver compares the VALUE, so an unset
    # TRACK_VIZ is what asks for the overlay.
    assert seen["TRACK_VIZ"] == "0"


def test_default_jobs_leaves_the_drivers_own_choice(tmp_path):
    """0 must leave JOBS unset rather than pass "0", which the driver would
    hand to -j and fan out to nothing."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import \
        MarkerlessTrackProcessor
    boxes = tmp_path / "boxes.json"
    boxes.write_text('[[1, 2, 3, 4]]', encoding="utf-8")
    seen = {}

    def _launcher(argv, env, log_path):
        """Capture the environment the driver would be started with."""
        del argv, log_path
        seen.update(env)
        return _DoneJob()

    processor = MarkerlessTrackProcessor(script=str(boxes),
                                         boxes_json=str(boxes),
                                         jobs=0,
                                         viz=True,
                                         launcher=_launcher)
    processor.launch("take.svo2", str(tmp_path / "bundle"), "30264679")

    assert "JOBS" not in seen
    # Likewise: asking for the overlay means saying nothing at all.
    assert "TRACK_VIZ" not in seen


class _DoneJob:
    """A launched job that has already finished."""

    def poll(self):
        """Finished."""
        return 0

    def wait(self, timeout=None):
        """Finished."""
        del timeout
        return 0


def test_boxes_are_unwrapped_into_what_stage_2_reads(tmp_path):
    """Regression from run_20260817_162250: stage 2 died on int('id').

    init_boxes.py WRITES records -- {"id", "box", "label"} under a
    "boxes" key -- but the BOXES env it READS expects a bare list of
    [x0, y0, x1, y1]. Handing the records over unchanged made stage 2
    iterate a dict, minutes into the run, after the arm had already
    executed the whole episode.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import _read_boxes
    written = tmp_path / "boxes.json"
    written.write_text(json.dumps({
        "frame":
        0,
        "source":
        "manual",
        "boxes": [{
            "id": 0,
            "box": [324, 434, 411, 555],
            "label": "domino"
        }, {
            "id": 1,
            "box": [452, 398, 521, 495],
            "label": "domino"
        }],
    }),
                       encoding="utf-8")

    assert json.loads(_read_boxes(str(written))) == [[324, 434, 411, 555],
                                                     [452, 398, 521, 495]]


def test_bare_box_lists_are_still_accepted(tmp_path):
    """A hand-written file in the form the env documents keeps working."""
    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import _read_boxes
    written = tmp_path / "boxes.json"
    written.write_text(json.dumps([[1, 2, 3, 4]]), encoding="utf-8")

    assert json.loads(_read_boxes(str(written))) == [[1, 2, 3, 4]]


def test_a_failing_pipeline_job_leaves_its_reason_in_a_log(tmp_path):
    """The job is detached, so its output has nowhere else to go.

    Without the log a failed stage is a missing track and no reason,
    noticed minutes later when the fit finds nothing.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.pybullet_helpers.track_pipeline import LOG_NAME, \
        MarkerlessTrackProcessor
    boxes = tmp_path / "boxes.json"
    boxes.write_text('{"boxes": [[1, 2, 3, 4]]}', encoding="utf-8")
    script = tmp_path / "driver.sh"
    script.write_text("#!/bin/sh\necho 'stage 3 exploded' >&2\nexit 3\n",
                      encoding="utf-8")
    script.chmod(0o755)
    bundle = tmp_path / "bundle"
    processor = MarkerlessTrackProcessor(script=str(script),
                                         boxes_json=str(boxes))

    job = processor.launch(str(tmp_path / "take.svo2"), str(bundle), "123")
    assert job is not None
    processor.wait_all(timeout=30)

    log = (bundle / LOG_NAME).read_text(encoding="utf-8")
    assert "stage 3 exploded" in log


def test_boxes_are_drawn_once_for_the_run(tmp_path):
    """One drag window per RUN, not per take.

    That is what makes an otherwise-interactive pipeline usable in a
    learning loop, and it is valid because a fixed-plan replay trains
    and tests on one arrangement.
    """
    session = _StubSession()
    processor = _StubProcessor()
    rec = EpisodeRecorder(session,
                          processor=processor,
                          track_dir=str(tmp_path))
    rec.open()
    drawn = []

    def _picker(svo, bundle, serial):
        """Stand in for the drag window."""
        drawn.append((svo, bundle, serial))
        return str(tmp_path / "boxes.json")

    rec.ensure_boxes(picker=_picker)
    for i in range(3):
        rec.start_episode(f"ep{i}")
        rec.stop_episode()

    assert len(drawn) == 1, "the window must open once, not once per episode"
    assert processor.boxes == str(tmp_path / "boxes.json")
    assert len(processor.launched) == 3


def test_a_failed_box_draw_still_records_the_takes(tmp_path, caplog):
    """Losing the boxes costs the post-processing, not the run: the takes are
    on disk and can be processed by hand."""
    processor = _StubProcessor()
    rec = EpisodeRecorder(_StubSession(),
                          processor=processor,
                          track_dir=str(tmp_path))
    rec.open()

    with caplog.at_level("ERROR"):
        assert rec.ensure_boxes(picker=lambda *_: None) is None

    assert "recorded but" in caplog.text
    rec.start_episode("ep")
    rec.stop_episode()
    assert len(processor.launched) == 1


def test_the_recorder_closes_an_in_flight_take():
    """close() is registered with atexit, and a session left recording would
    keep writing until the disk filled."""
    session = _StubSession()
    rec = EpisodeRecorder(session)
    rec.open()
    rec.start_episode("stamp")

    rec.close()

    assert session.closes == 1
    # Idempotent: atexit may fire after an explicit close.
    rec.close()
    assert session.closes == 1


# -- snapshot scene rebuild --------------------------------------------------
def _snapshot_perception(recorder, tmp_path, serial="32294776", scene=None):
    """A snapshot perception whose pipeline and loader are stubbed out."""

    def _runner(svo, bundle, cam):
        """Stand in for markerless stages 1-4."""
        del bundle, cam
        return svo + ".dominoes.json"

    return MarkerlessSnapshotPerception(recorder,
                                        serial=serial,
                                        runner=_runner,
                                        scene_loader=lambda p: scene or
                                        ("scene", p),
                                        work_dir=str(tmp_path),
                                        frames=3)


def _touch_svo(tmp_path, take_dir, serial="32294776", ext=".svo2"):
    """Create the recording a take is expected to contain."""
    directory = tmp_path / take_dir
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"zed_{serial}{ext}").write_text("", encoding="utf-8")
    return str(directory)


class _SnapshottingRecorder:
    """A recorder whose snapshot() writes a take dir on disk."""

    def __init__(self, tmp_path, serial="32294776", ext=".svo2"):
        self._tmp_path = tmp_path
        self._serial = serial
        self._ext = ext
        self.calls = 0
        self.serials = [serial, "30264679"]

    def snapshot(self, frames=5):
        """Record a short take; return its directory and meta."""
        del frames  # the stub writes one file regardless
        self.calls += 1
        take = _touch_svo(self._tmp_path,
                          f"snap{self.calls}",
                          serial=self._serial,
                          ext=self._ext)
        return take, {"svo_ext": self._ext}


def test_a_snapshot_is_a_second_take_on_the_same_open_session():
    """The whole design: a ZED admits one owner, so the scene look does not
    open cameras -- it takes a short take on the session already holding
    them."""
    session = _StubSession()
    rec = EpisodeRecorder(session)
    rec.open()

    take_dir, meta = rec.snapshot(frames=3)

    assert session.opens == 1, "a snapshot must not open cameras of its own"
    assert take_dir.startswith("take_")
    assert meta["svo_ext"] == ".svo2"
    assert session.recording is False, "the snapshot take is closed again"
    assert session.started[0][1] == 3, "max_frames bounds the snapshot"


def test_a_snapshot_is_not_recorded_as_an_episode_track():
    """``takes`` is what the fit consumes.

    A snapshot is an input to a task, not a record of an execution, and
    must not be mistaken for one.
    """
    session = _StubSession()
    rec = EpisodeRecorder(session)
    rec.open()
    rec.snapshot()

    assert not rec.takes
    assert len(rec.snapshots) == 1


@pytest.mark.usefixtures("recorder")
def test_a_snapshot_and_an_episode_take_do_not_overlap():
    """Sequenced, not concurrent: the real session refuses a second take, so
    the snapshot has to happen while no episode take is running."""
    session = _StubSession()
    rec = EpisodeRecorder(session)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   recorder=rec)

    rec.snapshot()  # between episodes
    ex.after_reset("train", 0, _state(0.0))  # episode take opens
    ex.after_episode(True)  # and closes
    rec.snapshot()  # between episodes again

    # Three takes, none refused: snapshot, episode, snapshot.
    assert len(session.started) == 3
    stamps = [stamp for stamp, _ in session.started]
    assert len(set(stamps)) == 3, \
        "take directories must be distinct, even within one second"


def test_snapshot_perception_owns_no_cameras():
    """open/close are no-ops by design; taking cameras here is the collision
    being avoided."""
    perception = MarkerlessSnapshotPerception(object(), serial="32294776")

    perception.open()
    perception.close()

    assert perception.has_perception is True


def test_observing_fits_the_snapshot_and_returns_the_scene(tmp_path):
    """The look a scene reset asks for: record, fit, read the scene JSON."""
    rec = _SnapshottingRecorder(tmp_path)
    perception = _snapshot_perception(rec, tmp_path)

    observation = perception.observe(settle_s=0.0)

    assert rec.calls == 1
    kind, path = observation
    assert kind == "scene"
    assert path.endswith(".dominoes.json")
    assert perception.scenes == [path]


def test_each_reset_fits_its_own_snapshot(tmp_path):
    """A scene rebuild is per episode, so a second look must not return the
    first one's fit."""
    rec = _SnapshottingRecorder(tmp_path)
    perception = _snapshot_perception(rec, tmp_path)

    first = perception.observe()
    second = perception.observe()

    assert first != second
    assert rec.calls == 2


def test_a_snapshot_missing_the_chosen_camera_says_so(tmp_path):
    """Fitting is single-camera, so the serial has to be one the recorder
    actually records."""
    rec = _SnapshottingRecorder(tmp_path, serial="30264679")
    perception = _snapshot_perception(rec, tmp_path, serial="32294776")

    with pytest.raises(FileNotFoundError, match="32294776"):
        perception.observe()


def test_the_svo_extension_comes_from_the_take(tmp_path):
    """SVO_EXT depends on the SDK major version, so it is read from the take's
    own meta rather than assumed."""
    rec = _SnapshottingRecorder(tmp_path, ext=".svo")
    perception = _snapshot_perception(rec, tmp_path)

    observation = perception.observe()

    assert observation[1].endswith(".svo.dominoes.json")


# -- two-camera snapshot fusion ----------------------------------------------
class _TwoCameraRecorder:
    """A recorder whose snapshot() writes BOTH cameras' recordings, as the real
    one does -- the session opens every camera it was given."""

    def __init__(self,
                 tmp_path,
                 serials=("30264679", "32294776"),
                 ext=".svo2",
                 omit=()):
        self._tmp_path = tmp_path
        self._ext = ext
        self._omit = set(omit)
        self.calls = 0
        self.serials = list(serials)

    def snapshot(self, frames=5):
        del frames
        self.calls += 1
        directory = self._tmp_path / f"snap{self.calls}"
        directory.mkdir(parents=True, exist_ok=True)
        for serial in self.serials:
            if serial in self._omit:
                continue
            (directory / f"zed_{serial}{self._ext}").write_text(
                "", encoding="utf-8")
        return str(directory), {"svo_ext": self._ext}


def _fusing_perception(recorder, tmp_path, **kwargs):
    """A two-camera snapshot perception with the pipeline stubbed out."""
    seen = {}

    def _multi_runner(svos, bundle):
        seen["svos"] = dict(svos)
        seen["bundle"] = bundle
        return os.path.join(bundle, "dominoes.json")

    perception = MarkerlessSnapshotPerception(recorder,
                                              serials=recorder.serials,
                                              multi_runner=_multi_runner,
                                              scene_loader=lambda p:
                                              ("scene", p),
                                              work_dir=str(tmp_path),
                                              frames=3,
                                              **kwargs)
    return perception, seen


def test_a_fused_snapshot_fits_both_cameras_from_one_take(tmp_path):
    """One take, two recordings, one fused scene: the second view costs no
    extra hardware time because the recorder already wrote it."""
    rec = _TwoCameraRecorder(tmp_path)
    perception, seen = _fusing_perception(rec, tmp_path)

    kind, path = perception.observe(settle_s=0.0)

    assert rec.calls == 1, "one take, not one per camera"
    assert set(seen["svos"]) == {"30264679", "32294776"}
    for serial, svo in seen["svos"].items():
        assert os.path.basename(svo) == f"zed_{serial}.svo2"
    assert kind == "scene"
    assert perception.scenes == [path]


def test_a_take_missing_one_camera_is_not_quietly_fitted_from_the_other(
        tmp_path):
    """A half-fused scene looks exactly like a fused one, right where the
    missing camera's view was the reason for fusing."""
    rec = _TwoCameraRecorder(tmp_path, omit=("32294776", ))
    perception, _ = _fusing_perception(rec, tmp_path)

    with pytest.raises(FileNotFoundError, match="32294776"):
        perception.observe()


def test_fusing_needs_exactly_two_cameras():
    """The fusion pairs a reference against one other."""
    utils.reset_config({"real_robot_snapshot_fuse_cameras": True})
    recorder = _TwoCameraRecorder(pathlib.Path("."), serials=("30264679", ))

    with pytest.raises(ValueError, match="exactly two"):
        build_snapshot_perception(recorder)


def test_per_camera_boxes_are_parsed_and_passed_through(tmp_path):
    """Boxes cannot be shared between cameras, so each one names its own."""
    utils.reset_config({
        "real_robot_snapshot_fuse_cameras":
        True,
        "real_robot_snapshot_boxes_json_by_camera":
        '{"30264679": "/a/boxes.json", "32294776": "/b/boxes.json"}',
    })
    recorder = _TwoCameraRecorder(tmp_path)

    perception = build_snapshot_perception(recorder)

    # pylint: disable-next=protected-access
    assert perception._boxes_json_by_camera == {
        "30264679": "/a/boxes.json",
        "32294776": "/b/boxes.json",
    }


def test_boxes_for_a_camera_that_is_not_recorded_are_refused(tmp_path):
    """Otherwise the first anyone hears of it is a drag window opening
    mid-run."""
    utils.reset_config({
        "real_robot_snapshot_fuse_cameras":
        True,
        "real_robot_snapshot_boxes_json_by_camera":
        '{"19824535": "/a/boxes.json"}',
    })

    with pytest.raises(ValueError, match="19824535"):
        build_snapshot_perception(_TwoCameraRecorder(tmp_path))


def test_unparseable_per_camera_boxes_say_what_the_shape_should_be(tmp_path):
    utils.reset_config({
        "real_robot_snapshot_fuse_cameras": True,
        "real_robot_snapshot_boxes_json_by_camera": "not json",
    })

    with pytest.raises(ValueError, match="JSON object"):
        build_snapshot_perception(_TwoCameraRecorder(tmp_path))


def test_single_camera_snapshots_are_unchanged_by_the_fusion_setting(tmp_path):
    """Off by default, and the one-camera path must not go near the fusion."""
    utils.reset_config({
        "real_robot_snapshot_fuse_cameras": False,
        "real_robot_snapshot_camera": "30264679"
    })

    perception = build_snapshot_perception(_TwoCameraRecorder(tmp_path))

    # pylint: disable-next=protected-access
    assert perception._serials == ["30264679"]


def test_snapshot_rebuild_needs_the_recorder():
    """It takes its snapshot on the recorder's session; without one it would
    have to open cameras, which is the collision being avoided."""
    utils.reset_config({
        "real_robot_execute": True,
        "real_robot_snapshot_rebuild": True,
        "real_robot_record_episodes": False,
    })
    with pytest.raises(ValueError, match="real_robot_record_episodes"):
        attach_real_robot(cast(Any, _StubEnv()))


# -- open-loop episodes ------------------------------------------------------
def _run_episode(ex, options, recorder, completed=True):
    """Drive ``ex`` through whole options, then end the episode.

    Each option contributes two actions, the second terminal, so a chunk
    is two actions long and the option count is recoverable from the
    shipped chunks.
    """
    obs = _state(0.0)
    for name in options:
        option = _option(name)
        ex.after_step(_act(option, terminal=False), obs)
        ex.after_step(_act(option, terminal=True), obs)
    ex.after_episode(completed)
    return recorder


def test_open_loop_ships_once_per_episode_not_once_per_option(recorder):
    """The point of the flag: one request for the whole episode.

    Per-boundary shipping calls execute_chunks once per option; open-
    loop calls it once, with every option's chunk in order.
    """
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   open_loop_episode=True)

    _run_episode(ex, ["Pick", "Place", "Push"], recorder)

    assert len(recorder.shipped) == 1, \
        "open-loop must ship the episode in a single execute_chunks call"
    (batch, ) = recorder.shipped
    assert len(batch) == 3, "every option's chunk must be in the batch"
    assert all(len(chunk) == 2 for chunk in batch)


def test_batching_ships_the_same_segments_as_shipping_one_at_a_time(recorder):
    """Batching changes WHEN the arm is told, not WHAT.

    The payload for chunk i must not depend on how many chunks travel
    with it -- otherwise open-loop would be commanding different motion,
    and any difference in how the arm behaves would be ours rather than
    the hardware's.
    """
    options = ["Pick", "Place", "Push"]

    def _chunks_for(ex):
        obs = _state(0.0)
        for name in options:
            option = _option(name)
            ex.after_step(_act(option, terminal=False), obs)
            ex.after_step(_act(option, terminal=True), obs)
        ex.after_episode(True)

    _chunks_for(_executor(_StubEnv(), observe_at_boundaries=False))
    one_at_a_time = [c for call in recorder.shipped for c in call]
    recorder.shipped.clear()
    _chunks_for(
        _executor(_StubEnv(),
                  observe_at_boundaries=False,
                  open_loop_episode=True))
    (batched, ) = recorder.shipped

    assert len(one_at_a_time) == len(batched)
    for eager, deferred in zip(one_at_a_time, batched):
        assert len(eager) == len(deferred)
        for a, b in zip(eager, deferred):
            assert np.allclose(a.arr, b.arr), \
                "the arm is being commanded different motion under open-loop"


def test_recording_starts_in_front_of_the_named_option(recorder, tmp_path):
    """The bridge runs unrecorded; the take opens, then the push ships.

    Only the cascade is scored, and on run_20260818_092302 it began 107
    s into a 131 s take -- so recording the pick-and-place cost most of
    the video and none of the evidence.
    """
    session = _StubSession()
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   open_loop_episode=True,
                   record_from_option="Push",
                   recorder=EpisodeRecorder(session, track_dir=str(tmp_path)))

    _run_episode(ex, ["Pick", "Place", "Push", "Wait"], recorder)

    prologue, measured = recorder.shipped
    assert len(prologue) == 2, "the bridge must ship as one unrecorded run"
    assert len(measured) == 2, "the take must cover Push and the Wait after it"
    assert len(session.started) == 1


def _shipments_before_the_take_opened(session, recorder):
    """Watch ``session`` and report what had already gone to the arm.

    Counting shipments at ``start_take`` is the only way to pin the
    ORDER. Asserting on the shipments alone cannot: an episode that
    sends everything and only then opens the take looks identical from
    the robot's side, and it is the exact failure worth guarding -- a
    recording with none of the motion in it.
    """
    seen = []
    start_take = session.start_take

    def _watched_start_take(*args, **kwargs):
        """Note how much had already been sent to the arm."""
        seen.append(len(recorder.shipped))
        return start_take(*args, **kwargs)

    session.start_take = _watched_start_take
    return seen


def _boxes_drawn_when(session, recorder, episode_recorder):
    """Note what had shipped, and whether the take was open, at draw time.

    The boxes are SAM-2 prompts for the take's first frame, so WHEN they
    are drawn is the whole of their correctness: run-start boxes
    describe an arrangement a later take never sees.
    """
    utils.reset_config({
        "real_robot_pick_boxes_at_start": True,
        "real_robot_snapshot_boxes_json": "",
    })
    seen = []
    ensure = episode_recorder.ensure_boxes

    def _watched_ensure_boxes(*args, **kwargs):
        """Snapshot the world at the moment of the draw."""
        seen.append({
            "shipments": len(recorder.shipped),
            "take_open": len(session.started) - len(session.stopped),
        })
        return ensure(*args, **kwargs)

    episode_recorder.ensure_boxes = _watched_ensure_boxes
    return seen


def test_boxes_are_drawn_at_the_boundary_when_the_take_opens_later(
        recorder, tmp_path):
    """After the prologue rearranges the row, before the take opens.

    The prologue picks and places two dominoes, so boxes drawn at run
    start point at where they used to be -- and stage 2 does not report
    an empty box, it fits a mask to whatever is inside it. Observed on
    run_20260818_140211, whose boxes were drawn 3 minutes before the
    take opened.
    """
    session = _StubSession()
    episode_recorder = EpisodeRecorder(session, track_dir=str(tmp_path))
    drawn = _boxes_drawn_when(session, recorder, episode_recorder)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   open_loop_episode=True,
                   record_from_option="Push",
                   recorder=episode_recorder)

    _run_episode(ex, ["Pick", "Place", "Push", "Wait"], recorder)

    assert len(drawn) == 1, "drawn exactly once"
    assert drawn[0]["shipments"] == 1, \
        "drawn AFTER the prologue, so the row is in its final arrangement"
    assert drawn[0]["take_open"] == 0, \
        "and BEFORE the take opens, so its first frame matches the boxes"


def test_boxes_are_still_drawn_at_run_start_when_the_take_opens_there(
        recorder, tmp_path):
    """Unchanged where it was already right.

    With no record_from_option the take brackets the whole episode, so
    run start IS the arrangement its first frame shows -- and drawing
    then keeps the human at the bench before anything moves.
    """
    session = _StubSession()
    episode_recorder = EpisodeRecorder(session, track_dir=str(tmp_path))
    drawn = _boxes_drawn_when(session, recorder, episode_recorder)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   open_loop_episode=True,
                   record_from_option="",
                   recorder=episode_recorder)

    _run_episode(ex, ["Pick", "Place", "Push", "Wait"], recorder)

    assert len(drawn) == 1
    assert drawn[0]["shipments"] == 0, "nothing had been sent to the arm yet"


def test_boxes_are_drawn_once_across_several_episodes(recorder, tmp_path):
    """A fixed-plan replay arranges the same row every episode, so the boundary
    draw is a per-RUN cost and must not become a per-take one -- a drag window
    opening each episode is what pick_boxes_at_start exists to avoid."""
    session = _StubSession()
    episode_recorder = EpisodeRecorder(session, track_dir=str(tmp_path))
    drawn = _boxes_drawn_when(session, recorder, episode_recorder)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   open_loop_episode=True,
                   record_from_option="Push",
                   recorder=episode_recorder)

    for _ in range(3):
        _run_episode(ex, ["Pick", "Place", "Push", "Wait"], recorder)

    assert len(drawn) == 3, "the call site runs every episode"
    assert episode_recorder.ensure_boxes() is None, \
        "but the draw itself happens once per run"


def test_the_take_opens_between_the_two_shipments(recorder, tmp_path):
    """Ordering is the whole correctness argument.

    A take opened after the push has already been sent would miss the
    first onset, and every interval is measured against that one -- so
    the failure would not be a missing residual but a wrong one.
    """
    session = _StubSession()
    before = _shipments_before_the_take_opened(session, recorder)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   open_loop_episode=True,
                   record_from_option="Push",
                   recorder=EpisodeRecorder(session, track_dir=str(tmp_path)))

    _run_episode(ex, ["Pick", "Place", "Push", "Wait"], recorder)

    assert before == [1], \
        "the take must open after the bridge and before the push"


def test_an_option_the_episode_never_runs_records_everything(
        recorder, tmp_path):
    """Too much video is slow; too little is a lost episode.

    The take must open BEFORE anything ships, not merely exist: an
    episode that sends every chunk and then starts recording has the
    same shipment count and none of the motion.
    """
    session = _StubSession()
    before = _shipments_before_the_take_opened(session, recorder)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   open_loop_episode=True,
                   record_from_option="Nudge",
                   recorder=EpisodeRecorder(session, track_dir=str(tmp_path)))

    _run_episode(ex, ["Pick", "Place", "Push"], recorder)

    (batch, ) = recorder.shipped
    assert len(batch) == 3, "an unknown option must not lose any motion"
    assert before == [0], "the whole batch must be inside the take"


def test_recording_from_the_first_option_ships_one_batch(recorder, tmp_path):
    """A boundary at index 0 has no prologue, and must not ship an empty
    request in front of the batch."""
    session = _StubSession()
    before = _shipments_before_the_take_opened(session, recorder)
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   human_reset=False,
                   open_loop_episode=True,
                   record_from_option="Push",
                   recorder=EpisodeRecorder(session, track_dir=str(tmp_path)))

    _run_episode(ex, ["Push", "Wait"], recorder)

    (batch, ) = recorder.shipped
    assert len(batch) == 2
    assert before == [0], "the whole batch must be inside the take"


def test_the_split_commands_the_same_motion_as_one_batch(recorder):
    """Splitting changes WHEN the arm is told, not WHAT.

    Same argument as batching itself: the payload for chunk i must not
    depend on which request it travels in, or the recorded episode would
    be different motion from the unrecorded one.
    """
    options = ["Pick", "Place", "Push", "Wait"]

    _run_episode(
        _executor(_StubEnv(),
                  observe_at_boundaries=False,
                  open_loop_episode=True), options, recorder)
    (whole, ) = recorder.shipped
    recorder.shipped.clear()
    _run_episode(
        _executor(_StubEnv(),
                  observe_at_boundaries=False,
                  open_loop_episode=True,
                  record_from_option="Push"), options, recorder)
    split = [chunk for call in recorder.shipped for chunk in call]

    assert len(whole) == len(split)
    for one, two in zip(whole, split):
        assert len(one) == len(two)
        for a, b in zip(one, two):
            assert np.allclose(a.arr, b.arr), \
                "splitting the batch changed the motion the arm is commanded"


def test_open_loop_preserves_option_order(recorder):
    """A batch is a plan, so the arm must run it in the order it was
    simulated."""
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   open_loop_episode=True)

    _run_episode(ex, ["Pick", "Place", "Push"], recorder)

    (batch, ) = recorder.shipped
    shipped = [chunk[0].get_option().name for chunk in batch]
    assert shipped == ["Pick", "Place", "Push"]


@pytest.mark.usefixtures("recorder")
def test_open_loop_leaves_the_twin_trajectory_bit_identical():
    """The whole safety argument for deferring: with the boundary look off,
    shipping is a pure write-only side effect, so *when* it happens cannot
    change what the rollout sees.

    Same actions through both paths; the observations handed back must
    match exactly, not approximately.
    """
    options = ["Pick", "Place", "Push"]

    per_boundary = _executor(_StubEnv(), observe_at_boundaries=False)
    open_loop = _executor(_StubEnv(),
                          observe_at_boundaries=False,
                          open_loop_episode=True)

    def _observed(ex):
        # A distinct observation per step, so a path that hands back
        # anything other than the one it was given is visible. With a
        # constant obs this assertion would hold for the wrong reasons.
        seen = []
        for i, name in enumerate(options):
            option = _option(name)
            seen.append(
                ex.after_step(_act(option, terminal=False),
                              _state(2.0 * i + 1.0)))
            seen.append(
                ex.after_step(_act(option, terminal=True),
                              _state(2.0 * i + 2.0)))
        ex.after_episode(True)
        return seen

    eager = _observed(per_boundary)
    deferred = _observed(open_loop)

    assert len(eager) == len(deferred)
    for a, b in zip(eager, deferred):
        assert a is b or a.allclose(b), \
            "deferring the ship changed what the rollout observed"


def test_open_loop_discards_an_episode_that_did_not_complete(recorder):
    """A prefix of a plan is not a plan.

    Half a bridge, or a transport with no place at the end of it, would
    be executed by an arm with nobody having decided it was a good idea.
    """
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   open_loop_episode=True)

    _run_episode(ex, ["Pick", "Place"], recorder, completed=False)

    assert recorder.shipped == [], \
        "an incomplete episode must ship nothing at all"


def test_open_loop_drops_a_half_option_but_ships_the_whole_ones(recorder):
    """The buffer holds whole options, so a trailing partial one is not
    shippable -- but the options that did finish still are."""
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   open_loop_episode=True)
    obs = _state(0.0)

    finished = _option("Pick")
    ex.after_step(_act(finished, terminal=False), obs)
    ex.after_step(_act(finished, terminal=True), obs)
    # A second option that never reaches its boundary.
    ex.after_step(_act(_option("Place"), terminal=False), obs)
    ex.after_episode(True)

    (batch, ) = recorder.shipped
    assert len(batch) == 1
    assert batch[0][0].get_option().name == "Pick"


def test_open_loop_does_not_leak_across_episodes(recorder):
    """If finish_execution never runs, the next episode must not inherit the
    last one's motion -- it would be driven against a rearranged scene."""
    ex = _executor(_StubEnv(),
                   observe_at_boundaries=False,
                   open_loop_episode=True,
                   human_reset=False)
    obs = _state(0.0)
    stale = _option("Pick")
    ex.after_step(_act(stale, terminal=False), obs)
    ex.after_step(_act(stale, terminal=True), obs)

    # No after_episode: the episode ended some other way.
    ex.after_reset("train", 0, _state(0.0))
    _run_episode(ex, ["Push"], recorder)

    (batch, ) = recorder.shipped
    assert len(batch) == 1, "stale options must not ride along"
    assert batch[0][0].get_option().name == "Push"


def test_per_boundary_path_is_unchanged_by_the_flag(recorder):
    """Off by default: the existing behaviour must be exactly what it was."""
    ex = _executor(_StubEnv(), observe_at_boundaries=False)

    _run_episode(ex, ["Pick", "Place", "Push"], recorder)

    assert len(recorder.shipped) == 3, \
        "per-boundary shipping still ships each option as it is simulated"


def test_open_loop_and_boundary_looks_are_mutually_exclusive():
    """A boundary look has to happen between the two options it separates, and
    open-loop leaves no such moment.

    Refuse rather than silently drop whichever was asked for second.
    """
    with pytest.raises(ValueError, match="no moment between two options"):
        _executor(_StubEnv(),
                  observe_at_boundaries=True,
                  open_loop_episode=True)


def test_after_episode_is_a_no_op_for_the_per_boundary_path(recorder):
    """Nothing is outstanding when every option shipped as it was simulated."""
    ex = _executor(_StubEnv(), observe_at_boundaries=False)
    _run_episode(ex, ["Pick"], recorder)
    before = len(recorder.shipped)

    ex.after_episode(True)

    assert len(recorder.shipped) == before
