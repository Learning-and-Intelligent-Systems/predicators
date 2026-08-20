"""The real-robot executor driving the real domino twin, end to end.

The executor's own tests use a stub env to cover its logic in isolation.
This file does the opposite: a **real** ``PyBulletDominoRealEnv`` with a real
physics client, driven through ``env.step`` exactly as the episode loop does,
so the whole closed loop is exercised -- ship a chunk, look at the scene,
convert what was seen, write it into the twin, read it back out through the
agent-facing observation.

No hardware either way. Most of these stub the two bridge helpers -- the seam
where predicators ends and the robot begins -- so they run without the private
submodule. The last one instead drives a genuine dry ``RealRobot`` and so
skips without it: it is the only place the real ``StepRequest`` / ``Segment``
construction is exercised, which stubs by their nature cannot check.
"""
# The component's predicate helpers (_Toppled_holds) and the env's
# component itself are what these tests assert on, so reading them is
# the point. babyrobot is imported inside one test body because it is
# optional and absent on CI.
# pylint: disable=protected-access,import-outside-toplevel,import-error
import json
from typing import Any, List

import numpy as np
import pytest
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from predicators import utils
from predicators.envs.pybullet_domino.real_geometry import _REAL_TO_ENV_BODY
from predicators.envs.pybullet_domino_real import PyBulletDominoRealEnv
from predicators.envs.pybullet_env import PyBulletEnv
from predicators.pybullet_helpers.real_robot_executor import attach_real_robot
from predicators.structs import Action, ParameterizedOption, State

_TABLE_Z = -0.041
_START_ID = 6
_TARGET_ID = 5
_STANDING_YAW = np.pi


def _base_quat(roll=0.0, yaw=_STANDING_YAW, pitch=0.0):
    """The base-frame quaternion of a domino at env ``(roll, pitch, yaw)``."""
    r_env = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    r_base = Rotation.from_euler(
        "z", -np.pi / 2).as_matrix() @ (r_env @ _REAL_TO_ENV_BODY.T)
    return list(Rotation.from_matrix(r_base).as_quat())


def _record(capture_id, base_xy, quat=None):
    """One scene-JSON domino record."""
    return {
        "id": capture_id,
        "center_base_m": [base_xy[0], base_xy[1], 0.03],
        "quat_base_xyzw": list(quat if quat is not None else _base_quat()),
        "dims_m": [0.15, 0.07, 0.029],
    }


class _StubDominoPose:
    """Stands in for babyrobot's DominoPose."""

    def __init__(self, capture_id, xyz, quat_xyzw):
        self.id = capture_id
        self.xyz = tuple(xyz)
        self.quat_xyzw = tuple(quat_xyzw)


class _StubDominoObservation:
    """Stands in for babyrobot's DominoObservation."""

    def __init__(self, dominoes):
        self.dominoes = list(dominoes)


class _StubRobot:
    """A robot with cameras that never moves anything."""
    has_perception = True
    dry = True


def _config(scene_path, **overrides):
    """The real-execution config for this env."""
    flags = {
        "env": "pybullet_domino_real",
        "pybullet_robot": "panda",
        "domino_real_scene": scene_path,
        "domino_real_table_z": _TABLE_Z,
        "domino_real_start_id": _START_ID,
        "domino_real_target_id": _TARGET_ID,
        "domino_use_domino_blocks_as_target": True,
        "domino_use_skill_factories": False,
        "domino_real_decorate": False,
        "real_robot_execute": True,
        "real_robot_observe_at_option_boundary": True,
        "real_robot_settle_s": 0.0,
        "real_robot_divergence_atol": 0.02,
        # These cover per-option execution and the twin sync. The human-gated
        # task rebuild is its own concern, in test_domino_real_online.py.
        "real_robot_human_reset": False,
        # ...so the captured scene is what these tests mean to run against,
        # which is what the stale-task guard asks callers to say out loud.
        "real_robot_allow_captured_scene_task": True,
    }
    flags.update(overrides)
    utils.reset_config(flags)


@pytest.fixture(scope="module", name="scene_path")
def scene_path_fixture(tmp_path_factory):
    """A two-domino scene: a green start and a purple target."""
    scene = {
        "frame":
        "robot_base",
        "units":
        "m",
        "dominoes":
        [_record(_START_ID, (0.0, 0.0)),
         _record(_TARGET_ID, (0.2, 0.0))],
    }
    path = tmp_path_factory.mktemp("rwe_integration") / "scene.json"
    path.write_text(json.dumps(scene), encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module", name="inner")
def inner_fixture(scene_path):
    """One real twin for the module -- building PyBullet per test is slow."""
    _config(scene_path)
    return PyBulletDominoRealEnv(use_gui=False)


@pytest.fixture(name="shipped")
def shipped_fixture(monkeypatch):
    """Stub the two bridge helpers and let a test queue up observations."""

    class _Bridge:
        """Records shipments; replies with whatever a test queued."""

        def __init__(self):
            self.chunks: List[Any] = []
            self.homed: List[List[float]] = []
            self.to_return: List[Any] = []

        def execute_chunks(self,
                           robot,
                           chunks,
                           layout,
                           observe=False,
                           settle_s=0.0):
            """Record the shipment; hand back the queued observations."""
            del robot, layout, settle_s
            self.chunks.extend(chunks)
            if not observe:
                return []
            return [self.to_return.pop(0) for _ in chunks if self.to_return]

        def reset_arm(self, robot, joints):
            """Record the homing request."""
            del robot
            self.homed.append(list(joints))
            return tuple(joints)

    bridge = _Bridge()
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.execute_chunks",
        bridge.execute_chunks)
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        bridge.reset_arm)
    return bridge


def _terminal_action(env):
    """A well-formed action for this env that ends its option."""
    arr = np.zeros(env.action_space.shape, dtype=np.float32)
    action = Action(arr)
    param_opt = ParameterizedOption("StubOption", [], Box(0, 1, (1, )),
                                    lambda s, m, o, p: Action(arr),
                                    lambda s, m, o, p: True,
                                    lambda s, m, o, p: False)
    option = param_opt.ground([], [0.5])
    option.terminal = lambda _obs: True
    action.set_option(option)
    return action


def _attach(env, robot):
    """Attach an executor built against the stubs."""
    executor = attach_real_robot(env, robot)
    assert executor is not None
    return executor


def test_attaching_leaves_the_env_the_env(inner, scene_path):
    """The env keeps its identity, which the old wrapper did not.

    A wrapper was a BaseEnv but NOT a PyBulletEnv, and the episode loop
    branches on exactly that (``cogman.py:225`` and ``:272``) to decide
    whether to pass ``render_obs`` -- so wrapping silently disabled
    rendering. Attaching cannot: there is still only one object.
    """
    _config(scene_path)
    _attach(inner, _StubRobot())

    assert isinstance(inner, PyBulletEnv)
    assert isinstance(inner, PyBulletDominoRealEnv)
    assert inner.get_name() == "pybullet_domino_real"


def test_simulate_never_reaches_the_executor(inner, scene_path, shipped):
    """Bilevel search calls ``simulate`` hundreds of times per option. If that
    path could drive hardware, the planner would move the arm while merely
    considering a candidate.

    This is the guarantee the executor design buys structurally --
    ``simulate`` goes through ``_step_once``, which has no hook --
    rather than by anyone remembering not to attach an executor to the
    planner's env.
    """
    _config(scene_path)
    _attach(inner, _StubRobot())
    state = inner.reset("test", 0)
    shipped.chunks.clear()
    shipped.homed.clear()

    for _ in range(3):
        inner.simulate(state, _terminal_action(inner))

    assert shipped.chunks == [], "simulate() shipped to the robot"
    assert shipped.homed == []


def test_reset_homes_the_arm_to_the_real_twins_joints(inner, scene_path,
                                                      shipped):
    """The homing joints come from the twin's own reset state, so the arm
    starts where the first option's waypoints do."""
    _config(scene_path)
    _attach(inner, _StubRobot())
    env = inner

    env.reset("test", 0)

    assert len(shipped.homed) == 1
    # 7 arm joints: the two finger entries are dropped by the layout.
    assert len(shipped.homed[0]) == 7


def test_perceived_topple_reaches_the_agent_through_the_twin(
        inner, scene_path, shipped):
    """The whole point of the closed loop, end to end.

    The cameras find the target on its face; that has to survive the
    conversion, the write into PyBullet, and the read back out --
    because the agent only ever sees the twin.
    """
    _config(scene_path)
    _attach(inner, _StubRobot())
    env = inner
    comp = inner._domino_component  # pylint: disable=protected-access
    target = comp.dominos[1]  # slot 1 <-> capture id 5

    obs = env.reset("test", 0)
    assert not comp._Toppled_holds(obs, [target]), \
        "the target starts standing; the test proves nothing otherwise"

    shipped.to_return = [
        _StubDominoObservation([
            _StubDominoPose(_TARGET_ID, (0.2, 0.0, 0.0145),
                            _base_quat(roll=np.pi / 2, yaw=0.0))
        ])
    ]
    returned = env.step(_terminal_action(env))

    # The chunk went out, and what came back is the twin's State, not the
    # library's observation type.
    assert len(shipped.chunks) == 1
    assert isinstance(returned, State)
    assert not isinstance(returned, _StubDominoObservation)
    # The twin now believes the target is down, and so does the agent.
    assert comp._Toppled_holds(returned, [target])
    assert abs(returned.get(target, "roll")) == pytest.approx(np.pi / 2,
                                                              abs=1e-3)


def test_the_correction_survives_the_next_step(inner, scene_path, shipped):
    """Not merely handed over once: the twin's BODIES hold the perceived pose,
    so the next step reads a corrected state rather than reverting to the sim's
    prediction.

    This is the difference between syncing and not syncing -- without
    the write, the correction would be overwritten on the very next
    action.
    """
    _config(scene_path)
    _attach(inner, _StubRobot())
    env = inner
    comp = inner._domino_component  # pylint: disable=protected-access
    target = comp.dominos[1]

    env.reset("test", 0)
    shipped.to_return = [
        _StubDominoObservation([
            _StubDominoPose(_TARGET_ID, (0.2, 0.0, 0.0145),
                            _base_quat(roll=np.pi / 2, yaw=0.0))
        ])
    ]
    env.step(_terminal_action(env))

    # A plain action with no option: no shipping, no looking, just the twin
    # advancing from the state perception put it in.
    after = env.step(Action(np.zeros(env.action_space.shape,
                                     dtype=np.float32)))

    assert comp._Toppled_holds(after, [target]), \
        "the perceived topple did not survive the next simulation step"


# -- against the genuine RealRobot -------------------------------------------


def test_closed_loop_against_a_real_dry_robot(inner, scene_path, tmp_path):
    """The same loop with NO stubbing between predicators and babyrobot.

    A real ``RealRobot`` (dry, so no arm exists) with file perception,
    reached through the real ``execute_chunks``. This is the only test
    that exercises the actual ``StepRequest`` / ``Segment`` construction
    and the gripper dedup, so a contract change on either side of the
    submodule boundary surfaces here rather than on the real scene.
    """
    pytest.importorskip("babyrobot")
    from babyrobot.realrobot.perception import FileDominoPerception
    from babyrobot.realrobot.real_robot import RealRobot

    # The scene the cameras will "see": the target lying on its face.
    seen = {
        "frame":
        "robot_base",
        "units":
        "m",
        "dominoes": [
            _record(_START_ID, (0.0, 0.0)),
            _record(_TARGET_ID, (0.2, 0.0), _base_quat(roll=np.pi / 2,
                                                       yaw=0.0)),
        ],
    }
    seen_path = tmp_path / "observed.json"
    seen_path.write_text(json.dumps(seen), encoding="utf-8")

    _config(scene_path)
    robot = RealRobot(perception=FileDominoPerception(str(seen_path)),
                      dry=True)
    try:
        assert robot.dry and robot.has_perception
        _attach(inner, robot)
        env = inner
        comp = inner._domino_component  # pylint: disable=protected-access
        target = comp.dominos[1]

        env.reset("test", 0)
        returned = env.step(_terminal_action(env))

        # The perceived topple made it all the way through the real
        # message types and into the agent-facing state.
        assert comp._Toppled_holds(returned, [target])
        # A gripper command was actually issued and is well formed.
        assert robot.last_gripper_command in ("open", "close")
    finally:
        robot.close()
