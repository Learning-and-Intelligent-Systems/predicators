"""Tests for PyBulletDominoRealEnv's perception -> State / Task conversions,
and for the PyBulletEnv primitives the real-world wrapper builds on.

These are pure conversions plus one simulator write: no hardware, no robot, and
**no babyrobot**. That is deliberate and is asserted below -- a suite that
silently skipped without the private submodule would hide exactly the
regressions these tests exist to catch.

The scene fixture is synthetic and tiny so the expected world poses can be
worked out by hand. The base -> world transplant is a rigid yaw(+pi/2) plus a
translation, i.e.

    world_x = 0.75 - base_y
    world_y = 0.72 + base_x
    world_z = base_z + z_off        (z_off = 0.4 - domino_real_table_z)

so every coordinate assertion below is checkable without running the code.
"""
import ast
import inspect
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pybullet as p
import pytest
from scipy.spatial.transform import Rotation

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.real_geometry import _REAL_TO_ENV_BODY, \
    Pose6D, domino_env_euler, domino_upright_yaw, pose_base_to_world
from predicators.envs.pybullet_domino_real import PyBulletDominoRealEnv, \
    _canonical_roll
from predicators.pybullet_helpers.real_robot_bridge import \
    gripper_joint_layout_from_robot
from predicators.settings import CFG
from predicators.structs import GroundAtom

_TABLE_Z = -0.041
_Z_OFF = 0.4 - _TABLE_Z  # 0.441
_START_ID = 6
_TARGET_ID = 5
# The default fixture heading. A standing domino whose width (body-y) axis
# points along world -x after the transplant's +pi/2 yaw, i.e. env yaw = pi.
_STANDING_YAW = np.pi


def _base_quat(roll: float = 0.0,
               yaw: float = _STANDING_YAW,
               pitch: float = 0.0) -> List[float]:
    """The base-frame quaternion of a domino at env ``(roll, pitch, yaw)``.

    Built by inverting the env<->real body-axis permutation and the
    transplant's +pi/2 world yaw, so it does not reuse
    ``domino_env_euler``'s own arithmetic: this is the orientation a real
    capture of such a domino would carry.

    Note a domino at rest is **not** the identity quaternion. The real
    body frame is x=L, y=W, z=H, so a STANDING domino has its body-x
    vertical; identity would mean the long axis lying along the table.
    """
    r_env = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    r_real = r_env @ _REAL_TO_ENV_BODY.T
    r_base = Rotation.from_euler("z", -np.pi / 2).as_matrix() @ r_real
    return list(Rotation.from_matrix(r_base).as_quat())


def _record(capture_id: int,
            base_xy: Tuple[float, float],
            *,
            quat: Optional[List[float]] = None,
            role: Optional[str] = None) -> Dict[str, Any]:
    """One scene-JSON domino record, in the shape a real capture emits."""
    rec: Dict[str, Any] = {
        "id": capture_id,
        "center_base_m": [base_xy[0], base_xy[1], 0.03],
        "quat_base_xyzw": list(quat if quat is not None else _base_quat()),
        "dims_m": [0.15, 0.07, 0.029],
    }
    if role is not None:
        rec["role"] = role
    return rec


def _write_scene(tmp_path,
                 records,
                 start_push_dir_base=None,
                 name="scene.json"):
    """Write a scene JSON and return its path."""
    scene = {"frame": "robot_base", "units": "m", "dominoes": records}
    if start_push_dir_base is not None:
        scene["start_push_dir_base"] = start_push_dir_base
    path = tmp_path / name
    path.write_text(json.dumps(scene), encoding="utf-8")
    return str(path)


class _StubDominoPose:
    """Stands in for babyrobot's DominoPose.

    The env reads observations structurally (``id`` / ``xyz`` /
    ``quat_xyzw``), which is what lets these tests run with the private
    submodule absent.
    """

    def __init__(self, capture_id, xyz, quat_xyzw=None):
        self.id = capture_id
        self.xyz = tuple(xyz)
        self.quat_xyzw = tuple(
            quat_xyzw if quat_xyzw is not None else _base_quat())


class _StubDominoObservation:
    """Stands in for babyrobot's DominoObservation."""

    def __init__(self, dominoes, stamp=0.0):
        self.stamp = stamp
        self.dominoes = list(dominoes)


def _config(scene_path):
    """Apply the config this env is actually run with."""
    utils.reset_config({
        "env": "pybullet_domino_real",
        "pybullet_robot": "panda",
        "domino_real_scene": scene_path,
        "domino_real_table_z": _TABLE_Z,
        "domino_real_start_id": _START_ID,
        "domino_real_target_id": _TARGET_ID,
        # The env's goal semantics assume targets are domino blocks; every
        # shipped config for it sets this.
        "domino_use_domino_blocks_as_target": True,
        "domino_use_skill_factories": False,
        "domino_real_decorate": False,
    })


def _make_env(scene_path):
    """Build the env against ``scene_path``."""
    _config(scene_path)
    return PyBulletDominoRealEnv(use_gui=False)


# The default scene: start and target 20cm apart along base +x, plus two
# movable blocks. Slot order follows scene order, so slot i <-> _SCENE_IDS[i].
_SCENE_RECORDS = [
    _record(_START_ID, (0.0, 0.0)),
    _record(11, (0.1, 0.05)),
    _record(12, (0.15, -0.05)),
    _record(_TARGET_ID, (0.2, 0.0)),
]
_SCENE_IDS = [_START_ID, 11, 12, _TARGET_ID]


@pytest.fixture(scope="module", name="scene_path")
def scene_path_fixture(tmp_path_factory):
    """The default scene JSON, written once for the module."""
    return _write_scene(tmp_path_factory.mktemp("domino_real"), _SCENE_RECORDS)


@pytest.fixture(scope="module", name="env")
def env_fixture(scene_path):
    """One env for the module -- building PyBullet per test is slow."""
    return _make_env(scene_path)


@pytest.fixture(autouse=True)
def _reapply_config(scene_path):
    """Re-apply the config before each test, since CFG is global."""
    _config(scene_path)


def test_env_has_no_module_level_babyrobot_import():
    """These conversions are pure predicators code, so this file runs on a
    checkout without the private submodule -- and must never skip.

    Checked on the parse tree rather than on ``sys.modules``, which
    another test in the same session may legitimately have populated.
    """
    source = inspect.getsourcefile(PyBulletDominoRealEnv)
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


# -- scene loading and roles -------------------------------------------------


def test_scene_ids_follow_scene_order(env):
    """Slot i holds capture id scene_ids[i]; that list is the id -> slot
    map."""
    assert env._scene_ids == _SCENE_IDS  # pylint: disable=protected-access


def test_domino_role_prefers_explicit_role_over_id(env):
    """A scene carrying explicit roles is trusted; a raw capture is keyed by
    id."""
    # pylint: disable=protected-access
    # Explicit role wins even when the id says otherwise.
    assert env._domino_role({"id": _START_ID, "role": "movable"}) == "movable"
    assert env._domino_role({"id": 999, "role": "target"}) == "target"
    # No role field: fall back to the configured ids.
    assert env._domino_role({"id": _START_ID}) == "start"
    assert env._domino_role({"id": _TARGET_ID}) == "target"
    assert env._domino_role({"id": 11}) == "movable"


def test_role_counts_from_scene(env):
    """(num_target, num_nontarget) drives the component's slot allocation."""
    del env  # counts are read from CFG's scene, not the instance
    n_target, n_nontarget = PyBulletDominoRealEnv._scene_role_counts()  # pylint: disable=protected-access
    assert (n_target, n_nontarget) == (1, 3)


# -- base -> world transplant ------------------------------------------------


def test_base_to_world_matches_hand_computed_values():
    """The transplant is yaw(+pi/2) then translate; pin it to known numbers."""
    world = pose_base_to_world(Pose6D((0.2, 0.1, 0.03), tuple(_base_quat())),
                               _Z_OFF)
    assert world.xyz == pytest.approx((0.75 - 0.1, 0.72 + 0.2, 0.03 + _Z_OFF))
    # Identity in base -> body-y along world -x -> yaw = pi.
    assert domino_upright_yaw(world) == pytest.approx(_STANDING_YAW)


# -- task construction: the two sources must agree ---------------------------


def test_scene_and_observation_tasks_agree(env, scene_path):
    """The captured scene and a live observation of the SAME poses build the
    same task.

    This is the anti-drift property the refactor exists for: both route
    through one conversion.
    """
    scene_task = env._build_task_from_scene()  # pylint: disable=protected-access
    records = json.loads(open(scene_path, encoding="utf-8").read())["dominoes"]
    obs = _StubDominoObservation([
        _StubDominoPose(r["id"], r["center_base_m"], r["quat_base_xyzw"])
        for r in records
    ])
    obs_task = env.task_from_observation(obs, "test")

    assert scene_task.init.allclose(obs_task.init)
    assert scene_task.goal == obs_task.goal


def test_goal_is_toppled_on_the_target(env):
    """Goal = Toppled(target), and only the target."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    assert len(task.goal) == 1
    atom = next(iter(task.goal))
    assert isinstance(atom, GroundAtom)
    assert atom.predicate.name == "Toppled"
    # The target is slot 3 -- the scene's 4th record, capture id _TARGET_ID.
    assert atom.objects[0].name == "domino_3"


def test_task_init_places_dominoes_at_transplanted_poses(env):
    """Each domino lands at its own base pose mapped into the world."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    for slot, rec in enumerate(_SCENE_RECORDS):
        bx, by, bz = rec["center_base_m"]
        dom = comp.dominos[slot]
        assert task.init.get(dom, "x") == pytest.approx(0.75 - by)
        assert task.init.get(dom, "y") == pytest.approx(0.72 + bx)
        assert task.init.get(dom, "z") == pytest.approx(bz + _Z_OFF)
        assert task.init.get(dom, "roll") == pytest.approx(0.0)


# -- start-domino yaw canonicalization ---------------------------------------


def _start_yaw(tmp_path, records, push_dir=None):
    """Build a one-off env for ``records`` and read the start domino's yaw."""
    path = _write_scene(tmp_path, records, push_dir, name="canon.json")
    env = _make_env(path)
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    slot = [r["id"] for r in records].index(_START_ID)
    return task.init.get(comp.dominos[slot], "yaw")


def test_start_yaw_flips_to_face_the_target(tmp_path):
    """A domino is 180-degree symmetric, so perception's yaw branch is
    arbitrary; the start must end up facing the way it has to topple.

    Identity orientation reads as yaw=pi, i.e. facing world -y. With the
    target at world +y, that faces away, so it flips to 0.
    """
    records = [
        _record(_START_ID, (0.0, 0.0)),
        _record(_TARGET_ID, (0.2, 0.0)),  # world +y of the start
    ]
    assert _start_yaw(tmp_path, records) == pytest.approx(0.0)


def test_start_yaw_left_alone_when_already_facing_the_target(tmp_path):
    """With the target the other way, the perceived branch already faces it."""
    records = [
        _record(_START_ID, (0.0, 0.0)),
        _record(_TARGET_ID, (-0.2, 0.0)),  # world -y of the start
    ]
    assert _start_yaw(tmp_path, records) == pytest.approx(_STANDING_YAW)


def test_explicit_start_push_dir_overrides_the_target_default(tmp_path):
    """``start_push_dir_base`` wins over "toward the target".

    The target sits at world +y (which alone would flip the yaw to 0),
    but the scene declares a push along base -x = world -y, which the
    perceived branch already faces -- so no flip.
    """
    records = [
        _record(_START_ID, (0.0, 0.0)),
        _record(_TARGET_ID, (0.2, 0.0)),
    ]
    assert _start_yaw(tmp_path, records,
                      push_dir=[-1.0, 0.0]) == pytest.approx(_STANDING_YAW)


# -- state_from_observation --------------------------------------------------


def test_state_from_observation_maps_capture_id_to_slot(env):
    """An observation names dominoes by capture id, not by slot."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    # Move ONLY capture id 12, which lives in slot 2.
    obs = _StubDominoObservation([_StubDominoPose(12, (0.3, 0.25, 0.03))])
    state = env.state_from_observation(obs, task.init)

    moved = comp.dominos[2]
    assert state.get(moved, "x") == pytest.approx(0.75 - 0.25)
    assert state.get(moved, "y") == pytest.approx(0.72 + 0.3)
    assert state.get(moved, "z") == pytest.approx(0.03 + _Z_OFF)
    # Every other slot is untouched.
    for slot in (0, 1, 3):
        other = comp.dominos[slot]
        for feat in ("x", "y", "z", "yaw"):
            assert state.get(other,
                             feat) == pytest.approx(task.init.get(other, feat))


def test_state_from_observation_leaves_a_held_domino_alone(env):
    """A domino in the gripper keeps the twin's pose, not perception's.

    Perception snaps every domino to a resting pose on the table, so it
    reports a held one lying where it would be if the hand let go.
    Writing that in teleports it out of the gripper -- and
    ``_set_state`` then rebuilds the grasp constraint around the wrong
    offset, leaving the twin holding something that is not there.
    """
    # pylint: disable=protected-access
    task = env._build_task_from_scene()
    comp = env._domino_component
    held = comp.dominos[2]
    carried = task.init.copy()
    carried.set(held, "is_held", 1.0)
    # Lifted well clear of the table, as a carry leaves it.
    carried.set(held, "z", task.init.get(held, "z") + 0.2)

    # Perception insists it is back on the table, at its old spot.
    obs = _StubDominoObservation([_StubDominoPose(12, (0.15, -0.05, 0.03))])
    state = env.state_from_observation(obs, carried)

    for feat in ("x", "y", "z", "yaw", "roll"):
        expected = carried.get(held, feat)
        assert state.get(held, feat) == pytest.approx(expected), \
            f"{feat} of the held domino was overwritten by perception"


def test_state_from_observation_carries_unseen_dominoes_forward(env):
    """Observations carry no visibility flag, so absent means unchanged."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    empty = _StubDominoObservation([])
    state = env.state_from_observation(empty, task.init)
    for slot in range(len(_SCENE_RECORDS)):
        dom = comp.dominos[slot]
        for feat in ("x", "y", "z", "yaw", "roll"):
            assert state.get(dom,
                             feat) == pytest.approx(task.init.get(dom, feat))


def test_state_from_observation_preserves_robot_and_joint_positions(env):
    """The robot's entry -- and the joint positions ``_set_state`` trusts --
    carry forward untouched.

    Without them ``_set_state`` falls back to IK, which drops wrist roll
    and corrupts a recorded grasp.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    robot = env._robot  # pylint: disable=protected-access
    obs = _StubDominoObservation([_StubDominoPose(11, (0.4, 0.1, 0.03))])
    state = env.state_from_observation(obs, task.init)

    assert np.allclose(state[robot], task.init[robot])
    assert isinstance(state, utils.PyBulletState)
    assert state.joint_positions is not None
    assert list(state.joint_positions) == list(task.init.joint_positions)


def test_state_from_observation_ignores_unknown_capture_ids(env):
    """A spurious detection has no slot to write into; dropping it beats
    aborting a run."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    obs = _StubDominoObservation([_StubDominoPose(4242, (0.1, 0.1, 0.03))])
    state = env.state_from_observation(obs, task.init)
    assert state.allclose(task.init)


def test_state_from_observation_canonicalizes_a_standing_start_yaw(env):
    """A look must not turn the opening push around.

    A domino is 180-degree symmetric, so perception returns whichever
    heading branch it likes, and Push takes its entire direction from
    that yaw. Every option boundary before the push is a look, so
    writing the raw branch back is enough to send the start away from
    the target -- which is what a real run did.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    # The task build flipped the start's yaw to 0 (target is at world +y).
    assert task.init.get(comp.dominos[0], "yaw") == pytest.approx(0.0)
    # Re-observing the raw branch keeps the heading the task was built with.
    obs = _StubDominoObservation(
        [_StubDominoPose(_START_ID, (0.0, 0.0, 0.03))])
    state = env.state_from_observation(obs, task.init)
    assert state.get(comp.dominos[0], "yaw") == pytest.approx(0.0)


def test_state_from_observation_leaves_a_toppled_start_yaw_alone(env):
    """Once the start has gone over there is no push left to orient.

    Its heading is then a real observation of which way it fell, and
    flipping it would misreport that -- the reason the correction did
    not canonicalize at all before.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    fallen = float(DominoComponent.fallen_threshold) + 0.2
    obs = _StubDominoObservation([
        _StubDominoPose(_START_ID, (0.0, 0.0, 0.03),
                        _base_quat(roll=fallen, yaw=_STANDING_YAW))
    ])
    state = env.state_from_observation(obs, task.init)
    # The raw branch survives: no flip toward the target.
    assert state.get(comp.dominos[0], "yaw") == pytest.approx(_STANDING_YAW)
    assert abs(state.get(comp.dominos[0], "roll")) >= \
        DominoComponent.fallen_threshold


# -- PyBulletEnv primitives --------------------------------------------------


def test_gripper_joint_layout_matches_the_robot(env):
    """The layout is read off the simulated robot, so the env and the action
    splitter cannot disagree."""
    robot = env._pybullet_robot  # pylint: disable=protected-access
    assert env.gripper_joint_layout() == gripper_joint_layout_from_robot(robot)
    layout = env.gripper_joint_layout()
    assert layout.left_finger_joint_idx != layout.right_finger_joint_idx
    assert layout.open_fingers != layout.closed_fingers


def test_sync_to_state_writes_perceived_poses_into_the_twin(env):
    """A perceived pose reaches the simulator's bodies.

    This is what makes perception visible to the agent at all: the next
    ``env.step`` reads its State back out of these bodies.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    pcid = env._physics_client_id  # pylint: disable=protected-access

    obs = _StubDominoObservation([_StubDominoPose(12, (0.3, 0.25, 0.03))])
    env.sync_to_state(env.state_from_observation(obs, task.init))

    pos, _ = p.getBasePositionAndOrientation(comp.dominos[2].id,
                                             physicsClientId=pcid)
    assert pos[0] == pytest.approx(0.75 - 0.25, abs=1e-4)
    assert pos[1] == pytest.approx(0.72 + 0.3, abs=1e-4)


def test_sync_to_state_zeroes_velocities_on_its_own(env, monkeypatch):
    """``sync_to_state`` clears momentum itself, not by luck.

    ``_set_state`` writes poses through
    ``resetBasePositionAndOrientation``, which does not touch
    velocities, so a body keeps whatever momentum the previous rollout
    gave it and drifts on the next ``stepSimulation``.

    The domino component happens to zero its own blocks at the end of
    ``reset_state``, which would mask a broken ``sync_to_state`` here.
    So that domain-specific hook is stubbed out: what remains is
    ``sync_to_state``'s own contribution, which is the part every future
    PyBullet-backed real environment depends on.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    pcid = env._physics_client_id  # pylint: disable=protected-access
    env.sync_to_state(task.init)

    monkeypatch.setattr(type(env), "_set_domain_specific_state",
                        lambda self, state: None)

    ids = [d.id for d in comp.dominos if d.id is not None]
    assert ids, "expected the component's dominoes to have pybullet bodies"
    for body in ids:
        p.resetBaseVelocity(body, [1.0, 2.0, 3.0], [4.0, 5.0, 6.0],
                            physicsClientId=pcid)
    assert all(
        np.linalg.norm(p.getBaseVelocity(b, physicsClientId=pcid)[0]) > 0
        for b in ids), "failed to spin the bodies up; test proves nothing"

    obs = _StubDominoObservation([_StubDominoPose(12, (0.3, 0.25, 0.03))])
    env.sync_to_state(env.state_from_observation(obs, task.init))

    for body in ids:
        linear, angular = p.getBaseVelocity(body, physicsClientId=pcid)
        assert np.allclose(linear, 0.0), f"body {body} kept linear velocity"
        assert np.allclose(angular, 0.0), f"body {body} kept angular velocity"


# -- knocked-over dominoes ---------------------------------------------------
# The closed loop's whole point is that the twin's guess about which dominoes a
# cascade felled gets corrected by looking. That only works if a toppled pose
# survives the conversion, so these pin the standing/toppled round trip.


@pytest.mark.parametrize("roll", [0.0, np.pi / 2, -np.pi / 2, 0.35])
def test_domino_env_euler_round_trips(roll):
    """A pose built from env angles reads back as those angles, upright or
    knocked over -- so nothing is lost between the cameras and the twin."""
    yaw = 0.6
    pose = pose_base_to_world(
        Pose6D((0.1, 0.2, 0.03), tuple(_base_quat(roll, yaw))), _Z_OFF)
    got_roll, got_pitch, got_yaw = domino_env_euler(pose)
    assert got_roll == pytest.approx(roll, abs=1e-9)
    assert got_pitch == pytest.approx(0.0, abs=1e-9)
    assert got_yaw == pytest.approx(yaw, abs=1e-9)


def test_domino_env_euler_agrees_with_the_standing_helper():
    """On a STANDING domino the general decomposition must reproduce the simple
    upright reading, which is derived independently."""
    for yaw in (0.0, 0.9, -2.1, np.pi):
        pose = pose_base_to_world(
            Pose6D((0.0, 0.0, 0.03), tuple(_base_quat(0.0, yaw))), _Z_OFF)
        assert domino_env_euler(pose)[0] == pytest.approx(0.0, abs=1e-9)
        assert domino_env_euler(pose)[2] == pytest.approx(
            domino_upright_yaw(pose), abs=1e-9)


def test_observed_topple_reaches_the_state_as_toppled(env):
    """A domino the cameras find on its face comes back with the roll that put
    it there -- and ``Toppled``, which is defined on roll, then holds.

    Before this, every perceived domino was written in standing, so the
    twin could never learn that a cascade had happened.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    target = comp.dominos[3]  # capture id 5, the purple target

    # pylint: disable=protected-access
    assert not comp._Toppled_holds(task.init, [target]), \
        "the target starts standing; test proves nothing otherwise"

    obs = _StubDominoObservation([
        _StubDominoPose(_TARGET_ID, (0.2, 0.0, 0.0145),
                        _base_quat(np.pi / 2, yaw=0.0))
    ])
    state = env.state_from_observation(obs, task.init)

    assert abs(state.get(target, "roll")) == pytest.approx(np.pi / 2)
    assert comp._Toppled_holds(state, [target])  # pylint: disable=protected-access


def test_a_standing_observation_stays_upright(env):
    """The counterpart: an upright domino is not spuriously reported as fallen,
    so the roll being written is a reading and not a constant."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    target = comp.dominos[3]

    obs = _StubDominoObservation([
        _StubDominoPose(_TARGET_ID, (0.2, 0.0, 0.03), _base_quat(0.0, yaw=0.0))
    ])
    state = env.state_from_observation(obs, task.init)

    assert state.get(target, "roll") == pytest.approx(0.0, abs=1e-9)
    assert comp._Upright_holds(state, [target])  # pylint: disable=protected-access
    assert not comp._Toppled_holds(state, [target])  # pylint: disable=protected-access


def test_toppled_pose_survives_the_write_into_pybullet(env):
    """The roll must round-trip through the simulator too.

    ``sync_to_state`` writes ``getQuaternionFromEuler([roll, 0, yaw])``
    and the next ``_get_state`` reads the angles back; if those two
    disagreed, the correction would be undone the moment the agent
    looked at the twin.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    target = comp.dominos[3]

    obs = _StubDominoObservation([
        _StubDominoPose(_TARGET_ID, (0.2, 0.0, 0.0145),
                        _base_quat(np.pi / 2, yaw=0.0))
    ])
    env.sync_to_state(env.state_from_observation(obs, task.init))

    read_back = env.get_observation()
    assert abs(read_back.get(target, "roll")) == pytest.approx(np.pi / 2,
                                                               abs=1e-4)
    assert comp._Toppled_holds(read_back, [target])  # pylint: disable=protected-access


def test_a_toppled_start_domino_is_not_yaw_canonicalized(tmp_path):
    """The start flip picks which of two symmetric headings faces the push.

    A domino already lying down has no push to orient, and flipping it
    would misreport which way it fell -- so the canonicalization is
    gated on the domino still standing.
    """
    scene = _write_scene(tmp_path, [
        _record(_START_ID, (0.0, 0.0), quat=_base_quat(np.pi / 2, yaw=0.0)),
        _record(_TARGET_ID, (0.2, 0.0)),
    ],
                         name="toppled_start.json")
    env = _make_env(scene)
    comp = env._domino_component  # pylint: disable=protected-access
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    start = comp.dominos[0]

    assert abs(task.init.get(start, "roll")) == pytest.approx(np.pi / 2)
    # The yaw is the perceived one, not flipped 180 to face the target.
    assert task.init.get(start, "yaw") == pytest.approx(0.0, abs=1e-9)


def test_unrepresentable_pitch_is_reported(env, caplog):
    """A domino propped diagonally has a pitch the (yaw, roll) state cannot
    hold.

    Dropping it silently would put the twin somewhere the cameras never
    saw, so it is logged.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    obs = _StubDominoObservation([
        _StubDominoPose(12, (0.3, 0.25, 0.03),
                        _base_quat(0.0, yaw=0.0, pitch=0.5))
    ])

    with caplog.at_level("WARNING"):
        env.state_from_observation(obs, task.init)

    assert "pitched" in caplog.text


# -- the 180-degree box symmetry ---------------------------------------------
# A domino is a box: turning it 180 degrees about its own width axis leaves it
# exactly where it was, and a marker-based pose estimate returns either
# representative arbitrarily. Roll is therefore only meaningful modulo pi.


@pytest.mark.parametrize("raw_deg, folded_deg", [
    (0.0, 0.0),
    (180.0, 0.0),
    (-180.0, 0.0),
    (5.0, 5.0),
    (-5.0, -5.0),
    (90.0, -90.0),
    (-90.0, -90.0),
    (175.0, -5.0),
])
def test_canonical_roll_folds_modulo_pi(raw_deg, folded_deg):
    """Standing folds to ~0; knocked over keeps its magnitude.

    Magnitude is what matters: ``Toppled`` and ``Upright`` are both
    defined on ``|roll|``, so a fold that flips the sign at +-90 degrees
    is free, while folding 180 -> 0 is the whole point.
    """
    got = _canonical_roll(np.deg2rad(raw_deg))
    assert np.rad2deg(got) == pytest.approx(folded_deg, abs=1e-9)


def test_no_domino_starts_toppled_when_perception_flips_it(tmp_path):
    """A capture where some dominoes come back 180-degree-flipped must still
    build a task whose dominoes are all standing.

    This is the regression. Perception legitimately reports a standing
    domino as ``roll = pi``; unfolded, ``Toppled`` (``|roll| >= 10 deg``)
    then holds for it in the task's own initial state, so a
    ``Toppled(target)`` goal is satisfied before anything moves and the
    planner returns an empty plan and reports success.
    """
    flipped = _base_quat(roll=np.pi)
    scene = _write_scene(tmp_path, [
        _record(_START_ID, (0.0, 0.0)),
        _record(11, (0.1, 0.05), quat=flipped),
        _record(_TARGET_ID, (0.2, 0.0), quat=flipped),
    ],
                         name="flipped.json")
    env = _make_env(scene)
    comp = env._domino_component  # pylint: disable=protected-access
    task = env._build_task_from_scene()  # pylint: disable=protected-access

    for slot in range(3):
        dom = comp.dominos[slot]
        # pylint: disable=protected-access
        assert comp._Upright_holds(task.init, [dom]), \
            f"{dom.name} did not start upright"
        assert not comp._Toppled_holds(task.init, [dom]), \
            f"{dom.name} started toppled"

    # And the goal is therefore not already satisfied by the initial state.
    assert not task.goal.issubset(utils.abstract(task.init, env.predicates))


def test_a_flipped_observation_is_still_upright(env):
    """The same fold applies mid-episode, not just at task construction."""
    task = env._build_task_from_scene()  # pylint: disable=protected-access
    comp = env._domino_component  # pylint: disable=protected-access
    target = comp.dominos[3]

    obs = _StubDominoObservation(
        [_StubDominoPose(_TARGET_ID, (0.2, 0.0, 0.03), _base_quat(np.pi))])
    state = env.state_from_observation(obs, task.init)

    assert state.get(target, "roll") == pytest.approx(0.0, abs=1e-9)
    # pylint: disable=protected-access
    assert comp._Upright_holds(state, [target])
    assert not comp._Toppled_holds(state, [target])


# -- the task evaluator ------------------------------------------------------
def test_evaluator_counts_the_scene_s_movable_dominoes(env):
    """A real task carries an evaluator budgeted by its own scene.

    The count has to come from the scene rather than the min-block flag:
    a real scene stages whatever the person put on the table.
    """
    old_blues = CFG.domino_min_block_num_blues
    # A budget the min-block flag could never satisfy: if the count were
    # taken from it rather than from the scene, DominoEvaluator's own
    # assertion would fire here (0.05 * 25 > 1.0).
    CFG.domino_min_block_num_blues = 25
    try:
        task = env._build_task_from_scene()  # pylint: disable=protected-access
    finally:
        CFG.domino_min_block_num_blues = old_blues

    assert task.evaluator is not None
    assert task.evaluator.goal == task.goal


def test_evaluator_explains_the_scoring_it_introduces(env):
    """The goal text says what the reward does, as the generator's does.

    An agent that is scored but not told how reads a rejected goal-
    reaching attempt as a fatal per-blue penalty.
    """
    task = env._build_task_from_scene()  # pylint: disable=protected-access

    assert task.goal_nl is not None
    assert "Scoring:" in task.goal_nl
    assert "never disqualifies a solve" in task.goal_nl


def test_no_evaluator_when_the_certificate_cannot_judge(env):
    """Targets that are not roll-tracked dominoes get no evaluator.

    Mirrors the generator's own gate: with a separate target type the
    certificate is blind to a direct robot knock on a target, so it
    would certify that at zero cost. Better to score nothing than to
    score it wrongly.
    """
    CFG.domino_use_domino_blocks_as_target = False
    try:
        # pylint: disable-next=protected-access
        assert env._evaluator_for(env._build_task_from_scene().init,
                                  set()) is None
    finally:
        CFG.domino_use_domino_blocks_as_target = True


def test_evaluator_refuses_a_scene_it_cannot_score(env):
    """Too many movables and a success stops outscoring a failure.

    DominoEvaluator asserts this itself, but that would fire mid-episode
    on the real robot; this names the scene's own numbers up front
    instead.
    """
    old_cost = CFG.domino_block_cost
    CFG.domino_block_cost = 0.9  # 2 movables -> 1.8, well over the bar
    try:
        with pytest.raises(ValueError, match="movable dominoes"):
            env._build_task_from_scene()  # pylint: disable=protected-access
    finally:
        CFG.domino_block_cost = old_cost
