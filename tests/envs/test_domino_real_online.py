"""Real-world active learning: the human-gated task rebuild.

The property under test is an *ordering* one, and it is the whole reason this
lands at task-request time rather than in ``env.reset``. The loop is

    env_task = env.get_train_tasks()[i]   # (1)
    cogman.reset(env_task)                # (2)
    run_episode_and_get_observations(...) # (3) calls env.reset(...)

and the task must be rebuilt at (1) because both callers have already
consumed it by (3). On the evaluation path (2) *solves* -- ``_solve_task``
resets with no override policy, so ``_reset_policy`` calls
``approach.solve`` -- and a reset at (3) would plan against a scene that no
longer exists. On the exploration path an override policy is set first, so
(2) does not solve; but the task is what ``env.reset`` initializes the twin
from, so a stale one starts the episode from the captured scene rather
than the one just arranged.

No hardware and no babyrobot: the robot is a fake whose ``reset_env`` records
the prompt and replies with whatever the test wants the cameras to have seen.
These must never skip.
"""
# The env's task caches, component and executor slot are what these tests
# assert on, so reading them is the point.
# pylint: disable=protected-access
import json
from typing import Any, List

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from predicators import utils
from predicators.envs.pybullet_domino.real_geometry import _REAL_TO_ENV_BODY
from predicators.envs.pybullet_domino_real import PyBulletDominoRealEnv
from predicators.pybullet_helpers.real_robot_executor import RealRobotExecutor
from predicators.settings import CFG
from predicators.structs import EnvironmentTask, Object

_TABLE_Z = -0.041
_START_ID = 6
_TARGET_ID = 5


def _base_quat(roll: float = 0.0,
               yaw: float = np.pi,
               pitch: float = 0.0) -> List[float]:
    """The base-frame quaternion of a domino at env ``(roll, pitch, yaw)``."""
    r_env = Rotation.from_euler("xyz", [roll, pitch, yaw]).as_matrix()
    r_base = Rotation.from_euler(
        "z", -np.pi / 2).as_matrix() @ (r_env @ _REAL_TO_ENV_BODY.T)
    return list(Rotation.from_matrix(r_base).as_quat())


class _StubDominoPose:
    """Stands in for babyrobot's DominoPose."""

    def __init__(self, capture_id: int, xyz: Any, quat_xyzw: Any) -> None:
        self.id = capture_id
        self.xyz = tuple(xyz)
        self.quat_xyzw = tuple(quat_xyzw)


class _StubDominoObservation:
    """Stands in for babyrobot's DominoObservation."""

    def __init__(self, dominoes: Any) -> None:
        self.dominoes = list(dominoes)


def _observation(target_base_x: float) -> _StubDominoObservation:
    """A two-domino look with the target at a chosen base-frame x."""
    return _StubDominoObservation([
        _StubDominoPose(_START_ID, (0.0, 0.0, 0.03), _base_quat()),
        _StubDominoPose(_TARGET_ID, (target_base_x, 0.0, 0.03), _base_quat()),
    ])


class _FakeRobot:
    """Records human-reset prompts; replies with queued observations."""

    has_perception = True
    dry = True

    def __init__(self, observations: Any) -> None:
        self.prompts = 0
        self._queue = list(observations)
        self.last_returned: Any = None

    def reset_env(self, req: Any) -> Any:
        """Stand in for home-arm + wait-for-human + look."""
        del req
        self.prompts += 1
        self.last_returned = self._queue.pop(0) if self._queue else None
        return self.last_returned


def _config(scene_path: str, **overrides: Any) -> None:
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
        "real_robot_observe_at_option_boundary": False,
        "real_robot_human_reset": True,
    }
    flags.update(overrides)
    utils.reset_config(flags)


@pytest.fixture(scope="module", name="scene_path")
def scene_path_fixture(tmp_path_factory: Any) -> str:
    """A captured scene with the target 20cm along base +x."""
    records = [{
        "id": i,
        "center_base_m": [x, 0.0, 0.03],
        "quat_base_xyzw": _base_quat(),
        "dims_m": [0.15, 0.07, 0.029],
    } for i, x in ((_START_ID, 0.0), (_TARGET_ID, 0.2))]
    path = tmp_path_factory.mktemp("domino_online") / "scene.json"
    path.write_text(json.dumps({
        "frame": "robot_base",
        "units": "m",
        "dominoes": records
    }),
                    encoding="utf-8")
    return str(path)


@pytest.fixture(scope="module", name="env")
def env_fixture(scene_path: str) -> PyBulletDominoRealEnv:
    """One real twin for the module -- building PyBullet per test is slow."""
    _config(scene_path)
    return PyBulletDominoRealEnv(use_gui=False)


@pytest.fixture(autouse=True)
def _clean_env(env: PyBulletDominoRealEnv, scene_path: str) -> Any:
    """Give each test a pristine shared env.

    The env is module-scoped because building PyBullet is slow, so its
    executor and its cached tasks -- the very things under test -- have
    to be cleared between tests.
    """
    _config(scene_path)
    env._executor = None
    env._train_tasks = []
    env._test_tasks = []
    yield
    env._executor = None
    env._train_tasks = []
    env._test_tasks = []


@pytest.fixture(name="attached")
def attached_fixture(env: PyBulletDominoRealEnv, monkeypatch: Any) -> Any:
    """Attach an executor whose robot traffic is recorded, not performed."""
    # reset_arm and execute_chunks would import babyrobot. The human reset is
    # what these tests are about, so it is the only one that does anything.
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_arm",
        lambda robot, joints: tuple(joints))
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.execute_chunks",
        lambda *a, **k: [])
    monkeypatch.setattr(
        "predicators.pybullet_helpers.real_robot_executor.reset_env",
        lambda robot, joints=None: robot.reset_env(None))

    def _attach(robot: Any, human_reset: bool = True) -> RealRobotExecutor:
        executor = RealRobotExecutor(env,
                                     robot,
                                     observe_at_boundaries=False,
                                     human_reset=human_reset)
        env.attach_executor(executor)
        return executor

    return _attach


def _target(env: PyBulletDominoRealEnv) -> Object:
    """The purple target domino, which sits in slot 1 of this scene."""
    comp = env._domino_component
    assert comp is not None, "env has no domino component"
    return comp.dominos[1]


# -- the ordering property ---------------------------------------------------
def test_the_task_is_rebuilt_before_the_approach_would_solve(
        env: PyBulletDominoRealEnv, attached: Any) -> None:
    """``get_train_tasks`` blocks for the human and returns the REBUILT task.

    That is the ordering fix: the caller solves against what it gets
    back here, so what it gets back must already reflect the new scene.
    """
    robot = _FakeRobot([_observation(target_base_x=0.30)])
    attached(robot)

    tasks = env.get_train_tasks()

    assert robot.prompts == 1, \
        "the human was not asked before the task was given"
    assert len(tasks) == 1
    # base_x 0.30 -> world y = 0.72 + 0.30; the captured scene had 0.2.
    assert tasks[0].init.get(_target(env), "y") == pytest.approx(0.72 + 0.30)


def test_each_episode_prompts_exactly_once(env: PyBulletDominoRealEnv,
                                           attached: Any) -> None:
    """One prompt per episode: asking again inside an episode must not re-
    prompt, and the next episode must."""
    robot = _FakeRobot([_observation(0.30), _observation(0.25)])
    attached(robot)

    env.get_train_tasks()
    assert robot.prompts == 1
    # Same episode: the task is already built, so no second prompt.
    env.get_train_tasks()
    env.get_train_tasks()
    assert robot.prompts == 1

    # An episode begins; that is what owes the next reset.
    env.reset("train", 0)
    env.get_train_tasks()
    assert robot.prompts == 2


def test_a_new_look_replaces_the_cached_task(env: PyBulletDominoRealEnv,
                                             attached: Any) -> None:
    """BaseEnv caches tasks, so without this every episode would replan against
    one frozen capture.

    The second episode must see the second look.
    """
    robot = _FakeRobot([_observation(0.30), _observation(0.10)])
    attached(robot)
    target = _target(env)

    first = env.get_train_tasks()[0]
    env.reset("train", 0)
    second = env.get_train_tasks()[0]

    assert first.init.get(target, "y") == pytest.approx(0.72 + 0.30)
    assert second.init.get(target, "y") == pytest.approx(0.72 + 0.10)


def test_test_split_is_rebuilt_too(env: PyBulletDominoRealEnv,
                                   attached: Any) -> None:
    """Evaluation episodes face a freshly arranged scene as well."""
    robot = _FakeRobot([_observation(0.28)])
    attached(robot)

    tasks = env.get_test_tasks()

    assert robot.prompts == 1
    assert isinstance(tasks[0], EnvironmentTask)
    assert tasks[0].init.get(_target(env), "y") == pytest.approx(0.72 + 0.28)


def test_no_human_reset_keeps_the_captured_scene(env: PyBulletDominoRealEnv,
                                                 attached: Any) -> None:
    """With human resets off nothing is prompted and nothing is rebuilt, which
    is what a fixed-plan replay depends on."""
    robot = _FakeRobot([_observation(0.30)])
    attached(robot, human_reset=False)
    # A replay opts in to those exact poses, as replay_plan does; without it
    # the stale-task guard refuses to plan against a snapshot while the
    # cameras are live.
    CFG.real_robot_allow_captured_scene_task = True

    tasks = env.get_train_tasks()

    assert robot.prompts == 0
    # The captured scene put the target at base_x 0.2.
    assert tasks[0].init.get(_target(env), "y") == pytest.approx(0.72 + 0.2)


def test_an_unattached_env_never_prompts(env: PyBulletDominoRealEnv) -> None:
    """A pure-sim run -- including every env the planner builds -- must not
    acquire a human in the loop."""
    tasks = env.get_train_tasks()

    assert tasks[0].init.get(_target(env), "y") == pytest.approx(0.72 + 0.2)
