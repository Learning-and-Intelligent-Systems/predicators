"""Offline end-to-end: the real-world active-learning loop, no hardware.

Everything here is real except the arm and the cameras: a genuine
``RealRobot(dry=True)`` with mock perception and an auto-confirming reset, the
real ``RealRobotExecutor``, the real ``CogMan``, and the real
``run_episode_and_get_observations``. Only the arm does not move and the
cameras are a stub.

It reproduces the exact three-call sequence ``main.py`` uses per interaction
request (``main.py:551-565``)::

    cogman.set_override_policy(request.act_policy)
    env_task = env.get_train_tasks()[request.train_task_idx]
    cogman.reset(env_task)
    run_episode_and_get_observations(cogman, env, "train", ...)

rather than invoking ``main()`` itself, which would add argument parsing,
result directories and an LLM-backed approach without testing anything more:
the ordering property lives entirely in those four lines. This is the only
test that exercises that ordering against real babyrobot objects, so it skips
without the private submodule.
"""
# The env's caches and the domino component are what this asserts on,
# and babyrobot is imported inside the test body because it is
# optional and absent on CI.
# pylint: disable=protected-access,import-outside-toplevel,import-error
import json
from typing import Any, List, cast

import numpy as np
import pytest
from gym.spaces import Box
from scipy.spatial.transform import Rotation

from predicators import utils
from predicators.approaches.base_approach import BaseApproach
from predicators.cogman import CogMan, run_episode_and_get_observations
from predicators.envs.pybullet_domino.real_geometry import _REAL_TO_ENV_BODY
from predicators.envs.pybullet_domino_real import PyBulletDominoRealEnv
from predicators.execution_monitoring import create_execution_monitor
from predicators.perception import create_perceiver
from predicators.pybullet_helpers.real_robot_executor import RealRobotExecutor
from predicators.structs import Action, ParameterizedOption

_TABLE_Z = -0.041
_START_ID = 6
_TARGET_ID = 5
_EPISODES = 2  # "two online-learning cycles"


def _base_quat(roll: float = 0.0, yaw: float = np.pi) -> List[float]:
    """The base-frame quaternion of a domino at env ``(roll, 0, yaw)``."""
    r_env = Rotation.from_euler("xyz", [roll, 0.0, yaw]).as_matrix()
    r_base = Rotation.from_euler(
        "z", -np.pi / 2).as_matrix() @ (r_env @ _REAL_TO_ENV_BODY.T)
    return list(Rotation.from_matrix(r_base).as_quat())


@pytest.fixture(name="scene_path")
def scene_path_fixture(tmp_path: Any) -> str:
    """A two-domino captured scene."""
    records = [{
        "id": i,
        "center_base_m": [x, 0.0, 0.03],
        "quat_base_xyzw": _base_quat(),
        "dims_m": [0.15, 0.07, 0.029],
    } for i, x in ((_START_ID, 0.0), (_TARGET_ID, 0.2))]
    path = tmp_path / "scene.json"
    path.write_text(json.dumps({
        "frame": "robot_base",
        "units": "m",
        "dominoes": records
    }),
                    encoding="utf-8")
    return str(path)


def _config(scene_path: str) -> None:
    """The closed-loop real config, with the arm and cameras faked."""
    utils.reset_config({
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
        "real_robot_dry": True,
        "real_robot_observe_at_option_boundary": True,
        "real_robot_human_reset": True,
        "real_robot_settle_s": 0.0,
        "horizon": 6,
    })


def _two_option_policy(env: PyBulletDominoRealEnv) -> Any:
    """A policy of two one-action options, one opening and one closing.

    Enough to produce two option boundaries per episode -- so two looks
    -- and both gripper states, so the commands reaching the hand can be
    checked for well-formedness.
    """
    layout = env.gripper_joint_layout()
    step = {"n": 0}

    def _action(fingers: float) -> Action:
        arr = np.zeros(env.action_space.shape, dtype=np.float32)
        arr[layout.left_finger_joint_idx] = fingers
        arr[layout.right_finger_joint_idx] = fingers
        return Action(arr)

    def policy(_state: Any) -> Action:
        fingers = (layout.open_fingers
                   if step["n"] == 0 else layout.closed_fingers)
        action = _action(fingers)
        param_opt = ParameterizedOption(f"Stub{step['n']}", [],
                                        Box(0, 1,
                                            (1, )), lambda s, m, o, p: action,
                                        lambda s, m, o, p: True,
                                        lambda s, m, o, p: False)
        option = param_opt.ground([], np.array([0.5], dtype=np.float32))
        option.terminal = lambda _obs: True  # every action ends its option
        action.set_option(option)
        step["n"] += 1
        return action

    return policy


def test_offline_end_to_end_active_learning(scene_path: str) -> None:
    """One human prompt per episode, one look per option boundary, and only
    well-formed gripper commands reaching the hand."""
    pytest.importorskip("babyrobot")
    from babyrobot.realrobot.observations.domino import DominoObservation, \
        DominoPose
    from babyrobot.realrobot.perception import MockDominoPerception
    from babyrobot.realrobot.real_robot import RealRobot

    _config(scene_path)
    env = PyBulletDominoRealEnv(use_gui=False)

    prompts = {"n": 0}

    def _auto_confirm() -> None:
        """Stand in for the human at the scene."""
        prompts["n"] += 1

    seen = DominoObservation(
        stamp=0.0,
        dominoes=(DominoPose(id=_START_ID,
                             xyz=(0.0, 0.0, 0.03),
                             quat_xyzw=tuple(_base_quat())),
                  DominoPose(id=_TARGET_ID,
                             xyz=(0.22, 0.0, 0.03),
                             quat_xyzw=tuple(_base_quat()))))

    class _CountingPerception(MockDominoPerception):
        """Mock perception that records how often it was asked to look."""

        def __init__(self) -> None:
            super().__init__(seen)
            self.looks = 0

        def observe(self, settle_s: float = 0.0) -> Any:
            """Record the look, then report the fixed scene."""
            self.looks += 1
            return super().observe(settle_s)

    perception = _CountingPerception()
    robot = RealRobot(perception=perception,
                      dry=True,
                      confirm_reset=_auto_confirm)
    try:
        executor = RealRobotExecutor(env,
                                     robot,
                                     observe_at_boundaries=True,
                                     settle_s=0.0,
                                     human_reset=True)
        env.attach_executor(executor)
        # Cast: CogMan is typed against BaseApproach, but with an override
        # policy set it only ever calls the three methods the stub has.
        cogman = CogMan(cast(BaseApproach, _StubApproach()),
                        create_perceiver("trivial"),
                        create_execution_monitor("trivial"))

        for _ in range(_EPISODES):
            cogman.set_override_policy(_two_option_policy(env))
            cogman.set_termination_function(lambda s: False)
            env_task = env.get_train_tasks()[0]
            cogman.reset(env_task)
            run_episode_and_get_observations(cogman,
                                             env,
                                             "train",
                                             0,
                                             max_num_steps=2,
                                             terminate_on_goal_reached=False)

        # One human reset per episode -- not zero (never asked) and not two
        # (asked again mid-episode, which the caching defeat could cause).
        assert prompts["n"] == _EPISODES
        assert executor.resets_done == _EPISODES
        # Each episode: one look for the reset, plus one per option boundary.
        assert perception.looks == _EPISODES * 3
        # And the hand only ever saw well-formed commands.
        assert robot.last_gripper_command in ("open", "close")
    finally:
        robot.close()


class _StubApproach:
    """The narrowest thing CogMan will accept.

    An override policy is always set, so the approach is never asked to
    solve; it exists because ``CogMan.reset`` calls these three methods.
    """

    @property
    def is_learning_based(self) -> bool:
        """Unused here."""
        return False

    @classmethod
    def get_name(cls) -> str:
        """Read by CogMan when deciding whether to render observations."""
        return "stub"

    def reset_for_new_episode(self) -> None:
        """Nothing to reset."""

    def get_execution_monitoring_info(self) -> List[Any]:
        """No info to hand the monitor."""
        return []
