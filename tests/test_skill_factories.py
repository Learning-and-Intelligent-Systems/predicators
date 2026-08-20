"""Tests for predicators/ground_truth_models/skill_factories/.

Covers: SkillConfig, Phase, PhaseSkill, create_wait_option,
        make_move_to_phase, create_move_to_skill,
        create_pick_skill, create_place_skill, create_push_skill.
"""
import numpy as np
import pybullet as p
import pytest
from gym.spaces import Box

from predicators import utils
from predicators.ground_truth_models.skill_factories.base import \
    _BIRRT_STEP_KEY, _BIRRT_TRAJ_KEY, Phase, PhaseAction, PhaseSkill, \
    SkillConfig, _fmt_option_params
from predicators.ground_truth_models.skill_factories.move_to import \
    create_move_to_skill, make_move_to_phase
from predicators.ground_truth_models.skill_factories.pick import \
    create_pick_skill
from predicators.ground_truth_models.skill_factories.place import \
    create_place_skill
from predicators.ground_truth_models.skill_factories.push import \
    create_push_skill, resolve_ee_yaw_offset
from predicators.ground_truth_models.skill_factories.wait import \
    create_wait_option, note_external_state_change
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.inverse_kinematics import \
    InverseKinematicsError
from predicators.pybullet_helpers.robots import \
    create_single_arm_pybullet_robot
from predicators.structs import Action, Object, ParameterizedOption, Type

# ---------------------------------------------------------------------------
# Type definitions reused across tests
# ---------------------------------------------------------------------------
_ROBOT_TYPE = Type("robot", ["x", "y", "z", "tilt", "wrist", "fingers"])
_OBJ_TYPE = Type("obj", ["x", "y", "z"])

# Finger state values matching PyBulletEnv class-var conventions for Fetch.
_OPEN_STATE = 0.04  # open_fingers feature value
_CLOSED_STATE = 0.01  # closed_fingers feature value

# EE pose used to home the Fetch robot.
_EE_HOME = (1.35, 0.75, 0.75)
_EE_HOME_ORN = None  # computed on first use


def _get_ee_home_pose() -> Pose:
    orn = p.getQuaternionFromEuler([0.0, np.pi / 2, -np.pi])
    return Pose(_EE_HOME, orn)


# ---------------------------------------------------------------------------
# Module-scoped fixture: create a Fetch robot exactly once for all tests.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", name="robot_scene")
def _setup_robot_scene():
    """Connect to PyBullet DIRECT, create a Fetch robot, yield both."""
    utils.reset_config({"seed": 123})
    physics_client_id = p.connect(p.DIRECT)
    robot = create_single_arm_pybullet_robot("fetch", physics_client_id,
                                             _get_ee_home_pose())
    yield physics_client_id, robot
    p.disconnect(physics_client_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fingers_state_to_joint(robot, finger_state: float) -> float:
    """Nearest open/closed joint value — mirrors
    PyBulletEnv._fingers_state_to_joint."""
    open_j = robot.open_fingers
    closed_j = robot.closed_fingers
    if abs(finger_state - open_j) <= abs(finger_state - closed_j):
        return open_j
    return closed_j


def _make_config(robot) -> SkillConfig:
    return SkillConfig(
        robot=robot,
        open_fingers_joint=robot.open_fingers,
        closed_fingers_joint=robot.closed_fingers,
        fingers_state_to_joint=_fingers_state_to_joint,
    )


def _make_robot_obj() -> Object:
    return Object("robot0", _ROBOT_TYPE)


def _make_obj() -> Object:
    return Object("obj0", _OBJ_TYPE)


def _build_state(
    robot_obj: Object,
    robot,
    ee_x: float,
    ee_y: float,
    ee_z: float,
    finger_state: float = _OPEN_STATE,
    obj: Object = None,  # type: ignore[assignment]
    obj_xyz=(0.0, 0.0, 0.0),
) -> utils.PyBulletState:
    """Build a PyBulletState at the specified EE position.

    Uses the robot's initial joint positions as the simulator state.
    When the EE position equals the home position, the state is fully
    self-consistent (joint positions match the EE pose in state
    features).
    """
    tilt = np.pi / 2
    wrist = -np.pi
    data = {
        robot_obj:
        np.array([ee_x, ee_y, ee_z, tilt, wrist, finger_state],
                 dtype=np.float32)
    }
    if obj is not None:
        data[obj] = np.array(obj_xyz, dtype=np.float32)
    joint_positions = list(robot.initial_joint_positions)
    return utils.PyBulletState(data, simulator_state=joint_positions)


def _make_home_state(
    robot_obj: Object,
    robot,
    finger_state: float = _OPEN_STATE,
    obj: Object = None,  # type: ignore[assignment]
    obj_xyz=(0.0, 0.0, 0.0),
) -> utils.PyBulletState:
    """Build a fully self-consistent PyBulletState at the robot's home pose.

    Resets the robot to its cached initial joint positions and reads the
    actual EE state from PyBullet, so that joint_positions and EE
    features are always mutually consistent regardless of prior robot
    manipulation.
    """
    robot.set_joints(robot.initial_joint_positions)
    raw = robot.get_state()  # [rx, ry, rz, qx, qy, qz, qw, rf]
    rx, ry, rz, qx, qy, qz, qw, _ = raw
    tilt_val = p.getEulerFromQuaternion([qx, qy, qz, qw])[1]
    wrist_val = p.getEulerFromQuaternion([qx, qy, qz, qw])[2]
    data = {
        robot_obj:
        np.array([rx, ry, rz, tilt_val, wrist_val, finger_state],
                 dtype=np.float32)
    }
    if obj is not None:
        data[obj] = np.array(obj_xyz, dtype=np.float32)
    return utils.PyBulletState(data,
                               simulator_state=list(
                                   robot.initial_joint_positions))


# ===========================================================================
# 1. SkillConfig
# ===========================================================================


class TestSkillConfig:
    """TestSkillConfig class."""

    def test_required_fields_stored(self, robot_scene):
        """Test required fields stored."""
        _, robot = robot_scene
        cfg = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
        )
        assert cfg.robot is robot
        assert cfg.open_fingers_joint == robot.open_fingers
        assert cfg.closed_fingers_joint == robot.closed_fingers

    def test_default_tolerances(self, robot_scene):
        """Test default tolerances."""
        _, robot = robot_scene
        cfg = _make_config(robot)
        assert cfg.move_to_pose_tol == pytest.approx(1e-4)
        assert cfg.max_vel_norm == pytest.approx(0.05)
        assert cfg.grasp_tol == pytest.approx(5e-4)
        assert cfg.collision_bodies == ()
        assert cfg.ik_validate is True
        assert cfg.robot_init_tilt == pytest.approx(0.0)
        assert cfg.robot_init_wrist == pytest.approx(0.0)

    def test_extra_dict_stored(self, robot_scene):
        """Test extra dict stored."""
        _, robot = robot_scene
        cfg = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            extra={"my_key": 42},
        )
        assert cfg.extra["my_key"] == 42

    def test_custom_tolerances(self, robot_scene):
        """Test custom tolerances."""
        _, robot = robot_scene
        cfg = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            move_to_pose_tol=5e-5,
            max_vel_norm=0.02,
            grasp_tol=2e-3,
        )
        assert cfg.move_to_pose_tol == pytest.approx(5e-5)
        assert cfg.max_vel_norm == pytest.approx(0.02)
        assert cfg.grasp_tol == pytest.approx(2e-3)


# ===========================================================================
# 2. Phase dataclass
# ===========================================================================


class TestPhase:
    """TestPhase class."""

    def test_move_to_pose_phase(self):
        """Test move to pose phase."""

        def dummy_target(_state, _objects, _params, _cfg):
            return None, None, "open"

        phase = Phase(name="TestMove",
                      action_type=PhaseAction.MOVE_TO_POSE,
                      target_fn=dummy_target)
        assert phase.name == "TestMove"
        assert phase.action_type == PhaseAction.MOVE_TO_POSE
        assert phase.terminal_fn is None
        assert phase.use_motion_planning is False  # default from CFG
        assert not phase.allow_shallow_held_object_contacts

    def test_change_fingers_phase(self):
        """Test change fingers phase."""

        def dummy_target(_state, _objects, _params, _cfg):
            return 0.04, 0.01

        phase = Phase(name="Grasp",
                      action_type=PhaseAction.CHANGE_FINGERS,
                      target_fn=dummy_target)
        assert phase.action_type == PhaseAction.CHANGE_FINGERS

    def test_custom_terminal_fn_stored(self):
        """Test custom terminal fn stored."""

        def my_terminal(_state, _objects, _params, _cfg):
            return True

        phase = Phase(
            name="CustomPhase",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.0, 0.0),
            terminal_fn=my_terminal,
        )
        assert phase.terminal_fn is my_terminal

    def test_no_motion_planning_flag(self):
        """Test no motion planning flag."""
        phase = Phase(
            name="IKMove",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=lambda s, o, p_, c: (None, None, "open"),
            use_motion_planning=False,
        )
        assert phase.use_motion_planning is False

    def test_move_to_phase_collision_metadata(self):
        """Test move-to phase stores collision metadata."""

        def dummy_pose(_state, _objects, _params, _cfg):
            return 0.0, 0.0, 0.0, 0.0

        phase = make_move_to_phase(
            "Move",
            dummy_pose,
            allow_shallow_held_object_contacts=True,
        )

        assert phase.allow_shallow_held_object_contacts


# ===========================================================================
# 3. PhaseSkill — structure and public-interface behaviour
# ===========================================================================


class TestPhaseSkill:
    """TestPhaseSkill class."""

    def _make_single_ik_skill(self, robot, target_pos):
        """One IK-mode MOVE_TO_POSE phase (no BiRRT, predictable terminal)."""
        config = _make_config(robot)
        robot_obj = _make_robot_obj()

        def target_fn(state, _objects, _params, cfg):
            x = state.get(robot_obj, "x")
            y = state.get(robot_obj, "y")
            z = state.get(robot_obj, "z")
            tilt = state.get(robot_obj, "tilt")
            wrist = state.get(robot_obj, "wrist")
            orn = p.getQuaternionFromEuler([0, tilt, wrist])
            current = Pose((x, y, z), orn)
            orn_tgt = p.getQuaternionFromEuler([0, cfg.robot_init_tilt, 0.0])
            target = Pose(target_pos, orn_tgt)
            return current, target, "open"

        phase = Phase(
            name="Move",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=target_fn,
            use_motion_planning=False,
        )
        skill = PhaseSkill("Test", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        return skill, robot_obj, phase

    def _make_single_cf_skill(self, robot, current_val, target_val):
        """One CHANGE_FINGERS phase with fixed current/target."""
        config = _make_config(robot)

        def target_fn(_state, _objects, _params, _cfg):
            return current_val, target_val

        phase = Phase(
            name="CF",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=target_fn,
        )
        skill = PhaseSkill("TestCF", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        return skill, phase

    def test_build_returns_parameterized_option(self, robot_scene):
        """Test build returns parameterized option."""
        _, robot = robot_scene
        skill, _robot_obj, _ = self._make_single_ik_skill(robot, _EE_HOME)
        opt = skill.build()
        assert isinstance(opt, ParameterizedOption)

    def test_build_name_and_types(self, robot_scene):
        """Test build name and types."""
        _, robot = robot_scene
        skill, _robot_obj, _ = self._make_single_ik_skill(robot, _EE_HOME)
        opt = skill.build()
        assert opt.name == "Test"
        assert opt.types == [_ROBOT_TYPE]

    def test_initiable_sets_phase_idx_zero(self, robot_scene):
        """Test initiable sets phase idx zero."""
        _, robot = robot_scene
        skill, _robot_obj, _ = self._make_single_ik_skill(robot, _EE_HOME)
        opt = skill.build()
        grounded = opt.ground([_make_robot_obj()], np.zeros(0))
        state = _build_state(_make_robot_obj(), robot, *_EE_HOME)
        assert grounded.initiable(state)
        assert grounded.memory["phase_idx"] == 0

    def test_change_fingers_terminal_when_at_target(self, robot_scene):
        """Test change fingers terminal when at target."""
        _, robot = robot_scene
        # current == target → (target-current)^2 = 0 < grasp_tol
        skill, _ = self._make_single_cf_skill(robot, 0.04, 0.04)
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert grounded.terminal(state)

    def test_change_fingers_not_terminal_when_far(self, robot_scene):
        """Test change fingers not terminal when far."""
        _, robot = robot_scene
        # current=0.04, target=0.00 → (0.00-0.04)^2 = 1.6e-3 > 1e-3
        skill, _ = self._make_single_cf_skill(robot, 0.04, 0.00)
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert not grounded.terminal(state)

    def test_ik_terminal_when_at_target(self, robot_scene):
        """Test ik terminal when at target."""
        _, robot = robot_scene
        # Target == current EE position → distance = 0 < tol
        skill, robot_obj, _ = self._make_single_ik_skill(robot, _EE_HOME)
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert grounded.terminal(state)

    def test_ik_not_terminal_when_far(self, robot_scene):
        """Test ik not terminal when far."""
        _, robot = robot_scene
        # Target is far from current EE (0.3m away in z)
        far_target = (_EE_HOME[0], _EE_HOME[1], _EE_HOME[2] - 0.3)
        skill, robot_obj, _ = self._make_single_ik_skill(robot, far_target)
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert not grounded.terminal(state)

    def test_multi_phase_terminal_only_on_last(self, robot_scene):
        """With 2 phases, terminal is False even when phase 0 would be
        terminal."""
        _, robot = robot_scene
        config = _make_config(robot)

        # Phase 0: CHANGE_FINGERS, immediately terminal (current==target).
        phase0 = Phase(
            name="CF",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.04, 0.04),
        )
        # Phase 1: CHANGE_FINGERS, NOT terminal (current 0.04, target 0.00).
        phase1 = Phase(
            name="CF2",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.04, 0.00),
        )
        skill = PhaseSkill("TwoPhase", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase0, phase1])
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        # Phase 0 is terminal, but we're on phase 0 of 2 → overall not terminal.
        assert not grounded.terminal(state)

    def test_policy_advances_phase_when_terminal(self, robot_scene):
        """Calling policy when phase is terminal bumps phase_idx."""
        _, robot = robot_scene
        config = _make_config(robot)

        # Phase 0: immediately terminal.
        phase0 = Phase(
            name="CF0",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.04, 0.04),
        )
        # Phase 1: not terminal.
        phase1 = Phase(
            name="CF1",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c: (0.04, 0.00),
        )
        skill = PhaseSkill("Advance", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase0, phase1])
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE)
        grounded.initiable(state)
        assert grounded.memory["phase_idx"] == 0
        # Phase 0 is terminal; policy should advance to phase 1.
        grounded.policy(state)
        assert grounded.memory["phase_idx"] == 1

    def test_custom_terminal_fn_overrides_default(self, robot_scene):
        """A custom terminal_fn takes precedence over distance-based
        terminal."""
        _, robot = robot_scene
        config = _make_config(robot)
        call_count = {"n": 0}

        def my_terminal(_state, _objects, _params, _cfg):
            call_count["n"] += 1
            return True

        phase = Phase(
            name="Custom",
            action_type=PhaseAction.CHANGE_FINGERS,
            target_fn=lambda s, o, p_, c:
            (0.04, 0.00),  # would not be terminal
            terminal_fn=my_terminal,
        )
        skill = PhaseSkill("CustomTerm", [_ROBOT_TYPE], Box(0, 1, (0, )),
                           config, [phase])
        opt = skill.build()
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        assert grounded.terminal(state)  # custom fn returns True
        assert call_count["n"] >= 1


# ===========================================================================
# 4. BiRRT trajectory caching and IK fallback
# ===========================================================================


class TestBiRRT:
    """Integration tests requiring a real PyBullet robot."""

    def test_birrt_not_terminal_before_first_policy_call(self, robot_scene):
        """With BiRRT mode, terminal is False until the first policy call."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()

        def target_fn(_state, _objects, _params, _cfg):
            orn = p.getQuaternionFromEuler([0, 0, 0])
            return Pose(_EE_HOME, orn), Pose(_EE_HOME, orn), "open"

        phase = Phase("Move",
                      PhaseAction.MOVE_TO_POSE,
                      target_fn,
                      use_motion_planning=True)
        skill = PhaseSkill("BT", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        opt = skill.build()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        grounded.initiable(state)
        # No trajectory in memory yet → NOT terminal.
        assert not grounded.terminal(state)

    def test_birrt_caches_trajectory_after_first_policy_call(
            self, robot_scene):
        """After the first policy call, a trajectory is cached in memory."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        home_orn = p.getQuaternionFromEuler([0, np.pi / 2, -np.pi])

        def target_fn(state, _objects, _params, _cfg):
            current_orn = p.getQuaternionFromEuler([
                0,
                state.get(robot_obj, "tilt"),
                state.get(robot_obj, "wrist"),
            ])
            current = Pose(
                (state.get(robot_obj, "x"), state.get(
                    robot_obj, "y"), state.get(robot_obj, "z")),
                current_orn,
            )
            # Target = home, same as current, so BiRRT trivially succeeds.
            target = Pose(_EE_HOME, home_orn)
            return current, target, "open"

        phase = Phase("Move",
                      PhaseAction.MOVE_TO_POSE,
                      target_fn,
                      use_motion_planning=True)
        skill = PhaseSkill("BT", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        opt = skill.build()
        grounded = opt.ground([robot_obj], np.zeros(0))

        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)

        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert action.arr.shape == robot.action_space.shape

        # Trajectory should now be cached.
        traj_key = _BIRRT_TRAJ_KEY.format(id(phase))
        assert traj_key in grounded.memory

    def test_birrt_terminal_after_trajectory_exhausted(self, robot_scene):
        """Terminal becomes True once all trajectory waypoints are consumed."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        home_orn = p.getQuaternionFromEuler([0, np.pi / 2, -np.pi])

        def target_fn(state, _objects, _params, _cfg):
            current_orn = p.getQuaternionFromEuler([
                0,
                state.get(robot_obj, "tilt"),
                state.get(robot_obj, "wrist"),
            ])
            current = Pose(
                (state.get(robot_obj, "x"), state.get(
                    robot_obj, "y"), state.get(robot_obj, "z")),
                current_orn,
            )
            # Same-position target → BiRRT path is short (a few waypoints).
            return current, Pose(_EE_HOME, home_orn), "open"

        phase = Phase("Move",
                      PhaseAction.MOVE_TO_POSE,
                      target_fn,
                      use_motion_planning=True)
        skill = PhaseSkill("BT", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        opt = skill.build()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)

        assert not grounded.terminal(state)  # no traj yet

        # Consume waypoints by calling policy until terminal.
        # Path length varies with IK rounding; 50 steps is more than enough.
        for _ in range(50):
            if grounded.terminal(state):
                break
            grounded.policy(state)
        else:
            pytest.fail(
                "BiRRT terminal never became True after 50 policy calls")

        assert grounded.terminal(state)

    def test_birrt_fallback_to_ik_when_traj_is_none(self, robot_scene):
        """When memory[traj_key]=None (BiRRT failure), policy uses IK fallback.

        We inject the failure directly into memory rather than depending
        on BiRRT actually failing, which is non-deterministic with
        limited budgets and no collision obstacles.
        """
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        home_orn = p.getQuaternionFromEuler([0, np.pi / 2, -np.pi])
        target_pos = (_EE_HOME[0], _EE_HOME[1], _EE_HOME[2] - 0.15)

        def target_fn(state, _objects, _params, _cfg):
            current_orn = p.getQuaternionFromEuler([
                0,
                state.get(robot_obj, "tilt"),
                state.get(robot_obj, "wrist"),
            ])
            current = Pose((state.get(robot_obj, "x"), state.get(
                robot_obj, "y"), state.get(robot_obj, "z")), current_orn)
            target = Pose(target_pos, home_orn)
            return current, target, "open"

        phase = Phase("Move",
                      PhaseAction.MOVE_TO_POSE,
                      target_fn,
                      use_motion_planning=True)
        skill = PhaseSkill("FB", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        opt = skill.build()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)

        # Simulate BiRRT failure: set traj = None in memory.
        traj_key = _BIRRT_TRAJ_KEY.format(id(phase))
        step_key = _BIRRT_STEP_KEY.format(id(phase))
        grounded.memory[traj_key] = None
        grounded.memory[step_key] = 0

        # Policy must not raise — IK fallback is activated.
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)

        # Fallback terminal is distance-based: target 0.15m away → not terminal.
        assert not grounded.terminal(state)


# ===========================================================================
# 5. Wait option
# ===========================================================================


class TestWaitOption:
    """TestWaitOption class."""

    def test_wait_always_initiable(self, robot_scene):
        """Test wait always initiable."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        assert grounded.initiable(state)

    def test_wait_never_terminal(self, robot_scene):
        """Test wait never terminal."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        for _ in range(5):
            assert not grounded.terminal(state)

    def test_wait_quiescence_terminates_when_scene_settles(self, robot_scene):
        """With wait_quiescence_eps set, Wait terminates after the non-robot
        scene stops moving for wait_quiescence_steps consecutive steps."""
        from dataclasses import \
            replace  # pylint: disable=import-outside-toplevel
        _, robot = robot_scene
        config = replace(_make_config(robot),
                         wait_quiescence_eps=1e-4,
                         wait_quiescence_steps=3)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        block = Object("block0", _OBJ_TYPE)

        def state_with_block_x(x):
            return _build_state(robot_obj,
                                robot,
                                *_EE_HOME,
                                obj=block,
                                obj_xyz=(x, 0.0, 0.0))

        grounded = opt.ground([robot_obj], np.zeros(0))
        assert grounded.initiable(state_with_block_x(0.5))
        # Block moving: never terminal, count resets.
        assert not grounded.terminal(state_with_block_x(0.5))
        assert not grounded.terminal(state_with_block_x(0.51))
        assert not grounded.terminal(state_with_block_x(0.52))
        # Block settles: three sub-eps deltas in a row terminate.
        settled = [state_with_block_x(0.52) for _ in range(4)]
        assert not grounded.terminal(settled[0])
        # Re-querying the SAME state must not stand in for physics steps.
        assert not grounded.terminal(settled[0])
        assert not grounded.terminal(settled[1])
        assert grounded.terminal(settled[2])
        # Re-initiating clears the tracking: a rerun of the same grounded
        # option must not terminate instantly on stale counts.
        assert grounded.initiable(settled[3])
        assert not grounded.terminal(settled[3])

    def test_wait_quiescence_survives_a_twin_resync(self, robot_scene):
        """Writing perception into the twin moves objects without the scene
        having moved.

        Counting that jolt as motion would zero the settle tally at
        every look, and on the real robot Wait would never see the scene
        rest.
        """
        from dataclasses import \
            replace  # pylint: disable=import-outside-toplevel
        _, robot = robot_scene
        config = replace(_make_config(robot),
                         wait_quiescence_eps=1e-4,
                         wait_quiescence_steps=3)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        block = Object("block0", _OBJ_TYPE)

        def state_with_block_z(z):
            return _build_state(robot_obj,
                                robot,
                                *_EE_HOME,
                                obj=block,
                                obj_xyz=(0.5, 0.0, z))

        grounded = opt.ground([robot_obj], np.zeros(0))
        assert grounded.initiable(state_with_block_z(0.475))
        # The first call only seeds the baseline; then two settled steps,
        # leaving the tally one short of the boundary.
        assert not grounded.terminal(state_with_block_z(0.475))
        assert not grounded.terminal(state_with_block_z(0.475))
        assert not grounded.terminal(state_with_block_z(0.475))
        # A look writes perception in, moving the block 4 mm -- far more than
        # the eps, so it would otherwise zero the tally.
        resynced = state_with_block_z(0.471)
        note_external_state_change(grounded, resynced)
        # The next settled step is still the boundary.
        assert grounded.terminal(state_with_block_z(0.471))

    def test_external_state_change_ignores_an_untracked_option(
            self, robot_scene):
        """Without quiescence tracking there is no tally to protect, so the
        hook has to leave the option alone rather than invent one."""
        _, robot = robot_scene
        opt = create_wait_option("Wait", _make_config(robot), _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)

        note_external_state_change(grounded, state)

        assert not grounded.memory

    def test_wait_quiescence_disabled_by_default(self, robot_scene):
        """Without wait_quiescence_eps the legacy never-terminate behavior
        holds even on a frozen scene."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        block = Object("block0", _OBJ_TYPE)
        grounded = opt.ground([robot_obj], np.zeros(0))
        for _ in range(6):
            state = _build_state(robot_obj,
                                 robot,
                                 *_EE_HOME,
                                 obj=block,
                                 obj_xyz=(0.5, 0.0, 0.0))
            assert not grounded.terminal(state)

    def test_wait_custom_name(self, robot_scene):
        """Test wait custom name."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Idle", config, _ROBOT_TYPE)
        assert opt.name == "Idle"

    def test_wait_default_name(self, robot_scene):
        """Test wait default name."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        assert opt.name == "Wait"

    def test_wait_policy_nudges_fingers_open(self, robot_scene):
        """When fingers are open, the action should nudge them more open."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE)
        action = grounded.policy(state)
        l_idx = robot.left_finger_joint_idx
        # Finger nudge should be positive (open direction).
        initial_fingers = state.joint_positions[l_idx]
        assert action.arr[l_idx] > initial_fingers

    def test_wait_policy_nudges_fingers_closed(self, robot_scene):
        """When fingers are closed, the action should nudge them more
        closed."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_CLOSED_STATE)
        action = grounded.policy(state)
        l_idx = robot.left_finger_joint_idx
        initial_fingers = state.joint_positions[l_idx]
        # Finger nudge should be negative (closed direction).
        assert action.arr[l_idx] < initial_fingers

    def test_wait_policy_action_within_bounds(self, robot_scene):
        """The action returned by wait must lie within the robot's action
        space."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        action = grounded.policy(state)
        assert robot.action_space.contains(action.arr)

    def test_wait_non_finger_joints_unchanged(self, robot_scene):
        """Wait must not move any joints except the two finger joints."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_wait_option("Wait", config, _ROBOT_TYPE)
        robot_obj = _make_robot_obj()
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _build_state(robot_obj, robot, *_EE_HOME)
        action = grounded.policy(state)
        l_idx = robot.left_finger_joint_idx
        r_idx = robot.right_finger_joint_idx
        for i, (act, orig) in enumerate(zip(action.arr,
                                            state.joint_positions)):
            if i not in (l_idx, r_idx):
                assert act == pytest.approx(orig, abs=1e-6), \
                    f"Joint {i} should not change in wait policy"


# ===========================================================================
# 6. make_move_to_phase
# ===========================================================================


class TestMakeMoveToPosePhase:
    """TestMakeMoveToPosePhase class."""

    def test_returns_phase_with_move_action_type(self):
        """Test returns phase with move action type."""
        phase = make_move_to_phase(
            "MoveTest",
            get_target_pose_fn=lambda s, o, p_, c: (1.0, 2.0, 3.0, 0.0),
            finger_status="open",
        )
        assert isinstance(phase, Phase)
        assert phase.action_type == PhaseAction.MOVE_TO_POSE
        assert phase.name == "MoveTest"
        assert phase.use_motion_planning is False  # default from CFG

    def test_explicit_open_finger_status(self, robot_scene):
        """Test explicit open finger status."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        phase = make_move_to_phase(
            "OpenMove",
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
            finger_status="open",
        )
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_CLOSED_STATE)  # state says closed
        _, _, returned_status = phase.target_fn(state, [robot_obj],
                                                np.zeros(0), config)
        assert returned_status == "open"  # explicit overrides state

    def test_explicit_closed_finger_status(self, robot_scene):
        """Test explicit closed finger status."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        phase = make_move_to_phase(
            "ClosedMove",
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
            finger_status="closed",
        )
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE)  # state says open
        _, _, returned_status = phase.target_fn(state, [robot_obj],
                                                np.zeros(0), config)
        assert returned_status == "closed"

    def test_inferred_open_finger_status(self, robot_scene):
        """When finger_status=None, infers 'open' from state with open
        fingers."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        phase = make_move_to_phase(
            "InferOpen",
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
            finger_status=None,
        )
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE)
        _, _, returned_status = phase.target_fn(state, [robot_obj],
                                                np.zeros(0), config)
        assert returned_status == "open"

    def test_inferred_closed_finger_status(self, robot_scene):
        """When finger_status=None, infers 'closed' from state with closed
        fingers."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        phase = make_move_to_phase(
            "InferClosed",
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
            finger_status=None,
        )
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_CLOSED_STATE)
        _, _, returned_status = phase.target_fn(state, [robot_obj],
                                                np.zeros(0), config)
        assert returned_status == "closed"

    def test_target_position_is_forwarded(self, robot_scene):
        """The target (x, y, z, yaw) from get_target_pose_fn is used."""
        _, robot = robot_scene
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        custom_target = (1.1, 2.2, 3.3, 0.5)
        phase = make_move_to_phase(
            "TargetCheck",
            get_target_pose_fn=lambda s, o, p_, c: custom_target,
        )
        state = _build_state(robot_obj, robot, *_EE_HOME)
        _, target_pose, _ = phase.target_fn(state, [robot_obj], np.zeros(0),
                                            config)
        assert target_pose.position == pytest.approx(custom_target[:3],
                                                     abs=1e-6)


# ===========================================================================
# 7. create_move_to_skill
# ===========================================================================


class TestCreateMoveToPoseSkill:
    """TestCreateMoveToPoseSkill class."""

    def test_returns_parameterized_option(self, robot_scene):
        """Test returns parameterized option."""
        _, robot = robot_scene
        config = _make_config(robot)
        opt = create_move_to_skill(
            "Move",
            [_ROBOT_TYPE],
            Box(0, 1, (0, )),
            config,
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
        )
        assert isinstance(opt, ParameterizedOption)
        assert opt.name == "Move"

    def test_policy_returns_valid_action(self, robot_scene):
        """Test policy returns valid action."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        config = _make_config(robot)
        robot_obj = _make_robot_obj()
        opt = create_move_to_skill(
            "Move",
            [_ROBOT_TYPE],
            Box(0, 1, (0, )),
            config,
            get_target_pose_fn=lambda s, o, p_, c: (*_EE_HOME, 0.0),
        )
        grounded = opt.ground([robot_obj], np.zeros(0))
        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)


# ===========================================================================
# 8. create_pick_skill — structure
# ===========================================================================


class TestCreatePickSkill:
    """TestCreatePickSkill class."""

    def _make_pick(self, robot):
        config = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            transport_z=0.8,
        )
        return create_pick_skill(
            name="Pick",
            types=[_ROBOT_TYPE, _OBJ_TYPE],
            config=config,
            get_target_pose_fn=lambda s, o, p_, c: (1.35, 0.75, 0.4, 0.0),
        )

    def test_returns_parameterized_option(self, robot_scene):
        """Test returns parameterized option."""
        _, robot = robot_scene
        opt = self._make_pick(robot)
        assert isinstance(opt, ParameterizedOption)
        assert opt.name == "Pick"

    def test_pick_policy_returns_valid_action(self, robot_scene):
        """Test pick policy returns valid action."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        robot_obj = _make_robot_obj()
        obj = _make_obj()
        opt = self._make_pick(robot)
        # Pick params: (grasp_z_offset) — use 0.02
        grounded = opt.ground([robot_obj, obj],
                              np.array([0.02], dtype=np.float32))
        state = _make_home_state(robot_obj,
                                 robot,
                                 obj=obj,
                                 obj_xyz=(1.35, 0.75, 0.4))
        grounded.initiable(state)
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)


# ===========================================================================
# 9. create_place_skill — structure
# ===========================================================================


class TestCreatePlaceSkill:
    """TestCreatePlaceSkill class."""

    def _make_place(self, robot):
        config = SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            transport_z=0.8,
        )
        return create_place_skill(
            name="Place",
            types=[_ROBOT_TYPE],
            config=config,
        )

    def test_returns_parameterized_option(self, robot_scene):
        """Test returns parameterized option."""
        _, robot = robot_scene
        opt = self._make_place(robot)
        assert isinstance(opt, ParameterizedOption)
        assert opt.name == "Place"

    def test_place_policy_returns_valid_action(self, robot_scene):
        """Test place policy returns valid action."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        robot_obj = _make_robot_obj()
        opt = self._make_place(robot)
        # Place params: (target_x, target_y, release_z, target_yaw)
        grounded = opt.ground([robot_obj],
                              np.array([0.75, 1.35, 0.55, 0.0],
                                       dtype=np.float32))
        state = _make_home_state(robot_obj, robot)
        grounded.initiable(state)
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)


# ===========================================================================
# 10. create_push_skill — structure
# ===========================================================================


class TestCreatePushSkill:
    """TestCreatePushSkill class."""

    @staticmethod
    def _make_push_config(robot):
        # robot_home_pos is required for create_push_skill
        return SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=_fingers_state_to_joint,
            robot_home_pos=_EE_HOME,
            transport_z=0.8,
        )

    def _make_push(self, robot):
        """Make push."""
        config = self._make_push_config(robot)
        return create_push_skill(
            name="Push",
            types=[_ROBOT_TYPE, _OBJ_TYPE],
            config=config,
            get_target_pose_fn=lambda s, o, p_, c: (1.35, 0.75, 0.4, 0.0),
        )

    def test_returns_parameterized_option(self, robot_scene):
        """Test returns parameterized option."""
        _, robot = robot_scene
        opt = self._make_push(robot)
        assert isinstance(opt, ParameterizedOption)
        assert opt.name == "Push"

    def test_push_policy_close_fingers_returns_valid_action(self, robot_scene):
        """First call lands in CloseFingers phase -> action within bounds."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123})
        robot_obj = _make_robot_obj()
        obj = _make_obj()
        opt = self._make_push(robot)
        # Push params: (approach_distance, contact_z_offset)
        grounded = opt.ground([robot_obj, obj],
                              np.array([0.05, 0.02], dtype=np.float32))
        state = _build_state(robot_obj,
                             robot,
                             *_EE_HOME,
                             finger_state=_OPEN_STATE,
                             obj=obj,
                             obj_xyz=(1.35, 0.75, 0.4))
        grounded.initiable(state)
        action = grounded.policy(state)
        assert isinstance(action, Action)
        assert robot.action_space.contains(action.arr)

    def test_ee_yaw_offset_comes_from_the_robot(self, robot_scene):
        """With no config override, the hand decides the push orientation."""
        _, robot = robot_scene
        utils.reset_config({"seed": 123, "skill_push_ee_yaw_offset": None})
        config = self._make_push_config(robot)
        # The fetch pushes with the 0.0 default.
        assert resolve_ee_yaw_offset(config) == robot.push_ee_yaw_offset == 0.0

    def test_ee_yaw_offset_config_override_wins(self, robot_scene):
        """Setting the flag forces one offset regardless of the robot."""
        _, robot = robot_scene
        utils.reset_config({
            "seed": 123,
            "skill_push_ee_yaw_offset": np.pi / 2
        })
        config = self._make_push_config(robot)
        assert resolve_ee_yaw_offset(config) == pytest.approx(np.pi / 2)
        assert robot.push_ee_yaw_offset == 0.0
        utils.reset_config({"seed": 123})

    def test_contact_phases_never_motion_planned(self, robot_scene):
        """Waypoint_2 (stroke) and Waypoint_3 (retreat) step IK straight at the
        target even when the config turns motion planning on.

        A collision-free planner asked for a goal pose inside the pushed
        object either fails or detours around it and strikes it from the
        wrong side, so only the free-space phases may follow the config.
        """
        _, robot = robot_scene
        utils.reset_config({
            "seed": 123,
            "skill_phase_use_motion_planning": True,
        })
        opt = self._make_push(robot)
        skill = opt.policy.__self__
        phases = {ph.name: ph for ph in skill._phases}  # pylint: disable=protected-access
        assert phases["Waypoint_0"].use_motion_planning
        assert phases["Waypoint_1"].use_motion_planning
        assert not phases["Waypoint_2"].use_motion_planning
        assert not phases["Waypoint_3"].use_motion_planning
        utils.reset_config({"seed": 123})


def test_fmt_option_params():
    """Params render compactly for failure messages, including empty."""
    assert _fmt_option_params(np.zeros(0, dtype=np.float32)) == "[]"
    assert _fmt_option_params(np.array([0.05, 0.02],
                                       dtype=np.float32)) == "[0.05, 0.02]"


class TestIkStallAbort:
    """Incremental-IK stall detection (_check_ik_stall)."""

    def _make_skill_and_phase(self, robot, target_pos):
        config = _make_config(robot)
        robot_obj = _make_robot_obj()

        def target_fn(state, _objects, _params, _cfg):
            x = state.get(robot_obj, "x")
            y = state.get(robot_obj, "y")
            z = state.get(robot_obj, "z")
            orn = p.getQuaternionFromEuler([0, np.pi / 2, -np.pi])
            return Pose((x, y, z), orn), Pose(target_pos, orn), "open"

        phase = Phase(
            name="Waypoint",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=target_fn,
            use_motion_planning=True,
            expect_contact=True,
        )
        skill = PhaseSkill("Push", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                           [phase])
        return skill, phase, robot_obj

    def test_stall_raises_after_window(self, robot_scene):
        """No end-effector progress for a full window aborts the option."""
        _, robot = robot_scene
        target = (_EE_HOME[0] + 0.5, _EE_HOME[1], _EE_HOME[2])
        skill, phase, robot_obj = self._make_skill_and_phase(robot, target)
        state = _build_state(robot_obj, robot, *_EE_HOME)
        memory: dict = {}
        params = np.zeros(0, dtype=np.float32)
        # First call initializes the best distance; the next window-1
        # no-progress calls only count up.
        for _ in range(PhaseSkill._ik_stall_window):  # pylint: disable=protected-access
            skill._check_ik_stall(phase, state, memory, [robot_obj], params)  # pylint: disable=protected-access
        with pytest.raises(utils.OptionExecutionFailure) as e:
            skill._check_ik_stall(phase, state, memory, [robot_obj], params)  # pylint: disable=protected-access
        assert "incremental-IK stalled" in str(e.value)
        # The message names the phase target and echoes the option params
        # (the agent's only channel for diagnosing which values failed).
        assert "m from the target (" in str(e.value)
        assert "commanded by params []" in str(e.value)

    def test_progress_resets_counter(self, robot_scene):
        """Steady progress toward the target never trips the abort."""
        _, robot = robot_scene
        target = (_EE_HOME[0] + 0.5, _EE_HOME[1], _EE_HOME[2])
        skill, phase, robot_obj = self._make_skill_and_phase(robot, target)
        memory: dict = {}
        params = np.zeros(0, dtype=np.float32)
        # 5 mm of progress per step (> _ik_stall_min_progress) for three
        # windows' worth of steps: no abort.
        for i in range(3 * PhaseSkill._ik_stall_window):  # pylint: disable=protected-access
            state = _build_state(robot_obj, robot, _EE_HOME[0] + 0.005 * i,
                                 _EE_HOME[1], _EE_HOME[2])
            skill._check_ik_stall(phase, state, memory, [robot_obj], params)  # pylint: disable=protected-access


# ---------------------------------------------------------------------------
# PhaseSkill._solve_goal_ik acceptance logic
# ---------------------------------------------------------------------------


class _FakeGoalIkRobot:
    """Scripted stand-in for the planning robot in goal-IK tests.

    The unvalidated one-shot IK returns an in-limits branch whose true
    forward kinematics misses the target by ``one_shot_error_m`` meters;
    validated IK returns a branch that hits the target exactly.
    """

    joint_lower_limits = [-3.0] * 7
    joint_upper_limits = [3.0] * 7
    initial_joint_positions = [0.0] * 7

    def __init__(self, target_pose: Pose, one_shot_error_m: float) -> None:
        self._target = target_pose
        self._one_shot_error_m = one_shot_error_m
        self.validated_calls = 0

    def set_joints(self, joints):
        """No-op; the fake tracks nothing."""

    def inverse_kinematics(self, target_pose, validate, set_joints=True):
        """Scripted joints; validated calls get the good solution."""
        del target_pose, set_joints  # scripted result
        if validate:
            self.validated_calls += 1
            return [0.1] * 7
        return [0.2] * 7

    def forward_kinematics(self, joints):
        """Good joints hit the target; others land short by the error."""
        x, y, z = self._target.position
        if joints == [0.1] * 7:
            return Pose((x, y, z))
        return Pose((x, y, z - self._one_shot_error_m))


class TestSolveGoalIk:
    """Every accepted goal config must hit the pose under FK."""

    def _make_skill(self, robot) -> PhaseSkill:
        config = _make_config(robot)
        phase = Phase(
            name="MoveToDrop",
            action_type=PhaseAction.MOVE_TO_POSE,
            target_fn=lambda *args: None,
            use_motion_planning=True,
        )
        return PhaseSkill("Place", [_ROBOT_TYPE], Box(0, 1, (0, )), config,
                          [phase])

    def test_inaccurate_one_shot_escalates_to_validated(self, robot_scene):
        """An in-limits one-shot whose FK misses by centimeters must be
        rejected and the same seed re-solved with validated IK.

        Regression test for run_20260716_133656: with
        ``pybullet_ik_validate False`` a 5.7 cm one-shot residual used
        to be accepted without any FK check, so BiRRT's goal collision
        check placed the carried domino inside the table and refused a
        valid Place.
        """
        _, robot = robot_scene
        skill = self._make_skill(robot)
        target = Pose((0.77, 1.34, 0.55))
        fake = _FakeGoalIkRobot(target, one_shot_error_m=0.057)
        result = skill._solve_goal_ik(  # pylint: disable=protected-access
            fake,
            target, [0.5] * 7,
            validate=False)
        assert result == [0.1] * 7
        assert fake.validated_calls == 1

    def test_accurate_one_shot_keeps_fast_path(self, robot_scene):
        """A one-shot within tolerance is accepted with no validated IK."""
        _, robot = robot_scene
        skill = self._make_skill(robot)
        target = Pose((0.77, 1.34, 0.55))
        fake = _FakeGoalIkRobot(target, one_shot_error_m=0.002)
        result = skill._solve_goal_ik(  # pylint: disable=protected-access
            fake,
            target, [0.5] * 7,
            validate=False)
        assert result == [0.2] * 7
        assert fake.validated_calls == 0

    def test_all_branches_inaccurate_raises(self, robot_scene):
        """When no branch hits the pose, goal IK raises instead of handing
        BiRRT a wrong goal configuration."""
        _, robot = robot_scene
        skill = self._make_skill(robot)
        target = Pose((0.77, 1.34, 0.55))
        fake = _FakeGoalIkRobot(target, one_shot_error_m=0.057)
        fake.forward_kinematics = lambda joints: Pose(  # type: ignore
            (target.position[0], target.position[1], target.position[2] - 0.057
             ))
        with pytest.raises(InverseKinematicsError):
            skill._solve_goal_ik(  # pylint: disable=protected-access
                fake,
                target, [0.5] * 7,
                validate=False)
