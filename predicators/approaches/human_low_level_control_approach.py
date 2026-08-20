"""A human low-level control approach that allows users to control the robot
end effector via keyboard input from the terminal.

Example usage:
python predicators/main.py --env pybullet_circuit \
    --approach human_low_level_control --seed 0 \
    --pybullet_max_vel_norm 0.1 \
    --pybullet_sim_steps_per_action 5 --use_gui
"""

import select
import sys
import termios
import tty
from typing import Any, Callable, List, Optional, Set

import numpy as np
import pybullet as p
from gym.spaces import Box

from predicators import utils
from predicators.approaches.base_approach import BaseApproach
from predicators.pybullet_helpers.controllers import \
    get_move_end_effector_to_pose_action
from predicators.pybullet_helpers.geometry import Pose
from predicators.pybullet_helpers.robots import SingleArmPyBulletRobot
from predicators.settings import CFG
from predicators.structs import Action, ParameterizedOption, Predicate, \
    State, Task, Type


class HumanLowLevelControlApproach(BaseApproach):
    """A human-in-the-loop approach for low-level robot control via keyboard.

    Unlike the HumanOptionControlApproach which selects high-level options via
    terminal, this approach generates raw joint position Actions based on
    keyboard input from the terminal.

    Key mappings:
        W/S: Forward/Backward (+/- X)
        A/D: Left/Right (+/- Y)
        Q/E: Up/Down (+/- Z)
        R/F: Tilt forward/backward (pitch)
        Z/X: Rotate wrist left/right (yaw)
        Space: Toggle gripper open/close
    """

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Track gripper state (True = open, False = closed)
        self._gripper_open = True
        # Cache for the robot instance
        self._robot: Optional[SingleArmPyBulletRobot] = None
        # Track if we've printed the instructions
        self._instructions_printed = False
        # Step counter for periodic status
        self._step_count = 0
        # Store original terminal settings for restoration
        self._original_terminal_settings: Optional[List[Any]] = None
        # Track last key to avoid repeated toggle
        self._last_gripper_toggle_step = -10

    @classmethod
    def get_name(cls) -> str:
        return "human_low_level_control"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_robot(self) -> SingleArmPyBulletRobot:
        """Get or create a robot instance for IK calculations."""
        if self._robot is None:
            self._robot = _get_shadow_robot_for_env()
        return self._robot

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Create a policy that reads keyboard input and generates actions."""
        del task, timeout  # Unused parameters

        # Print instructions once at the start
        self._print_instructions()

        # Set terminal to raw mode for non-blocking single-char input
        self._setup_terminal()

        def _policy(state: State) -> Action:
            try:
                return self._get_action_from_keyboard(state)
            except Exception as e:
                self._restore_terminal()
                raise e

        return _policy

    def _setup_terminal(self) -> None:
        """Set terminal to raw mode for single-character input."""
        try:
            self._original_terminal_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        except Exception:  # pylint: disable=broad-except
            # Terminal setup may fail in non-TTY environments
            self._original_terminal_settings = None

    def _restore_terminal(self) -> None:
        """Restore terminal to original settings."""
        if self._original_terminal_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN,
                                  self._original_terminal_settings)
            except Exception:  # pylint: disable=broad-except
                pass

    def _print_instructions(self) -> None:
        """Print keyboard control instructions to terminal."""
        if self._instructions_printed:
            return
        self._instructions_printed = True

        print("\n" + "=" * 60)
        print("HUMAN LOW-LEVEL CONTROL MODE")
        print("=" * 60)
        print("\n*** KEEP THIS TERMINAL WINDOW FOCUSED ***")
        print("*** Press keys here (not in PyBullet window) ***")
        print("\nKeyboard Controls:")
        print("  W/S  : Move Forward/Backward (+/- X)")
        print("  A/D  : Move Left/Right (+/- Y)")
        print("  Q/E  : Move Up/Down (+/- Z)")
        print("  R/F  : Tilt forward/backward (pitch)")
        print("  Z/X  : Rotate wrist left/right (yaw)")
        print("  Space: Toggle gripper open/close")
        print("  Ctrl+C: Exit")
        print("\nCurrent Settings:")
        print(f"  Move speed: {CFG.human_control_move_speed} m/step")
        print(f"  Rotation speed: {CFG.human_control_rot_speed} rad/step")
        print(f"  Max velocity: {CFG.pybullet_max_vel_norm} m/step")
        print("=" * 60 + "\n")

    def _get_pressed_key(self) -> Optional[str]:
        """Get the most recent pressed key from terminal (non-blocking).

        Drains the entire input buffer and returns only the LAST key
        pressed. This prevents input lag from buffered keystrokes when
        holding a key.
        """
        last_key: Optional[str] = None

        # Drain ALL available input, keeping only the last character
        while True:
            readable, _, _ = select.select([sys.stdin], [], [], 0)
            if not readable:
                break

            try:
                char = sys.stdin.read(1)
                if char:
                    last_key = char.lower()
                else:
                    break
            except Exception:  # pylint: disable=broad-except
                break

        return last_key

    def _get_action_from_keyboard(self, state: State) -> Action:
        """Read keyboard events and generate an action.

        Args:
            state: Current PyBullet state with simulator_state containing
                   joint_positions and physics_client_id

        Returns:
            Action containing joint positions for the robot
        """
        self._step_count += 1

        # Some envs (e.g. pybullet_blocks) pass simulator_state as a raw
        # joint-positions list rather than the dict that PyBulletEnv
        # builds. Only the mobile-base branch below needs
        # physics_client_id/robot_id, so look them up lazily.
        assert isinstance(state, utils.PyBulletState)
        assert state.simulator_state is not None
        sim_state_dict = (state.simulator_state if isinstance(
            state.simulator_state, dict) else {})
        physics_client_id = sim_state_dict.get("physics_client_id")

        # Get the most recent pressed key (only one key per frame)
        key = self._get_pressed_key()

        # Initialize deltas
        dx, dy, dz = 0.0, 0.0, 0.0
        d_tilt, d_wrist = 0.0, 0.0
        toggle_gripper = False

        # Process key
        if key is not None:
            if key == 'w':
                dx = CFG.human_control_move_speed
            elif key == 's':
                dx = -CFG.human_control_move_speed
            elif key == 'a':
                dy = CFG.human_control_move_speed
            elif key == 'd':
                dy = -CFG.human_control_move_speed
            elif key == 'q':
                dz = CFG.human_control_move_speed
            elif key == 'e':
                dz = -CFG.human_control_move_speed
            elif key == 'r':
                d_tilt = CFG.human_control_rot_speed
            elif key == 'f':
                d_tilt = -CFG.human_control_rot_speed
            elif key == 'z':
                d_wrist = CFG.human_control_rot_speed
            elif key == 'x':
                d_wrist = -CFG.human_control_rot_speed
            elif key == ' ':
                # Debounce gripper toggle (at least 5 steps apart)
                if self._step_count - self._last_gripper_toggle_step > 5:
                    toggle_gripper = True
                    self._last_gripper_toggle_step = self._step_count

        # Handle gripper toggle
        if toggle_gripper:
            self._gripper_open = not self._gripper_open
            gripper_state = "OPEN" if self._gripper_open else "CLOSED"
            print(f"[Step {self._step_count}] Gripper: {gripper_state}")

        # Get current joint positions
        current_joint_positions = state.joint_positions

        # Check if there's any spatial movement
        spatial_movement = (dx != 0 or dy != 0 or dz != 0 or d_tilt != 0
                            or d_wrist != 0)

        # If only toggling gripper with no spatial movement, handle it directly
        if toggle_gripper and not spatial_movement:
            # Get robot for accessing finger joint indices
            robot = self._get_robot()
            # Directly set finger joints to fully open/closed for fast response
            # The motor controllers will move the fingers as fast as possible
            action_arr = np.array(current_joint_positions, dtype=np.float32)
            target_finger_pos = (robot.open_fingers if self._gripper_open else
                                 robot.closed_fingers)
            action_arr[robot.left_finger_joint_idx] = target_finger_pos
            action_arr[robot.right_finger_joint_idx] = target_finger_pos
            return self._pad_base_action(action_arr)

        # If no movement at all, return true no-op to prevent drift
        if not spatial_movement and not toggle_gripper:
            # Return current joint positions as-is (no IK, no drift)
            action_arr = np.array(current_joint_positions, dtype=np.float32)
            return self._pad_base_action(action_arr)

        # Get robot for IK
        robot = self._get_robot()
        if hasattr(robot, "base_action_dim") and robot.base_action_dim > 0:
            robot_id = sim_state_dict.get("robot_id")
            if robot_id is not None:
                if physics_client_id is None:
                    raise ValueError(
                        "physics_client_id not found in simulator_state; "
                        "mobile-base envs must populate it.")
                base_pos, base_orn = p.getBasePositionAndOrientation(
                    robot_id, physicsClientId=physics_client_id)
                robot.set_base_pose(  # type: ignore[attr-defined]
                    Pose(base_pos, base_orn))

        # Get current EE pose from forward kinematics on the shadow
        # robot. This avoids relying on env-specific state feature
        # names (some envs use x/y/z, others pose_x/pose_z, cover has
        # no y at all) and gives the true world-frame pose.
        current_pose = robot.forward_kinematics(current_joint_positions)
        current_x, current_y, current_z = current_pose.position

        target_x = current_x + dx
        target_y = current_y + dy
        target_z = current_z + dz

        # Apply tilt/wrist deltas to the current orientation. tilt = pitch,
        # wrist = yaw. Roll is preserved.
        if d_tilt or d_wrist:
            roll, pitch, yaw = p.getEulerFromQuaternion(
                current_pose.orientation)
            target_orn = p.getQuaternionFromEuler(
                [roll, pitch + d_tilt, yaw + d_wrist])
        else:
            target_orn = current_pose.orientation

        target_pose = Pose((target_x, target_y, target_z), target_orn)

        # Finger status
        finger_status = "open" if self._gripper_open else "closed"

        # Print movement feedback
        if key is not None and key != ' ':
            print(f"[Step {self._step_count}] Key: '{key}' | "
                  f"Pos: ({current_x:.2f}, {current_y:.2f}, {current_z:.2f})")

        # Generate action via IK
        try:
            action = get_move_end_effector_to_pose_action(
                robot=robot,
                current_joint_positions=current_joint_positions,
                current_pose=current_pose,
                target_pose=target_pose,
                finger_status=finger_status,
                max_vel_norm=CFG.pybullet_max_vel_norm,
                finger_action_nudge_magnitude=1e-3,
                validate=CFG.pybullet_ik_validate,
            )
        except utils.OptionExecutionFailure:
            # IK failed, return no-op action
            action_arr = np.array(current_joint_positions, dtype=np.float32)
            action = self._pad_base_action(action_arr)

        # Validate action is within action space bounds
        if not self._action_space.contains(action.arr):
            print(f"[Step {self._step_count}] "
                  "Warning: Action out of bounds,"
                  " staying in place")
            action_arr = np.array(current_joint_positions, dtype=np.float32)
            action = self._pad_base_action(action_arr)

        return action

    def __del__(self) -> None:
        """Restore terminal settings on cleanup."""
        self._restore_terminal()

    def _pad_base_action(self, action_arr: np.ndarray) -> Action:
        """Pad action with zero base deltas when the action space expects
        it."""
        extra_dim = self._action_space.shape[0] - action_arr.shape[0]
        if extra_dim > 0:
            zeros = np.zeros(extra_dim, dtype=np.float32)
            action_arr = np.concatenate([action_arr, zeros])
        action_arr = np.clip(action_arr, self._action_space.low,
                             self._action_space.high)
        return Action(action_arr)


def _get_shadow_robot_for_env() -> SingleArmPyBulletRobot:
    """Create a shadow robot for IK calculations.

    IK is base-pose-dependent (each env may translate/rotate the fetch's
    base), so we instantiate a fetch at the active env's base pose. We
    deliberately bypass each subclass's ``initialize_pybullet`` override
    to avoid loading env-specific bodies (tables/blocks/fans/etc.) that
    the IK does not need; we just connect a fresh DIRECT client, drop a
    ground plane, and ask the env class for a robot.
    """
    # pylint: disable=import-outside-toplevel
    from predicators.envs.base_env import BaseEnv
    from predicators.envs.pybullet_env import PyBulletEnv

    # pylint: enable=import-outside-toplevel

    env_name = CFG.env
    env_cls = None
    for cls in utils.get_all_subclasses(BaseEnv):
        if cls.__abstractmethods__:
            continue
        if not issubclass(cls, PyBulletEnv):
            continue
        if cls.get_name() == env_name:
            env_cls = cls
            break
    if env_cls is None:
        raise NotImplementedError(
            f"human_low_level_control: no PyBulletEnv subclass registered "
            f"for env name {env_name!r}.")

    physics_client_id = p.connect(p.DIRECT)
    p.resetSimulation(physicsClientId=physics_client_id)
    p.loadURDF(utils.get_env_asset_path("urdf/plane.urdf"), [0, 0, 0],
               useFixedBase=True,
               physicsClientId=physics_client_id)
    p.setGravity(0., 0., -10., physicsClientId=physics_client_id)
    return env_cls._create_pybullet_robot(  # pylint: disable=protected-access
        physics_client_id)
