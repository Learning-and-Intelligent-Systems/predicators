"""Ground-truth options for the Kitchen environment."""

from typing import ClassVar, Dict, Sequence, Set

import numpy as np
import os
from gym.spaces import Box
import mujoco  # for quaternion operations

from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.structs import Action, Array, GroundAtom, Object, ParameterizedOption, ParameterizedTerminal, Predicate, State, Type

import torch
from predicators.DS_models.gen_demo_model import DynamicalSystem

from scipy.spatial.transform import Rotation as R

class RoboKitchenGroundTruthOptionFactory(GroundTruthOptionFactory):
    """Ground-truth options for the RoboKitchen environment."""

    moveto_tol: ClassVar[float] = 0.03  # for terminating moving
    max_delta_mag: ClassVar[float] = 1.0  # don't move more than this per step
    max_push_mag: ClassVar[float] = 0.05  # for pushing forward
    # A reasonable home position for the end effector.
    home_pos: ClassVar[Pose3D] = (0.0, 0.37, 2.1)
    # Keep pushing a bit even if the On classifier holds.
    push_lr_thresh_pad: ClassVar[float] = 0.02
    push_microhandle_thresh_pad: ClassVar[float] = 0.02
    turn_knob_tol: ClassVar[float] = 0.02  # for twisting the knob

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"robo_kitchen"}

    @classmethod
    def get_options(cls, env_name: str, types: Dict[str, Type], predicates: Dict[str, Predicate], action_space: Box) -> Set[ParameterizedOption]:

        # assert _MJKITCHEN_IMPORTED, "See kitchen.py"

        # Define quaternions using MuJoCo's utilities
        down_quat = np.zeros(4)
        mujoco.mju_euler2Quat(down_quat, np.array([-np.pi, 0.0, -np.pi / 2]), "xyz")

        # End effector facing forward (e.g., toward the knobs.)
        fwd_quat = np.zeros(4)
        mujoco.mju_euler2Quat(fwd_quat, np.array([-np.pi / 2, 0.0, -np.pi / 2]), "xyz")

        # Angled quaternion
        angled_quat = np.zeros(4)
        mujoco.mju_euler2Quat(angled_quat, np.array([-3 * np.pi / 4, 0.0, -np.pi / 2]), "xyz")

        # Types
        hinge = types["hinge_type"]
        gripper = types["gripper_type"]
        handle = types["handle_type"]
        base = types["base_type"]

        options: Set[ParameterizedOption] = set()

        # DS_move_option - always initiable, empty policy, never terminates
        def _DS_move_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            if "model" not in memory:
                # Define model architecture
                class SimpleDS(torch.nn.Module):
                    def __init__(self):
                        super().__init__()
                        self.D = torch.nn.Parameter(torch.zeros(3, 3))

                    def forward(self, x):
                        if isinstance(x, np.ndarray):
                            x = torch.from_numpy(x).float()
                        if len(x.shape) == 1:
                            x = x.unsqueeze(0)
                        return torch.matmul(x, self.D.T).squeeze(0)

                # Create model and load state dict
                model = SimpleDS()
                current_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(current_dir, "..", "..", "DS_models", "models", "model.pt")
                model.load_state_dict(torch.load(model_path, map_location="cpu"))
                model.eval()
                memory["model"] = model

                gripper, handle, base = objects
                handle_quat = np.array([state.get(handle, "qx"), state.get(handle, "qy"), state.get(handle, "qz"), state.get(handle, "qw")])
                handle_pos = np.array([state.get(handle, "x"), state.get(handle, "y"), state.get(handle, "z")])
                memory["handle_init_rot"] = R.from_quat(handle_quat).as_matrix()
                memory["handle_init_pos"] = handle_pos


            return True

        def vee_operator(w):
            return np.array([w[2, 1], w[0, 2], w[1, 0]])


        def _DS_move_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            # Get objects
            gripper, _, base = objects
            handle_init_pos = memory["handle_init_pos"]
            handle_init_rot = memory["handle_init_rot"]
            # Get positions
            gripper_pos = np.array([state.get(gripper, "x"), state.get(gripper, "y"), state.get(gripper, "z")])
            gripper_quat = np.array([state.get(gripper, "qx"), state.get(gripper, "qy"), state.get(gripper, "qz"), state.get(gripper, "qw")])
            gripper_rot = R.from_quat(gripper_quat).as_matrix()
            # handle_pos = np.array([state.get(handle, "x"), state.get(handle, "y"), state.get(handle, "z")]
            # Compute relative position in world frame
            rel_pos_world = gripper_pos - handle_init_pos
            # Transform relative position to handle frame
            pos_in_handle = handle_init_rot.T @ rel_pos_world
            # Transform gripper rotation to handle frame
            rot_in_handle = handle_init_rot.T @ gripper_rot

            expected_relative_rot_handle = R.from_quat(np.array([0.5, 0.5, 0.5, -0.5]))

            # Compute the difference between the expected relative rotation and the actual relative rotation
            rel_rot_diff = expected_relative_rot_handle * R.from_matrix(rot_in_handle).inv()
            angular_w_handle = vee_operator(rel_rot_diff.as_matrix())
            # move that difference to the base frame
            world_w = handle_init_rot @ angular_w_handle

            robot_base_pos = np.array([state.get(base, "x"), state.get(base, "y"), state.get(base, "z")])
            robot_base_quat = np.array([state.get(base, "qx"), state.get(base, "qy"), state.get(base, "qz"), state.get(base, "qw")])
            robot_base_rot = R.from_quat(robot_base_quat).as_matrix()

            robot_base_w = robot_base_rot.T @ world_w
            mag = np.linalg.norm(robot_base_w)
            if mag > 1:
                robot_base_w = robot_base_w / mag

            net = memory["model"]
            with torch.no_grad():
                velocity_in_handle = net(torch.from_numpy(pos_in_handle).float())

            # Transform velocity back to world frame
            # Since handle frame is just translated, velocity transforms directly
            velocity_world = handle_init_rot @ velocity_in_handle.numpy()
            velocity_robot_base = robot_base_rot.T @ velocity_world
            # Create action array
            arr = np.zeros(7, dtype=np.float32)
            arr[:3] = velocity_robot_base
            arr[3:6] = 0.3 * robot_base_w

            # Clip the action to the action space limits
            action_low = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
            action_high = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
            arr = np.clip(arr, action_low, action_high)

            return Action(arr)

        def _DS_move_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Never terminates
            return False

        DS_move_option = ParameterizedOption(
            "DS_move_option",
            types=[gripper, handle, base],
            # Unused params
            params_space=Box(-5, 5, (1,)),
            policy=_DS_move_option_policy,
            initiable=_DS_move_option_initiable,
            terminal=_DS_move_option_terminal,
        )

        options.add(DS_move_option)

        # GripperOpen_option
        def _GripperOpen_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Always initiable
            return True

        def _GripperOpen_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            return Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32))  # Open gripper

        def _GripperOpen_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Never terminates
            return RoboKitchenEnv._GripperOpen_holds(state, objects)  # Check if gripper is open

        GripperOpen_option = ParameterizedOption(
            "GripperOpen_option",
            types=[gripper],  # Adjust type requirements as needed.
            params_space=Box(-5, 5, (1,)),
            policy=_GripperOpen_option_policy,
            initiable=_GripperOpen_option_initiable,
            terminal=_GripperOpen_option_terminal,
        )
        options.add(GripperOpen_option)

        # GripperClose_option
        def _GripperClose_option_initiable(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Always initiable
            return True

        def _GripperClose_option_policy(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> Action:
            return Action(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float32))  # Close gripper

        def _GripperClose_option_terminal(state: State, memory: Dict, objects: Sequence[Object], params: Array) -> bool:
            # Never terminates
            return RoboKitchenEnv._GripperClosed_holds(state, objects)  # Check if gripper is closed

        GripperClose_option = ParameterizedOption(
            "GripperClose_option",
            types=[gripper],  # Adjust type requirements as needed.
            params_space=Box(-5, 5, (1,)),
            policy=_GripperClose_option_policy,
            initiable=_GripperClose_option_initiable,
            terminal=_GripperClose_option_terminal,
        )

        options.add(GripperClose_option)

        # Dummy initiable: always returns True.
        def _Dummy_initiable(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
            return True

        def _Dummy_policy(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Action:
            # Here we assume an action vector of size 7.
            return Action(np.zeros(7, dtype=np.float32))

        def _Dummy_terminal(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
            return True

        DummyOption = ParameterizedOption(
            "DummyOption",
            types=[],  # Adjust type requirements as needed.
            params_space=Box(-1.0, 1.0, (3,)),  # Example parameter space.
            policy=_Dummy_policy,
            initiable=_Dummy_initiable,
            terminal=_Dummy_terminal,
        )

        options.add(DummyOption)

        return options
