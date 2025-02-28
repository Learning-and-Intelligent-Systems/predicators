"""Ground-truth options for the Kitchen environment."""

from typing import ClassVar, Dict, Sequence, Set

import numpy as np
from gym.spaces import Box
import mujoco # for quaternion operations

from predicators.envs.robo_kitchen import RoboKitchenEnv
from predicators.ground_truth_models import GroundTruthOptionFactory
from predicators.pybullet_helpers.geometry import Pose3D
from predicators.structs import Action, Array, GroundAtom, Object, \
    ParameterizedOption, ParameterizedTerminal, Predicate, State, Type


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
    def get_options(cls, env_name: str, types: Dict[str, Type],
                    predicates: Dict[str, Predicate],
                    action_space: Box) -> Set[ParameterizedOption]:

        # assert _MJKITCHEN_IMPORTED, "See kitchen.py"

        # Define quaternions using MuJoCo's utilities
        down_quat = np.zeros(4)
        mujoco.mju_euler2Quat(down_quat, np.array([-np.pi, 0.0, -np.pi / 2]), 'xyz')

        # End effector facing forward (e.g., toward the knobs.)
        fwd_quat = np.zeros(4)
        mujoco.mju_euler2Quat(fwd_quat, np.array([-np.pi / 2, 0.0, -np.pi / 2]), 'xyz')

        # Angled quaternion
        angled_quat = np.zeros(4)
        mujoco.mju_euler2Quat(angled_quat, np.array([-3 * np.pi / 4, 0.0, -np.pi / 2]), 'xyz')

        # Types
        gripper_type = types["gripper_type"]
        on_off_type = types["rich_object_type"]
        kettle_type = types["rich_object_type"]
        surface_type = types["rich_object_type"]
        switch_type = types["rich_object_type"]
        knob_type = types["rich_object_type"]
        hinge_type = types["hinge_type"]

        # Predicates
        # OnTop = predicates["OnTop"]

        options: Set[ParameterizedOption] = set()

        # Dummy initiable: always returns True.
        def _Dummy_initiable(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
            return True

        # Dummy policy: returns a zero action (adjust the dimension as needed).
        def _Dummy_policy(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Action:
            # Here we assume an action vector of size 7.
            return Action(np.zeros(7, dtype=np.float32))

        # Dummy terminal: always returns True.
        def _Dummy_terminal(state: State, memory: dict, objects: Sequence[Object], params: Array) -> bool:
            return True

        # Creating a dummy option.
        DummyOption = ParameterizedOption(
            "DummyOption",
            types=[],  # Adjust type requirements as needed.
            params_space=Box(-1.0, 1.0, (3, )),  # Example parameter space.
            policy=_Dummy_policy,
            initiable=_Dummy_initiable,
            terminal=_Dummy_terminal
        )
        
        # Add the dummy option to the set of options.
        options.add(DummyOption)

        return options
