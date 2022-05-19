import glob
import importlib.util
import logging
import os
import sys
from typing import Sequence, List

import numpy as np
import pybullet as p

from predicators.src import utils
from predicators.src.settings import CFG
from predicators.src.structs import JointsState, Pose3D

"""
This implementation is heavily based on the pybullet-planning repository
by Caelan Garrett (https://github.com/caelan/pybullet-planning/).
"""


def _install_ikfast_module(ikfast_dir: str) -> None:
    """One-time install an IKFast module for a specific robot.

    Assumes there is a subdirectory in envs/assets/ikfast with a
    setup.py file for the robot. See the panda_arm subdirectory for an
    example.
    """
    # TODO: consider directly calling setup.py instead of using an OS command.
    cmds = [
        # Go to the subdirectory with the setup.py file.
        f"cd {ikfast_dir}",
        # Run the setup.py file.
        "python setup.py",
    ]
    # Execute the command.
    cmd = "; ".join(cmds)
    os.system(cmd)


def _install_ikfast_if_required(robot_name: str, module_name: str) -> str:
    """
    If IKFast has been previously installed, there should be a file with
    extension .so, starting with name module_name, in the ikfast_dir.

    We check if this file exists, if not we install IKFast to compile it.
    """
    ikfast_dir = os.path.join(utils.get_env_asset_path("ikfast"), robot_name)
    glob_pattern = os.path.join(ikfast_dir, f"{module_name}*.so")
    so_filepaths = glob.glob(glob_pattern)
    if len(so_filepaths) > 1:
        raise ValueError("More than one .so file found for IKFast.")

    # We need to install.
    if not so_filepaths:
        logging.warning(f"IKFast module for {robot_name} not found; installing.")
        _install_ikfast_module(ikfast_dir)
        so_filepaths = glob.glob(glob_pattern)
        if len(so_filepaths) != 1:
            raise RuntimeError(
                f"Encountered {len(so_filepaths)} .so files after installing IKFast."
            )

    module_filepath = so_filepaths[0]
    return module_filepath


def _import_ikfast(robot_name: str):
    """
    Imports the MoveIt IKFast solver for the given robot. If the solver is not
    already installed, it will be installed automatically when this function is
    called for the first time.
    """
    module_name = f"ikfast_{robot_name}"
    if module_name in sys.modules:
        # IKFast already imported for robot so just return it.
        ikfast = sys.modules[module_name]
        return ikfast

    # Otherwise, we load the IKFast module and build it if it isn't already built.
    module_filepath = _install_ikfast_if_required(robot_name, module_name)

    # Import the module.
    # See https://docs.python.org/3/library/importlib.html.
    spec = importlib.util.spec_from_file_location(module_name, module_filepath)
    assert spec is not None, "IKFast module could not be found."
    ikfast = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = ikfast
    spec.loader.exec_module(ikfast)
    logging.debug(f"Loaded IKFast module for {robot_name} from {module_filepath}")

    return ikfast


def get_ikfast_supported_robots() -> List[str]:
    ikfast_base_dir = utils.get_env_asset_path("ikfast")
    return [
        path
        for path in os.listdir(ikfast_base_dir)
        if os.path.isdir(os.path.join(ikfast_base_dir, path)) and path != "__pycache__"
    ]


def ikfast_inverse_kinematics(
    robot_name: str,
    target_position: Pose3D,
    target_orientation: Sequence[float],
    physics_client_id: int,
) -> JointsState:
    """Runs IK and returns a joints state.

    TODO: describe the assumptions about the target position and orientation
    in terms of what joints they're referring to.

    Note that this will automatically compile IKFast for the given robot if it
    hasn't been compiled already when this function is called for the first time.

    """
    ikfast = _import_ikfast(robot_name)

    # IKFast expects matrix representation of orientation.
    matrix_target_orn = (
        np.array(
            p.getMatrixFromQuaternion(
                target_orientation, physicsClientId=physics_client_id
            )
        )
        .reshape((3, 3))
        .tolist()
    )
    # TODO: understand third argument. Waiting for Caelan reply on this.
    # This is a temporary thing until I understand what the hell is happening.
    # ipdb> self._joint_lower_limits[6]
    # -2.8973
    # ipdb> self._joint_upper_limits[6]
    # 2.8973
    rng = np.random.default_rng(CFG.seed)
    for _ in range(CFG.pybullet_max_ik_iters):
        what_is_this_thing = rng.uniform(-2.8973, 2.8973)
        solutions = ikfast.get_ik(
            matrix_target_orn, list(target_position), [what_is_this_thing]
        )
        if solutions:
            break
    assert solutions
    return solutions[0]
